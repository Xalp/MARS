"""
Causal Self-Block Diffusion Language Model Trainer V2 (CSBD3LM-v2)
Difference from v1: Noisy Block is strictly CAUSAL (no reverse reasoning).
"""

import torch
from functools import partial
import transformers
from torch.nn.attention.flex_attention import create_block_mask

from .csbd3lm import CSBD3LMTrainer

def csbd_v2_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the CSBD3LM-v2 attention mask.
    
    Structure:
    - Clean Context: Strict Causal (Yellow)
    - Noisy Block: Strict Causal (Red) <- CHANGED from Bidirectional
    - Noisy sees Clean: Inter-Block Causal (Blue)
    
    This effectively makes the entire graph Causal, just with different "offsets" for Noisy.
    """

    # Indicate whether token belongs to Noisy (0) or Clean (1)
    x0_flag_q = (q_idx >= n).int()
    x0_flag_kv = (kv_idx >= n).int()
    
    # Logical positions (aligned to 0..L-1)
    pos_q = torch.where(x0_flag_q == 1, q_idx - n, q_idx)
    pos_kv = torch.where(x0_flag_kv == 1, kv_idx - n, kv_idx)

    block_q = pos_q // block_size
    block_kv = pos_kv // block_size

    # **1. Clean Context Self-Attention (Yellow)**
    # Strict Causal
    clean_causal = (x0_flag_q == 1) & (x0_flag_kv == 1) & (pos_q >= pos_kv)
    
    # **2. Noisy Block Self-Attention (Red - Modified)**
    # Rule: Causal (Token level) AND same block
    # Previously was: (block_q == block_kv)
    # Now: (block_q == block_kv) & (pos_q >= pos_kv)
    noisy_block_causal = (x0_flag_q == 0) & (x0_flag_kv == 0) & (block_q == block_kv) & (pos_q >= pos_kv)

    # **3. Noisy sees Clean Context (Blue)**
    # Rule: Inter-Block Causal (Fixed, No Leakage)
    noisy_sees_clean = (x0_flag_q == 0) & (x0_flag_kv == 1) & (block_q > block_kv)

    # **4. Combine Masks**
    return clean_causal | noisy_block_causal | noisy_sees_clean


class CSBD3LMv2Trainer(CSBD3LMTrainer):
    """
    Trainer for Causal Self-Block Diffusion V2.
    Removes bidirectional attention in Noisy blocks.
    """
    
    def compute_loss(
        self,
        model: transformers.PreTrainedModel,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        # We reuse almost everything from CSBD3LM, but swap the mask function.
        # Since logic is inside compute_loss, we must copy-paste or duplicate structure.
        # To ensure we have the fix, we copy the current state of CSBD3LM logic.
        
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        token_cnt_per_seq = torch.sum(labels != -100, dim=1, keepdim=True)

        # === 1. Sample diffusion timesteps ===
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # === 2. Apply stochastic masking ===
        masked_indices = (torch.rand((b, l), device=input_ids.device) < p_mask) & (
            labels != -100
        )
        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass (CSBD3LM v2 Mask Logic) ===
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            attention_mask = csbd_v2_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l * 2)[:, None],
                kv_idx=torch.arange(l * 2)[None, :],
                block_size=self.block_size,
                n=l,
            )
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l)
            )
            attention_mask = attention_mask.to(input_ids.device)
        elif (
            self.accelerator.unwrap_model(model).config._attn_implementation
            == "flex_attention"
        ):
            attention_mask = create_block_mask(
                partial(csbd_v2_mask, block_size=self.block_size, n=l),
                B=None,
                H=None,
                Q_LEN=l * 2,
                KV_LEN=l * 2,
            )
        else:
            raise NotImplementedError

        base_pos = (
            torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        )
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits
        logits = logits[:, :l]

        # === 4. Degenerate cases ===
        if not masked_indices.any():
            self.epoch_meter.update(
                split="train" if model.training else "eval", 
                nll_sum=logits.sum() * 0.0, 
                token_cnt=token_cnt_per_seq.sum(),
            )
            return (
                (logits.sum() * 0.0, outputs) if return_outputs else logits.sum() * 0.0
            )

        # === 5. Loss weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_indices=masked_indices
        )

        # === 6. Cross-entropy ===
        # Handle potential mismatch between input_ids and labels (e.g. EOS vs IM_END)
        # We trust labels as the ground truth if they are not -100.
        mismatch = (input_ids[masked_indices] != labels[masked_indices])
        if mismatch.any():
            input_ids[masked_indices] = labels[masked_indices]

        token_loss = torch.nn.functional.cross_entropy(
            logits[masked_indices], labels[masked_indices], reduction="none"
        )
        token_loss = token_loss * loss_weights[masked_indices]

        # === 7. Normalize ===
        if self.loss_normalization_type == "batch":
            token_loss_normalized = token_loss / b
        elif self.loss_normalization_type == "sequence":
            token_loss_normalized = token_loss / token_cnt_per_seq.expand(-1, l)[masked_indices] / b
        elif self.loss_normalization_type == "token":
            token_loss_normalized = token_loss / token_cnt_per_seq.sum()
        else:
            raise ValueError("Invalid loss_normalization_type.")
        loss = token_loss_normalized.sum()

        self.epoch_meter.update(
            split="train" if model.training else "eval", 
            nll_sum=token_loss.sum(), 
            token_cnt=token_cnt_per_seq.sum(),
        )
        return (loss, outputs) if return_outputs else loss
