"""
Causal Self-Block Diffusion Language Model Trainer (CSBD3LM)
"""

import torch
from functools import partial
import transformers
from torch.nn.attention.flex_attention import create_block_mask

from .bd3lm import BD3LMTrainer

def csbd_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the Causal Block Diffusion attention mask.
    
    Structure:
    - Clean Context (kv_idx >= n, q_idx >= n): Strict Causal (Yellow)
    - Noisy Block (kv_idx < n, q_idx < n): Block Diagonal Bidirectional (Orange)
    - Noisy sees Clean (kv_idx >= n, q_idx < n): Block Causal (Blue)
      - Noisy Block i sees Clean Tokens strictly before the start of Block i.
    - Clean sees Noisy: Forbidden.
    
    Args:
        n: Sequence length (split point between Noisy and Clean).
    """

    # Indicate whether token belongs to Noisy (0) or Clean (1)
    x0_flag_q = (q_idx >= n).int()
    x0_flag_kv = (kv_idx >= n).int()
    
    # Logical positions (aligned to 0..L-1)
    # If Clean: pos = idx - n
    # If Noisy: pos = idx
    pos_q = torch.where(x0_flag_q == 1, q_idx - n, q_idx)
    pos_kv = torch.where(x0_flag_kv == 1, kv_idx - n, kv_idx)

    # Compute block indices based on logical positions
    block_q = pos_q // block_size
    block_kv = pos_kv // block_size

    # **1. Clean Context Self-Attention (Yellow)**
    # Rule: Strict Causal (Token level)
    # q is Clean, k is Clean.
    clean_causal = (x0_flag_q == 1) & (x0_flag_kv == 1) & (pos_q >= pos_kv)
    
    # **2. Noisy Block Self-Attention (Orange)**
    # Rule: Block Diagonal (Bidirectional within block)
    # q is Noisy, k is Noisy.
    noisy_block_bi = (x0_flag_q == 0) & (x0_flag_kv == 0) & (block_q == block_kv)

    # **3. Noisy sees Clean Context (Blue)**
    # Rule: Block Causal (Inter-Block only)
    # Noisy tokens should ONLY see clean tokens from STRICTLY PREVIOUS blocks.
    # Allowing them to see clean tokens in the current block (even if previous) is LEAKAGE
    # because at inference time, the current block's clean tokens do not exist yet.
    noisy_sees_clean = (x0_flag_q == 0) & (x0_flag_kv == 1) & (block_q > block_kv)

    # **4. Combine Masks**
    return clean_causal | noisy_block_bi | noisy_sees_clean


class CSBD3LMTrainer(BD3LMTrainer):
    """
    Trainer for Causal Self-Block Diffusion.
    """
    
    def compute_loss(
        self,
        model: transformers.PreTrainedModel,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        # We override compute_loss ONLY to change the mask generation logic.
        # But compute_loss in BD3LM is long. 
        # To avoid code duplication, we can monkey-patch or copy-paste.
        # Given the instruction "copy many things", copy-paste is safer/clearer.
        
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

        # === 3. Forward pass (CSBD3LM Mask Logic) ===
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            attention_mask = csbd_mask(
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
                partial(csbd_mask, block_size=self.block_size, n=l),
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
