"""
Samplers for CSBD3LM (v1 and v2).
"""

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import SamplerOutput
from dllm.core.samplers.bd3lm import BD3LMSampler, BD3LMSamplerConfig, diffusion_step_block
from dllm.core.samplers.utils import get_num_transfer_tokens

def build_csbd_attention_mask(
    x: torch.Tensor,
    block_size: int,
    pad_token_id: int,
    start_pos_noisy: int, # The position where Noisy block starts
    is_v2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build attention mask for CSBD3LM inference.
    
    Args:
        x: [B, T] input ids.
        start_pos_noisy: The index in sequence where the 'current generation block' starts.
                         Indices < start_pos_noisy are Context (Clean).
                         Indices >= start_pos_noisy are Generation (Noisy).
    
    Rules:
    1. Clean Context (< start_pos_noisy): Strict Causal.
    2. Noisy Generation (>= start_pos_noisy):
       - If is_v2=False: Block-Bidirectional (can see other Noisy in same block).
       - If is_v2=True: Strict Causal.
    3. Cross (Noisy -> Clean):
       - Inter-Block Only: Noisy at Block B can only see Clean at Block < B.
       - Any Clean token in Block B is invisible.
    """
    B, T = x.shape
    device = x.device

    # Per-sample valid mask (exclude padding)
    valid = x != pad_token_id
    
    # RoPE position_ids: logical positions 0..T-1 (ignoring padding gaps)
    pos_raw = torch.cumsum(valid.to(torch.long), dim=-1)
    logical_pos = pos_raw - 1
    position_ids = torch.where(valid, logical_pos, torch.zeros_like(logical_pos)).to(device=device, dtype=torch.long)

    # Physical positions and Blocks
    pos = torch.arange(T, device=device)
    block_ids = torch.div(pos, block_size, rounding_mode="floor") #[T]
    
    # Expand to batch
    block_ids_b = block_ids.view(1, T).expand(B, -1) #[B, T]
    
    # Identify Noisy vs Clean
    # Ideally we use `pos` comparison.
    is_noisy = (pos >= start_pos_noisy).view(1, T).expand(B, -1) # [B, T]
    is_clean = ~is_noisy
    
    # Build mask matrix [B, 1, T, T]
    # Q: [B, 1, T, 1], K: [B, 1, 1, T]
    
    # Factors for mask:
    # 1. Padding: valid_q & valid_k
    valid_q = valid.unsqueeze(2) # [B, T, 1]
    valid_k = valid.unsqueeze(1) # [B, 1, T]
    
    # 2. Causal (General Time Arrow): pos_k <= pos_q
    # Used for Clean->Clean and V2 Noisy->Noisy
    causal_mask = pos.view(1, T, 1) >= pos.view(1, 1, T) # [1, T, T] broadcastable
    
    # 3. Block constraint: block_k == block_q
    same_block = block_ids.view(1, T, 1) == block_ids.view(1, 1, T)
    
    # 4. Inter-Block constraint: block_k < block_q
    prev_block = block_ids.view(1, T, 1) > block_ids.view(1, 1, T)
    
    # --- Assemble Regions ---
    
    is_clean_q = is_clean.unsqueeze(2) # [B, T, 1]
    is_clean_k = is_clean.unsqueeze(1) # [B, 1, T]
    is_noisy_q = is_noisy.unsqueeze(2)
    is_noisy_k = is_noisy.unsqueeze(1)
    
    # A. Clean -> Clean: Strict Causal
    mask_cc = is_clean_q & is_clean_k & causal_mask
    
    # B. Noisy -> Noisy
    if is_v2:
        # V2: Strict Causal
        mask_nn = is_noisy_q & is_noisy_k & causal_mask
    else:
        # V1: Same Block (Bidirectional)
        # Assuming we only generate ONE block at a time effectively or multiple blocks?
        # The sampler loop generates multiple blocks.
        # But `x` contains ALL blocks generated so far + current.
        # Wait, if we generate Block N, and then Block N+1.
        # Block N+1 Noisy -> Block N Noisy?
        # Once Block N is done, it's considered "Clean" (fixed) in the next step?
        # The sampler logic maintains `x` as the canvas.
        # Once we move to next block loop, the previous block is "Context".
        # `start_pos_noisy` passed by sampler is `T_prefix`.
        # So `x` before `T_prefix` is treated as CLEAN context.
        # `x` after `T_prefix` is the CURRENT Noisy block.
        # So Noisy->Noisy only happens within the current block.
        mask_nn = is_noisy_q & is_noisy_k # All noisy can see all noisy (since limited to current block)
    
    # C. Noisy -> Clean: Inter-Block Only
    # Must be strictly previous block.
    # Mask: (NoisyQ & CleanK) & (BlockQ > BlockK)
    mask_nc = is_noisy_q & is_clean_k & prev_block
    
    # Combine
    base_mask = (mask_cc | mask_nn | mask_nc) & (valid_q & valid_k)
    
    return base_mask.unsqueeze(1), position_ids # [B, 1, T, T]


@dataclass
class CSBD3LMSampler(BD3LMSampler):
    """Sampler for CSBD3LM (v1)."""
    
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: BD3LMSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        # Copy-paste of BD3LMSampler.sample but replacing build_staircase_attention_mask
        # with build_csbd_attention_mask(..., is_v2=False)
        
        # To avoid massive duplication, I'll monkey-patch or copy the necessary parts.
        # Since I cannot modify `BD3LMSampler` methods easily without copy-paste,
        # I will duplicate the `sample` method here. It's the cleanest way to ensure correctness.
        
        if config is None:
            config = BD3LMSamplerConfig()

        steps = kwargs.get("steps", config.steps)
        steps_per_block = kwargs.get("steps_per_block", config.steps_per_block)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        
        # V2 Flag
        is_v2 = getattr(self, "is_v2", False)

        assert block_size >= 1
        assert steps >= 1

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        pad_id = self.tokenizer.pad_token_id

        if right_shift_logits:
            inputs = [[bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs]

        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs]

        prompt_lens = [p.shape[0] for p in inputs]
        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        max_prompt_len = max(prompt_lens)

        x = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=self.model.device)
        for b, p in enumerate(inputs):
            L = prompt_lens[b]
            offset = max_prompt_len - L
            x[b, offset : offset + L] = p

        unmasked_index = (x != mask_id) & (x != pad_id)
        if cfg_keep_tokens:
            keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=self.model.device))
            unmasked_index = unmasked_index & (~keep_mask)

        num_blocks = math.ceil(max_new_tokens / block_size)
        if steps_per_block is None:
            steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        generated = 0

        for b_idx in range(num_blocks):
            T_prefix = x.shape[1]
            offset = T_prefix % block_size
            block_room = block_size if offset == 0 else block_size - offset

            cur_block_len = min(block_room, max_new_tokens - generated)
            if cur_block_len <= 0:
                break

            # --- 2.1 Prefix ---
            x_prefix = x
            
            # MASK CHANGE: Use CSBD mask logic.
            # For prefix, `start_pos_noisy` = T_prefix (everything is clean).
            prefix_attn, prefix_pos = build_csbd_attention_mask(
                x=x_prefix,
                block_size=block_size,
                pad_token_id=pad_id,
                start_pos_noisy=T_prefix, # All clean
                is_v2=is_v2
            )

            out_prefix = self.model(
                x_prefix,
                attention_mask=prefix_attn,
                position_ids=prefix_pos,
                use_cache=True,
            )
            cond_past = out_prefix.past_key_values
            cond_prefix_last_logits = out_prefix.logits[:, -1:, :]

            if cfg_scale > 0.0:
                un_x_prefix = x_prefix.clone()
                un_x_prefix[unmasked_index] = mask_id
                out_un_prefix = self.model(
                    un_x_prefix,
                    attention_mask=prefix_attn, # Same mask for uncond? Yes.
                    position_ids=prefix_pos,
                    use_cache=True,
                )
                uncond_past = out_un_prefix.past_key_values
                uncond_prefix_last_logits = out_un_prefix.logits[:, -1:, :]
            else:
                uncond_past = None
                uncond_prefix_last_logits = None

            # --- 2.2 New Block ---
            new_block = torch.full(
                (B, cur_block_len), mask_id, dtype=torch.long, device=self.model.device
            )
            x = torch.cat([x, new_block], dim=1)
            
            # Update unmasked index logic (omitted complex CFG logic for brevity if safe)
            unmasked_index = torch.cat(
                [unmasked_index, torch.zeros((B, cur_block_len), dtype=torch.bool, device=self.model.device)],
                dim=1
            )

            B_cur, T_total = x.shape
            block_mask_index = x[:, -cur_block_len:] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            # MASK CHANGE: Full mask.
            # `start_pos_noisy` = T_prefix.
            full_attention_mask, full_position_ids = build_csbd_attention_mask(
                x=x,
                block_size=block_size,
                pad_token_id=pad_id,
                start_pos_noisy=T_prefix,
                is_v2=is_v2
            )

            attn_block = full_attention_mask[:, :, T_prefix:T_total, :]
            pos_block = full_position_ids[:, T_prefix:T_total]

            # --- 3. Diffusion Loop ---
            for i_step in range(effective_steps):
                x_block = x[:, T_prefix:T_total]
                mask_block = x_block == mask_id
                if not mask_block.any():
                    break

                cond_logits_block = self.model(
                    x_block,
                    attention_mask=attn_block,
                    position_ids=pos_block,
                    past_key_values=copy.deepcopy(cond_past),
                    use_cache=False,
                ).logits
                logits_block = cond_logits_block

                if cfg_scale > 0.0:
                    un_logits_block = self.model(
                        x_block,
                        attention_mask=attn_block,
                        position_ids=pos_block,
                        past_key_values=copy.deepcopy(uncond_past),
                        use_cache=False,
                    ).logits
                    logits_block = un_logits_block + (cfg_scale + 1.0) * (cond_logits_block - un_logits_block)

                if right_shift_logits:
                    if cfg_scale > 0.0:
                         prefix_last_logits = uncond_prefix_last_logits + (cfg_scale + 1.0) * (cond_prefix_last_logits - uncond_prefix_last_logits)
                    else:
                        prefix_last_logits = cond_prefix_last_logits
                    
                    shifted = torch.empty_like(logits_block)
                    shifted[:, 0:1, :] = prefix_last_logits
                    shifted[:, 1:, :] = logits_block[:, :-1, :]
                    logits_block = shifted

                x_block_updated = diffusion_step_block(
                    logits=logits_block,
                    x_block=x_block,
                    mask_block=mask_block,
                    num_transfer_step=num_transfer_tokens[:, i_step],
                    temperature=temperature,
                    remasking=remasking,
                )
                x[:, T_prefix:T_total] = x_block_updated
                if histories is not None:
                    histories.append(x.clone())

            if self.tokenizer.eos_token_id in x[:, T_prefix:T_total]:
                break
            generated += cur_block_len

        if not return_dict:
            return x
        else:
            return SamplerOutput(sequences=x, histories=histories)


@dataclass
class CSBD3LMv2Sampler(CSBD3LMSampler):
    """Sampler for CSBD3LM-v2 (Strict Causal Noisy Block)."""
    is_v2: bool = True
