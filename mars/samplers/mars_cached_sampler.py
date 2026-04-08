"""
KV-Cache-based MARS Samplers.

Two variants:
1. MARSCachedSampler: sliding window with KV cache (per-step)
2. MARSBlockCachedSampler: block-by-block with KV cache
   - Prefix KV computed once per block
   - Inner loop fills the block using cached prefix (deepcopy per step)
   - After block filled, extend cache with filled block
   - Batch: fast samples wait for slow ones at block boundaries
"""

import copy
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .mars_sampler import (
    SamplerOutput,
    MARSSamplerConfig,
    add_gumbel_noise,
    build_pure_causal_attention_mask,
    left_to_right_step,
)


def build_cached_attention_mask(
    prefix_tokens: torch.Tensor,
    new_input: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Build 4D causal attention mask for cached forward.
    Handles PAD tokens in both prefix (cache) and new_input.

    Args:
        prefix_tokens: [B, T_prefix] tokens in cache (may contain PADs)
        new_input: [B, new_len] new tokens being forwarded (may contain PADs)
        pad_token_id: padding token id

    Returns:
        [B, 1, new_len, T_prefix + new_len] boolean mask
    """
    B, T_prefix = prefix_tokens.shape
    new_len = new_input.shape[1]
    device = prefix_tokens.device

    prefix_valid = (prefix_tokens != pad_token_id)  # [B, T_prefix]
    new_valid = (new_input != pad_token_id)           # [B, new_len]

    # Query validity: PAD queries attend to nothing
    new_valid_q = new_valid.unsqueeze(2)  # [B, new_len, 1]

    # Prefix part: valid new tokens attend to valid prefix positions
    prefix_part = prefix_valid.unsqueeze(1).expand(B, new_len, T_prefix)
    prefix_part = prefix_part & new_valid_q

    # New part: causal among new tokens, both q and k must be valid
    causal = torch.tril(torch.ones(new_len, new_len, dtype=torch.bool, device=device))
    new_valid_k = new_valid.unsqueeze(1)  # [B, 1, new_len]
    new_part = causal.unsqueeze(0) & new_valid_q & new_valid_k

    mask = torch.cat([prefix_part, new_part], dim=2)
    return mask.unsqueeze(1)  # [B, 1, new_len, T_prefix + new_len]


def compute_cached_position_ids(
    prefix_tokens: torch.Tensor,
    new_input: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Compute per-sample logical position IDs for new tokens.
    Valid tokens get sequential positions continuing from prefix.
    PAD tokens get position 0 (masked out anyway).

    Returns: [B, new_len]
    """
    B = prefix_tokens.shape[0]
    new_len = new_input.shape[1]
    device = prefix_tokens.device

    num_valid_prefix = (prefix_tokens != pad_token_id).sum(dim=1)  # [B]
    new_valid = (new_input != pad_token_id)  # [B, new_len]

    # Cumulative count of valid tokens within new_input (0-indexed)
    cum_valid = torch.cumsum(new_valid.long(), dim=1) - 1  # [B, new_len]

    # Offset by prefix valid count: valid tokens get sequential positions
    position_ids = torch.where(
        new_valid,
        cum_valid + num_valid_prefix.unsqueeze(1),
        torch.zeros(B, new_len, dtype=torch.long, device=device),
    )
    return position_ids


def trim_kv_cache(past_key_values, max_length: int):
    """Trim DynamicCache in-place to max_length entries."""
    past_key_values.crop(max_length)


@dataclass
class MARSCachedSampler:
    """
    KV-cache-based sampler for MARS models.

    After the initial full forward, each step only processes
    (N_accepted + block_size) new tokens using cached prefix KV.
    Supports batch>1 with per-sample variable acceptance.
    """
    model: object
    tokenizer: object

    def __post_init__(self):
        self.global_forward_passes = 0
        self.global_tokens_accepted = 0
        self.global_samples = 0

    def get_global_stats(self) -> dict:
        """Aggregate stats across all processes."""
        forward_passes = torch.tensor(self.global_forward_passes, device=self.model.device)
        tokens_accepted = torch.tensor(self.global_tokens_accepted, device=self.model.device)
        samples = torch.tensor(self.global_samples, device=self.model.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(forward_passes)
            torch.distributed.all_reduce(tokens_accepted)
            torch.distributed.all_reduce(samples)
        fp = forward_passes.item()
        ta = tokens_accepted.item()
        s = samples.item()
        stats = {
            "samples": s,
            "total_forwards": fp,
            "total_tokens_accepted": ta,
            "avg_tokens_per_forward": ta / fp if fp > 0 else 0.0,
        }
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if fp > 0 and is_main:
            print(f"[MARSCached Global Stats] {stats}")
        return stats

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MARSSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        if config is None:
            config = MARSSamplerConfig()

        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        confidence_threshold = kwargs.get("confidence_threshold", config.confidence_threshold)
        max_accept_per_step = kwargs.get("max_accept_per_step", config.max_accept_per_step)
        acceptance_metric = kwargs.get("acceptance_metric", config.acceptance_metric)

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

        # Left-padded prompts
        x_prefix = torch.full(
            (B, max_prompt_len), pad_id, dtype=torch.long, device=self.model.device
        )
        for b, p in enumerate(inputs):
            L = prompt_lens[b]
            offset = max_prompt_len - L
            x_prefix[b, offset:offset + L] = p

        done = torch.zeros(B, dtype=torch.bool, device=self.model.device)
        generated = 0
        total_forward_passes = 0
        total_tokens_accepted = 0
        kv_cache = None
        accepted_ids = None

        max_steps = steps if steps else max_new_tokens

        for step_idx in range(max_steps):
            if done.all() or generated >= max_new_tokens:
                break

            T_prefix = x_prefix.shape[1]

            # Fresh MASK block each step (PAD for done samples)
            mask_block_input = torch.full(
                (B, block_size), mask_id, dtype=torch.long, device=self.model.device
            )
            if done.any():
                mask_block_input[done] = pad_id

            if kv_cache is None:
                # ====== Step 0: full forward ======
                x_full = torch.cat([x_prefix, mask_block_input], dim=1)
                attn_mask, pos_ids = build_pure_causal_attention_mask(x_full, pad_id)

                out = self.model(
                    x_full,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    use_cache=True,
                )
                kv_cache = out.past_key_values
                total_forward_passes += 1

                if right_shift_logits:
                    block_logits = out.logits[:, T_prefix - 1:T_prefix + block_size - 1, :]
                else:
                    block_logits = out.logits[:, T_prefix:T_prefix + block_size, :]
            else:
                # ====== Cached step: forward [accepted | MASKs] ======
                cache_len = kv_cache.get_seq_length()
                new_input = torch.cat([accepted_ids, mask_block_input], dim=1)
                N_prev = accepted_ids.shape[1]

                # Build mask/pos based on actual cache contents
                cached_prefix = x_prefix[:, :cache_len]
                attn_mask = build_cached_attention_mask(cached_prefix, new_input, pad_id)
                pos_ids = compute_cached_position_ids(cached_prefix, new_input, pad_id)

                out = self.model(
                    new_input,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                kv_cache = out.past_key_values
                total_forward_passes += 1

                if right_shift_logits:
                    block_logits = out.logits[:, N_prev - 1:N_prev - 1 + block_size, :]
                else:
                    block_logits = out.logits[:, N_prev:N_prev + block_size, :]

            # ====== Accept tokens left-to-right ======
            x_block = mask_block_input.clone()
            block_mask = (x_block == mask_id) & (
                ~done.unsqueeze(1).expand(B, block_size)
            )

            x_block_updated, n_accepted = left_to_right_step(
                logits=block_logits,
                x_block=x_block,
                mask_block=block_mask,
                mask_id=mask_id,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                max_accept=max_accept_per_step,
                acceptance_metric=acceptance_metric,
            )
            total_tokens_accepted += sum(n_accepted)

            # Per-sample accepted count
            n_max = max(n_accepted) if n_accepted else 1
            n_max = max(n_max, 1)
            n_max = min(n_max, max_new_tokens - generated)
            if n_max <= 0:
                break

            # Build per-sample accepted_ids: real tokens + PAD padding
            # Each sample commits only its own accepted tokens, rest are PAD
            accepted_ids = torch.full(
                (B, n_max), pad_id, dtype=torch.long, device=self.model.device
            )
            for b in range(B):
                if not done[b] and b < len(n_accepted):
                    n_b = min(n_accepted[b], n_max)
                    accepted_ids[b, :n_b] = x_block_updated[b, :n_b]

            # Trim KV cache: remove only the block_size MASK entries.
            trim_kv_cache(kv_cache, kv_cache.get_seq_length() - block_size)

            # Grow prefix (PADs in accepted_ids are handled by attention mask)
            x_prefix = torch.cat([x_prefix, accepted_ids], dim=1)
            generated += n_max

            # Per-sample EOS check
            for b in range(B):
                if not done[b] and self.tokenizer.eos_token_id in accepted_ids[b]:
                    done[b] = True

        # ====== Stats ======
        self.global_forward_passes += total_forward_passes
        self.global_tokens_accepted += total_tokens_accepted
        self.global_samples += 1

        # ====== Output: pad remaining to match expected length ======
        remaining = max_new_tokens - generated
        if remaining > 0:
            pad_tail = torch.full(
                (B, remaining), pad_id, dtype=torch.long, device=self.model.device
            )
            x_prefix = torch.cat([x_prefix, pad_tail], dim=1)

        if not return_dict:
            return x_prefix
        else:
            return SamplerOutput(sequences=x_prefix, histories=None)


@dataclass
class MARSBlockCachedSampler:
    """
    Block-by-block sampler with KV cache for MARS models.

    Inspired by block diffusion's dual cache:
    1. Prefix forward once per block -> KV cache
    2. Inner loop: fill block using deepcopy(prefix_cache)
    3. After all samples fill the block, extend cache with filled block
    4. Move to next block

    Batch: fast samples wait for slow ones at block boundaries.
    """
    model: object
    tokenizer: object

    def __post_init__(self):
        self.global_forward_passes = 0
        self.global_tokens_accepted = 0
        self.global_samples = 0

    def get_global_stats(self) -> dict:
        forward_passes = torch.tensor(self.global_forward_passes, device=self.model.device)
        tokens_accepted = torch.tensor(self.global_tokens_accepted, device=self.model.device)
        samples = torch.tensor(self.global_samples, device=self.model.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(forward_passes)
            torch.distributed.all_reduce(tokens_accepted)
            torch.distributed.all_reduce(samples)
        fp = forward_passes.item()
        ta = tokens_accepted.item()
        s = samples.item()
        stats = {
            "samples": s,
            "total_forwards": fp,
            "total_tokens_accepted": ta,
            "avg_tokens_per_forward": ta / fp if fp > 0 else 0.0,
        }
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if fp > 0 and is_main:
            print(f"[MARSBlockCached Global Stats] {stats}")
        return stats

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MARSSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        if config is None:
            config = MARSSamplerConfig()

        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        confidence_threshold = kwargs.get("confidence_threshold", config.confidence_threshold)
        max_accept_per_step = kwargs.get("max_accept_per_step", config.max_accept_per_step)
        acceptance_metric = kwargs.get("acceptance_metric", config.acceptance_metric)

        assert block_size >= 1

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

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

        # Left-padded prompts
        x_prefix = torch.full(
            (B, max_prompt_len), pad_id, dtype=torch.long, device=self.model.device
        )
        for b, p in enumerate(inputs):
            L = prompt_lens[b]
            offset = max_prompt_len - L
            x_prefix[b, offset:offset + L] = p

        done = torch.zeros(B, dtype=torch.bool, device=self.model.device)
        generated = 0
        total_forward_passes = 0
        total_tokens_accepted = 0

        # ====== Prefix forward -> KV cache ======
        prefix_attn, prefix_pos = build_pure_causal_attention_mask(x_prefix, pad_id)
        out = self.model(
            x_prefix, attention_mask=prefix_attn, position_ids=prefix_pos,
            use_cache=True,
        )
        prefix_cache = out.past_key_values
        prefix_last_logit = out.logits[:, -1:, :]
        total_forward_passes += 1

        num_blocks = math.ceil(max_new_tokens / block_size)

        for block_idx in range(num_blocks):
            if done.all() or generated >= max_new_tokens:
                break

            cur_bs = min(block_size, max_new_tokens - generated)

            # Initialize block with MASKs (PAD for done samples)
            block = torch.full(
                (B, cur_bs), mask_id, dtype=torch.long, device=self.model.device
            )
            if done.any():
                block[done] = pad_id

            # Build attention mask and position IDs for block (stable within inner loop)
            cache_len = prefix_cache.get_seq_length()
            cached_prefix = x_prefix[:, :cache_len]
            block_attn = build_cached_attention_mask(cached_prefix, block, pad_id)
            block_pos = compute_cached_position_ids(cached_prefix, block, pad_id)

            block_filled = done.clone()

            # ====== Inner loop: fill the block ======
            for inner_step in range(cur_bs):
                if block_filled.all():
                    break

                block_mask = (block == mask_id) & ~block_filled.unsqueeze(1).expand_as(block)
                if not block_mask.any():
                    break

                # Forward block with deepcopy of prefix cache
                out_block = self.model(
                    block,
                    attention_mask=block_attn,
                    position_ids=block_pos,
                    past_key_values=copy.deepcopy(prefix_cache),
                    use_cache=True,
                )
                total_forward_passes += 1

                block_logits = out_block.logits
                if right_shift_logits:
                    shifted = torch.empty_like(block_logits)
                    shifted[:, 0:1, :] = prefix_last_logit
                    shifted[:, 1:, :] = block_logits[:, :-1, :]
                    block_logits = shifted

                # Accept tokens left-to-right
                block_updated, n_accepted = left_to_right_step(
                    logits=block_logits,
                    x_block=block,
                    mask_block=block_mask,
                    mask_id=mask_id,
                    temperature=temperature,
                    confidence_threshold=confidence_threshold,
                    max_accept=max_accept_per_step,
                    acceptance_metric=acceptance_metric,
                )
                total_tokens_accepted += sum(n_accepted)
                block = block_updated

                # Per-sample: check if block is fully filled
                for b in range(B):
                    if not block_filled[b] and not (block[b] == mask_id).any():
                        block_filled[b] = True

            # ====== Block filled: extend prefix cache ======
            # Forward the filled block to compute correct KV and extend cache
            out_ext = self.model(
                block,
                attention_mask=block_attn,
                position_ids=block_pos,
                past_key_values=prefix_cache,
                use_cache=True,
            )
            prefix_cache = out_ext.past_key_values
            prefix_last_logit = out_ext.logits[:, -1:, :]
            total_forward_passes += 1

            # Update prefix
            x_prefix = torch.cat([x_prefix, block], dim=1)
            generated += cur_bs

            # EOS check
            for b in range(B):
                if not done[b] and eos_id in block[b]:
                    done[b] = True

        # ====== Stats ======
        self.global_forward_passes += total_forward_passes
        self.global_tokens_accepted += total_tokens_accepted
        self.global_samples += B

        # ====== Output ======
        remaining = max_new_tokens - generated
        if remaining > 0:
            pad_tail = torch.full(
                (B, remaining), pad_id, dtype=torch.long, device=self.model.device
            )
            x_prefix = torch.cat([x_prefix, pad_tail], dim=1)

        if not return_dict:
            return x_prefix
        else:
            return SamplerOutput(sequences=x_prefix, histories=None)
