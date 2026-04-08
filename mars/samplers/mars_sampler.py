"""
MARS Sampler for models trained with 100% masking.

The key difference from standard block diffusion samplers:
- Standard: Tokens are revealed based on confidence (most confident first).
- MARS: Tokens are revealed LEFT-TO-RIGHT within the block.

Left-to-Right Logic:
1. Model predicts all masked positions in the current block.
2. Starting from the leftmost masked position, tokens pass a confidence threshold.
3. Accept tokens consecutively from left until one fails the threshold OR we hit max_accept.
4. If no threshold is set (None), accept exactly 1 token per step (always the leftmost).

This aligns with training where the model always saw a fully masked block
and learned to predict left-to-right (causal intra-block).
"""

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class SamplerOutput:
    sequences: torch.Tensor
    histories: list | None = None


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves
    perplexity score but reduces generation quality. Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def build_pure_causal_attention_mask(
    x: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a strict causal attention mask.
    No block boundary constraints. Every token can attend to all previous valid tokens.
    """
    B, T = x.shape
    device = x.device

    # Per-sample valid mask (exclude padding)
    valid = x != pad_token_id

    # Position IDs: logical positions 0..T-1 (ignoring padding gaps)
    pos_raw = torch.cumsum(valid.to(torch.long), dim=-1)
    logical_pos = pos_raw - 1
    position_ids = torch.where(valid, logical_pos, torch.zeros_like(logical_pos)).to(device=device, dtype=torch.long)

    # Physical positions for causal mask
    pos = torch.arange(T, device=device)

    valid_q = valid.unsqueeze(2)  # [B, T, 1]
    valid_k = valid.unsqueeze(1)  # [B, 1, T]

    # Causal (General Time Arrow): pos_k <= pos_q
    causal_mask = pos.view(1, T, 1) >= pos.view(1, 1, T)  # [1, T, T] broadcastable

    # Combine
    base_mask = causal_mask & valid_q & valid_k

    return base_mask.unsqueeze(1), position_ids  # [B, 1, T, T]


@dataclass
class MARSSamplerConfig:
    """Config for MARSSampler."""
    max_new_tokens: int = 128
    max_length: int = None
    block_size: int = 32
    steps: int = 128
    steps_per_block: int | None = None
    temperature: float = 0.0
    cfg_scale: float = 0.0
    right_shift_logits: bool = False
    # Confidence threshold for accepting tokens.
    # If None, accept exactly 1 token per step.
    # If set (e.g., 0.5), accept consecutive tokens from left as long as conf >= threshold.
    confidence_threshold: float | None = None
    # Max tokens to accept per step (even if more pass threshold).
    max_accept_per_step: int | None = None
    # Acceptance metric: "probability" (default), "entropy", or "margin".
    # - probability: accept if P(top token) >= threshold (higher = more confident)
    # - entropy: accept if H(p) <= threshold (lower = more confident)
    # - margin: accept if P(top1) - P(top2) >= threshold (higher = more confident)
    acceptance_metric: str = "probability"
    return_dict: bool = False


def left_to_right_step(
    logits: torch.Tensor,  # [B, L, V]
    x_block: torch.Tensor,  # [B, L]
    mask_block: torch.Tensor,  # [B, L] bool
    mask_id: int,
    temperature: float,
    confidence_threshold: float | None,
    max_accept: int | None,
    acceptance_metric: str = "probability",
) -> tuple[torch.Tensor, list[int]]:
    """
    One generation step with left-to-right acceptance.

    Args:
        logits: Model predictions for the block.
        x_block: Current block tokens.
        mask_block: Boolean mask of which positions are still masked.
        mask_id: The mask token ID.
        temperature: Sampling temperature (0 = greedy).
        confidence_threshold: If set, accept consecutive tokens from left with conf >= this.
        max_accept: Maximum tokens to accept per step.
        acceptance_metric: "probability", "entropy", or "margin".

    Returns:
        Tuple of (updated x_block, list of accepted counts per batch element).
    """
    B, L, V = logits.shape
    device = logits.device

    if not mask_block.any():
        return x_block, [0] * B

    # Sample predictions
    if temperature > 0:
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]
    else:
        x0 = torch.argmax(logits, dim=-1)  # [B, L]

    # Compute per-position acceptance score based on metric
    p = F.softmax(logits, dim=-1)  # [B, L, V]
    if acceptance_metric == "entropy":
        # H(p) = -sum(p * log(p)); accept if H <= threshold (low entropy = confident)
        log_p = torch.log(p + 1e-10)
        x0_conf = -(p * log_p).sum(dim=-1)  # [B, L]
    elif acceptance_metric == "margin":
        # margin = P(top1) - P(top2); accept if margin >= threshold
        top2 = torch.topk(p, k=2, dim=-1).values  # [B, L, 2]
        x0_conf = top2[:, :, 0] - top2[:, :, 1]  # [B, L]
    else:
        # probability (default): P(chosen token); accept if P >= threshold
        x0_conf = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, L]

    # For entropy, lower is more confident; for probability/margin, higher is more confident
    accept_if_below = (acceptance_metric == "entropy")

    # Build output
    x_block_new = x_block.clone()
    accepted_per_batch = []

    for b in range(B):
        # Find masked positions for this batch element
        masked_positions = mask_block[b].nonzero(as_tuple=True)[0]  # 1D tensor of indices
        if len(masked_positions) == 0:
            continue

        # Iterate LEFT-TO-RIGHT through masked positions
        accepted_count = 0
        max_to_accept = max_accept if max_accept is not None else L  # Default: accept up to block_size

        for pos in masked_positions:
            pos_idx = pos.item()
            conf = x0_conf[b, pos_idx].item()

            # Check threshold (if set)
            if confidence_threshold is not None:
                if accept_if_below:
                    # Entropy: reject if conf > threshold (high entropy = uncertain)
                    if conf > confidence_threshold and accepted_count > 0:
                        break
                else:
                    # Probability/margin: reject if conf < threshold
                    if conf < confidence_threshold and accepted_count > 0:
                        break

            # Accept this token
            x_block_new[b, pos_idx] = x0[b, pos_idx]
            accepted_count += 1

            # Check max accept limit
            if accepted_count >= max_to_accept:
                break

            # If no threshold, accept only 1 token (the leftmost)
            if confidence_threshold is None:
                break

        accepted_per_batch.append(accepted_count)

    return x_block_new, accepted_per_batch


@dataclass
class MARSSampler:
    """
    Sampler for MARS models.

    Uses left-to-right token acceptance within each block.
    """
    model: object
    tokenizer: object

    def __post_init__(self):
        self.global_forward_passes = 0
        self.global_tokens_accepted = 0
        self.global_samples = 0

    def get_global_stats(self) -> dict:
        """Aggregate stats across all processes and return as dict. Also prints on rank 0."""
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
            print(f"[MARS Global Stats] {stats}")
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

        # Pull args
        steps = kwargs.get("steps", config.steps)
        steps_per_block = kwargs.get("steps_per_block", config.steps_per_block)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
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

        # Normalize inputs
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

        # Initialize with left-padded prompts
        x = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=self.model.device)
        for b, p in enumerate(inputs):
            L = prompt_lens[b]
            offset = max_prompt_len - L
            x[b, offset:offset + L] = p

        histories = [x.clone()] if return_dict else None
        generated = 0

        # Statistics tracking
        total_forward_passes = 0
        total_tokens_accepted = 0

        # Per-sample done tracking
        done = torch.zeros(B, dtype=torch.bool, device=self.model.device)

        # Append initial block of MASK tokens (always keep block_size MASKs at end)
        initial_mask = torch.full((B, block_size), mask_id, dtype=torch.long, device=self.model.device)
        x = torch.cat([x, initial_mask], dim=1)

        # =====================================================
        # Sliding Window: always keep block_size MASKs at the end
        # Each step: forward block -> accept N tokens -> update cache -> slide
        # =====================================================
        max_steps = steps if steps else max_new_tokens
        for step_idx in range(max_steps):
            if done.all():
                break
            if generated >= max_new_tokens:
                break

            T_total = x.shape[1]
            T_prefix = T_total - block_size

            # --- 1. Single forward of entire x (no cache split) ---
            full_attn, full_pos = build_pure_causal_attention_mask(
                x=x,
                pad_token_id=pad_id,
            )

            x_block = x[:, T_prefix:T_total]
            mask_block = (x_block == mask_id) & (~done.unsqueeze(1).expand(B, block_size))

            if not mask_block.any():
                break

            out_full = self.model(
                x,
                attention_mask=full_attn,
                position_ids=full_pos,
            )
            total_forward_passes += 1

            full_logits = out_full.logits
            logits_block = full_logits[:, T_prefix:T_total, :]

            if right_shift_logits:
                shifted = torch.empty_like(logits_block)
                shifted[:, 0:1, :] = full_logits[:, T_prefix-1:T_prefix, :]
                shifted[:, 1:, :] = logits_block[:, :-1, :]
                logits_block = shifted

            # --- 3. Accept tokens left-to-right ---
            x_block_updated, n_accepted = left_to_right_step(
                logits=logits_block,
                x_block=x_block,
                mask_block=mask_block,
                mask_id=mask_id,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                max_accept=max_accept_per_step,
                acceptance_metric=acceptance_metric,
            )
            total_tokens_accepted += sum(n_accepted)
            x[:, T_prefix:T_total] = x_block_updated

            # Calculate how many tokens were accepted
            n_new = max(n_accepted) if n_accepted else 0
            if n_new == 0:
                n_new = 1  # always advance by at least 1

            n_new = min(n_new, max_new_tokens - generated)
            if n_new <= 0:
                break

            # --- 4. Slide: append new MASKs to replace accepted tokens ---
            new_masks = torch.full((B, n_new), mask_id, dtype=torch.long, device=self.model.device)
            # Done samples get pad
            if done.any():
                new_masks[done] = pad_id
            x = torch.cat([x, new_masks], dim=1)
            generated += n_new

            # Per-sample EOS check (in the accepted region)
            for b in range(B):
                if not done[b]:
                    accepted_region = x[b, T_prefix:T_prefix + block_size]
                    if self.tokenizer.eos_token_id in accepted_region:
                        done[b] = True

            if histories is not None:
                histories.append(x.clone())

        # =====================================================
        # Accumulate global statistics
        # =====================================================
        self.global_forward_passes += total_forward_passes
        self.global_tokens_accepted += total_tokens_accepted
        self.global_samples += 1

        # =====================================================
        # Output
        # =====================================================
        if not return_dict:
            return x
        else:
            return SamplerOutput(sequences=x, histories=histories)
