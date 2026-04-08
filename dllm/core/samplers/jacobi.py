"""
Jacobi Decoding Sampler.

Training-free parallel decoding: initialize all remaining positions with random
tokens, run a standard causal forward pass, replace each position with the
argmax prediction, and accept consecutive tokens from the left that match
the previous guess (fixed-point convergence).

This is a baseline for MARS -- it achieves multi-token generation from a
standard AR model without any training or architectural changes, but its
acceptance rate is limited by the pretrained model's ability to predict
from incorrect prefixes.

Reference: Santilli et al., "Accelerating Transformer Inference via Jacobi Iteration" (2023)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from dllm.core.samplers.base import SamplerOutput
from dllm.core.samplers.bd3lm import BD3LMSampler, BD3LMSamplerConfig
from dllm.core.samplers.full_mask import build_pure_causal_attention_mask


@dataclass
class JacobiSamplerConfig(BD3LMSamplerConfig):
    pass


@dataclass
class JacobiSampler(BD3LMSampler):
    """
    Jacobi decoding sampler for standard AR models.

    Initializes all max_new_tokens positions with random tokens, then
    iteratively runs causal forward passes. Each iteration replaces every
    position with its argmax prediction. Tokens are "accepted" (frozen)
    from the left when the prediction matches the previous guess
    (fixed-point convergence). Iteration stops when all positions converge
    or max steps is reached.
    """

    def __post_init__(self):
        super().__post_init__()
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
            print(f"[Jacobi Global Stats] {stats}")
        return stats

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: JacobiSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        if config is None:
            config = JacobiSamplerConfig()

        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        temperature = kwargs.get("temperature", config.temperature)
        return_dict = kwargs.get("return_dict", config.return_dict)

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        # Get vocab size from underlying model (handle DDP wrapper)
        model_unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
        vocab_size = model_unwrapped.config.vocab_size

        # Normalize inputs
        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=model_unwrapped.device) for p in inputs]

        prompt_lens = [p.shape[0] for p in inputs]
        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        max_prompt_len = max(prompt_lens)

        # Left-pad prompts
        x = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=inputs[0].device)
        for b, p in enumerate(inputs):
            L = prompt_lens[b]
            offset = max_prompt_len - L
            x[b, offset:offset + L] = p

        # Initialize all max_new_tokens positions with random tokens
        random_init = torch.randint(0, vocab_size, (B, max_new_tokens), device=x.device)
        x = torch.cat([x, random_init], dim=1)  # [B, max_prompt_len + max_new_tokens]

        # Track which positions have converged (frozen from left)
        cursor = torch.zeros(B, dtype=torch.long, device=x.device)  # per-sample accepted count
        done = torch.zeros(B, dtype=torch.bool, device=x.device)

        total_forward_passes = 0
        total_tokens_accepted = 0

        max_iters = kwargs.get("steps", max_new_tokens * 2)  # generous iteration budget

        for step_idx in range(max_iters):
            if done.all():
                break

            # Forward pass with causal attention over the full sequence
            attn_mask, pos_ids = build_pure_causal_attention_mask(x, pad_token_id=pad_id)
            out = self.model(x, attention_mask=attn_mask, position_ids=pos_ids)
            total_forward_passes += 1

            # AR right-shifted: logits at position t predict token t+1
            # For generation positions [max_prompt_len : max_prompt_len + max_new_tokens],
            # predictions come from logits at [max_prompt_len-1 : max_prompt_len + max_new_tokens - 1]
            logits = out.logits  # [B, T, V]
            preds = torch.argmax(logits[:, max_prompt_len - 1:max_prompt_len + max_new_tokens - 1, :], dim=-1)  # [B, max_new_tokens]

            # Current guesses in the generation region
            current_guesses = x[:, max_prompt_len:max_prompt_len + max_new_tokens]  # [B, max_new_tokens]

            # For each sample, find consecutive matches from cursor position
            step_accepted = 0
            for b in range(B):
                if done[b]:
                    continue

                c = cursor[b].item()
                n_acc = 0

                # Check consecutive fixed-point matches starting from cursor
                for i in range(c, max_new_tokens):
                    if preds[b, i] == current_guesses[b, i]:
                        # Fixed point: prediction matches guess
                        n_acc += 1
                    else:
                        # Not converged yet; update this and all remaining positions
                        break

                # Always accept at least 1 token (the argmax prediction at cursor)
                if n_acc == 0:
                    n_acc = 1

                # Advance cursor
                cursor[b] += n_acc
                step_accepted += n_acc

                # Check EOS in newly accepted region
                accepted_region = preds[b, c:c + n_acc]
                if eos_id in accepted_region:
                    done[b] = True

                if cursor[b] >= max_new_tokens:
                    done[b] = True

            total_tokens_accepted += step_accepted

            # Update all generation positions with new predictions
            # (only non-frozen positions matter, but updating all is simpler)
            for b in range(B):
                if not done[b]:
                    # Update positions from cursor onwards with new predictions
                    c = cursor[b].item()
                    x[b, max_prompt_len + c:max_prompt_len + max_new_tokens] = preds[b, c:max_new_tokens]
                # Also write accepted positions (they are frozen but ensure consistency)
                c = cursor[b].item()
                x[b, max_prompt_len:max_prompt_len + c] = preds[b, :c]

        self.global_forward_passes += total_forward_passes
        self.global_tokens_accepted += total_tokens_accepted
        self.global_samples += 1

        if not return_dict:
            return x
        else:
            return SamplerOutput(sequences=x, histories=None)
