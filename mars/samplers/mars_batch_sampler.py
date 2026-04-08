"""
Batch-friendly MARS Sampler.

Simple version: full-sequence forward each step with causal attention.
No KV cache -- correctness first, optimize later.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .mars_sampler import (
    SamplerOutput,
    add_gumbel_noise,
    build_pure_causal_attention_mask,
)


@dataclass
class MARSBatchSamplerConfig:
    max_new_tokens: int = 256
    block_size: int = 4
    temperature: float = 0.0
    confidence_threshold: float = 0.7
    right_shift_logits: bool = False


@dataclass
class MARSBatchSampler:
    model: object
    tokenizer: object

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MARSBatchSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        if config is None:
            config = MARSBatchSamplerConfig()

        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        confidence_threshold = kwargs.get("confidence_threshold", config.confidence_threshold)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        return_dict = kwargs.get("return_dict", False)

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        bos_id = self.tokenizer.bos_token_id

        if right_shift_logits:
            inputs = [[bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs]
        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs]

        B = len(inputs)
        prompt_lens = [p.shape[0] for p in inputs]
        max_prompt_len = max(prompt_lens)

        # Align max_new_tokens to block_size
        max_new_tokens = ((max_new_tokens + block_size - 1) // block_size) * block_size

        # Build [left-padded prompt | MASK * max_new_tokens]
        total_len = max_prompt_len + max_new_tokens
        x = torch.full((B, total_len), mask_id, dtype=torch.long, device=self.model.device)
        for b, p in enumerate(inputs):
            L = prompt_lens[b]
            offset = max_prompt_len - L
            x[b, :offset] = pad_id
            x[b, offset:offset + L] = p

        cursors = torch.full((B,), max_prompt_len, dtype=torch.long, device=self.model.device)
        done = torch.zeros(B, dtype=torch.bool, device=self.model.device)

        total_forward_passes = 0
        total_tokens_accepted = 0

        max_steps = max_new_tokens
        for step in range(max_steps):
            if done.all():
                break

            # Build causal attention mask
            # All tokens treated as clean -> pure causal
            attn_mask, position_ids = build_pure_causal_attention_mask(
                x=x,
                pad_token_id=pad_id,
            )

            # Forward
            out = self.model(
                input_ids=x,
                attention_mask=attn_mask,
                position_ids=position_ids,
            )
            logits = out.logits  # (B, total_len, V)
            total_forward_passes += 1

            # Right shift logits
            if right_shift_logits:
                shifted = torch.empty_like(logits)
                shifted[:, 0:1, :] = logits[:, 0:1, :]
                shifted[:, 1:, :] = logits[:, :-1, :]
                logits = shifted

            # Predictions
            if temperature > 0:
                preds = torch.argmax(add_gumbel_noise(logits, temperature=temperature), dim=-1)
            else:
                preds = torch.argmax(logits, dim=-1)

            probs = F.softmax(logits.float(), dim=-1)
            pred_conf = torch.gather(probs, dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)

            # Per-sample left-to-right acceptance from cursor
            for b in range(B):
                if done[b]:
                    continue

                cursor = cursors[b].item()
                accepted = 0

                while cursor < total_len and x[b, cursor].item() == mask_id:
                    conf = pred_conf[b, cursor].item()
                    token = preds[b, cursor].item()

                    if conf < confidence_threshold and accepted > 0:
                        break

                    x[b, cursor] = token
                    cursor += 1
                    accepted += 1

                    if token == eos_id:
                        done[b] = True
                        break

                cursors[b] = cursor
                total_tokens_accepted += accepted

                if cursor >= total_len:
                    done[b] = True

        # Stats
        stats = {
            "total_forwards": total_forward_passes,
            "total_tokens_accepted": total_tokens_accepted,
            "avg_tokens_per_forward": total_tokens_accepted / max(total_forward_passes, 1),
            "samples": B,
        }
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main:
            print(f"[MARSBatch Stats] {stats}")

        if not return_dict:
            return x
        else:
            return SamplerOutput(sequences=x, histories=None)
