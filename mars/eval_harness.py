"""
MARS Evaluation Harness for lm-eval-harness.

Supports three sampling modes:
1. MARSBatchSampler (batch_size > 1): batch inference without cache
2. MARSCachedSampler (use_cache=True): KV-cached sliding window (bs=1)
3. MARSSampler (default): full forward each step (bs=1)

Usage:
    accelerate launch --num_processes 8 mars/eval_harness.py \
        --tasks gsm8k_cot --num_fewshot 0 \
        --model mars --apply_chat_template \
        --model_args "pretrained=<path>,max_new_tokens=256,steps=256,block_size=4,cfg=0.0,right_shift_logits=True,confidence_threshold=0.95"
"""

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

import accelerate
import torch
import torch.nn.functional as F
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from tqdm import tqdm

import dllm
from mars.samplers import MARSSampler, MARSCachedSampler, MARSBatchSampler


@dataclass
class BaseEvalConfig:
    """Configuration for MARS evaluation harness."""

    # Sampler parameters (mirroring BD3LMSamplerConfig fields)
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    right_shift_logits: bool = False

    # Generation parameters
    max_new_tokens: int = 128
    max_length: int = 2048
    steps: int = 128
    block_size: int = 32

    # Model / harness parameters
    pretrained: str = ""
    dtype: str | torch.dtype = "auto"
    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False
    device: str = "cuda"


class BaseEvalHarness(LM):
    """
    Base evaluation harness that handles model loading, tokenizer setup,
    and distributed inference scaffolding. Subclasses override generate_until
    to use specific samplers.
    """

    def __init__(
        self,
        config: BaseEvalConfig | None = None,
        **kwargs,
    ):
        super().__init__()

        # Initialize config if not provided
        if config is None:
            config = BaseEvalConfig()

        # Pull args from config, allow kwargs to override
        pretrained = kwargs.get("pretrained", config.pretrained)
        dtype = kwargs.get("dtype", config.dtype)
        batch_size = kwargs.get("batch_size", config.batch_size)
        mc_num = kwargs.get("mc_num", config.mc_num)
        is_check_greedy = kwargs.get("is_check_greedy", config.is_check_greedy)
        device = kwargs.get("device", config.device)
        cfg = kwargs.get("cfg", config.cfg_scale)
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        block_size = kwargs.get("block_size", config.block_size)
        max_length = kwargs.get("max_length", config.max_length)
        remasking = kwargs.get("remasking", config.remasking)
        right_shift_logits = kwargs.get(
            "right_shift_logits", config.right_shift_logits
        )

        accelerator = accelerate.Accelerator()

        # Get GLOBAL rank from torch.distributed (not accelerator)
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        # Use accelerator for device placement
        self.model = dllm.utils.get_model(
            SimpleNamespace(
                model_name_or_path=pretrained, dtype=get_dtype(dtype)
            )
        )
        self.model.eval()

        if accelerator.num_processes > 1:
            # Let accelerator handle device placement
            self.model = accelerator.prepare(self.model)
            self.device = accelerator.device
            self.accelerator = accelerator
        else:
            # Single GPU
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.tokenizer = dllm.utils.get_tokenizer(
            SimpleNamespace(
                model_name_or_path=pretrained, model=self.model
            )
        )

        # Sampler params
        self.mask_id = self.tokenizer.mask_token_id
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.max_new_tokens = int(max_new_tokens)
        self.block_size = int(block_size)
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.remasking = remasking
        self.is_check_greedy = is_check_greedy
        self.right_shift_logits = right_shift_logits

        # Loglikelihood params
        self.mc_num = int(mc_num)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.0

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply a chat template to a list of chat history between user and model."""
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            enable_thinking=False,
        )
        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def generate_until(self, requests: list[Instance]):
        """Generate greedily until a stopping sequence.

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a
            tuple (context, gen_kwargs).
            context: str -- Context string
            gen_kwargs: dict -- Keyword arguments for generation (e.g. until)
        :return: list[str]
            A list of model generated continuations.
        """
        raise NotImplementedError(
            "BaseEvalHarness.generate_until should not be called directly. "
            "Use MARSEvalHarness or JacobiEvalHarness instead."
        )

    def loglikelihood(self, requests: list[Instance]):
        raise NotImplementedError(
            "loglikelihood not supported for this model"
        )

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError(
            "loglikelihood_rolling not supported for this model"
        )


@register_model("mars")
@register_model("full_mask")
class MARSEvalHarness(BaseEvalHarness):
    """
    Evaluation harness for MARS (Mask AutoRegressive Speculation).
    Uses left-to-right confidence-based token acceptance within each block.

    Sampling modes:
    - batch_size > 1: MARSBatchSampler (pure causal batch inference)
    - use_cache=True: MARSCachedSampler (KV-cached sliding window, bs=1)
    - default:        MARSSampler (full forward each step, bs=1)
    """

    def __init__(self, config=None, **kwargs):
        # Parse MARS-specific args before parent __init__
        self._confidence_threshold = (
            float(kwargs.pop("confidence_threshold"))
            if "confidence_threshold" in kwargs
            else None
        )
        self._max_accept_per_step = (
            int(kwargs.pop("max_accept_per_step"))
            if "max_accept_per_step" in kwargs
            else None
        )
        self._use_cache = (
            kwargs.pop("use_cache", "false").lower() in ("true", "1", "yes")
            if "use_cache" in kwargs
            else False
        )
        self._acceptance_metric = (
            kwargs.pop("acceptance_metric", "probability")
            if "acceptance_metric" in kwargs
            else "probability"
        )
        super().__init__(config=config, **kwargs)

    def generate_until(self, requests: list[Instance]):
        out = [None] * len(requests)
        batch_size = self.batch_size

        confidence_threshold = self._confidence_threshold
        max_accept_per_step = self._max_accept_per_step
        acceptance_metric = self._acceptance_metric

        # ----------------------------------------------------------------
        # Path 1: Batch inference (batch_size > 1)
        # ----------------------------------------------------------------
        if batch_size > 1:
            sampler = MARSBatchSampler(
                model=self.model, tokenizer=self.tokenizer
            )
            print(
                f"[MARS Config] Using MARSBatchSampler, "
                f"confidence_threshold={confidence_threshold}, "
                f"acceptance_metric={acceptance_metric}, "
                f"batch_size={batch_size}"
            )

            for i in tqdm(
                range(0, len(requests), batch_size),
                desc=f"Generating (MARS, bs={batch_size})...",
            ):
                batch = requests[i : i + batch_size]
                batch_prompts = []
                batch_stop_tokens = []
                for instance in batch:
                    context, gen_kwargs = instance.args
                    prompt_ids = self.tokenizer(context)["input_ids"]
                    batch_prompts.append(prompt_ids)
                    batch_stop_tokens.append(gen_kwargs["until"])

                generated_ids = sampler.sample(
                    inputs=batch_prompts,
                    max_new_tokens=self.max_new_tokens,
                    block_size=self.block_size,
                    temperature=0.0,
                    confidence_threshold=confidence_threshold,
                    right_shift_logits=self.right_shift_logits,
                )

                max_prompt_len = max(len(p) for p in batch_prompts)
                for b in range(len(batch)):
                    generated_answer = self.tokenizer.decode(
                        generated_ids[b][max_prompt_len:],
                        skip_special_tokens=False,
                    )
                    # Remove mask/pad tokens from output
                    for special in [
                        self.tokenizer.mask_token,
                        self.tokenizer.pad_token,
                    ]:
                        if special:
                            generated_answer = generated_answer.replace(
                                special, ""
                            )

                    for stop_seq in batch_stop_tokens[b]:
                        if stop_seq in generated_answer:
                            generated_answer = generated_answer.split(
                                stop_seq
                            )[0]

                    generated_answer_ids = self.tokenizer(generated_answer)[
                        "input_ids"
                    ]
                    generated_answer = self.tokenizer.decode(
                        generated_answer_ids, skip_special_tokens=True
                    )
                    out[i + b] = generated_answer

                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()

        # ----------------------------------------------------------------
        # Path 2: KV-cached sliding window (use_cache=True, bs=1)
        # ----------------------------------------------------------------
        elif self._use_cache:
            sampler = MARSCachedSampler(
                model=self.model, tokenizer=self.tokenizer
            )
            print(
                f"[MARS Config] Using MARSCachedSampler (KV cache), "
                f"confidence_threshold={confidence_threshold}, "
                f"acceptance_metric={acceptance_metric}, "
                f"max_accept_per_step={max_accept_per_step}"
            )

            for i in tqdm(
                range(len(requests)),
                desc="Generating (MARSCached, bs=1)...",
            ):
                instance = requests[i]
                context, gen_kwargs = instance.args
                prompt_ids = torch.tensor(
                    self.tokenizer(context)["input_ids"],
                    device=self.device,
                    dtype=torch.long,
                )
                prompt = [prompt_ids]
                stop_tokens = gen_kwargs["until"]
                generated_ids = sampler.sample(
                    inputs=prompt,
                    steps=self.steps,
                    max_new_tokens=self.max_new_tokens,
                    block_size=self.block_size,
                    temperature=0.0,
                    cfg_scale=self.cfg,
                    right_shift_logits=self.right_shift_logits,
                    confidence_threshold=confidence_threshold,
                    max_accept_per_step=max_accept_per_step,
                    acceptance_metric=acceptance_metric,
                )
                generated_answer = self.tokenizer.decode(
                    generated_ids[0][prompt[0].shape[0] :],
                    skip_special_tokens=False,
                )
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

                generated_answer_ids = self.tokenizer(generated_answer)[
                    "input_ids"
                ]
                generated_answer = self.tokenizer.decode(
                    generated_answer_ids, skip_special_tokens=True
                )
                out[i] = generated_answer
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()

            stats = sampler.get_global_stats()
            task_name = requests[0].task_name if requests else ""
            self._save_sampler_stats(stats, task_name=task_name)

        # ----------------------------------------------------------------
        # Path 3: Default -- full forward each step (bs=1)
        # ----------------------------------------------------------------
        else:
            sampler = MARSSampler(
                model=self.model, tokenizer=self.tokenizer
            )
            print(
                f"[MARS Config] Using MARSSampler, "
                f"confidence_threshold={confidence_threshold}, "
                f"acceptance_metric={acceptance_metric}, "
                f"max_accept_per_step={max_accept_per_step}"
            )

            for i in tqdm(
                range(len(requests)),
                desc="Generating (MARS, bs=1)...",
            ):
                instance = requests[i]
                context, gen_kwargs = instance.args
                prompt_ids = torch.tensor(
                    self.tokenizer(context)["input_ids"],
                    device=self.device,
                    dtype=torch.long,
                )
                prompt = [prompt_ids]
                stop_tokens = gen_kwargs["until"]
                generated_ids = sampler.sample(
                    inputs=prompt,
                    steps=self.steps,
                    max_new_tokens=self.max_new_tokens,
                    block_size=self.block_size,
                    temperature=0.0,
                    cfg_scale=self.cfg,
                    right_shift_logits=self.right_shift_logits,
                    confidence_threshold=confidence_threshold,
                    max_accept_per_step=max_accept_per_step,
                    acceptance_metric=acceptance_metric,
                )
                generated_answer = self.tokenizer.decode(
                    generated_ids[0][prompt[0].shape[0] :],
                    skip_special_tokens=False,
                )
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

                generated_answer_ids = self.tokenizer(generated_answer)[
                    "input_ids"
                ]
                generated_answer = self.tokenizer.decode(
                    generated_answer_ids, skip_special_tokens=True
                )
                out[i] = generated_answer
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()

            stats = sampler.get_global_stats()
            task_name = requests[0].task_name if requests else ""
            self._save_sampler_stats(stats, task_name=task_name)

        return out

    def _save_sampler_stats(self, stats: dict, task_name: str = ""):
        """Save sampler stats JSON to output dir (rank 0 only)."""
        is_main = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        if not is_main or not stats:
            return
        # Add config info
        stats["confidence_threshold"] = self._confidence_threshold
        stats["acceptance_metric"] = self._acceptance_metric
        stats["max_accept_per_step"] = self._max_accept_per_step
        stats["block_size"] = self.block_size
        stats["model"] = self.tokenizer.name_or_path
        stats["task"] = task_name
        # Save next to model checkpoint
        out_dir = os.path.dirname(self.tokenizer.name_or_path)
        if not out_dir or not os.path.isdir(out_dir):
            out_dir = "."
        metric_str = (
            self._acceptance_metric
            if self._acceptance_metric != "probability"
            else ""
        )
        suffix = (
            f"_ct{self._confidence_threshold}"
            if self._confidence_threshold is not None
            else "_no_ct"
        )
        if metric_str:
            suffix = f"_{metric_str}{suffix}"
        task_suffix = f"_{task_name}" if task_name else ""
        path = os.path.join(
            out_dir, f"sampler_stats{suffix}{task_suffix}.json"
        )
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[MARS] Sampler stats saved to {path}")


@register_model("jacobi")
class JacobiEvalHarness(BaseEvalHarness):
    """
    Evaluation harness for Jacobi decoding on standard AR models.
    Training-free baseline: random init + causal forward + fixed-point convergence.
    """

    def generate_until(self, requests: list[Instance]):
        from dllm.core.samplers.jacobi import JacobiSampler

        out = []
        sampler = JacobiSampler(
            model=self.model, tokenizer=self.tokenizer
        )

        print(
            f"[Jacobi Config] max_new_tokens={self.max_new_tokens}, "
            f"steps={self.steps}"
        )

        for instance in tqdm(requests, desc="Generating (Jacobi)..."):
            context, gen_kwargs = instance.args
            prompt_ids = torch.tensor(
                self.tokenizer(context)["input_ids"],
                device=self.device,
                dtype=torch.long,
            )
            prompt = [prompt_ids]
            stop_tokens = gen_kwargs["until"]
            generated_ids = sampler.sample(
                inputs=prompt,
                steps=self.steps,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
            )
            generated_answer = self.tokenizer.decode(
                generated_ids[0][prompt[0].shape[0] :],
                skip_special_tokens=False,
            )
            # Remove pad tokens from output
            if self.tokenizer.pad_token:
                generated_answer = generated_answer.replace(
                    self.tokenizer.pad_token, ""
                )

            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            generated_answer_ids = self.tokenizer(generated_answer)[
                "input_ids"
            ]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            out.append(generated_answer)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        stats = sampler.get_global_stats()
        return out


if __name__ == "__main__":
    cli_evaluate()
