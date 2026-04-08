"""
Stage 2: MARS Training (with SFT loss).
Trains the model to predict masked blocks with auxiliary AR loss on clean stream.
Loss = CE(noisy_logits, labels) + ar_weight * CE(clean_logits, labels)

Usage:
    accelerate launch --config_file configs/zero2.yaml --num_processes 8 \
        train/train_mars.py \
        --model_name_or_path models/ar_sft/checkpoint-final \
        --output_dir models/mars_blk4 \
        --dataset_args allenai/Dolci-Instruct-SFT \
        --num_train_epochs 5 \
        --learning_rate 5e-6 \
        --per_device_train_batch_size 48 \
        --block_size 4 \
        --right_shift_logits \
        --ar_weight 1.0 \
        --max_length 512 \
        --bf16
"""

import os
from dataclasses import dataclass, field
from functools import partial

import torch
import transformers
import accelerate

import dllm
from dllm.pipelines.a2d import A2DQwen2Config, A2DQwen3Config, A2DQwen3InstructConfig
from mars.trainers import MARSTrainer

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "models/ar_sft/checkpoint-final"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/Dolci-Instruct-SFT"
    max_length: int = 512
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/mars_blk4"
    group_by_length: bool = True
    num_train_epochs: int = 5
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 48
    per_device_eval_batch_size: int = 16
    block_size: int = 4
    right_shift_logits: bool = False
    ar_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for AR loss on clean stream. 0 = pure mask, 1 = equal weight."},
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    base_config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
    )
    a2d_config_map = {
        "qwen2": A2DQwen2Config,
        "qwen3": A2DQwen3InstructConfig,
    }
    A2DConfigClass = a2d_config_map.get(base_config.model_type)
    if A2DConfigClass is None:
        raise ValueError(f"Unsupported model type: {base_config.model_type}. Supported: {list(a2d_config_map.keys())}")
    config = A2DConfigClass.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
    )
    config.model_type = A2DConfigClass.model_type

    model = dllm.utils.get_model(model_args=model_args, config=config)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args, config=config)

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_mdlm_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start MARS training (with SFT loss)...")
    trainer = MARSTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        block_size=training_args.block_size,
        right_shift_logits=training_args.right_shift_logits,
        ar_weight=training_args.ar_weight,
        data_collator=(
            dllm.core.trainers.bd3lm.AppendEOSBlockWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                ),
                block_size=training_args.block_size,
            )
        ),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
