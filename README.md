# MARS: Enabling Autoregressive Models Multi-Token Generation

Code for the paper: *MARS: Enabling Autoregressive Models Multi-Token Generation*

MARS enables existing AR instruction-tuned models to generate multiple tokens per forward pass with **zero architectural changes** and a **single checkpoint**. The AR model remains fully functional -- MARS adds multi-token prediction as an additional capability through masked fine-tuning.

## Setup

```bash
git clone https://github.com/Xalp/MARS.git
cd MARS
bash setup_env.sh
conda activate mars
```

## Repository Structure

```
MARS/
├── mars/                          # Core MARS code
│   ├── trainers/
│   │   ├── attention_mask.py      # MARS training attention mask
│   │   └── mars_trainer.py        # MARSTrainer (with/without SFT loss)
│   ├── samplers/
│   │   ├── mars_sampler.py        # Sliding-window sampler
│   │   ├── mars_cached_sampler.py # KV-cached sampler (per-step + block-level)
│   │   └── mars_batch_sampler.py  # Batch sampler
│   └── eval_harness.py            # lm-eval integration
├── dllm/                          # Base infrastructure (trainers, model configs, data utils)
├── train/                         # Training entry points
│   ├── train_ar_sft.py            # Stage 1: Standard AR SFT
│   ├── train_mars.py              # Stage 2: MARS (with SFT loss)
│   └── train_mars_no_sft.py       # Stage 2: MARS (without SFT loss)
└── scripts/                       # Experiment scripts
    ├── train/                     # Training pipelines
    ├── eval/                      # Evaluation scripts
    └── benchmark/                 # Speed benchmarks
```

## Training

MARS uses a two-stage training pipeline:

**Stage 1: AR SFT** -- Standard next-token prediction fine-tuning.
**Stage 2: MARS** -- Masked block prediction with optional auxiliary AR loss.

### Quick Start (0.5B)

```bash
export BASE_DIR=/path/to/experiment/root
bash scripts/train/train_0.5b.sh
```

### Training Commands

```bash
# Stage 1: AR SFT
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes 8 train/train_ar_sft.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir ${BASE_DIR}/models/ar_sft \
    --dataset_args allenai/Dolci-Instruct-SFT \
    --num_train_epochs 5 --learning_rate 5e-6 \
    --per_device_train_batch_size 48 --max_length 512 --bf16

# Stage 2: MARS (with SFT loss, block_size=4)
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes 8 train/train_mars.py \
    --model_name_or_path ${BASE_DIR}/models/ar_sft/checkpoint-final \
    --output_dir ${BASE_DIR}/models/mars_blk4 \
    --dataset_args allenai/Dolci-Instruct-SFT \
    --num_train_epochs 5 --learning_rate 5e-6 \
    --per_device_train_batch_size 48 --max_length 512 \
    --block_size 4 --right_shift_logits --ar_weight 1.0 --bf16
```

## Evaluation

### One-token mode (Table 2)

```bash
accelerate launch --num_processes 8 mars/eval_harness.py \
    --tasks gsm8k_cot --num_fewshot 0 \
    --model mars --apply_chat_template \
    --model_args "pretrained=${BASE_DIR}/models/mars_blk4/checkpoint-final,max_new_tokens=256,steps=256,block_size=4,cfg=0.0,right_shift_logits=True"
```

### Multi-token mode with threshold (Table 4)

```bash
accelerate launch --num_processes 8 mars/eval_harness.py \
    --tasks gsm8k_cot --num_fewshot 0 \
    --model mars --apply_chat_template \
    --model_args "pretrained=${BASE_DIR}/models/mars_blk4/checkpoint-final,max_new_tokens=256,steps=256,block_size=4,cfg=0.0,right_shift_logits=True,confidence_threshold=0.95"
```

### Block-cached batch inference (Table 5)

```bash
python scripts/benchmark/bench_block_cached.py \
    --mars_model ${BASE_DIR}/models/mars_blk4/checkpoint-final \
    --ar_model ${BASE_DIR}/models/ar_sft/checkpoint-final \
    --batch_size 16 --block_sizes "4,8,16,32" --threshold 0.95
```

## Reproducing Paper Results

| Script | Paper Reference |
|--------|----------------|
| `scripts/train/train_0.5b.sh` | Table 2 (0.5B models) |
| `scripts/train/train_7b.sh` | Table 2 (7B models) |
| `scripts/train/train_blk8_16.sh` | Table 3 (block size ablation) |
| `scripts/train/train_bd3lm_baseline.sh` | Table 2 (BD3LM baseline) |
| `scripts/eval/eval_main.sh` | Table 2 (main results) |
| `scripts/eval/eval_threshold_095.sh` | Table 4 (multi-token at tau=0.95) |
| `scripts/eval/eval_threshold_sweep.sh` | Appendix Table 15 (full sweep) |
| `scripts/eval/eval_acceptance_metrics.sh` | Appendix Figure 7 (metric comparison) |
| `scripts/benchmark/bench_block_cached.sh` | Table 5 (wall-clock speedup) |
| `scripts/benchmark/bench_full_sweep.sh` | Figure 5 (parameter sweep) |

## Hyperparameters

| | 0.5B | 7B |
|---|---|---|
| Base model | Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct |
| Dataset | Dolci-Instruct-SFT (~2M) | Dolci-Instruct-SFT (~2M) |
| Epochs (both stages) | 5 | 5 |
| Learning rate | 5e-6 | 5e-6 |
| Effective batch size | 384 | 384 |
| Max sequence length | 512 | 512 |
| Block sizes tested | 4, 8, 16 | 4 |
| Hardware | 8x H200 | 8x H200 |
