"""
Benchmark MARSBlockCachedSampler vs AR at batch_size=16.
Tests different block sizes (4, 8, 16, 32) to find optimal.

Usage: python scripts/benchmark/bench_block_cached.py \
    --mars_model <path> --ar_model <path> --batch_size 16 --threshold 0.95
"""
import argparse
import json
import os
import re
import time
import torch
import transformers
from datasets import load_dataset

from dllm.pipelines.a2d import A2DQwen2Config
from mars.samplers import MARSBlockCachedSampler


def load_gsm8k(limit=None):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    questions = [row["question"] for row in ds]
    answers = [row["answer"].split("####")[-1].strip().replace(",", "") for row in ds]
    return questions, answers


def extract_answer(text):
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def benchmark_block_cached(model, tokenizer, questions, batch_size, threshold,
                           block_size, max_new_tokens=256):
    sampler = MARSBlockCachedSampler(model=model, tokenizer=tokenizer)
    all_responses = []
    total_tokens = 0

    t_start = time.time()
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i:i + batch_size]
        batch_prompts = []
        for q in batch_q:
            messages = [{"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(tokenizer.encode(prompt, add_special_tokens=False))

        output = sampler.sample(
            inputs=batch_prompts,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=0.0,
            cfg_scale=0.0,
            confidence_threshold=threshold,
            right_shift_logits=True,
        )

        max_pl = max(len(p) for p in batch_prompts)
        for b in range(len(batch_q)):
            gen = output[b, max_pl:].tolist()
            trimmed = []
            for tid in gen:
                if tid in (tokenizer.eos_token_id, tokenizer.mask_token_id,
                           tokenizer.pad_token_id):
                    break
                trimmed.append(tid)
            all_responses.append(tokenizer.decode(trimmed))
            total_tokens += len(trimmed)

    elapsed = time.time() - t_start
    stats = sampler.get_global_stats()
    return all_responses, total_tokens, elapsed, stats


def benchmark_ar(model_path, tokenizer, questions, batch_size, max_new_tokens=256):
    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.model_type = "qwen2"
    if hasattr(config, "auto_map"):
        delattr(config, "auto_map")
    ar_model = transformers.Qwen2ForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16,
    ).cuda().eval()

    all_responses = []
    total_tokens = 0

    t_start = time.time()
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i:i + batch_size]
        batch_texts = []
        for q in batch_q:
            messages = [{"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(prompt)

        tokenizer.padding_side = "left"
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            outputs = ar_model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        for b in range(len(batch_q)):
            gen_tokens = outputs[b, input_len:]
            text_tokens = []
            for t in gen_tokens:
                tid = t.item()
                if tid == tokenizer.eos_token_id:
                    break
                text_tokens.append(tid)
            all_responses.append(tokenizer.decode(text_tokens))
            total_tokens += len(text_tokens)

    elapsed = time.time() - t_start
    del ar_model
    torch.cuda.empty_cache()
    return all_responses, total_tokens, elapsed


def compute_accuracy(responses, answers):
    correct = sum(1 for r, a in zip(responses, answers) if extract_answer(r) == a)
    return correct / len(answers) if answers else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mars_model", type=str, required=True)
    parser.add_argument("--ar_model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_sizes", type=str, default="4,8,16,32")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    block_sizes = [int(x) for x in args.block_sizes.split(",")]

    print(f"Loading GSM8K (limit={args.limit})...")
    questions, answers = load_gsm8k(args.limit)

    print(f"Loading MARS model from {args.mars_model}...")
    config = A2DQwen2Config.from_pretrained(args.mars_model, trust_remote_code=True)
    config.model_type = A2DQwen2Config.model_type
    mars_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.mars_model, config=config, torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda().eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.mars_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results = []

    for bs in block_sizes:
        print(f"\n{'='*60}")
        print(f"MARS BlockCached bs={args.batch_size}, block_size={bs}, threshold={args.threshold}")
        print(f"{'='*60}")

        responses, total_tokens, elapsed, stats = benchmark_block_cached(
            mars_model, tokenizer, questions, args.batch_size, args.threshold,
            bs, args.max_new_tokens,
        )
        acc = compute_accuracy(responses, answers)
        tps = total_tokens / elapsed if elapsed > 0 else 0

        print(f"  Accuracy:    {acc*100:.1f}%")
        print(f"  Tokens/sec:  {tps:.1f}")
        print(f"  Time:        {elapsed:.2f}s")
        print(f"  Avg tok/fwd: {stats['avg_tokens_per_forward']:.3f}")

        results.append({
            "model": f"MARS+BlockCache(blk{bs})", "batch_size": args.batch_size,
            "block_size": bs, "threshold": args.threshold, "accuracy": acc,
            "total_tokens": total_tokens, "time_sec": elapsed, "tokens_per_sec": tps,
        })

    del mars_model
    torch.cuda.empty_cache()

    # AR baseline
    print(f"\n{'='*60}")
    print(f"AR bs={args.batch_size}")
    print(f"{'='*60}")

    responses, total_tokens, elapsed = benchmark_ar(
        args.ar_model, tokenizer, questions, args.batch_size, args.max_new_tokens,
    )
    acc = compute_accuracy(responses, answers)
    tps = total_tokens / elapsed if elapsed > 0 else 0

    print(f"  Accuracy:    {acc*100:.1f}%")
    print(f"  Tokens/sec:  {tps:.1f}")
    print(f"  Time:        {elapsed:.2f}s")

    results.append({
        "model": "AR", "batch_size": args.batch_size, "block_size": None,
        "threshold": None, "accuracy": acc,
        "total_tokens": total_tokens, "time_sec": elapsed, "tokens_per_sec": tps,
    })

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Model':<30} {'Accuracy':<10} {'Tok/sec':<10} {'Time(s)':<10}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['model']:<30} {r['accuracy']*100:>7.1f}%  {r['tokens_per_sec']:>8.1f}  {r['time_sec']:>8.2f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
