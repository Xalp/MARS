"""
Full sweep benchmark: MARS (no-cache / block-cache) vs AR on GSM8K.

Sweeps over block_size and batch_size combinations.
Reports accuracy, tokens/sec, wall-clock time for each.

Usage:
  python scripts/benchmark/bench_full_sweep.py \
      --mars_model <path> --ar_model <path> \
      --block_sizes "1,2,4,8,16,32,64,128,256" \
      --batch_sizes "1,2,4,8,16,32" \
      --threshold 0.95 --limit 256 --output results/sweep.csv
"""
import argparse
import csv
import json
import os
import re
import time
import torch
import transformers
from datasets import load_dataset

from dllm.pipelines.a2d import A2DQwen2Config
from mars.samplers import MARSBatchSampler
from mars.samplers import MARSBlockCachedSampler


def load_gsm8k(limit=256):
    ds = load_dataset("openai/gsm8k", "main", split="test")
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


def compute_accuracy(responses, answers):
    correct = sum(1 for r, a in zip(responses, answers) if extract_answer(r) == a)
    return correct / len(answers) if answers else 0


def make_prompts(tokenizer, questions):
    """Pre-tokenize all questions."""
    prompts = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(tokenizer.encode(text, add_special_tokens=False))
    return prompts


def decode_outputs(output, batch_prompts, tokenizer):
    """Decode batch output to response strings."""
    max_pl = max(len(p) for p in batch_prompts)
    responses = []
    total_tokens = 0
    stop_ids = {tokenizer.eos_token_id, tokenizer.mask_token_id, tokenizer.pad_token_id}
    for b in range(len(batch_prompts)):
        gen = output[b, max_pl:].tolist()
        trimmed = []
        for tid in gen:
            if tid in stop_ids:
                break
            trimmed.append(tid)
        responses.append(tokenizer.decode(trimmed))
        total_tokens += len(trimmed)
    return responses, total_tokens


# ============================================================
# Benchmark runners
# ============================================================

def run_mars_nocache(model, tokenizer, prompts, batch_size, block_size, threshold, max_new_tokens):
    sampler = MARSBatchSampler(model=model, tokenizer=tokenizer)
    all_responses = []
    total_tokens = 0
    t0 = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        output = sampler.sample(
            inputs=batch, max_new_tokens=max_new_tokens, block_size=block_size,
            temperature=0.0, confidence_threshold=threshold, right_shift_logits=True,
        )
        resps, toks = decode_outputs(output, batch, tokenizer)
        all_responses.extend(resps)
        total_tokens += toks

    return all_responses, total_tokens, time.time() - t0


def run_mars_block_cache(model, tokenizer, prompts, batch_size, block_size, threshold, max_new_tokens):
    sampler = MARSBlockCachedSampler(model=model, tokenizer=tokenizer)
    all_responses = []
    total_tokens = 0
    t0 = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        output = sampler.sample(
            inputs=batch, max_new_tokens=max_new_tokens, block_size=block_size,
            temperature=0.0, cfg_scale=0.0, confidence_threshold=threshold,
            right_shift_logits=True,
        )
        resps, toks = decode_outputs(output, batch, tokenizer)
        all_responses.extend(resps)
        total_tokens += toks

    return all_responses, total_tokens, time.time() - t0


def run_ar(ar_model, tokenizer, prompts_text, batch_size, max_new_tokens):
    """AR with HuggingFace generate (KV cache by default)."""
    all_responses = []
    total_tokens = 0
    t0 = time.time()

    tokenizer.padding_side = "left"
    for i in range(0, len(prompts_text), batch_size):
        batch_texts = prompts_text[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            outputs = ar_model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        input_len = inputs["input_ids"].shape[1]
        for b in range(len(batch_texts)):
            gen = outputs[b, input_len:]
            toks = []
            for t in gen:
                tid = t.item()
                if tid == tokenizer.eos_token_id:
                    break
                toks.append(tid)
            all_responses.append(tokenizer.decode(toks))
            total_tokens += len(toks)

    return all_responses, total_tokens, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mars_model", type=str, required=True)
    parser.add_argument("--ar_model", type=str, required=True)
    parser.add_argument("--block_sizes", type=str, default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--output", type=str, default="results/full_sweep.csv")
    parser.add_argument("--model_name", type=str, default="0.5B",
                        help="Label for output (e.g., 0.5B or 7B)")
    args = parser.parse_args()

    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"=== Full Sweep: {args.model_name} ===")
    print(f"Block sizes: {block_sizes}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Threshold: {args.threshold}")
    print(f"Limit: {args.limit} questions")

    # Load data
    questions, answers = load_gsm8k(args.limit)

    # Load MARS model
    print(f"\nLoading MARS model: {args.mars_model}")
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

    prompts = make_prompts(tokenizer, questions)
    # Also make text prompts for AR
    prompts_text = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        prompts_text.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    results = []

    # ============================================================
    # MARS no-cache sweep
    # ============================================================
    for bs in batch_sizes:
        for blk in block_sizes:
            label = f"MARS(nocache) blk={blk} bs={bs}"
            print(f"\n>>> {label}")
            try:
                resps, total_tok, elapsed = run_mars_nocache(
                    mars_model, tokenizer, prompts, bs, blk,
                    args.threshold, args.max_new_tokens)
                acc = compute_accuracy(resps, answers)
                tps = total_tok / elapsed if elapsed > 0 else 0
                print(f"    Acc={acc*100:.1f}%  Tok/s={tps:.1f}  Time={elapsed:.1f}s")
                results.append({
                    "model_size": args.model_name, "method": "MARS_nocache",
                    "block_size": blk, "batch_size": bs,
                    "accuracy": round(acc * 100, 1), "tokens_per_sec": round(tps, 1),
                    "time_sec": round(elapsed, 2), "total_tokens": total_tok,
                })
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({
                    "model_size": args.model_name, "method": "MARS_nocache",
                    "block_size": blk, "batch_size": bs,
                    "accuracy": None, "tokens_per_sec": None,
                    "time_sec": None, "total_tokens": None, "error": str(e),
                })

    # ============================================================
    # MARS block-cache sweep
    # ============================================================
    for bs in batch_sizes:
        for blk in block_sizes:
            label = f"MARS(block_cache) blk={blk} bs={bs}"
            print(f"\n>>> {label}")
            try:
                resps, total_tok, elapsed = run_mars_block_cache(
                    mars_model, tokenizer, prompts, bs, blk,
                    args.threshold, args.max_new_tokens)
                acc = compute_accuracy(resps, answers)
                tps = total_tok / elapsed if elapsed > 0 else 0
                print(f"    Acc={acc*100:.1f}%  Tok/s={tps:.1f}  Time={elapsed:.1f}s")
                results.append({
                    "model_size": args.model_name, "method": "MARS_block_cache",
                    "block_size": blk, "batch_size": bs,
                    "accuracy": round(acc * 100, 1), "tokens_per_sec": round(tps, 1),
                    "time_sec": round(elapsed, 2), "total_tokens": total_tok,
                })
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({
                    "model_size": args.model_name, "method": "MARS_block_cache",
                    "block_size": blk, "batch_size": bs,
                    "accuracy": None, "tokens_per_sec": None,
                    "time_sec": None, "total_tokens": None, "error": str(e),
                })

    del mars_model
    torch.cuda.empty_cache()

    # ============================================================
    # AR sweep (only batch_size varies, no block_size)
    # ============================================================
    print(f"\nLoading AR model: {args.ar_model}")
    ar_config = transformers.AutoConfig.from_pretrained(args.ar_model, trust_remote_code=True)
    ar_config.model_type = "qwen2"
    if hasattr(ar_config, "auto_map"):
        delattr(ar_config, "auto_map")
    ar_model = transformers.Qwen2ForCausalLM.from_pretrained(
        args.ar_model, config=ar_config, torch_dtype=torch.bfloat16,
    ).cuda().eval()

    for bs in batch_sizes:
        label = f"AR bs={bs}"
        print(f"\n>>> {label}")
        try:
            resps, total_tok, elapsed = run_ar(
                ar_model, tokenizer, prompts_text, bs, args.max_new_tokens)
            acc = compute_accuracy(resps, answers)
            tps = total_tok / elapsed if elapsed > 0 else 0
            print(f"    Acc={acc*100:.1f}%  Tok/s={tps:.1f}  Time={elapsed:.1f}s")
            results.append({
                "model_size": args.model_name, "method": "AR",
                "block_size": None, "batch_size": bs,
                "accuracy": round(acc * 100, 1), "tokens_per_sec": round(tps, 1),
                "time_sec": round(elapsed, 2), "total_tokens": total_tok,
            })
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                "model_size": args.model_name, "method": "AR",
                "block_size": None, "batch_size": bs,
                "accuracy": None, "tokens_per_sec": None,
                "time_sec": None, "total_tokens": None, "error": str(e),
            })

    del ar_model
    torch.cuda.empty_cache()

    # ============================================================
    # Save results
    # ============================================================
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_size", "method", "block_size", "batch_size",
            "accuracy", "tokens_per_sec", "time_sec", "total_tokens",
        ])
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in writer.fieldnames}
            writer.writerow(row)

    # JSON
    json_path = args.output.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Method':<25} {'BlkSz':<6} {'BS':<4} {'Acc%':<7} {'Tok/s':<9} {'Time':<8}")
    print(f"{'='*90}")
    for r in results:
        blk = str(r.get("block_size") or "-")
        acc = f"{r['accuracy']:.1f}" if r.get("accuracy") is not None else "ERR"
        tps = f"{r['tokens_per_sec']:.1f}" if r.get("tokens_per_sec") is not None else "ERR"
        t = f"{r['time_sec']:.1f}" if r.get("time_sec") is not None else "ERR"
        print(f"{r['method']:<25} {blk:<6} {r['batch_size']:<4} {acc:<7} {tps:<9} {t:<8}")

    print(f"\nResults saved to {args.output} and {json_path}")


if __name__ == "__main__":
    main()
