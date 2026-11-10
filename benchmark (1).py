"""
Llama-3-8B inference benchmark: 16-bit vs 4-bit AWQ.

Measures, per model:
  - time-to-first-token (TTFT, ms)        -> prefill latency
  - decode throughput (tokens/s)          -> tokens after the first / decode time
  - VRAM: weights after load + peak during generation (GB)
  - MMLU accuracy (optional, 0- or k-shot, letter log-likelihood)

Usage:
  python benchmark.py                          # both default models, speed + memory
  python benchmark.py --mmlu --mmlu-limit 20   # also MMLU, 20 questions per subject
  python benchmark.py --models fp16            # just one variant
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

MODELS = {
    "fp16": "meta-llama/Meta-Llama-3-8B-Instruct",
    "awq4": "casperhansen/llama-3-8b-instruct-awq",
}

PROMPTS = [
    "Explain how synthetic aperture radar forms an image, in simple terms.",
    "Write a Python function that computes the FFT of a signal using NumPy and plots its magnitude.",
    "Summarise the trade-offs between floating-point and fixed-point arithmetic on FPGAs.",
    "What causes inter-symbol interference in an optical link, and how does an equalizer fix it?",
    "List five practical tips for debugging embedded firmware that talks over I2C.",
]

GB = 1024 ** 3


# --------------------------------------------------------------------------- multi-GPU helpers
def gpus():
    return range(torch.cuda.device_count())


def sync_all():
    for i in gpus():
        torch.cuda.synchronize(i)


def reset_peak_all():
    for i in gpus():
        torch.cuda.reset_peak_memory_stats(i)


def allocated_gb():
    return sum(torch.cuda.memory_allocated(i) for i in gpus()) / GB


def peak_gb():
    return sum(torch.cuda.max_memory_allocated(i) for i in gpus()) / GB


# --------------------------------------------------------------------------- timing
class TimingStreamer(BaseStreamer):
    """Records a timestamp every time generate() emits tokens.

    generate() first calls put() with the prompt ids, then once per new token,
    so timestamps[1] is the arrival time of the first generated token.
    """

    def __init__(self):
        self.timestamps = []

    def put(self, value):
        sync_all()
        self.timestamps.append(time.perf_counter())

    def end(self):
        pass


def build_inputs(tok, prompt, device):
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
    return ids.to(device)


@torch.inference_mode()
def run_once(model, tok, prompt, max_new_tokens):
    input_ids = build_inputs(tok, prompt, model.device)
    streamer = TimingStreamer()
    sync_all()
    t0 = time.perf_counter()
    out = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,  # fixed output length -> comparable runs
        do_sample=False,
        streamer=streamer,
        pad_token_id=tok.eos_token_id,
    )
    sync_all()
    t_end = time.perf_counter()

    n_new = out.shape[1] - input_ids.shape[1]
    t_first = streamer.timestamps[1]
    ttft = t_first - t0
    decode_time = t_end - t_first
    tps = (n_new - 1) / decode_time if n_new > 1 else float("nan")
    return {"ttft_s": ttft, "tokens_per_s": tps, "new_tokens": n_new, "prompt_tokens": input_ids.shape[1]}


def speed_benchmark(model, tok, max_new_tokens, repeats, warmup):
    for _ in range(warmup):
        run_once(model, tok, PROMPTS[0], 16)

    reset_peak_all()
    runs = []
    for _ in range(repeats):
        for p in PROMPTS:
            runs.append(run_once(model, tok, p, max_new_tokens))
    peak = peak_gb()

    return {
        "ttft_ms_median": 1000 * statistics.median(r["ttft_s"] for r in runs),
        "tokens_per_s_median": statistics.median(r["tokens_per_s"] for r in runs),
        "peak_vram_gb": peak,
        "runs": len(runs),
    }


# --------------------------------------------------------------------------- MMLU
CHOICES = ["A", "B", "C", "D"]


def format_q(row, with_answer):
    s = row["question"].strip() + "\n"
    for letter, choice in zip(CHOICES, row["choices"]):
        s += f"{letter}. {choice}\n"
    s += "Answer:"
    if with_answer:
        s += f" {CHOICES[row['answer']]}\n\n"
    return s


@torch.inference_mode()
def mmlu_eval(model, tok, limit_per_subject, shots):
    from datasets import load_dataset

    test = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")

    letter_ids = [tok.encode(" " + c, add_special_tokens=False)[-1] for c in CHOICES]

    by_subject = {}
    for row in test:
        by_subject.setdefault(row["subject"], []).append(row)
    dev_by_subject = {}
    for row in dev:
        dev_by_subject.setdefault(row["subject"], []).append(row)

    from tqdm.auto import tqdm

    n_total = sum(min(len(r), limit_per_subject) if limit_per_subject else len(r) for r in by_subject.values())
    bar = tqdm(total=n_total, desc="  MMLU", mininterval=5)
    correct = total = 0
    for subject, rows in sorted(by_subject.items()):
        header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        fewshot = "".join(format_q(r, True) for r in dev_by_subject.get(subject, [])[:shots])
        for row in rows[:limit_per_subject] if limit_per_subject else rows:
            prompt = header + fewshot + format_q(row, False)
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            logits = model(ids).logits[0, -1, letter_ids]
            pred = int(torch.argmax(logits))
            correct += int(pred == row["answer"])
            total += 1
            bar.update(1)
            bar.set_postfix(acc=f"{correct / total:.3f}")
    bar.close()
    return {"mmlu_accuracy": correct / total, "mmlu_questions": total, "mmlu_shots": shots}


# --------------------------------------------------------------------------- main
def load(model_id):
    torch.cuda.empty_cache()
    reset_peak_all()
    tok = AutoTokenizer.from_pretrained(model_id)
    # "auto" splits the model across every visible GPU (e.g. Kaggle's 2x T4).
    # CPU offload is disabled so a model that doesn't fit fails loudly instead of running slowly.
    max_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in gpus()}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", max_memory=max_memory
    )
    model.eval()
    placement = sorted({str(d) for d in getattr(model, "hf_device_map", {}).values()})
    print(f"  model placed on GPU(s): {', '.join(placement) or '0'}")
    sync_all()
    weights_gb = allocated_gb()
    return model, tok, weights_gb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="keys from MODELS or HF model ids")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--mmlu", action="store_true", help="also run MMLU")
    ap.add_argument("--mmlu-limit", type=int, default=0, help="questions per subject (0 = all)")
    ap.add_argument("--mmlu-shots", type=int, default=5)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "This benchmark needs a CUDA GPU."
    gpu = torch.cuda.get_device_properties(0)
    n = torch.cuda.device_count()
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in gpus()) / GB
    print(f"GPU: {n} x {gpu.name} ({total:.1f} GB total)\n")
    try:
        import awq_ext  # noqa: F401
        print("AutoAWQ GPU kernels: available\n")
    except ImportError:
        print("AutoAWQ GPU kernels: NOT available - AWQ speed will be much slower than normal\n")

    results = {"gpu": f"{n} x {gpu.name}", "gpu_mem_gb": total, "models": {}}
    for key in args.models:
        model_id = MODELS.get(key, key)
        print(f"=== {key}: {model_id}")
        model, tok, weights_gb = load(model_id)
        r = {"model_id": model_id, "weights_vram_gb": weights_gb}
        print("  running speed test...", flush=True)
        r.update(speed_benchmark(model, tok, args.max_new_tokens, args.repeats, args.warmup))
        show(r)
        results["models"][key] = r
        Path(args.out).write_text(json.dumps(results, indent=2))  # save speed results before MMLU
        if args.mmlu:
            m = mmlu_eval(model, tok, args.mmlu_limit, args.mmlu_shots)
            r.update(m)
            show(m)
            Path(args.out).write_text(json.dumps(results, indent=2))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print_table(results["models"])
    print(f"\nSaved {args.out}")


def show(d):
    for k, v in d.items():
        print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}", flush=True)


def print_table(models):
    cols = [("weights_vram_gb", "Weights VRAM (GB)"), ("peak_vram_gb", "Peak VRAM (GB)"),
            ("tokens_per_s_median", "Tokens/s"), ("ttft_ms_median", "TTFT (ms)"), ("mmlu_accuracy", "MMLU acc")]
    print("| Model | " + " | ".join(c[1] for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for key, r in models.items():
        cells = [f"{r[c]:.3f}" if c in r else "-" for c, _ in cols]
        print(f"| {key} | " + " | ".join(cells) + " |")
    if "fp16" in models and "awq4" in models:
        a, b = models["fp16"], models["awq4"]
        print(f"\nVRAM reduction (weights): {100 * (1 - b['weights_vram_gb'] / a['weights_vram_gb']):.1f}%")
        if "mmlu_accuracy" in a and "mmlu_accuracy" in b:
            print(f"MMLU change: {100 * (b['mmlu_accuracy'] - a['mmlu_accuracy']):+.2f} points")


if __name__ == "__main__":
    main()
