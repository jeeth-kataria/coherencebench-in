#!/usr/bin/env python3
"""
CoherenceBench-IN — GPU Evaluation Runner
==========================================
Runs the 584-instance English benchmark against multiple LLMs on a CUDA GPU.
Uses 4-bit quantization (bitsandbytes) — requires as little as 3 GB VRAM for 3B models.

Usage
-----
    python scripts/run_evaluation.py --models llama3 qwen mistral
    python scripts/run_evaluation.py --models llama3 gpt4o_mini --openai-key sk-...
    python scripts/run_evaluation.py --models all

Checkpointing
-------------
Each instance result is written to results/{run_name}_results.jsonl immediately.
If interrupted, re-running the same command resumes from where it stopped.

Environment variables
---------------------
    OPENAI_API_KEY   Required for gpt4o_mini / gpt4o models
    HF_TOKEN         Required for gated models (e.g. official Meta Llama repos)
                     Not needed for the defaults below — they use ungated copies.
"""
import os, sys, re, json, time, gc, argparse, logging
from pathlib import Path
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")

# ── Paths (auto-resolved relative to this script's location) ──────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
DATA_BENCH  = REPO_ROOT / "data" / "benchmark" / "english"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Evaluation settings ───────────────────────────────────────────────────────
MAX_CTX_TOKENS = 16_000   # tiktoken tokens — truncate documents longer than this
MAX_NEW_TOKENS = 15       # we only need one word; 15 is very generous
BATCH_TIMEOUT  = 120      # seconds — skip instance and log if it hangs

SYSTEM_PROMPT = (
    "You are a precise document analyst. "
    "Read the document carefully and answer the question about its internal consistency. "
    "Your answer must be exactly ONE word: either CONSISTENT or INCONSISTENT. "
    "Do not explain. Do not add punctuation. Output only the single word."
)

# ── Model registry ────────────────────────────────────────────────────────────
# All models here are ungated (no HF token required).
# For official Meta gated repos, set HF_TOKEN env var.
MODEL_REGISTRY = {
    "llama3": {
        "hf_id":    "unsloth/Llama-3.2-3B-Instruct",
        "run_name": "llama3_3b",
        "vram_gb":  3.5,     # 4-bit
    },
    "qwen": {
        "hf_id":    "Qwen/Qwen2.5-7B-Instruct",
        "run_name": "qwen25_7b",
        "vram_gb":  5.5,
    },
    "mistral": {
        "hf_id":    "mistralai/Mistral-7B-Instruct-v0.3",
        "run_name": "mistral_7b",
        "vram_gb":  5.5,
    },
    "gpt4o_mini": {
        "api":      True,
        "model_id": "gpt-4o-mini",
        "run_name": "gpt4o_mini",
    },
    "gpt4o": {
        "api":      True,
        "model_id": "gpt-4o",
        "run_name": "gpt4o",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark() -> list[dict]:
    index_file = DATA_BENCH / "index.jsonl"
    if not index_file.exists():
        log.error(f"Benchmark index not found at {index_file}")
        log.error("Make sure you cloned the full repo with its data/benchmark/english/ directory.")
        sys.exit(1)
    instances = [json.loads(l) for l in open(index_file)]
    log.info(f"Loaded {len(instances)} benchmark instances")
    return instances


def load_instance_text(inst_id: str) -> str:
    fpath = DATA_BENCH / f"{inst_id}.json"
    with open(fpath) as f:
        return json.load(f)["text"]


def build_prompt(text: str, question: str) -> str:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        q_tokens   = len(enc.encode(question))
        doc_budget = MAX_CTX_TOKENS - q_tokens - 512
        doc_tokens = enc.encode(text)
        if len(doc_tokens) > doc_budget:
            text = enc.decode(doc_tokens[:doc_budget]) + " [...document truncated...]"
    except Exception:
        # Fallback: rough char-based truncation (4 chars ≈ 1 token)
        char_limit = (MAX_CTX_TOKENS - 512) * 4
        if len(text) > char_limit:
            text = text[:char_limit] + " [...document truncated...]"
    return f"Document:\n{text}\n\n---\n{question}"


def parse_answer(response: str) -> str:
    """Returns 'INCONSISTENT', 'CONSISTENT', or 'UNPARSEABLE'."""
    if re.search(r"\bINCONSISTENT\b", response, re.IGNORECASE):
        return "INCONSISTENT"
    if re.search(r"\bCONSISTENT\b", response, re.IGNORECASE):
        return "CONSISTENT"
    return "UNPARSEABLE"


def load_done_ids(results_file: Path) -> set:
    if not results_file.exists():
        return set()
    ids = set()
    with open(results_file) as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                pass
    return ids


def write_record(out_f, inst: dict, response: str):
    pred    = parse_answer(response)
    correct = (pred == inst["answer"])
    record  = {
        "id":             inst["id"],
        "dimension":      inst["dimension"],
        "distance":       inst["distance"],
        "context_tokens": inst["context_tokens"],
        "gold":           inst["answer"],
        "pred":           pred,
        "correct":        correct,
        "raw_response":   response.strip()[:120],
    }
    out_f.write(json.dumps(record) + "\n")
    out_f.flush()
    return pred == "UNPARSEABLE"


def print_summary(run_name: str, results_file: Path):
    import pandas as pd
    rows = [json.loads(l) for l in open(results_file)]
    df   = pd.DataFrame(rows)
    acc  = df["correct"].mean()
    log.info(f"\n{'─'*50}")
    log.info(f"  {run_name}  overall accuracy: {acc:.1%}  ({df['correct'].sum()}/{len(df)})")
    for dim, sub in df.groupby("dimension"):
        log.info(f"    {dim:<28}: {sub['correct'].mean():.1%}")
    unparseable = (df["pred"] == "UNPARSEABLE").sum()
    if unparseable:
        log.warning(f"  Unparseable responses: {unparseable}/{len(df)}")
    log.info(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Local GPU inference (4-bit quantized via bitsandbytes)
# ─────────────────────────────────────────────────────────────────────────────

def run_local_model(hf_id: str, run_name: str, instances: list):
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )
        from tqdm import tqdm
    except ImportError as e:
        log.error(f"Missing package: {e}. Run: pip install -r scripts/requirements_gpu.txt")
        sys.exit(1)

    if not torch.cuda.is_available():
        log.error("No CUDA GPU found. This script requires a CUDA GPU.")
        log.error("If on a SLURM cluster, request a GPU node first:")
        log.error("  srun --gres=gpu:1 --pty bash")
        sys.exit(1)

    results_file = RESULTS_DIR / f"{run_name}_results.jsonl"
    done_ids     = load_done_ids(results_file)
    remaining    = [i for i in instances if i["id"] not in done_ids]

    if not remaining:
        log.info(f"  {run_name}: already complete ({len(instances)} instances).")
        return

    if done_ids:
        log.info(f"  {run_name}: resuming from {len(done_ids)} done → {len(remaining)} remaining.")

    # Check VRAM
    vram_free_gb = (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated(0)
    ) / 1024**3
    vram_required = MODEL_REGISTRY.get(run_name, {}).get("vram_gb", 6)
    if vram_free_gb < vram_required:
        log.warning(
            f"  Only {vram_free_gb:.1f} GB VRAM free, model needs ~{vram_required:.1f} GB. "
            "May still work — trying..."
        )

    log.info(f"  Loading tokenizer: {hf_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    except Exception as e:
        log.error(f"Failed to load tokenizer for {hf_id}: {e}")
        log.error("If this is a gated model, set: export HF_TOKEN=hf_...")
        raise

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4-bit quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_use_double_quant = True,
    )

    log.info(f"  Loading model in 4-bit on GPU...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config = quant_config,
            device_map          = "auto",
            trust_remote_code   = True,
        )
    except Exception as e:
        log.error(f"Model load failed: {e}")
        raise
    model.eval()

    gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
    log.info(f"  Model loaded. GPU memory used: {gpu_mem:.1f} GB")
    log.info(f"  Evaluating {len(remaining)} instances...")

    unparseable = skipped = 0
    with open(results_file, "a") as out_f:
        for inst in tqdm(remaining, desc=run_name, unit="inst", dynamic_ncols=True):
            try:
                text     = load_instance_text(inst["id"])
                user_msg = build_prompt(text, inst["question"])

                messages  = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

                # Guard context length
                if inputs["input_ids"].shape[1] > MAX_CTX_TOKENS + 512:
                    inputs["input_ids"]      = inputs["input_ids"][:, -(MAX_CTX_TOKENS + 512):]
                    inputs["attention_mask"] = inputs["attention_mask"][:, -(MAX_CTX_TOKENS + 512):]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens = MAX_NEW_TOKENS,
                        do_sample      = False,
                        pad_token_id   = tokenizer.pad_token_id,
                    )

                new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

                is_unparseable = write_record(out_f, inst, response)
                if is_unparseable:
                    unparseable += 1

            except torch.cuda.OutOfMemoryError:
                log.warning(f"  OOM on instance {inst['id']} — skipping and clearing cache.")
                torch.cuda.empty_cache()
                # Write a blank record so we don't retry
                write_record(out_f, inst, "UNPARSEABLE_OOM")
                skipped += 1

            except KeyboardInterrupt:
                log.warning("  Interrupted — progress saved. Re-run to resume.")
                break

            except Exception as e:
                log.warning(f"  Error on {inst['id']}: {e} — skipping.")
                write_record(out_f, inst, "")
                skipped += 1

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"  ✅ {run_name} done. Unparseable: {unparseable}, Skipped: {skipped}")
    print_summary(run_name, results_file)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI API inference
# ─────────────────────────────────────────────────────────────────────────────

def run_openai_model(model_id: str, run_name: str, instances: list,
                     api_key: Optional[str] = None):
    try:
        from openai import OpenAI
        from tqdm import tqdm
    except ImportError:
        log.error("openai package not installed. Run: pip install openai")
        sys.exit(1)

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        log.error(
            f"OPENAI_API_KEY not set — cannot run {run_name}.\n"
            "  Set with: export OPENAI_API_KEY=sk-...\n"
            "  Or pass:  --openai-key sk-..."
        )
        return

    client       = OpenAI(api_key=key)
    results_file = RESULTS_DIR / f"{run_name}_results.jsonl"
    done_ids     = load_done_ids(results_file)
    remaining    = [i for i in instances if i["id"] not in done_ids]

    if not remaining:
        log.info(f"  {run_name}: already complete ({len(instances)} instances).")
        return
    if done_ids:
        log.info(f"  {run_name}: resuming — {len(remaining)} remaining.")

    log.info(f"  Evaluating {len(remaining)} instances with {model_id}...")
    unparseable = 0
    last_call   = 0.0

    with open(results_file, "a") as out_f:
        for inst in tqdm(remaining, desc=run_name, unit="inst", dynamic_ncols=True):
            # Rate limit: 1 req/sec
            elapsed = time.time() - last_call
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

            text     = load_instance_text(inst["id"])
            user_msg = build_prompt(text, inst["question"])
            response = ""

            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model    = model_id,
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_msg},
                        ],
                        max_tokens  = MAX_NEW_TOKENS,
                        temperature = 0.0,
                    )
                    response  = resp.choices[0].message.content or ""
                    last_call = time.time()
                    break
                except KeyboardInterrupt:
                    log.warning("  Interrupted — progress saved.")
                    out_f.flush()
                    sys.exit(0)
                except Exception as e:
                    wait = min(2 ** attempt, 60)
                    log.warning(f"  API error (attempt {attempt+1}): {e} — retrying in {wait}s...")
                    time.sleep(wait)

            is_unparseable = write_record(out_f, inst, response)
            if is_unparseable:
                unparseable += 1

    log.info(f"  ✅ {run_name} done. Unparseable: {unparseable}/{len(remaining)}")
    print_summary(run_name, results_file)


# ─────────────────────────────────────────────────────────────────────────────
# Results tables & figures
# ─────────────────────────────────────────────────────────────────────────────

def generate_outputs():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")   # headless on server
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np

    MODEL_NAMES = {
        "llama3_3b":  "Llama-3.2-3B",
        "qwen25_7b":  "Qwen2.5-7B",
        "mistral_7b": "Mistral-7B",
        "gpt4o_mini": "GPT-4o-mini",
        "gpt4o":      "GPT-4o",
    }
    DIMS = ["entity_consistency", "temporal_coherence", "causal_chain"]
    DIM_SHORT = {"entity_consistency": "Entity", "temporal_coherence": "Temporal", "causal_chain": "Causal"}

    all_results = {}
    for run_name, display in MODEL_NAMES.items():
        fpath = RESULTS_DIR / f"{run_name}_results.jsonl"
        if fpath.exists():
            df = pd.DataFrame([json.loads(l) for l in open(fpath)])
            def bucket(n):
                if n < 6000: return "4K–6K"
                if n < 12000: return "6K–12K"
                return "12K+"
            df["ctx_bucket"] = df["context_tokens"].apply(bucket)
            all_results[display] = df

    if not all_results:
        log.warning("No results to analyse yet.")
        return

    # ── Table 3: accuracy by dimension ──────────────────────────────
    rows = []
    for model_name, df in all_results.items():
        row = {"Model": model_name}
        for dim in DIMS:
            sub = df[df["dimension"] == dim]
            row[DIM_SHORT[dim]] = f"{sub['correct'].mean():.1%}" if len(sub) else "N/A"
        row["Overall"] = f"{df['correct'].mean():.1%}"
        rows.append(row)
    t3 = pd.DataFrame(rows).set_index("Model")
    t3.to_csv(RESULTS_DIR / "table3_accuracy_by_dimension.csv")
    log.info(f"\nTable 3 — Accuracy by Dimension:\n{t3.to_string()}")

    # ── Table 4: distance breakdown (INCONSISTENT only) ─────────────
    rows = []
    for model_name, df in all_results.items():
        inc = df[df["gold"] == "INCONSISTENT"]
        row = {"Model": model_name}
        for dist in ["near", "mid", "far"]:
            sub = inc[inc["distance"] == dist]
            row[dist.capitalize()] = f"{sub['correct'].mean():.1%}" if len(sub) else "N/A"
        rows.append(row)
    t4 = pd.DataFrame(rows).set_index("Model")
    t4.to_csv(RESULTS_DIR / "table4_accuracy_by_distance.csv")
    log.info(f"\nTable 4 — Distance Breakdown:\n{t4.to_string()}")

    # ── Figure 2: accuracy vs context length ────────────────────────
    BUCKETS = ["4K–6K", "6K–12K", "12K+"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, dim in zip(axes, DIMS):
        for (mname, df), c in zip(all_results.items(), colors):
            sub  = df[df["dimension"] == dim]
            accs = [sub[sub["ctx_bucket"] == b]["correct"].mean() for b in BUCKETS]
            ax.plot(BUCKETS, accs, marker="o", label=mname, color=c, linewidth=2)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Chance (50%)")
        ax.set_title(DIM_SHORT[dim], fontsize=12, fontweight="bold")
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(all_results) + 1,
               bbox_to_anchor=(0.5, -0.12), fontsize=9)
    plt.suptitle("Figure 2 — Accuracy vs. Context Length by Dimension",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = RESULTS_DIR / "fig2_accuracy_vs_context_length.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {fig2_path}")

    # ── Figure 3: bar chart by dimension ────────────────────────────
    x     = np.arange(len(DIMS))
    width = 0.8 / max(len(all_results), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (mname, df) in enumerate(all_results.items()):
        accs   = [df[df["dimension"] == d]["correct"].mean() for d in DIMS]
        offset = (i - len(all_results) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, accs, width * 0.9,
                        label=mname, color=colors[i % len(colors)])
        for bar, acc in zip(bars, accs):
            if not np.isnan(acc):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{acc:.0%}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([DIM_SHORT[d] for d in DIMS], fontsize=11)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("Figure 3 — Accuracy by Dimension", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig3_path = RESULTS_DIR / "fig3_accuracy_by_dimension.png"
    plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {fig3_path}")

    log.info("\n✅ All tables and figures saved to results/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CoherenceBench-IN evaluation runner (CUDA GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_evaluation.py --models llama3
  python scripts/run_evaluation.py --models llama3 qwen mistral
  python scripts/run_evaluation.py --models all
  python scripts/run_evaluation.py --models gpt4o_mini --openai-key sk-...
  python scripts/run_evaluation.py --models llama3 qwen --tables-only
""",
    )
    p.add_argument(
        "--models", nargs="+",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        required=True,
        help="Model(s) to evaluate. Use 'all' for all 5.",
    )
    p.add_argument(
        "--openai-key", default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var).",
    )
    p.add_argument(
        "--tables-only", action="store_true",
        help="Skip inference — just regenerate tables/figures from existing results.",
    )
    p.add_argument(
        "--results-dir", default=None,
        help=f"Override results directory (default: {RESULTS_DIR})",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.results_dir:
        global RESULTS_DIR
        RESULTS_DIR = Path(args.results_dir)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.tables_only:
        log.info("Generating tables and figures from existing results...")
        generate_outputs()
        return

    models = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models

    instances = load_benchmark()

    for model_key in models:
        cfg = MODEL_REGISTRY[model_key]
        log.info(f"\n{'='*55}")
        log.info(f"  Running model: {model_key}")
        log.info(f"{'='*55}")

        if cfg.get("api"):
            run_openai_model(
                model_id  = cfg["model_id"],
                run_name  = cfg["run_name"],
                instances = instances,
                api_key   = args.openai_key,
            )
        else:
            run_local_model(
                hf_id     = cfg["hf_id"],
                run_name  = cfg["run_name"],
                instances = instances,
            )

    log.info("\nAll requested models done. Generating tables and figures...")
    generate_outputs()
    log.info("\n✅ Phase 3 complete. Results saved to results/")


if __name__ == "__main__":
    main()
