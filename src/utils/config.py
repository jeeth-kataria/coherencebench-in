"""
CoherenceBench-IN Configuration

Central config for model names, paths, hyperparameters, and constants.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark"
RESULTS_DIR = PROJECT_ROOT / "results"

# ─── Benchmark Parameters ────────────────────────────────────────────
DIMENSIONS = ["entity_consistency", "temporal_coherence", "causal_chain"]

LANGUAGES = {
    "en": {"name": "English", "priority": 1, "target_instances": 600},
    "hi": {"name": "Hindi", "priority": 3, "target_instances": 200},
    "ta": {"name": "Tamil", "priority": 3, "target_instances": 100},
}

CONTEXT_LENGTH_BUCKETS = {
    "short": {"min_tokens": 4_000, "max_tokens": 8_000, "proportion": 0.30},
    "medium": {"min_tokens": 8_000, "max_tokens": 32_000, "proportion": 0.50},
    "long": {"min_tokens": 32_000, "max_tokens": 65_000, "proportion": 0.20},
}

CORRUPTION_DISTANCE_BUCKETS = {
    "near": {"max_tokens": 2_000},
    "mid": {"min_tokens": 2_000, "max_tokens": 8_000},
    "far": {"min_tokens": 8_000},
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# ─── Models ──────────────────────────────────────────────────────────
EVAL_MODELS = {
    "llama-3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "type": "open",
        "platform": "colab",
        "quantization": "4bit",
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "type": "open",
        "platform": "kaggle",
        "quantization": "4bit",
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "type": "open",
        "platform": "kaggle",
        "quantization": "4bit",
    },
    "gemma-2-9b": {
        "hf_id": "google/gemma-2-9b-it",
        "type": "open",
        "platform": "kaggle",
        "quantization": "4bit",
    },
    "aya-23-8b": {
        "hf_id": "CohereForAI/aya-23-8B",
        "type": "open",
        "platform": "kaggle",
        "quantization": "4bit",
    },
    "gpt-4o-mini": {
        "hf_id": None,
        "type": "api",
        "platform": "openai",
        "quantization": None,
    },
    "gpt-4o": {
        "hf_id": None,
        "type": "api",
        "platform": "openai",
        "quantization": None,
    },
}

# ─── Evaluation ──────────────────────────────────────────────────────
PROMPT_TEMPLATE = """Read the following passage carefully, then answer the question.

Passage: {passage}

Question: {question}

Answer concisely in 1-3 words."""

MIN_ENTITY_DENSITY = 5  # minimum named entities per text chunk
QUALITY_THRESHOLD = 0.70  # minimum human-validated quality rate
MIN_ACCURACY_CLEAN = 0.80  # sanity check: models should score this on clean
MIN_ACCURACY_DROP = 0.15  # minimum delta to validate corruption difficulty

# ─── Source Corpora ──────────────────────────────────────────────────
SOURCE_CORPORA = {
    # NOTE: Use 'wikimedia/wikipedia', NOT the old 'wikipedia'.
    # The legacy 'wikipedia' dataset relied on a .py script which is no longer supported.
    "wikipedia_en": {
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.en",
        "language": "en",
    },
    "wikipedia_hi": {
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.hi",
        "language": "hi",
    },
    "wikipedia_ta": {
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.ta",
        "language": "ta",
    },
    "gutenberg": {
        "source": "gutenberg.org",
        "language": "en",
        "note": "Public domain books, manually curated subset",
    },
}
