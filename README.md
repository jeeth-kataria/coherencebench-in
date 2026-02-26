# CoherenceBench-IN

**Evaluating Long-Context Discourse Coherence in Large Language Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

CoherenceBench-IN is a benchmark for evaluating whether Large Language Models maintain discourse coherence — entity consistency, temporal logic, and causal chain integrity — across long contexts (4K–32K+ tokens).

Current long-context benchmarks (RULER, LongBench, InfiniteBench) test **retrieval** — can the model find a fact? CoherenceBench-IN tests **coherence** — does the model notice when a document contradicts itself?

### Key Features

- **3 Coherence Dimensions:** Entity Consistency, Temporal Coherence, Causal Chain Integrity
- **Controlled Incoherence Injection:** Programmatically corrupt clean texts at configurable token distances
- **Multi-Language:** English (primary), Hindi, Tamil/Telugu (extensions)
- **Distance Analysis:** Measures how performance degrades as corruption moves farther from the question

## Methodology

1. Take long-form texts (Wikipedia, Project Gutenberg)
2. Inject controlled incoherences (entity swaps, date violations, causal breaks)
3. Generate comprehension questions where answers differ for clean vs. corrupted text
4. Evaluate LLMs — do they catch the corruption or blindly accept it?

## Project Structure

```
coherencebench-in/
├── src/
│   ├── corruption_engines/     # Entity, temporal, causal corruption
│   ├── data_pipeline/          # Source corpus loading and filtering
│   ├── evaluation/             # Model inference and scoring
│   └── utils/                  # Shared utilities
├── data/
│   ├── raw/                    # Downloaded source corpora
│   ├── processed/              # Filtered candidate texts
│   └── benchmark/              # Final benchmark instances
├── notebooks/                  # Colab/Jupyter notebooks
├── paper/                      # LaTeX paper source
├── references/                 # Literature notes
├── scripts/                    # CLI scripts
├── tests/                      # Unit tests
└── results/                    # Evaluation results
```

## Models Evaluated

| Model | Parameters | Access |
|-------|-----------|--------|
| LLaMA-3.2-3B-Instruct | 3B | Colab (4-bit) |
| Qwen2.5-7B-Instruct | 7B | Kaggle (4-bit) |
| Mistral-7B-Instruct-v0.3 | 7B | Kaggle (4-bit) |
| Gemma-2-9B-it | 9B | Kaggle (4-bit) |
| Aya-23-8B | 8B | Kaggle (4-bit) |
| GPT-4o-mini | — | API |
| GPT-4o | — | API (sample) |

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/coherencebench-in.git
cd coherencebench-in

# Install dependencies
pip install -r requirements.txt

# Run evaluation on a single model
python scripts/evaluate.py --model "Qwen/Qwen2.5-3B-Instruct" --split test
```

## Setup (Google Colab)

Open `notebooks/00_setup.ipynb` in Google Colab for a ready-to-run environment with all dependencies.

## Citation

```bibtex
@article{kataria2026coherencebench,
  title={CoherenceBench-IN: Evaluating Long-Context Discourse Coherence in Large Language Models Across English and Indian Languages},
  author={Kataria, Jeeth},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Source texts from Wikipedia and Project Gutenberg (public domain)
- AI4Bharat for Indian language resources
- Evaluation run on Google Colab and Kaggle free tier
