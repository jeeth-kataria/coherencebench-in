# Research Log — CoherenceBench-IN

## Entry 1 — 27 February 2026 (Phase 0: Setup)

**Hours this week:** 2  
**Phase:** 0 — Pre-Research Setup

### What I did:
- Created GitHub repo structure (`coherencebench-in/`)
- Set up full project directory: `src/`, `data/`, `notebooks/`, `paper/`, `references/`, `scripts/`, `tests/`, `results/`
- Created Python package structure with `__init__.py` files
- Wrote central config (`src/utils/config.py`) with all model IDs, benchmark parameters, language priorities
- Created Colab setup notebook (`notebooks/00_setup.ipynb`) — installs deps, mounts Drive, tests GPU, loads a 4-bit model, tests Wikipedia + spaCy
- Created `requirements.txt`, `README.md`, `LICENSE`, `.gitignore`
- Updated plan.md: Indian languages deprioritized to last phase (English-first strategy)

### Key decision:
- **Language priority changed:** English is now the core benchmark (600+ instances). Hindi and Tamil/Telugu are last-priority extensions added only after all 3 English corruption engines are complete, validated, and evaluated. English-only is a publishable paper on its own.

### What's next (immediately):
- Phase 0 remaining: Read RULER paper, write summary
- Then: Phase 1, Week 1 — start benchmark landscape table

### Blockers:
- None

---
