#!/usr/bin/env bash
# =============================================================================
# CoherenceBench-IN — GPU Server Setup Script
# Run this once on your college GPU server to install everything.
# Usage: bash scripts/setup_gpu.sh
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/jeeth-kataria/coherencebench-in.git"
ENV_NAME="coherencebench"
PYTHON_VERSION="3.11"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── 1. Check CUDA ────────────────────────────────────────────────
log "Checking GPU / CUDA..."
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found. Are you on a GPU node? Use: srun --gres=gpu:1 --pty bash"
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "unknown")
log "CUDA version: ${CUDA_VER}"

# ── 2. Conda env ─────────────────────────────────────────────────
log "Setting up conda environment '${ENV_NAME}'..."
if ! command -v conda &>/dev/null; then
    warn "conda not found — trying module load..."
    module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || \
        die "conda not available. Ask your sysadmin or install Miniconda:
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3
  source \$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Conda env '${ENV_NAME}' already exists — will update packages."
else
    log "Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
log "Active env: $(conda info --envs | grep '*' | awk '{print $1}')"

# ── 3. Clone or update repo ──────────────────────────────────────
REPO_DIR="$HOME/coherencebench-in"
if [ -d "${REPO_DIR}/.git" ]; then
    log "Repo already exists at ${REPO_DIR} — pulling latest..."
    cd "${REPO_DIR}" && git pull
else
    log "Cloning repo to ${REPO_DIR}..."
    git clone "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}"
fi

# ── 4. Install Python packages ───────────────────────────────────
log "Installing Python packages (GPU edition)..."

# Pick torch version matching CUDA
if python -c "import torch; torch.cuda.is_available()" &>/dev/null; then
    warn "PyTorch already installed — skipping torch install."
else
    CUDA_SHORT=$(echo "${CUDA_VER}" | sed 's/\.//' | cut -c1-3)  # e.g. 118, 121, 124
    if [[ "${CUDA_SHORT}" -ge 121 ]]; then
        TORCH_IDX="https://download.pytorch.org/whl/cu121"
    elif [[ "${CUDA_SHORT}" -ge 118 ]]; then
        TORCH_IDX="https://download.pytorch.org/whl/cu118"
    else
        TORCH_IDX="https://download.pytorch.org/whl/cu117"
    fi
    log "Installing PyTorch for CUDA ${CUDA_SHORT} from ${TORCH_IDX}..."
    pip install torch torchvision --index-url "${TORCH_IDX}"
fi

pip install -r "${REPO_DIR}/scripts/requirements_gpu.txt"

# ── 5. Download spaCy model (needed if running corpus pipeline) ──
python -m spacy download en_core_web_sm 2>/dev/null || warn "spaCy model already downloaded."

# ── 6. Verify installation ───────────────────────────────────────
log "Verifying installation..."
python - <<'EOF'
import torch, transformers, bitsandbytes, tqdm, tiktoken, openai
print(f"  torch         {torch.__version__}  — CUDA: {torch.cuda.is_available()}")
print(f"  transformers  {transformers.__version__}")
print(f"  bitsandbytes  {bitsandbytes.__version__}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name}  ({vram:.1f} GB VRAM)")
else:
    print("  ⚠️  CUDA not available — check nvidia-smi and torch install")
EOF

log "─────────────────────────────────────────────────────"
log "✅ Setup complete!"
log ""
log "Next steps:"
log "  conda activate ${ENV_NAME}"
log "  cd ${REPO_DIR}"
log "  python scripts/run_evaluation.py --models llama3 qwen mistral"
log ""
log "With OpenAI API:"
log "  export OPENAI_API_KEY=sk-..."
log "  python scripts/run_evaluation.py --models llama3 qwen mistral gpt4o_mini"
log "─────────────────────────────────────────────────────"
