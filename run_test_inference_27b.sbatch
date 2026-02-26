#!/usr/bin/env bash
#SBATCH -A r00877
#SBATCH -p gpu-interactive
#SBATCH --job-name=medgemma27b-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=122G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

PROJECT_DIR="/geode2/home/u010/alikh/BigRed200/CBR-eICU-Data/Project-files"
MODEL_27B="/N/scratch/alikh/models/google--medgemma-27b-it"

log "Job start: id=${SLURM_JOB_ID:-unknown} node=${SLURMD_NODENAME:-unknown}"
log "Changing directory to ${PROJECT_DIR}"
cd "${PROJECT_DIR}"

log "Loading project environment variables from .env"
set -a
source .env
set +a

log "Overriding model selection to 27B-only for this run"
export MODEL_PATH="${MODEL_27B}"
export FALLBACK_MODEL_PATH=""
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

log "Loading CUDA module"
module load cuda/12.6

log "Activating virtual environment"
source venv/bin/activate

log "Preflight: Python and Torch versions"
python3 - <<'PY'
import torch
print("python ok")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda devices:", torch.cuda.device_count())
PY

log "Preflight: checking 27B local model directory"
if [[ ! -f "${MODEL_27B}/config.json" ]]; then
  log "ERROR: Missing ${MODEL_27B}/config.json"
  exit 2
fi

log "Launching test_inference.py (27B only)"
python3 -u test_inference.py
log "Job completed successfully"
