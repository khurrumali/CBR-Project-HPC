#!/usr/bin/env bash
# Helper script to launch a modest 30-minute GPU interactive session.
# Usage: ./gpu_session.sh

set -euo pipefail

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source .env
fi

ACCOUNT="${SLURM_ACCOUNT:-r00877}"
PARTITION="${SLURM_PARTITION:-gpu-interactive}"
GPUS="${SLURM_GPUS:-1}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
MEMORY="${SLURM_MEM:-64G}"
WALLTIME="${SLURM_TIME:-00:30:00}"
JOB_NAME="${SLURM_JOB_NAME:-medgemma-infer}"

echo "Launching GPU session:"
echo "  account=$ACCOUNT partition=$PARTITION gpus=$GPUS cpus=$CPUS_PER_TASK mem=$MEMORY time=$WALLTIME job=$JOB_NAME"

exec srun \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
  --gpus="$GPUS" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEMORY" \
  --time="$WALLTIME" \
  --job-name="$JOB_NAME" \
  --pty bash
