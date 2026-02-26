#!/bin/bash
# Helper script to launch GPU session
# Usage: ./gpu_session.sh

source .env
echo "Requesting GPU session for account $SLURM_ACCOUNT..."
srun -A $SLURM_ACCOUNT -p gpu-interactive --gpus=1 --pty bash
