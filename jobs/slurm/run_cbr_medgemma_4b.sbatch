#!/usr/bin/env bash
#SBATCH -A r00877
#SBATCH -p gpu-interactive
#SBATCH --job-name=cbr-medgemma-4b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/cbr_medgemma_%j.out

set -euo pipefail

# Navigate to project root
cd /N/u/alikh/BigRed200/CBR-eICU-Data/Project-files

# Load environment and modules
source ./env_setup.sh

# Optional: ensure output directory exists
mkdir -p logs/slurm

# Run the CBR MedGemma script for patient 141765 using the 4B model
python3 scripts/run_cbr_medgemma.py \
    --age 87 \
    --gender "Female" \
    --problem "Sepsis" \
    --model_path /N/scratch/alikh/models/google--medgemma-4b-it
