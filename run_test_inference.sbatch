#!/usr/bin/env bash
#SBATCH -A r00877
#SBATCH -p gpu-interactive
#SBATCH --job-name=medgemma4b-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /geode2/home/u010/alikh/BigRed200/CBR-eICU-Data/Project-files

# Keep env in sync with project defaults.
set -a
source .env
set +a

# BigRed200 guide indicates CUDA modules on GPU nodes.
module load cuda/12.6
source venv/bin/activate

python3 test_inference.py
