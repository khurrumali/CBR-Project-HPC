module load python/3.12.11
module load craype-accel-amd-gfx90a
export HSA_OVERRIDE_GFX_VERSION=9.0.10
source venv/bin/activate
echo "Environment ready. ROCm version:"
python3 -c "import torch; print(f'Pytorch-ROCm: {torch.__version__}'); print(f'GPU Detected: {torch.cuda.is_available()}')"
