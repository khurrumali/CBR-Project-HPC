import torch
import sys

print(f"Python: {sys.version}")
print(f"Pytorch version: {torch.__version__}")
print(f"CUDA Available (maps to ROCm on this system): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU detected by Pytorch.")
    # Check for library issues
    try:
        import torch.utils.cpp_extension
        print("ROCm / HIP likely not correctly initialized in this shell.")
    except Exception as e:
        print(f"Issue checking extension: {e}")
