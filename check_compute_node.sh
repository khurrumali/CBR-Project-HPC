#!/bin/bash
echo "--- System Info ---"
uname -a
echo "--- Path Info ---"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "--- ROCm / AMD Info ---"
echo "HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "--- Device Files ---"
ls -l /dev/kfd /dev/dri/render* 2>/dev/null
echo "--- Module Info ---"
module list 2>&1
echo "--- ROCm Tools ---"
which rocm-smi 2>/dev/null || echo "rocm-smi not found"
which rocminfo 2>/dev/null || echo "rocminfo not found"
/opt/rocm/bin/rocm-smi --showhw 2>/dev/null || echo "/opt/rocm/bin/rocm-smi not found"
