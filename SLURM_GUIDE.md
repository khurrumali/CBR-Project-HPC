# BigRed200 Slurm GPU Guide

This guide explains how to use the NVIDIA GPUs on BigRed200.

## 🚀 Launching a GPU Session

Run this command from the login node to start an interactive session:

```bash
srun -A r00877 -p gpu-interactive --gpus=1 --mem=64G --pty bash
```

### 📋 Command Flags Explained
| Flag | Name | Description |
| :--- | :--- | :--- |
| `-A r00877` | Account | Your project billing account. |
| `-p gpu-interactive` | Partition | Use this for testing/development (up to 4 hours). |
| `--gpus=1` | GPUs | Requests 1 NVIDIA A100 GPU. |
| `--mem=64G` | Memory | Requests 64GB of system RAM. |
| `--pty bash` | Interactive | Opens a terminal shell inside the compute node. |

---

## 🛠️ Setting up the Environment (Inside the Node)

Once you are on the compute node (your prompt will change to `alikh@nidXXXX`), run:

```bash
module load cuda/12.6
source venv/bin/activate
```

### 🔍 Quick Verification
Check if the GPU is visible to the system:
```bash
nvidia-smi
```

Check if the GPU is visible to PyTorch:
```bash
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

---

## 📝 Rules & Best Practices
1. **Never run models on the login node.** It will slow down the system for everyone and your job will be killed.
2. **Use the Scratch Space** (`/N/scratch/alikh/`) for storing large model weights and checkpoints.
3. **Exit when done.** Type `exit` to release the GPU resource so others can use it.
