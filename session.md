# Session Log - 2026-02-25

> [!IMPORTANT]
> **CLUSTER EXECUTION PROTOCOL**: All model execution, training, inference, and fine-tuning MUST happen via `srun` on a compute node. The login node (`login2`) is for editing and orchestration only.

## 22:21 - Session Started
- Initialized Python environment.
- Encountered workspace validation issues; user disabled validation.

## 22:34 - Python Environment Setup
- Created virtual environment in `/N/u/alikh/BigRed200/CBR-eICU-Data/Project-files/venv`.
- Loaded `python/3.12.11` module for modern Python support.
- Saved `requirements.txt` with essential libraries:
    - Database: `psycopg2-binary`, `sqlalchemy`
    - Data: `pandas`, `numpy`, `scipy`
    - Environment: `python-dotenv`, `configobj`
    - Knowledge Graph: `pyoxigraph`
    - LLM: `langchain` suite
    - Visualization: `jupyter`, `matplotlib`, `seaborn`

## 23:14 - Project Configuration
- Initialized Git repository.
- Created `.gitignore` to protect environment and caches.
- Created `.env` with primary `DATA_PATH` for eICU demo.

## 23:33 - System & GPU Evaluation
- Performed system audit: 128-core AMD EPYC, 256GB RAM on SLES 15 SP6.
- Confirmed GPU access matches BigRed200 structure (MI210 AMD GPUs on compute nodes, not login node).
- Identified necessary `srun` commands for GPU interactive sessions.

## 23:47 - Storage Configuration
- Created scratch directories: `/N/scratch/alikh/models` and `/N/scratch/alikh/hf_cache`.
- Configured `.env` to point `MODELS_PATH` and `HF_HOME` to high-performance 100TB scratch space.

## 23:55 - Current Status
- Resolving Ollama vs. Local Transformers installation strategy for AMD GPUs.
- Catching up on session logging.

## 00:20 - MedGemma 27B Download Success
- User acknowledged model license terms.
- Successfully downloaded `google/medgemma-27b-it` with HF token.
- Local path: `/N/scratch/alikh/models/google--medgemma-27b-it`.

## 00:30 - Database Setup & Verification
- Identified corruption in provided `eicu_v2_0_1.sqlite3.gz` (SHA256 mismatch).
- Rebuilt SQLite database from verified CSVs in `/N/scratch/alikh/eicu_demo.sqlite`.
- **All 31 tables loaded**, including clinical data (medication, infusion, charting, etc.).
- Total database size: 281MB.
- Total patients in demo: 2,520.
- Updated `.env` with `DATABASE_URL="sqlite:////N/scratch/alikh/eicu_demo.sqlite"`.
## 00:45 - Inference Script Development
- Developed `test_inference.py` to:
    - Extract comprehensive patient profile (Demographics, Diagnosis, Meds, Labs) from SQLite.
    - Load `MedGemma 27B` from scratch in `BF16` (optimized for MI210 GPUs).
    - Generate a family-friendly "Layman's Report".
## 01:25 - Hardware Correction & Quantization
- **Correction**: Verified hardware is **NVIDIA A100-SXM4-40GB**, not AMD Instinct.
- Installed **NVIDIA/CUDA version of PyTorch** (v2.5.1+cu124).
- Updated `test_inference.py` to use **4-bit quantization** (`bitsandbytes`) to fit the 27B MedGemma model (~54GB) into the 40GB A100 VRAM.
- Verified successful GPU detection via `debug_gpu.py` on compute node.

## 01:30 - Execution Ready
- Environment is fully configured for NVIDIA A100.
- User ready to run `python3 test_inference.py` on compute node.

## 02:00 - Case Construction Script
- Created `build_case.py`: CLI script that accepts a `patientunitstayid` and outputs structured JSON.
- Fetches data from 7 clinical tables: diagnosis, admissionDx, carePlanGeneral, carePlanEOL, pastHistory, allergy, medication.
- JSON has 5 top-level keys: metadata, patient, clinical_data, semantic_clinical_summary, triage_context.
- Computes derived fields: BMI, age parsing (handles "> 89"), chronic condition flags, drug categories, PRN flags, drug allergy flags, offset minute→hour conversions.
- Tested on patient 141765 (12KB, 14 meds) and 346380 (27KB, 42 diagnoses, 4 allergies). Invalid IDs error gracefully.
- Usage: `python3 build_case.py <id>` or `python3 build_case.py <id> -o output.json`

## 01:35 - A100 OOM Fix for MedGemma 27B
- Patched `test_inference.py` model loading to reduce VRAM pressure.
- Added explicit `max_memory` caps (`GPU:34GiB`, `CPU:220GiB`) with `device_map='auto'`.
- Enabled `offload_folder` and `offload_state_dict` to spill to disk safely.
- Enabled allocator hint `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Added retry path: if 4-bit quantized load fails, retry with BF16 + CPU/disk offload.
- Verified script syntax with `python -m py_compile test_inference.py`.

## 02:05 - MedGemma 4B Default Setup
- Switched inference default model to `/N/scratch/alikh/models/google--medgemma-4b-it`.
- Added `.env` entries `MODEL_PATH` and `FALLBACK_MODEL_PATH`.
- Kept optional fallback path to `google--medgemma-27b-it` only if needed.
- Verified `test_inference.py` syntax with `python -m py_compile`.

## 02:18 - Batch Job Setup for Inference
- Added `run_test_inference.sbatch` to run `test_inference.py` via Slurm batch.
- Script requests: `-A r00877`, `-p gpu-interactive`, `--gpus=1`, `--cpus-per-task=8`, `--mem=64G`, `--time=01:00:00`.
- Includes environment setup: `.env`, `module load cuda/12.6`, and `venv` activation.
- Validated with `sbatch --test-only` (no inference executed).

## 02:25 - Tokenizer Dependency Fix for MedGemma 4B
- Added missing tokenizer/runtime deps to `requirements.txt`: `bitsandbytes`, `sentencepiece`, `protobuf`, `tiktoken`.
- Installed missing `tiktoken` in `venv`.
- Updated `test_inference.py` tokenizer loading to `use_fast=False` for both primary and fallback model loads.
- Verified syntax with `python -m py_compile test_inference.py`.

## 02:32 - MedGemma 4B Root Cause & Guardrail
- Confirmed local `google--medgemma-4b-it` directory was incomplete (only README/cache metadata).
- Added `ensure_model_dir_ready()` preflight in `test_inference.py` to fail fast with actionable message when model files are missing.
- `HF_TOKEN` is currently missing from `.env`, so model re-download cannot proceed until token is set.

## 02:40 - 27B Batch Script with Step Logging
- Added `run_test_inference_27b.sbatch` for non-interactive MedGemma 27B runs.
- Requests: `-A r00877`, `-p gpu-interactive`, `--gpus=1`, `--cpus-per-task=16`, `--mem=122G`, `--time=04:00:00`.
- Enforces 27B-only mode via env overrides (`MODEL_PATH=...27b`, `FALLBACK_MODEL_PATH=''`).
- Adds timestamped logging for each stage (env load, module load, venv activation, preflight checks, inference start/end).
- Includes model-directory preflight (`config.json` required) before launching inference.

## 02:50 - Repository Snapshot Commit
- Prepared a full repository commit including scripts, Slurm configs, inference code updates, logs, and session notes.
- Snapshot created at user request: "commit everything with session note".
