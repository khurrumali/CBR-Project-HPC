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
    - Store all interactions in `inference_logs/google--medgemma-27b-it/` as `.txt` files.
- Verified patient `141765` as initial test subject.
## 00:31 - Scheduler Account Requirement
- Interactive and compute commands must include project account flag.## 00:55 - Pytorch (ROCm) Installation
- Identified missing `torch` and `transformers` in virtual environment.
- Installed `torch` with ROCM 6.2 support (AMD optimal) from `pytorch.org` index.
- Updated `requirements.txt` to include `torch`, `transformers`, and `accelerate`.
- Verified installation in `venv`.

## 01:10 - Next Step
- User to run `./gpu_session.sh` from `Project-files/` and then execute `python3 test_inference.py`.
- Required format: `srun -A r00877 <commands>`.
