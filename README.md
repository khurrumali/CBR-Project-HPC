# CBR-eICU-Project (BigRed200)

This project implements a **Case-Based Reasoning (CBR)** cycle for clinical decision support using the eICU Collaborative Research Database. It leverages Knowledge Graphs (KG) for case retrieval and MedGemma LLMs for case adaptation and clinical reasoning.

## 🚀 Overview

The system follows the four stages of the CBR cycle:
1. **Retrieve**: Finds similar past patient cases from a Knowledge Graph.
2. **Reuse**: Provides the retrieved cases as context to a medical LLM.
3. **Revise**: The LLM suggests clinical actions or predictions for the target case.
4. **Retain**: (In development) Storing successful outcomes back into the case library.

## 🛠️ Setup & Installation

### Prerequisites
- Access to **BigRed200** HPC.
- Python 3.12+ (managed via `venv`).
- CUDA 12.6+ for GPU acceleration.

### Environment Setup
Run the provided setup script or follow these manual steps:

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📈 Usage

### Running the CBR Cycle (Interactive/Script)
To run an end-to-end CBR inference for a specific patient:

```bash
python scripts/run_cbr_medgemma.py --id 141765 --model_path /N/scratch/alikh/models/google--medgemma-27b-it
```

### HPC Execution (Slurm)
Submit batch jobs using the provided sbatch scripts:

```bash
sbatch jobs/slurm/run_test_inference_27b.sbatch
```

Refer to [SLURM_GUIDE.md](SLURM_GUIDE.md) for detailed instructions on launching GPU sessions and managing compute nodes.

## 📂 Project Structure

- `src/`: Core logic and modules.
  - `cbr/`: Implementations of retrieval, adaptation, and evaluation.
  - `inference/`: Model loading, prompt construction, and state management.
  - `knowledge_graph/`: KG construction and querying (RDF/SPARQL).
- `scripts/`: Entry point scripts for running experiments and data exploration.
- `jobs/slurm/`: Batch scripts for BigRed200.
- `logs/`: Execution logs and inference outputs.
- `tests/`: Unit tests and logic verification.

## 📚 Dependencies
- **LLM Inference**: `transformers`, `torch`, `accelerate`, `bitsandbytes`.
- **Database/KG**: `pyoxigraph` (RDF), `pandas`, `numpy`.
- **Server/API**: `fastmcp` (Model Context Protocol).

