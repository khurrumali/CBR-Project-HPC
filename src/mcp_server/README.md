# eICU project MCP Server

This server coordinates the various tools developed for the eICU project, including Knowledge Graph management, clinical inference, and Slurm cluster operations.

## Architecture
The server is built using [FastMCP](https://github.com/jlowin/fastmcp) and runs over `stdio`. It wraps existing Python scripts in `src/` to provide a unified interface for agentic assistants.

## Capabilities

### Tools

#### Knowledge Graph
- `build_kg`: Rebuild the Oxigraph RDF store from the SQLite source.
- `query_kg`: Execute SPARQL queries (e.g., finding patients by clinical concepts via the SKOS layer).
- `get_kg_stats`: Quick summary of triples and entity counts.

#### CBR + MedGemma (persistent model)
- `load_medgemma(model_path, use_gpu)`: Load MedGemma into memory **once**. The model stays resident — subsequent `cbr_adapt` calls reuse it without any reload overhead.
- `medgemma_status()`: Check whether MedGemma is loaded and which path was used.
- `cbr_adapt(age, problem, gender, ethnicity, retrieval_k, adapt_k)`: Full CBR pipeline for a new patient — no database lookup required.
  - Retrieves similar past cases from the KG
  - Runs deterministic rule-based pre-analysis (triage priority, acuity score, organ-support adjustments)
  - Injects rule output into the LLM prompt
  - Generates adapted clinical plan using the resident MedGemma
- `unload_medgemma()`: Release model from memory and clear GPU cache.

#### Patient Cases & Inference
- `build_patient_case`: Generate the structured JSON case file for any `patientunitstayid`.
- `list_inference_reports`: Browse generated medical reports.

#### Cluster
- `submit_slurm_inference`: Launch GPU inference jobs on the cluster.
- `check_job_status`: View Slurm queue status.

### Resources
- `kg://stats`: Real-time KG metrics.
- `config://env`: Project environment configuration (sanitized).

## Typical workflow
```
# 1. Load model once (call from MCP client)
load_medgemma(model_path="/N/scratch/alikh/models/google--medgemma-4b-it", use_gpu=True)

# 2. Run CBR for as many patients as needed — no reload
cbr_adapt(age=87, problem="Sepsis", gender="Female", ethnicity="Caucasian")
cbr_adapt(age=60, problem="Rhythm disturbance (atrial, supraventricular)", gender="Male")

# 3. Release when done
unload_medgemma()
```

## Usage
To start the server manually:
```bash
source venv/bin/activate
python src/mcp_server/server.py
```

To use with a desktop MCP client, add the following to your configuration:
```json
{
  "mcpServers": {
    "eicu-coordinator": {
      "command": "/N/u/alikh/BigRed200/CBR-eICU-Data/Project-files/venv/bin/python3",
      "args": ["/N/u/alikh/BigRed200/CBR-eICU-Data/Project-files/src/mcp_server/server.py"]
    }
  }
}
```
