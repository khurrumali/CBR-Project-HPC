# src/cbr — Case-Based Reasoning (CBR) Engine

This package implements a stateless Case-Based Reasoning engine for the eICU clinical project. Unlike traditional CBR systems that maintain a separate "Case Library," this implementation treats the **eICU SQLite database** as the source of truth and the **Oxigraph Knowledge Graph** as a semantic index for discovery.

## Architecture

The engine is designed to be **stateless** and **on-demand**:
- **Source**: eICU SQLite database (`DATABASE_URL`).
- **Index**: Oxigraph RDF Store (`OXIGRAPH_STORE_PATH`).
- **Retrieval**: Candidates are discovered via SPARQL (SKOS concept layer) and ranked using composite similarity.
- **Identification**: The `patientunitstayid` is used as the canonical `case_id` throughout the system.

## CBR Cycle → Module Mapping

| CBR Phase | Module | Implementation Detail |
|-----------|--------|-----------------------|
| **Retrieve** | `retrieval.py` | Discovers candidate `patientunitstayid`s via KG; constructs full case JSONs via `build_case.py`; ranks by weighted similarity. |
| **Reuse** | `adaptation.py` | Three strategies: `null_adapt` (baseline), `rule_based_adapt` (deterministic clinical rules, CPU-only), `llm_adapt` (MedGemma, GPU node). |
| **Revise** | `evaluation.py` | Evaluates the quality of an adapted solution (Outcome scoring). |
| **Retain** | *(None)* | **Stateless**: New cases are treated as part of the source dataset once added to the SQLite/KG upstream. |

## Mandatory Retrieval Fields

The `retrieve()` function and associated CLI require three mandatory inputs to locate similar patients:

1.  **Name**: A patient identifier or label (used for logging and report generation).
2.  **Age**: Patient age in years (used for ±15 year range filtering in the KG).
3.  **Problem**: A clinical keyword (e.g., "sepsis", "hypertension") matched via the SKOS hierarchy.

## Usage

### CLI Tool
The recommended way to test retrieval is via the provided script:

```bash
# Basic usage
python scripts/cbr_retrieve.py --name "John Doe" --age 72 --problem "sepsis"

# Detailed scoring breakdown
python scripts/cbr_retrieve.py --name "Jane Doe" --age 45 --problem "hypertension" --verbose
```

### Python API
You can integrate the engine directly into clinical workflows:

```python
from src.cbr.retrieval import retrieve

results = retrieve(
    name="Clinical Query",
    age=65,
    problem="atrial fibrillation",
    top_k=5
)

for res in results:
    print(f"Match: {res.case_id}, Score: {res.score}")
    print(f"Details: {res.details}")
```

## Similarity Scoring Strategy

The `retrieval.py` module computes a composite score ($0.0$ to $1.0$) based on:
- **Diagnosis Match (50%)**: Boolean match if the queried problem exists in the candidate's diagnosis list.
- **Age Closeness (30%)**: Linear decline in similarity over a 40-year difference.
- **Medication Richness (20%)**: A proxy for clinical complexity based on the number of unique drugs ordered.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Path to the eICU SQLite database (used for case construction). |
| `OXIGRAPH_STORE_PATH` | Path to the Oxigraph RDF store (used for candidate discovery). |
| `MODEL_PATH` | Path to the local MedGemma model (for LLM adaptation). |

## Integration with Project Tools

This engine depends on existing project infrastructure:
- **`src/inference/build_case.py`**: Used to transform raw SQLite rows into the structured JSON cases analyzed by the CBR engine.
- **`src/knowledge_graph/`**: The `build_kg.py` script must be run once to populate the RDF index used for retrieval.
- **`src/mcp_server/`**: The MCP server can wrap `src.cbr.retrieval` to provide agentic assistants with similar-case lookup capabilities.

## Design Principles
- **Lazy Loading**: Heavy dependencies (like `pyoxigraph`) are only imported when a retrieval is actually triggered.
- **Source-Centric**: No duplication of data; cases are generated from the primary database on the fly.
- **Type Safety**: Uses Python `dataclasses` and `typing` for clear interfaces between CBR phases.