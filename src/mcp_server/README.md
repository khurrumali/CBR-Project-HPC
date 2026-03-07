# eICU project MCP Server

This server coordinates the various tools developed for the eICU project, including Knowledge Graph management, clinical inference, and Slurm cluster operations.

## Architecture
The server is built using [FastMCP](https://github.com/jlowin/fastmcp) and runs over `stdio`. It wraps existing Python scripts in `src/` to provide a unified interface for agentic assistants.

## Capabilities

### Tools
- `build_kg`: Rebuild the Oxigraph RDF store from the SQLite source.
- `query_kg`: Execute SPARQL queries (e.g., finding patients by clinical concepts via the SKOS layer).
- `get_kg_stats`: Quick summary of triples and entity counts.
- `build_patient_case`: Generate the structured JSON case file for any `patientunitstayid`.
- `submit_slurm_inference`: Launch GPU inference jobs on the cluster.
- `check_job_status`: View Slurm queue status.
- `list_inference_reports`: Browse generated medical reports.

### Resources
- `kg://stats`: Real-time KG metrics.
- `config://env`: Project environment configuration (sanitized).

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
