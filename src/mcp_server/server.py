#!/usr/bin/env python3
"""
mcp_server/server.py – MCP Server for eICU Project Coordination.
Integrates Knowledge Graph, Inference, and Slurm job management.
"""

import json
import logging
import os
import subprocess
from typing import Optional

from fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eicu-mcp-server")

# Base project path
PROJECT_ROOT = "/N/u/alikh/BigRed200/CBR-eICU-Data/Project-files"
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv/bin/python3")

mcp = FastMCP("eICU Coordinator")


# --- Helper: Run project scripts ---
def run_project_script(script_rel_path: str, args: list[str] = None):
    script_path = os.path.join(PROJECT_ROOT, script_rel_path)
    cmd = [VENV_PYTHON, script_path] + (args or [])
    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_rel_path}: {e.stderr}")
        return f"Error: {e.stderr}\nOutput: {e.stdout}"


# --- Tools: Knowledge Graph ---


@mcp.tool()
def build_kg(full_rebuild: bool = True) -> str:
    """
    Build or rebuild the Oxigraph RDF Knowledge Graph from SQLite.
    :param full_rebuild: If True, wipes the existing store and starts fresh.
    """
    args = []  # build_kg.py defaults to full rebuild unless --stats is passed
    return run_project_script("src/knowledge_graph/build_kg.py", args)


@mcp.tool()
def query_kg(sparql_query: str) -> str:
    """
    Execute a SPARQL query against the persistent Oxigraph store.
    """
    return run_project_script(
        "src/knowledge_graph/query_kg.py", ["--query", sparql_query]
    )


@mcp.tool()
def get_kg_stats() -> str:
    """
    Retrieve statistics (triple counts, entity counts) from the Knowledge Graph.
    """
    return run_project_script("src/knowledge_graph/build_kg.py", ["--stats"])


@mcp.tool()
def fetch_diagnosis_rdf(diagnosis_string: str) -> str:
    """
    Fetch RDF triples for a diagnosis.
    If a URI is provided, it fetches triples for that URI.
    If a keyword is provided, it searches for matching diagnoses.
    """
    if (
        diagnosis_string.startswith("http")
        or diagnosis_string.startswith("data:")
        or len(diagnosis_string) == 12
    ):
        return run_project_script(
            "src/knowledge_graph/fetch_diagnosis.py", ["--uri", diagnosis_string]
        )
    else:
        return run_project_script(
            "src/knowledge_graph/fetch_diagnosis.py", ["--search", diagnosis_string]
        )


# --- Tools: Patient Cases & Inference ---


@mcp.tool()
def build_patient_case(patient_id: int, output_file: Optional[str] = None) -> str:
    """
    Construct a comprehensive JSON case file for a patient.
    """
    args = [str(patient_id)]
    if output_file:
        args += ["-o", output_file]
    return run_project_script("src/inference/build_case.py", args)


@mcp.tool()
def list_inference_reports() -> str:
    """
    List all generated layman reports in the logs directory.
    """
    log_dir = os.path.join(PROJECT_ROOT, "logs/inference")
    results = []
    if not os.path.exists(log_dir):
        return "No inference logs directory found."

    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.endswith(".txt"):
                results.append(os.path.relpath(os.path.join(root, f), log_dir))

    return "\n".join(results) if results else "No reports found."


# --- Tools: CBR (Case-Based Reasoning) ---


@mcp.tool()
def cbr_retrieve_and_suggest(
    name: str, age: int, problem: str, top_k: int = 3, adapt: bool = True
) -> str:
    """
    Perform a full CBR cycle: Retrieve similar cases and optionally Suggest an adaptation.
    :param name: Patient identifier.
    :param age: Patient age.
    :param problem: Clinical problem keyword (e.g., 'sepsis').
    :param top_k: Number of similar cases to retrieve.
    :param adapt: If True, applies null_adapt (baseline) to the top result.
    """
    # 1. Retrieval
    args = [
        "--name",
        name,
        "--age",
        str(age),
        "--problem",
        problem,
        "--top_k",
        str(top_k),
        "--json",
    ]
    retrieval_json = run_project_script("scripts/cbr_retrieve.py", args)

    try:
        results = json.loads(retrieval_json)
    except Exception as e:
        return f"Retrieval failed or returned non-JSON: {retrieval_json}"

    if not results:
        return "No similar cases found."

    output = {
        "query": {"name": name, "age": age, "problem": problem},
        "retrieved_cases": results,
    }

    # 2. Adaptation (Simple Suggestion)
    if adapt and results:
        # For the demo/MCP tool, we use the top match and show a suggestion block.
        # We import adaptation here to keep server start-up light.
        top_case_id = results[0]["case_id"]

        # We need the full case for adaptation logic
        case_data_str = run_project_script("src/inference/build_case.py", [top_case_id])
        try:
            retrieved_case_json = json.loads(case_data_str)
            # In a real scenario, we'd define a field map.
            # Here we just show the 'Identity' adaptation as a baseline.
            output["suggestion"] = {
                "strategy": "null_adapt",
                "applied_to_case_id": top_case_id,
                "note": "Identity adaptation: the retrieved solution is presented as-is for the new context.",
            }
        except:
            output["suggestion"] = {
                "error": "Could not build full case for adaptation."
            }

    return json.dumps(output, indent=2)


# --- Tools: Slurm & Cluster ---


@mcp.tool()
def submit_slurm_inference(model: str = "4b") -> str:
    """
    Submit a Slurm batch job for inference (default 4B model).
    :param model: "4b" or "27b"
    """
    sbatch_file = "jobs/slurm/run_test_inference.sbatch"
    if model == "27b":
        sbatch_file = "jobs/slurm/run_test_inference_27b.sbatch"

    cmd = ["sbatch", os.path.join(PROJECT_ROOT, sbatch_file)]
    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Slurm submission failed: {e.stderr}"


@mcp.tool()
def check_job_status() -> str:
    """
    Check the status of your jobs in the Slurm queue.
    """
    try:
        # Get current user
        user = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()
        result = subprocess.run(
            ["squeue", "-u", user], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Failed to get job status: {e.stderr}"


# --- Resources ---


@mcp.resource("kg://stats")
def kg_stats() -> str:
    """Returns the current Knowledge Graph statistics."""
    return run_project_script("src/knowledge_graph/build_kg.py", ["--stats"])


@mcp.resource("config://env")
def project_config() -> str:
    """Returns the project environment configuration (sanitized)."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        return "No .env file found."
    with open(env_path, "r") as f:
        lines = f.readlines()
        # Hide sensitive tokens if any
        sanitized = []
        for line in lines:
            if "TOKEN" in line.upper() or "KEY" in line.upper():
                sanitized.append(line.split("=")[0] + "=********")
            else:
                sanitized.append(line)
        return "".join(sanitized)


if __name__ == "__main__":
    mcp.run()
