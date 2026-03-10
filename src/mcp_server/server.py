#!/usr/bin/env python3
"""
mcp_server/server.py – MCP Server for eICU Project Coordination.
Integrates Knowledge Graph, Inference, and Slurm job management.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eicu-mcp-server")

# Resolve project root dynamically: server.py lives at src/mcp_server/server.py
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "bin", "python3")

mcp = FastMCP("eICU Coordinator")

DEFAULT_MODEL_PATH = "/N/scratch/alikh/models/google--medgemma-4b-it"


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


# --- Tools: CBR (Case-Based Reasoning) + MedGemma ---


@mcp.tool()
def load_medgemma(
    model_path: str = DEFAULT_MODEL_PATH,
    use_gpu: bool = True,
) -> str:
    """
    Load MedGemma into memory once. The model stays resident for all subsequent
    cbr_adapt calls — no reload overhead per patient.
    :param model_path: Local path to MedGemma weights directory.
    :param use_gpu: Use CUDA if available (requires GPU node).
    """
    from src.inference import model_state
    try:
        return model_state.load(model_path, use_gpu=use_gpu)
    except Exception as e:
        return f"ERROR loading model: {e}"


@mcp.tool()
def unload_medgemma() -> str:
    """Release MedGemma from memory and free GPU cache."""
    from src.inference import model_state
    return model_state.unload()


@mcp.tool()
def medgemma_status() -> str:
    """Check whether MedGemma is currently loaded and which path it was loaded from."""
    from src.inference import model_state
    if model_state.is_loaded():
        return f"Loaded: {model_state.current_model_path()}"
    return "Not loaded. Call load_medgemma() first."


@mcp.tool()
def cbr_adapt(
    age: int,
    problem: str,
    gender: Optional[str] = None,
    ethnicity: Optional[str] = None,
    retrieval_k: int = 10,
    adapt_k: int = 4,
    max_new_tokens: int = 512,
) -> str:
    """
    Full CBR cycle for a new patient — no database lookup, no model reload.

    Pipeline: build case → retrieve similar cases from KG →
              rule-based pre-analysis → LLM adaptation (resident MedGemma).

    Requires load_medgemma() to have been called first.

    :param age: Patient age in years.
    :param problem: Primary clinical problem / admission diagnosis (e.g. 'Sepsis').
    :param gender: Patient gender (optional).
    :param ethnicity: Patient ethnicity (optional).
    :param retrieval_k: Number of similar cases to retrieve from KG.
    :param adapt_k: Top-k cases to pass through to adaptation.
    :param max_new_tokens: Max tokens for LLM generation.
    """
    from src.inference import model_state
    from src.cbr.retrieval import retrieve
    from src.cbr.adaptation import rule_based_adapt, build_llm_prompt

    if not model_state.is_loaded():
        return "ERROR: Model not loaded. Call load_medgemma() first."

    # Build minimal target case
    def _infer_age_group(a: int) -> str:
        if a < 18: return "<18"
        if a < 40: return "18-39"
        if a < 60: return "40-59"
        if a < 80: return "60-79"
        return "80+"

    target_case: Dict[str, Any] = {
        "metadata": {"patient_unit_stay_id": None, "source": "mcp_direct_input"},
        "demographics": {
            "age": age,
            "age_raw": str(age),
            "age_group": _infer_age_group(age),
            "gender": gender or "Unknown",
            "ethnicity": ethnicity or "Unknown",
            "apache_admission_dx": problem,
            "admitting_diagnosis": problem,
            "primary_diagnosis_string": problem,
        },
        "triage_context": {"severity_flags": []},
        "acute_physiology_24h": {},
        "organ_support": {},
        "key_labs_24h": {},
        "comorbidities": {"flags": {}, "evidence": {}},
    }

    # Retrieve
    results = retrieve(name=f"MCP_age{age}", age=age, problem=problem, top_k=retrieval_k)
    if not results and "(" in problem:
        broad = problem.split("(")[0].strip()
        results = retrieve(name=f"MCP_age{age}_RELAXED", age=age, problem=broad, top_k=retrieval_k)

    results = results[:adapt_k] if results else []

    # Rule-based pre-analysis
    rule_result: Optional[Dict[str, Any]] = None
    rule_section = ""
    if results:
        rule_result = rule_based_adapt(target_case, results[0].source_case)
        notes = "\n".join(f"  - {n}" for n in rule_result.get("comparison_notes", [])) or "  (none)"
        adjustments = "\n".join(f"  - {a}" for a in rule_result.get("recommended_adjustments", [])) or "  (none)"
        rule_section = (
            "=== Rule-Based Pre-Analysis ===\n"
            f"Triage Priority: {rule_result['adapted_priority']}\n"
            f"New Patient Acuity: {rule_result['new_acuity']}\n"
            f"Retrieved Case Acuity: {rule_result['retrieved_acuity']}\n"
            f"Acuity Delta: {rule_result['acuity_delta']}\n\n"
            f"Physiological Comparison:\n{notes}\n\n"
            f"Recommended Adjustments:\n{adjustments}\n"
            "=== End Pre-Analysis ===\n\n"
        )

    # Build prompt
    retrieved_solution = results[0].source_case if results else {}
    instructions = (
        "You are an expert clinical reasoning assistant. "
        "Review the new patient's facts and the similar past case. "
        "A deterministic rule-based pre-analysis is provided — use it as structured context.\n\n"
        f"{rule_section}"
        "Adapt the clinical decision-making (interventions, organ support, escalation) "
        "for the new patient. Return a structured clinical plan."
    )
    prompt = build_llm_prompt(target_case, retrieved_solution, retrieved_solution, instructions=instructions)

    # Generate with resident model
    try:
        response = model_state.generate(prompt, max_new_tokens=max_new_tokens)
    except Exception as e:
        return f"ERROR during generation: {e}"

    output = {
        "query": {"age": age, "problem": problem, "gender": gender, "ethnicity": ethnicity},
        "retrieved_cases": len(results),
        "rule_based_priority": rule_result["adapted_priority"] if rule_result else None,
        "llm_adaptation": response,
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
