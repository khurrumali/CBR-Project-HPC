import pytest
import subprocess
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Adjust this path if necessary to match the structure
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_cbr_medgemma.py"


def test_run_cbr_medgemma_arguments():
    """Test that the script accepts the correct arguments by running it with --help"""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "--age AGE" in result.stdout
    assert "--gender GENDER" in result.stdout
    assert "--ethnicity ETHNICITY" in result.stdout
    assert "--problem PROBLEM" in result.stdout
    assert "--model_path MODEL_PATH" in result.stdout
    assert "--retrieval_k RETRIEVAL_K" in result.stdout
    assert "--adapt_k ADAPT_K" in result.stdout
    assert "--gpu" in result.stdout


@pytest.fixture
def mock_target_case():
    return {
        "metadata": {"patient_unit_stay_id": 141765},
        "demographics": {
            "age": 60,
            "gender": "Female",
            "apache_admission_dx": "Sepsis"
        },
        "triage_context": {"severity_flags": ["hypotension (MAP<65)"]},
        "acute_physiology_24h": {
            "map": {"value": 50},
            "heart_rate": {"value": 110},
            "gcs": {"total": 14}
        },
        "organ_support": {"mechanical_ventilation": {"flag": True}}
    }

@pytest.fixture
def mock_retrieval_output():
    # Simulate output from cbr_retrieve.py --json --full
    return [
        {
            "case_id": "123",
            "score": 0.9,
            "age": 62,
            "gender": "Female",
            "details": {},
            "source_case": {
                "metadata": {"patient_unit_stay_id": 123},
                "demographics": {"age": 62, "gender": "Female", "apache_admission_dx": "Sepsis"},
                "triage_context": {"severity_flags": ["hypotension (MAP<65)"]},
                "acute_physiology_24h": {"map": {"min": 55}, "heart_rate": {"max": 105}, "gcs": {"total": 15}},
                "organ_support": {"mechanical_ventilation": {"flag": True}},
                "icu_los": {"hospital_mortality": False, "los_days": 4.5},
                "apache": {"actual_hospital_mortality": "Alive"}
            }
        },
        {
            "case_id": "456",
            "score": 0.85,
            "age": 58,
            "gender": "Male",
            "details": {},
            "source_case": {
                "metadata": {"patient_unit_stay_id": 456},
                "demographics": {"age": 58, "gender": "Male", "apache_admission_dx": "Severe Sepsis"},
                "triage_context": {"severity_flags": ["tachypnea (RR>30)"]},
                "acute_physiology_24h": {"map": {"min": 60}, "heart_rate": {"max": 115}, "gcs": {"total": 13}},
                "organ_support": {"vasopressor_use": {"flag": True}},
                "icu_los": {"hospital_mortality": True, "los_days": 10.2},
                "apache": {"actual_hospital_mortality": "Expired"}
            }
        }
    ]

def test_direct_patient_args(mock_retrieval_output, monkeypatch):
    """
    Test that direct patient info (--age, --gender, --ethnicity, --problem)
    is forwarded correctly to the cbr_retrieve.py subprocess call.
    Retrieval is done via subprocess (not a direct import of retrieve()).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_cbr_medgemma", str(SCRIPT_PATH))
    run_cbr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_cbr)

    # Mock subprocess.run to simulate cbr_retrieve.py returning JSON results
    mock_proc = MagicMock()
    mock_proc.stdout = json.dumps(mock_retrieval_output)
    mock_proc.returncode = 0

    test_args = [
        "run_cbr_medgemma.py",
        "--age", "87",
        "--age_raw", "87",
        "--age_group", "80+",
        "--gender", "Female",
        "--ethnicity", "Caucasian",
        "--problem", "Sepsis",
        "--model_path", "dummy/path",
        "--gpu",
    ]

    with patch.object(sys, "argv", test_args), \
         patch("subprocess.run", return_value=mock_proc) as mock_subproc, \
         patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()), \
         patch("transformers.TextStreamer"):
        run_cbr.main()

    # Verify subprocess was called and forwarded the correct patient parameters
    mock_subproc.assert_called_once()
    cmd = mock_subproc.call_args[0][0]  # the command list
    assert "--age" in cmd and "87" in cmd
    assert "--problem" in cmd and "Sepsis" in cmd
    assert "--gender" in cmd and "Female" in cmd


def test_direct_patient_args_age_group_inference(monkeypatch):
    """_build_case_from_args infers age_group when not supplied."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_cbr_medgemma", str(SCRIPT_PATH))
    run_cbr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_cbr)

    class FakeArgs:
        age = 87
        age_raw = None
        age_group = None
        gender = "Female"
        ethnicity = "Caucasian"
        problem = "Pneumonia"

    case = run_cbr._build_case_from_args(FakeArgs())
    assert case["demographics"]["age"] == 87
    assert case["demographics"]["age_group"] == "80+"
    assert case["demographics"]["age_raw"] == "87"
    assert case["demographics"]["gender"] == "Female"
    assert case["demographics"]["ethnicity"] == "Caucasian"
    assert case["demographics"]["apache_admission_dx"] == "Pneumonia"
    assert case["metadata"]["source"] == "direct_input"


def test_prompt_construction(mock_target_case, mock_retrieval_output):
    """
    Test the internal prompt building logic of run_cbr_medgemma without triggering inference.
    We import the functions directly for unit testing.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # We load the script as a module to test its functions
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_cbr_medgemma", str(SCRIPT_PATH))
    run_cbr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_cbr)
    
    prompt = run_cbr.build_cbr_prompt(mock_target_case, mock_retrieval_output)
    
    # Assertions on prompt content
    assert "You are an expert Critical Care AI assistant" in prompt
    assert "ID: 141765" in prompt
    assert "Sepsis" in prompt
    assert "MAP: 50" in prompt
    
    # Assert retrieved cases are in the prompt
    assert "SIMILAR CASE 1 (Similarity: 0.90)" in prompt
    assert "ID: 123" in prompt
    assert "Mortality: Alive" in prompt
    
    assert "SIMILAR CASE 2 (Similarity: 0.85)" in prompt
    assert "ID: 456" in prompt
    assert "Mortality: Expired" in prompt

