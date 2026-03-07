import pytest

from src.cbr.adaptation import (
    _resolve_path,
    null_adapt,
    rule_based_adapt,
    substitution_adapt,
    _acuity_score,
    _triage_priority,
    _compare_vitals,
    _get_organ_flag,
    build_llm_prompt,
    llm_adapt,
)
from src.cbr.retrieval import _numeric_closeness, compute_similarity

# ---------------------------------------------------------------------------
# Retrieval / Similarity Tests
# ---------------------------------------------------------------------------

def test_numeric_closeness():
    # Exact match
    assert _numeric_closeness(70, 70, 40) == 1.0
    # Boundary match (max_diff)
    assert _numeric_closeness(70, 30, 40) == 0.0
    assert _numeric_closeness(70, 110, 40) == 0.0
    # Halfway
    assert _numeric_closeness(70, 50, 40) == 0.5
    # None handling
    assert _numeric_closeness(None, 70, 40) == 0.0


def test_compute_similarity_basic():
    query_age = 60
    query_problem = "Sepsis"

    # Mock candidate case structure (as returned by build_case.py)
    candidate = {
        "metadata": {"patient_unit_stay_id": "12345"},
        "patient": {"demographics": {"age": 60, "gender": "Female"}},
        "clinical_data": {
            "diagnosis": [{"diagnosis_string": "Sepsis (ICD-10)"}],
            "medication": [{"drugname": "Norepinephrine"}] * 5,  # 5 meds
        },
    }

    result = compute_similarity(query_age, query_problem, candidate)

    assert result.case_id == "12345"
    assert result.score > 0.8  # High score due to exact age and diagnosis match
    assert result.details["diagnosis_match"] == 1.0
    assert result.details["age_closeness"] == 1.0
    assert result.details["medication_richness"] == 0.5  # 5/10


def test_compute_similarity_no_match():
    query_age = 20
    query_problem = "Fracture"

    candidate = {
        "metadata": {"patient_unit_stay_id": "999"},
        "patient": {"demographics": {"age": 80}},  # Far age
        "clinical_data": {
            "diagnosis": [{"diagnosis_string": "Pneumonia"}],  # Different diagnosis
            "medication": [],  # No meds
        },
    }

    result = compute_similarity(query_age, query_problem, candidate)
    assert result.details["diagnosis_match"] == 0.0
    assert result.details["age_closeness"] == 0.0
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Adaptation Tests
# ---------------------------------------------------------------------------


def test_null_adapt():
    retrieved_sol = {"treatment": "Ventilation", "dose": "High"}
    new_prob = {"patient": "New"}

    adapted = null_adapt(new_prob, retrieved_sol)

    assert adapted == retrieved_sol
    assert adapted is not retrieved_sol  # Should be a deep copy


def test_resolve_path():
    data = {"a": {"b": {"c": 42}}}
    assert _resolve_path(data, "a.b.c") == 42
    assert _resolve_path(data, "a.x") is None
    assert _resolve_path(data, "") is None


def test_substitution_adapt_string_map():
    new_problem = {"patient": {"weight_kg": 85}}
    retrieved_solution = {
        "medication": "Heparin",
        "dose_per_kg": 10,
        "patient_weight": 70,
    }

    # Map 'patient_weight' in solution to 'patient.weight_kg' in new problem
    field_map = {"patient_weight": "patient.weight_kg"}

    adapted = substitution_adapt(new_problem, retrieved_solution, field_map=field_map)

    assert adapted["patient_weight"] == 85
    assert adapted["medication"] == "Heparin"  # Unchanged


def test_substitution_adapt_complex_map():
    new_problem = {"patient": {"age": 45}}
    retrieved_sol = {"risk_level": "low"}

    # Use transform and default
    field_map = {
        "patient_age": {"from": "patient.age", "transform": lambda x: x + 1},
        "department": {"default": "ICU-A"},
    }

    adapted = substitution_adapt(new_problem, retrieved_sol, field_map=field_map)

    assert adapted["patient_age"] == 46
    assert adapted["department"] == "ICU-A"

    assert adapted["department"] == "ICU-A"

def test_substitution_adapt_callable():
    new_problem = {"val": 10}
    retrieved_sol = {"total": 0}

    def sum_logic(new, ret_sol, ret_prob):
        return new["val"] + 5

    field_map = {"total": sum_logic}

    adapted = substitution_adapt(new_problem, retrieved_sol, field_map=field_map)
    assert adapted["total"] == 15


# ---------------------------------------------------------------------------
# Rule-Based Adaptation Tests
# ---------------------------------------------------------------------------

class TestAcuityScoring:
    """Tests for the acuity scoring helpers."""

    def test_empty_flags(self):
        assert _acuity_score([]) == 0.0

    def test_known_flags(self):
        flags = ["hypotension (MAP<65)", "mechanical ventilation"]
        score = _acuity_score(flags)
        # hypotension=1.5, mechanical ventilation=2.0
        assert score == 3.5

    def test_unknown_flag_gets_base_weight(self):
        score = _acuity_score(["some unknown flag"])
        assert score == 0.5

    def test_all_severity_flags_from_output_json(self):
        """Regression test against the real flags from output.json case 426976."""
        flags = [
            "hypotension (MAP<65)",
            "tachypnea (RR>30)",
            "hypoxemia (SpO2<90)",
            "hypothermia (T<36)",
            "altered mental status (GCS=-3)",
            "mechanical ventilation",
            "vasopressor use (Norepinephrine)",
        ]
        score = _acuity_score(flags)
        # 1.5 + 1.0 + 1.5 + 0.8 + 1.5 + 2.0 + 1.8 = 10.1
        assert score == 10.1


class TestTriagePriority:
    def test_high(self):
        assert _triage_priority(6.0) == "High"
        assert _triage_priority(10.1) == "High"

    def test_medium(self):
        assert _triage_priority(3.0) == "Medium"
        assert _triage_priority(5.9) == "Medium"

    def test_low(self):
        assert _triage_priority(0.0) == "Low"
        assert _triage_priority(2.9) == "Low"


class TestCompareVitals:
    def test_no_data(self):
        assert _compare_vitals({}, {}) == []

    def test_map_difference(self):
        new = {"map": {"value": 50.0}}
        ret = {"map": {"value": 70.0}}
        notes = _compare_vitals(new, ret)
        assert len(notes) == 1
        assert "lower" in notes[0]

    def test_spo2_improvement(self):
        new = {"oxygenation": {"spo2_min": 95.0}}
        ret = {"oxygenation": {"spo2_min": 88.0}}
        notes = _compare_vitals(new, ret)
        assert len(notes) == 1
        assert "better" in notes[0]

    def test_small_differences_ignored(self):
        """Differences within tolerance should produce no notes."""
        new = {"map": {"value": 68.0}, "heart_rate": {"value": 82.0}}
        ret = {"map": {"value": 70.0}, "heart_rate": {"value": 80.0}}
        notes = _compare_vitals(new, ret)
        assert notes == []


class TestGetOrganFlag:
    def test_dict_with_flag_true(self):
        assert _get_organ_flag({"vent": {"flag": True}}, "vent") is True

    def test_dict_with_flag_false(self):
        assert _get_organ_flag({"vent": {"flag": False}}, "vent") is False

    def test_bool_directly(self):
        assert _get_organ_flag({"dialysis": True}, "dialysis") is True

    def test_missing_key(self):
        assert _get_organ_flag({}, "dialysis") is False


class TestRuleBasedAdapt:
    """Integration tests for the full rule_based_adapt function."""

    @pytest.fixture
    def new_patient_high_acuity(self):
        """New patient with many severity flags (high acuity)."""
        return {
            "demographics": {"age": 72, "gender": "Male"},
            "triage_context": {
                "severity_flags": [
                    "hypotension (MAP<65)",
                    "tachypnea (RR>30)",
                    "hypoxemia (SpO2<90)",
                    "mechanical ventilation",
                    "vasopressor use (Norepinephrine)",
                ]
            },
            "acute_physiology_24h": {
                "map": {"value": 55.0},
                "heart_rate": {"value": 110.0},
                "oxygenation": {"spo2_min": 85.0},
                "gcs": {"total": 8},
            },
            "organ_support": {
                "mechanical_ventilation": {"flag": True},
                "vasopressor_use": {"flag": True, "agents": ["Norepinephrine"]},
                "dialysis": {"flag": False},
            },
            "comorbidities": {
                "flags": {"chf": True, "copd": True, "ckd": False},
            },
        }

    @pytest.fixture
    def retrieved_case_moderate(self):
        """Retrieved case with moderate severity and hospital mortality."""
        return {
            "demographics": {"age": 69, "gender": "Male"},
            "triage_context": {
                "severity_flags": [
                    "hypotension (MAP<65)",
                    "tachypnea (RR>30)",
                ]
            },
            "acute_physiology_24h": {
                "map": {"value": 58.0},
                "heart_rate": {"value": 89.0},
                "oxygenation": {"spo2_min": 88.0},
                "gcs": {"total": 12},
            },
            "organ_support": {
                "mechanical_ventilation": {"flag": False},
                "vasopressor_use": {"flag": True, "agents": ["Norepinephrine"]},
                "dialysis": {"flag": False},
            },
            "comorbidities": {
                "flags": {"chf": False, "copd": True, "ckd": False},
            },
            "icu_los": {"hospital_mortality": True, "los_days": 7.7},
        }

    def test_high_acuity_gets_high_priority(
        self, new_patient_high_acuity, retrieved_case_moderate
    ):
        result = rule_based_adapt(
            new_patient_high_acuity, retrieved_case_moderate
        )
        assert result["adapted_priority"] == "High"
        assert result["new_acuity"] > result["retrieved_acuity"]
        assert result["strategy"] == "rule_based"
        assert "adapted_at" in result

    def test_comparison_notes_generated(
        self, new_patient_high_acuity, retrieved_case_moderate
    ):
        result = rule_based_adapt(
            new_patient_high_acuity, retrieved_case_moderate
        )
        notes = result["comparison_notes"]
        # HR difference is 110 - 89 = 21 > 10, so should appear
        assert any("Heart rate" in n for n in notes)

    def test_organ_support_escalation(
        self, new_patient_high_acuity, retrieved_case_moderate
    ):
        result = rule_based_adapt(
            new_patient_high_acuity, retrieved_case_moderate
        )
        adjustments = result["recommended_adjustments"]
        # New patient has mech vent, retrieved does not
        assert any("mechanical ventilation" in a for a in adjustments)

    def test_mortality_warning(
        self, new_patient_high_acuity, retrieved_case_moderate
    ):
        result = rule_based_adapt(
            new_patient_high_acuity, retrieved_case_moderate
        )
        adjustments = result["recommended_adjustments"]
        assert any("mortality" in a.lower() for a in adjustments)

    def test_comorbidity_notes(
        self, new_patient_high_acuity, retrieved_case_moderate
    ):
        result = rule_based_adapt(
            new_patient_high_acuity, retrieved_case_moderate
        )
        adjustments = result["recommended_adjustments"]
        # New patient has chf, retrieved does not
        assert any("chf" in a.lower() for a in adjustments)

    def test_original_solution_preserved(
        self, new_patient_high_acuity, retrieved_case_moderate
    ):
        result = rule_based_adapt(
            new_patient_high_acuity, retrieved_case_moderate
        )
        # The original should be a deep copy
        assert result["original_retrieved_solution"] == retrieved_case_moderate
        assert result["original_retrieved_solution"] is not retrieved_case_moderate

    def test_low_acuity_patient(self):
        """A patient with no severity flags should get Low priority."""
        new = {
            "demographics": {"age": 40},
            "triage_context": {"severity_flags": []},
            "acute_physiology_24h": {},
            "organ_support": {},
            "comorbidities": {"flags": {}},
        }
        ret = {
            "demographics": {"age": 42},
            "triage_context": {"severity_flags": []},
            "acute_physiology_24h": {},
            "organ_support": {},
            "comorbidities": {"flags": {}},
            "icu_los": {},
        }
        result = rule_based_adapt(new, ret)
        assert result["adapted_priority"] == "Low"
        assert result["new_acuity"] == 0.0


# ---------------------------------------------------------------------------
# LLM Adapt Tests (prompt-only mode, no GPU needed)
# ---------------------------------------------------------------------------

class TestLlmAdapt:
    def test_prompt_only_mode(self):
        """llm_adapt with run_local=False should return prompt without inference."""
        new = {"demographics": {"age": 55}}
        ret = {"demographics": {"age": 60}}
        result = llm_adapt(new, ret, run_local=False)
        assert result["strategy"] == "llm_prompt_only"
        assert result["adapted_solution"] is None
        assert "prompt" in result
        assert len(result["prompt"]) > 0
        assert "adapted_at" in result

    def test_prompt_contains_case_data(self):
        new = {"demographics": {"age": 55}, "triage_context": {"severity_flags": ["hypotension"]}}
        ret = {"demographics": {"age": 60}, "triage_context": {"severity_flags": []}}
        result = llm_adapt(new, ret, run_local=False)
        assert "hypotension" in result["prompt"]
        assert "55" in result["prompt"]


class TestBuildLlmPrompt:
    def test_prompt_structure(self):
        prompt = build_llm_prompt(
            {"patient": "new"}, {"patient": "old"}, {"plan": "treat"}
        )
        assert "New patient problem" in prompt
        assert "Retrieved" in prompt
        assert "JSON" in prompt

