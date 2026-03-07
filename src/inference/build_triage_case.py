#!/usr/bin/env python3
"""
build_triage_case.py — eICU Triage Feature Extraction

Builds a structured JSON triage-feature vector for a given patientunitstayid.
All acute-physiology and lab values use the **first-24-hour** window
(offsets 0–1440 minutes from ICU admission) to capture the clinical
snapshot at the moment of ICU admission.

Feature groups:
  1. Demographics & diagnostics
  2. Chronic comorbidities
  3. Acute physiology (worst-24h vitals + GCS)
  4. Key labs (worst-24h)
  5. Organ-support flags (ventilation, vasopressors, dialysis)
  6. Code status / goals of care
  7. ICU length of stay (outcome bucket)
  8. APACHE score (if available)

Usage:
    python3 build_triage_case.py <patientunitstayid>
    python3 build_triage_case.py <patientunitstayid> --output triage.json
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

# Shared helpers ----------------------------------------------------------
from src.inference.utils import (
    DATASET_TAG,
    DB_PATH,
    DRUG_CATEGORY_PATTERNS,
    VASOPRESSOR_REGEX,
    age_group,
    classify_drug_category,
    compute_bmi,
    fetch_patient,
    fetch_rows,
    fetch_rows_24h,
    get_connection,
    is_chronic_keyword,
    minutes_to_hours,
    parse_age,
    safe_float,
    safe_int,
    table_exists,
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CASE_TYPE = "icu_triage"
CREATED_FOR = "cbr_triage_library"

# First-24h window (minutes from ICU admission)
WINDOW_MIN = 0
WINDOW_MAX = 1440  # 24 h

# ── Comorbidity detection patterns ──────────────────────────────────
# Each key is a comorbidity flag; value is a list of keywords matched
# against pastHistory.pasthistorypath  /  pasthistoryvalue (case-insensitive).
COMORBIDITY_PATTERNS = {
    "chf": [
        "heart failure", "chf", "cardiomyopathy",
        "congestive heart", "systolic dysfunction", "diastolic dysfunction",
        "reduced ejection", "hfref", "hfpef",
    ],
    "copd": [
        "copd", "chronic obstructive", "emphysema",
        "chronic bronchitis",
    ],
    "ckd": [
        "chronic kidney", "ckd", "chronic renal", "esrd",
        "end stage renal", "end-stage renal",
    ],
    "cirrhosis": [
        "cirrhosis", "hepatic failure", "liver failure",
        "portal hypertension", "hepatic encephalopathy",
    ],
    "immunosuppression": [
        "immunosuppression", "immunodeficiency", "aids",
        "hiv", "transplant", "immunosuppressed",
        "chemotherapy", "organ transplant",
    ],
    "cancer": [
        "cancer", "malignancy", "malignant", "metastat",
        "lymphoma", "leukemia", "carcinoma", "melanoma",
        "sarcoma", "myeloma", "tumor", "neoplasm",
        "oncologic", "oncology",
    ],
}

# ── Lab-name mapping ────────────────────────────────────────────────
# Maps our internal key -> list of eICU labname spellings (case-insensitive).
LAB_NAME_MAP = {
    "lactate": ["lactate", "lactic acid"],
    "creatinine": ["creatinine"],
    "platelets": ["platelets", "platelets x 1000", "platelet count", "plt"],
    "bilirubin": [
        "total bilirubin", "bilirubin", "bilirubin, total",
        "T. Bilirubin",
    ],
    "wbc": [
        "wbc", "wbc x 1000", "white blood cell", "WBC x 1000",
        "WBC's", "leukocytes",
    ],
}

# ── GCS nurseCharting label patterns ────────────────────────────────
GCS_TOTAL_LABELS = [
    "glasgow coma score",
    "gcs total",
]
GCS_EYES_LABELS = ["eyes", "gcs - loss of eye opening"]
GCS_VERBAL_LABELS = ["verbal", "gcs - verbal"]
GCS_MOTOR_LABELS = ["motor", "gcs - motor"]

# ── Dialysis treatment keywords ─────────────────────────────────────
DIALYSIS_KEYWORDS = [
    "dialysis", "crrt", "cvvh", "cvvhd", "cvvhdf",
    "hemodialysis", "ultrafiltration", "renal replacement",
    "continuous renal", "peritoneal dialysis", "scuf",
]

# ── FiO2 nurseCharting label patterns ──────────────────────────────
FIO2_LABELS = [
    "fio2", "fio2 (%)", "fi02", "fi02 (%)",
    "fio2(%)", "oxygen concentration",
]


# ══════════════════════════════════════════════
# Section Builders
# ══════════════════════════════════════════════

def build_metadata(patient_id):
    """Build metadata header."""
    return {
        "dataset": DATASET_TAG,
        "case_type": CASE_TYPE,
        "created_for": CREATED_FOR,
        "patient_unit_stay_id": patient_id,
        "first_24h_window_min": f"{WINDOW_MIN}-{WINDOW_MAX}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ── 1. Demographics & Diagnostics ──────────────────────────────────

def build_demographics(conn, patient_row, patient_id):
    """Extract age, gender, admitting diagnosis from patient + diagnosis tables."""
    age_val = parse_age(patient_row.get("age"))

    # Primary admitting diagnosis: try admissionDx first, then APACHE dx
    admitting_dx = None
    try:
        adx_rows = fetch_rows(conn, "admissionDx", patient_id)
        if adx_rows:
            best = max(adx_rows, key=lambda r: len(r.get("admitdxpath", "") or ""))
            admitting_dx = best.get("admitdxname") or best.get("admitdxtext")
    except sqlite3.OperationalError:
        pass

    if not admitting_dx:
        admitting_dx = patient_row.get("apacheadmissiondx")

    # Also get the primary diagnosis string from diagnosis table
    primary_dx_string = None
    try:
        dx_rows = fetch_rows(conn, "diagnosis", patient_id)
        primary = [r for r in dx_rows if str(r.get("diagnosispriority", "")).lower() == "primary"]
        if primary:
            primary_dx_string = primary[0].get("diagnosisstring")
        elif dx_rows:
            earliest = min(dx_rows, key=lambda r: r.get("diagnosisoffset", 99999) or 99999)
            primary_dx_string = earliest.get("diagnosisstring")
    except sqlite3.OperationalError:
        pass

    return {
        "age": age_val,
        "age_raw": patient_row.get("age"),
        "age_group": age_group(age_val),
        "gender": patient_row.get("gender"),
        "ethnicity": patient_row.get("ethnicity"),
        "admitting_diagnosis": admitting_dx,
        "primary_diagnosis_string": primary_dx_string,
        "apache_admission_dx": patient_row.get("apacheadmissiondx"),
        "unit_type": patient_row.get("unittype"),
        "unit_admit_source": patient_row.get("unitadmitsource"),
        "hospital_admit_source": patient_row.get("hospitaladmitsource"),
    }


# ── 2. Chronic Comorbidities ──────────────────────────────────────

def _match_comorbidity(text, keywords):
    """Check if any keyword appears in text (case-insensitive)."""
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def build_comorbidities(conn, patient_id):
    """
    Detect 6 major comorbidity flags from pastHistory + diagnosis tables.
    Returns a dict of {flag_name: bool} plus matched evidence strings.
    """
    flags = {k: False for k in COMORBIDITY_PATTERNS}
    evidence = {k: [] for k in COMORBIDITY_PATTERNS}

    # -- pastHistory table --
    try:
        ph_rows = fetch_rows(conn, "pastHistory", patient_id)
        for row in ph_rows:
            path = row.get("pasthistorypath", "") or ""
            value = row.get("pasthistoryvalue", "") or ""
            value_text = row.get("pasthistoryvaluetext", "") or ""
            combined = f"{path} {value} {value_text}"
            for flag, keywords in COMORBIDITY_PATTERNS.items():
                if _match_comorbidity(combined, keywords):
                    flags[flag] = True
                    parts = path.split("/")
                    label = parts[-1].strip() if parts else path
                    evidence[flag].append(label)
    except sqlite3.OperationalError:
        pass

    # -- diagnosis table (chronic diagnoses) --
    try:
        dx_rows = fetch_rows(conn, "diagnosis", patient_id)
        for row in dx_rows:
            ds = row.get("diagnosisstring", "") or ""
            for flag, keywords in COMORBIDITY_PATTERNS.items():
                if _match_comorbidity(ds, keywords):
                    flags[flag] = True
                    parts = ds.split("|")
                    label = parts[-1].strip() if parts else ds
                    if label not in evidence[flag]:
                        evidence[flag].append(label)
    except sqlite3.OperationalError:
        pass

    # -- apacheApsVar (chronic health fields if present) --
    try:
        if table_exists(conn, "apacheApsVar"):
            aps_rows = fetch_rows(conn, "apacheApsVar", patient_id)
            if aps_rows:
                aps = aps_rows[0]
                chronic_map = {
                    "immunosuppression": ["aids", "immunosuppression"],
                    "cirrhosis": ["hepaticfailure", "cirrhosis"],
                    "cancer": ["lymphoma", "metastaticcancer", "leukemia"],
                }
                for flag, aps_fields in chronic_map.items():
                    for field in aps_fields:
                        val = safe_int(aps.get(field))
                        if val and val >= 1:
                            flags[flag] = True
                            evidence[flag].append(f"apacheApsVar.{field}=1")
    except sqlite3.OperationalError:
        pass

    # Deduplicate evidence
    evidence = {k: list(dict.fromkeys(v)) for k, v in evidence.items()}

    return {
        "flags": flags,
        "evidence": evidence,
        "any_comorbidity": any(flags.values()),
        "comorbidity_count": sum(flags.values()),
    }


# ── 3. Acute Physiology (worst-24h) ───────────────────────────────

def build_acute_physiology(conn, patient_id):
    """
    Extract worst-24h vital signs from vitalPeriodic, vitalAperiodic,
    nurseCharting, and (fallback) apacheApsVar.
    """
    result = {
        "map": {"value": None, "hypotension_flag": None, "source": None},
        "heart_rate": {"value": None, "tachycardia_flag": None},
        "respiratory_rate": {"value": None, "tachypnea_flag": None},
        "oxygenation": {
            "spo2_min": None, "fio2_max": None,
            "pao2_fio2_ratio": None, "hypoxemia_flag": None,
        },
        "temperature": {
            "max_value": None, "min_value": None,
            "fever_flag": None, "hypothermia_flag": None,
        },
        "gcs": {"total": None, "eyes": None, "verbal": None, "motor": None, "ams_flag": None},
        "data_source": [],
    }

    nc_rows = []  # will be shared between GCS and FiO2 sections

    # ── vitalPeriodic (primary for MAP, HR, RR, SpO2, Temp) ──
    try:
        vp_rows = fetch_rows_24h(conn, "vitalPeriodic", patient_id, "observationoffset",
                                 WINDOW_MIN, WINDOW_MAX)
        if vp_rows:
            result["data_source"].append("vitalPeriodic")

            # MAP: MIN(systemicmean) — exclude nulls and 0
            maps = [safe_float(r.get("systemicmean")) for r in vp_rows]
            maps = [v for v in maps if v is not None and v > 0]
            if maps:
                result["map"]["value"] = round(min(maps), 1)
                result["map"]["source"] = "vitalPeriodic.systemicmean"

            # Heart rate: MAX
            hrs = [safe_float(r.get("heartrate")) for r in vp_rows]
            hrs = [v for v in hrs if v is not None and v > 0]
            if hrs:
                result["heart_rate"]["value"] = round(max(hrs), 1)

            # Respiratory rate: MAX
            rrs = [safe_float(r.get("respiration")) for r in vp_rows]
            rrs = [v for v in rrs if v is not None and v > 0]
            if rrs:
                result["respiratory_rate"]["value"] = round(max(rrs), 1)

            # SpO2: MIN
            spo2s = [safe_float(r.get("sao2")) for r in vp_rows]
            spo2s = [v for v in spo2s if v is not None and v > 0]
            if spo2s:
                result["oxygenation"]["spo2_min"] = round(min(spo2s), 1)

            # Temperature: MAX and MIN
            temps = [safe_float(r.get("temperature")) for r in vp_rows]
            temps = [v for v in temps if v is not None and v > 25]  # filter artifacts
            if temps:
                result["temperature"]["max_value"] = round(max(temps), 1)
                result["temperature"]["min_value"] = round(min(temps), 1)
    except sqlite3.OperationalError:
        pass

    # ── vitalAperiodic (supplementary MAP) ──
    try:
        va_rows = fetch_rows_24h(conn, "vitalAperiodic", patient_id, "observationoffset",
                                 WINDOW_MIN, WINDOW_MAX)
        if va_rows:
            result["data_source"].append("vitalAperiodic")
            ni_maps = [safe_float(r.get("noninvasivemean")) for r in va_rows]
            ni_maps = [v for v in ni_maps if v is not None and v > 0]
            if ni_maps:
                ni_min = round(min(ni_maps), 1)
                if result["map"]["value"] is None or ni_min < result["map"]["value"]:
                    result["map"]["value"] = ni_min
                    result["map"]["source"] = "vitalAperiodic.noninvasivemean"
    except sqlite3.OperationalError:
        pass

    # ── nurseCharting -> GCS ──
    try:
        nc_rows = fetch_rows_24h(conn, "nurseCharting", patient_id, "nursingchartoffset",
                                 WINDOW_MIN, WINDOW_MAX)
        if nc_rows:
            gcs_totals = []
            gcs_eyes_vals = []
            gcs_verbal_vals = []
            gcs_motor_vals = []

            for row in nc_rows:
                label = (row.get("nursingchartcelltypevalname") or "").lower()
                val_str = row.get("nursingchartvalue")
                val = safe_int(val_str) if val_str else None

                if val is not None and val > 0:
                    if any(gl in label for gl in GCS_TOTAL_LABELS):
                        gcs_totals.append(val)
                    elif any(gl in label for gl in GCS_EYES_LABELS):
                        gcs_eyes_vals.append(val)
                    elif any(gl in label for gl in GCS_VERBAL_LABELS):
                        gcs_verbal_vals.append(val)
                    elif any(gl in label for gl in GCS_MOTOR_LABELS):
                        gcs_motor_vals.append(val)

            if gcs_totals:
                result["gcs"]["total"] = min(gcs_totals)
                result["data_source"].append("nurseCharting_GCS")
            elif gcs_eyes_vals and gcs_verbal_vals and gcs_motor_vals:
                result["gcs"]["total"] = min(gcs_eyes_vals) + min(gcs_verbal_vals) + min(gcs_motor_vals)
                result["data_source"].append("nurseCharting_GCS_components")

            if gcs_eyes_vals:
                result["gcs"]["eyes"] = min(gcs_eyes_vals)
            if gcs_verbal_vals:
                result["gcs"]["verbal"] = min(gcs_verbal_vals)
            if gcs_motor_vals:
                result["gcs"]["motor"] = min(gcs_motor_vals)
    except sqlite3.OperationalError:
        pass

    # ── nurseCharting -> FiO2 ──
    try:
        if nc_rows:
            fio2_vals = []
            for row in nc_rows:
                label = (row.get("nursingchartcelltypevalname") or "").lower()
                if any(fl in label for fl in FIO2_LABELS):
                    val = safe_float(row.get("nursingchartvalue"))
                    if val is not None:
                        if val > 1:
                            val = val / 100.0
                        if 0.21 <= val <= 1.0:
                            fio2_vals.append(val)
            if fio2_vals:
                result["oxygenation"]["fio2_max"] = round(max(fio2_vals), 2)
                result["data_source"].append("nurseCharting_FiO2")
    except sqlite3.OperationalError:
        pass

    # ── respiratoryCare -> FiO2 fallback ──
    try:
        rc_rows = fetch_rows_24h(conn, "respiratoryCare", patient_id, "respcarestartoffset",
                                 WINDOW_MIN, WINDOW_MAX)
        if rc_rows:
            for row in rc_rows:
                for col_name in row:
                    if "fio2" in col_name.lower():
                        val = safe_float(row.get(col_name))
                        if val is not None:
                            if val > 1:
                                val = val / 100.0
                            if 0.21 <= val <= 1.0:
                                current = result["oxygenation"]["fio2_max"]
                                if current is None or val > current:
                                    result["oxygenation"]["fio2_max"] = round(val, 2)
                                    if "respiratoryCare_FiO2" not in result["data_source"]:
                                        result["data_source"].append("respiratoryCare_FiO2")
    except sqlite3.OperationalError:
        pass

    # ── apacheApsVar (fallback / validation) ──
    try:
        if table_exists(conn, "apacheApsVar"):
            aps_rows = fetch_rows(conn, "apacheApsVar", patient_id)
            if aps_rows:
                aps = aps_rows[0]
                result["data_source"].append("apacheApsVar")

                if result["map"]["value"] is None:
                    v = safe_float(aps.get("meanbp"))
                    if v and v > 0:
                        result["map"]["value"] = round(v, 1)
                        result["map"]["source"] = "apacheApsVar.meanbp"

                if result["heart_rate"]["value"] is None:
                    v = safe_float(aps.get("heartrate"))
                    if v and v > 0:
                        result["heart_rate"]["value"] = round(v, 1)

                if result["respiratory_rate"]["value"] is None:
                    v = safe_float(aps.get("respiratoryrate"))
                    if v and v > 0:
                        result["respiratory_rate"]["value"] = round(v, 1)

                if result["temperature"]["max_value"] is None:
                    v = safe_float(aps.get("temperature"))
                    if v and v > 25:
                        result["temperature"]["max_value"] = round(v, 1)
                        result["temperature"]["min_value"] = round(v, 1)

                # GCS from APACHE (eyes + verbal + motor)
                if result["gcs"]["total"] is None:
                    eyes = safe_int(aps.get("eyes"))
                    verbal = safe_int(aps.get("verbal"))
                    motor = safe_int(aps.get("motor"))
                    if eyes is not None and verbal is not None and motor is not None:
                        result["gcs"]["total"] = eyes + verbal + motor
                        result["gcs"]["eyes"] = eyes
                        result["gcs"]["verbal"] = verbal
                        result["gcs"]["motor"] = motor

                # FiO2 from APACHE
                if result["oxygenation"]["fio2_max"] is None:
                    v = safe_float(aps.get("fio2"))
                    if v is not None:
                        if v > 1:
                            v = v / 100.0
                        if 0.21 <= v <= 1.0:
                            result["oxygenation"]["fio2_max"] = round(v, 2)

                # PaO2/FiO2 ratio from APACHE
                pao2 = safe_float(aps.get("pao2"))
                fio2 = result["oxygenation"]["fio2_max"]
                if pao2 and fio2 and fio2 > 0:
                    ratio = round(pao2 / fio2, 1)
                    result["oxygenation"]["pao2_fio2_ratio"] = ratio
    except sqlite3.OperationalError:
        pass

    # ── Compute clinical flags ──
    map_val = result["map"]["value"]
    result["map"]["hypotension_flag"] = map_val < 65 if map_val is not None else None

    hr_val = result["heart_rate"]["value"]
    result["heart_rate"]["tachycardia_flag"] = hr_val > 130 if hr_val is not None else None

    rr_val = result["respiratory_rate"]["value"]
    result["respiratory_rate"]["tachypnea_flag"] = rr_val > 30 if rr_val is not None else None

    spo2_val = result["oxygenation"]["spo2_min"]
    result["oxygenation"]["hypoxemia_flag"] = spo2_val < 90 if spo2_val is not None else None

    temp_max = result["temperature"]["max_value"]
    temp_min = result["temperature"]["min_value"]
    result["temperature"]["fever_flag"] = temp_max >= 38.0 if temp_max is not None else None
    result["temperature"]["hypothermia_flag"] = temp_min < 36.0 if temp_min is not None else None

    gcs_total = result["gcs"]["total"]
    result["gcs"]["ams_flag"] = gcs_total < 15 if gcs_total is not None else None

    return result


# ── 4. Key Labs (worst-24h) ────────────────────────────────────────

def build_key_labs(conn, patient_id):
    """
    Extract worst-24h lab values: lactate, creatinine, platelets,
    bilirubin (optional), WBC (optional).
    """
    result = {
        "lactate": {"value": None, "elevated_flag": None, "unit": "mmol/L"},
        "creatinine": {"value": None, "renal_dysfunction_flag": None, "unit": "mg/dL"},
        "platelets": {"value": None, "thrombocytopenia_flag": None, "unit": "x10^3/uL"},
        "bilirubin": {"value": None, "unit": "mg/dL"},
        "wbc": {"value": None, "unit": "x10^3/uL"},
        "available_labs_24h": [],
        "data_source": None,
    }

    try:
        lab_rows = fetch_rows_24h(conn, "lab", patient_id, "labresultrevisedoffset",
                                  WINDOW_MIN, WINDOW_MAX)
    except sqlite3.OperationalError:
        return result

    if not lab_rows:
        return result

    result["data_source"] = "lab"

    lab_values = {k: [] for k in LAB_NAME_MAP}
    seen_labs = set()

    for row in lab_rows:
        lab_name = (row.get("labname") or "").strip()
        lab_result = row.get("labresult")
        seen_labs.add(lab_name)

        val = safe_float(lab_result)
        if val is None:
            val = safe_float(row.get("labresulttext"))
        if val is None:
            continue

        lab_lower = lab_name.lower()
        for key, spellings in LAB_NAME_MAP.items():
            if any(s.lower() == lab_lower for s in spellings):
                lab_values[key].append(val)
                break

    result["available_labs_24h"] = sorted(seen_labs)

    # Aggregate: MAX for lactate/creatinine/bilirubin/wbc, MIN for platelets
    if lab_values["lactate"]:
        v = round(max(lab_values["lactate"]), 2)
        result["lactate"]["value"] = v
        result["lactate"]["elevated_flag"] = v >= 2.0

    if lab_values["creatinine"]:
        v = round(max(lab_values["creatinine"]), 2)
        result["creatinine"]["value"] = v
        result["creatinine"]["renal_dysfunction_flag"] = v > 2.0

    if lab_values["platelets"]:
        v = round(min(lab_values["platelets"]), 1)
        result["platelets"]["value"] = v
        result["platelets"]["thrombocytopenia_flag"] = v < 100

    if lab_values["bilirubin"]:
        result["bilirubin"]["value"] = round(max(lab_values["bilirubin"]), 2)

    if lab_values["wbc"]:
        result["wbc"]["value"] = round(max(lab_values["wbc"]), 2)

    return result


# ── 5. Organ Support Flags ─────────────────────────────────────────

def _detect_vasopressors_infusion(conn, patient_id):
    """Check infusionDrug table for vasopressor infusions in first 24h."""
    try:
        rows = fetch_rows_24h(conn, "infusionDrug", patient_id, "infusionoffset",
                              WINDOW_MIN, WINDOW_MAX)
        agents = set()
        for row in rows:
            drug = row.get("drugname") or ""
            if VASOPRESSOR_REGEX.search(drug):
                clean = drug.split("(")[0].strip()
                agents.add(clean)
        return bool(agents), list(agents), "infusionDrug"
    except sqlite3.OperationalError:
        return False, [], None


def _detect_vasopressors_treatment(conn, patient_id):
    """Check treatment table for vasopressor treatment strings in first 24h."""
    try:
        rows = fetch_rows_24h(conn, "treatment", patient_id, "treatmentoffset",
                              WINDOW_MIN, WINDOW_MAX)
        agents = set()
        for row in rows:
            ts = (row.get("treatmentstring") or "").lower()
            if "vasopressor" in ts or "inotropic agent" in ts:
                parts = ts.split("|")
                agent = parts[-1].strip() if parts else ts
                agents.add(agent)
        return bool(agents), list(agents), "treatment"
    except sqlite3.OperationalError:
        return False, [], None


def _detect_vasopressors_medication(conn, patient_id):
    """Check medication table for IV vasopressor orders in first 24h."""
    try:
        rows = fetch_rows(conn, "medication", patient_id)
        agents = set()
        for row in rows:
            start = safe_int(row.get("drugstartoffset"))
            if start is None or not (WINDOW_MIN <= start <= WINDOW_MAX):
                continue
            drug = row.get("drugname") or ""
            route = (row.get("routeadmin") or "").lower()
            if VASOPRESSOR_REGEX.search(drug) and any(r in route for r in ["iv", "intraven", "central"]):
                clean = drug.split(" ")[0].strip()
                agents.add(clean)
        return bool(agents), list(agents), "medication"
    except sqlite3.OperationalError:
        return False, [], None


def build_organ_support(conn, patient_id):
    """
    Detect organ-support flags:
      - Mechanical ventilation (respiratoryCare, apacheApsVar)
      - Vasopressor use (infusionDrug -> treatment -> medication cascade)
      - Dialysis (treatment table)
    """
    result = {
        "mechanical_ventilation": {
            "flag": False, "source": None, "details": None,
        },
        "vasopressor_use": {
            "flag": False, "agents": [], "source": None,
        },
        "dialysis": {
            "flag": False, "source": None, "details": None,
        },
    }

    # ── Mechanical Ventilation ──
    # 1) respiratoryCare table
    try:
        rc_rows = fetch_rows_24h(conn, "respiratoryCare", patient_id, "respcarestartoffset",
                                 WINDOW_MIN, WINDOW_MAX)
        if rc_rows:
            result["mechanical_ventilation"]["flag"] = True
            result["mechanical_ventilation"]["source"] = "respiratoryCare"
            airway_types = set()
            for row in rc_rows:
                at = row.get("airwaytype")
                if at:
                    airway_types.add(at)
            if airway_types:
                result["mechanical_ventilation"]["details"] = list(airway_types)
    except sqlite3.OperationalError:
        pass

    # 2) apacheApsVar fallback for ventilation
    if not result["mechanical_ventilation"]["flag"]:
        try:
            if table_exists(conn, "apacheApsVar"):
                aps_rows = fetch_rows(conn, "apacheApsVar", patient_id)
                if aps_rows:
                    aps = aps_rows[0]
                    intubated = safe_int(aps.get("intubated"))
                    ventilator = safe_int(aps.get("ventilator"))
                    if (intubated and intubated >= 1) or (ventilator and ventilator >= 1):
                        result["mechanical_ventilation"]["flag"] = True
                        result["mechanical_ventilation"]["source"] = "apacheApsVar"
        except sqlite3.OperationalError:
            pass

    # ── Vasopressor Use (cascade: infusionDrug -> treatment -> medication) ──
    found, agents, source = _detect_vasopressors_infusion(conn, patient_id)
    if not found:
        found, agents, source = _detect_vasopressors_treatment(conn, patient_id)
    if not found:
        found, agents, source = _detect_vasopressors_medication(conn, patient_id)

    result["vasopressor_use"]["flag"] = found
    result["vasopressor_use"]["agents"] = agents
    result["vasopressor_use"]["source"] = source

    # ── Dialysis ──
    try:
        tx_rows = fetch_rows_24h(conn, "treatment", patient_id, "treatmentoffset",
                                 WINDOW_MIN, WINDOW_MAX)
        for row in tx_rows:
            ts = (row.get("treatmentstring") or "").lower()
            if any(kw in ts for kw in DIALYSIS_KEYWORDS):
                result["dialysis"]["flag"] = True
                result["dialysis"]["source"] = "treatment"
                result["dialysis"]["details"] = row.get("treatmentstring")
                break
    except sqlite3.OperationalError:
        pass

    return result


# ── 6. Code Status / Goals of Care ────────────────────────────────

def build_code_status(conn, patient_id):
    """Extract code status from carePlanGeneral within first 24h."""
    result = {
        "status": None,
        "full_code_flag": None,
        "all_entries": [],
        "source": None,
    }

    try:
        cpg_rows = fetch_rows_24h(conn, "carePlanGeneral", patient_id, "cplitemoffset",
                                  WINDOW_MIN, WINDOW_MAX)
    except sqlite3.OperationalError:
        try:
            cpg_rows = fetch_rows(conn, "carePlanGeneral", patient_id)
        except sqlite3.OperationalError:
            return result

    code_entries = []
    for row in cpg_rows:
        grp = (row.get("cplgroup") or "").lower()
        if "code status" in grp or "care limitation" in grp:
            val = row.get("cplitemvalue") or ""
            offset = safe_int(row.get("cplitemoffset"))
            code_entries.append({"value": val, "offset_min": offset})

    if code_entries:
        result["source"] = "carePlanGeneral"
        result["all_entries"] = code_entries

        valid = [e for e in code_entries if e["offset_min"] is not None]
        if valid:
            latest = max(valid, key=lambda e: e["offset_min"])
            result["status"] = latest["value"]
        else:
            result["status"] = code_entries[0]["value"]

        status_lower = (result["status"] or "").lower()
        result["full_code_flag"] = "full" in status_lower and "no" not in status_lower

    return result


# ── 7. ICU Length of Stay ──────────────────────────────────────────

def build_icu_los(patient_row):
    """Compute ICU LOS from unitdischargeoffset."""
    offset_min = safe_float(patient_row.get("unitdischargeoffset"))
    if offset_min is None or offset_min <= 0:
        return {
            "los_hours": None,
            "los_days": None,
            "los_bucket": "unknown",
            "discharge_status": patient_row.get("unitdischargestatus"),
            "hospital_discharge_status": patient_row.get("hospitaldischargestatus"),
            "hospital_mortality": None,
        }

    los_hours = round(offset_min / 60.0, 1)
    los_days = round(offset_min / 1440.0, 1)

    if los_days < 3:
        bucket = "<3d"
    elif los_days < 7:
        bucket = "3-7d"
    else:
        bucket = ">=7d"

    discharge_status = patient_row.get("hospitaldischargestatus")
    hospital_mortality = None
    if discharge_status:
        hospital_mortality = discharge_status.lower() in ("expired", "dead")

    return {
        "los_hours": los_hours,
        "los_days": los_days,
        "los_bucket": bucket,
        "discharge_status": patient_row.get("unitdischargestatus"),
        "hospital_discharge_status": discharge_status,
        "hospital_mortality": hospital_mortality,
    }


# ── 8. APACHE Score ───────────────────────────────────────────────

def build_apache_score(conn, patient_id):
    """Fetch APACHE score and predictions from apachePatientResult."""
    result = {
        "apache_score": None,
        "predicted_hospital_mortality": None,
        "actual_icu_mortality": None,
        "actual_hospital_mortality": None,
        "source": None,
    }

    try:
        if not table_exists(conn, "apachePatientResult"):
            return result
        rows = fetch_rows(conn, "apachePatientResult", patient_id)
        if rows:
            r = rows[0]
            result["apache_score"] = safe_float(r.get("apachescore"))
            result["predicted_hospital_mortality"] = safe_float(r.get("predictedhospitalmortality"))
            result["actual_icu_mortality"] = r.get("actualicumortality")
            result["actual_hospital_mortality"] = r.get("actualhospitalmortality")
            result["source"] = "apachePatientResult"
    except sqlite3.OperationalError:
        pass

    return result


# ── Triage Context (severity summary) ─────────────────────────────

def build_triage_context(demographics, acute_phys, labs, organ_support, apache, los):
    """
    Synthesize a severity summary from all extracted features.
    Produces risk flags and a multi-institute comparison block.
    """
    severity_flags = []

    if acute_phys["map"].get("hypotension_flag"):
        severity_flags.append("hypotension (MAP<65)")
    if acute_phys["heart_rate"].get("tachycardia_flag"):
        severity_flags.append("tachycardia (HR>130)")
    if acute_phys["respiratory_rate"].get("tachypnea_flag"):
        severity_flags.append("tachypnea (RR>30)")
    if acute_phys["oxygenation"].get("hypoxemia_flag"):
        severity_flags.append("hypoxemia (SpO2<90)")
    if acute_phys["temperature"].get("fever_flag"):
        severity_flags.append("fever (T>=38)")
    if acute_phys["temperature"].get("hypothermia_flag"):
        severity_flags.append("hypothermia (T<36)")
    if acute_phys["gcs"].get("ams_flag"):
        gcs_val = acute_phys["gcs"]["total"]
        severity_flags.append(f"altered mental status (GCS={gcs_val})")
    if labs["lactate"].get("elevated_flag"):
        severity_flags.append(f"lactic acidosis (lactate={labs['lactate']['value']})")
    if labs["creatinine"].get("renal_dysfunction_flag"):
        severity_flags.append(f"renal dysfunction (Cr={labs['creatinine']['value']})")
    if labs["platelets"].get("thrombocytopenia_flag"):
        severity_flags.append(f"thrombocytopenia (plt={labs['platelets']['value']})")
    if organ_support["mechanical_ventilation"]["flag"]:
        severity_flags.append("mechanical ventilation")
    if organ_support["vasopressor_use"]["flag"]:
        agents = ", ".join(organ_support["vasopressor_use"]["agents"]) or "unknown"
        severity_flags.append(f"vasopressor use ({agents})")
    if organ_support["dialysis"]["flag"]:
        severity_flags.append("renal replacement therapy")

    return {
        "severity_flags": severity_flags,
        "severity_count": len(severity_flags),
        "apache_score": apache.get("apache_score"),
        "predicted_mortality": apache.get("predicted_hospital_mortality"),
        "multi_institute_comparison_fields": {
            "age_group": demographics.get("age_group"),
            "gender": demographics.get("gender"),
            "primary_diagnosis": demographics.get("admitting_diagnosis"),
            "unit_type": demographics.get("unit_type"),
            "los_bucket": los.get("los_bucket"),
            "hospital_mortality": los.get("hospital_mortality"),
        },
    }


# ══════════════════════════════════════════════
# Main Orchestrator
# ══════════════════════════════════════════════

def build_triage_case(patient_id):
    """
    Construct the full triage-feature JSON for a given patientunitstayid.
    All acute values use the first-24h window (offset 0-1440 min).
    """
    conn = get_connection()
    try:
        patient_row = fetch_patient(conn, patient_id)
        if not patient_row:
            print(f"ERROR: patientunitstayid {patient_id} not found.", file=sys.stderr)
            sys.exit(1)

        # Build all sections
        metadata = build_metadata(patient_id)
        demographics = build_demographics(conn, patient_row, patient_id)
        comorbidities = build_comorbidities(conn, patient_id)
        acute_physiology = build_acute_physiology(conn, patient_id)
        key_labs = build_key_labs(conn, patient_id)
        organ_support = build_organ_support(conn, patient_id)
        code_status = build_code_status(conn, patient_id)
        icu_los = build_icu_los(patient_row)
        apache = build_apache_score(conn, patient_id)

        triage_context = build_triage_context(
            demographics, acute_physiology, key_labs,
            organ_support, apache, icu_los,
        )

        case = {
            "metadata": metadata,
            "demographics": demographics,
            "comorbidities": comorbidities,
            "acute_physiology_24h": acute_physiology,
            "key_labs_24h": key_labs,
            "organ_support": organ_support,
            "code_status": code_status,
            "icu_los": icu_los,
            "apache": apache,
            "triage_context": triage_context,
        }

        # Add data-source counts to metadata
        metadata["sections_populated"] = {
            k: v is not None and v != {} and v != []
            for k, v in case.items()
            if k != "metadata"
        }

        return case

    finally:
        conn.close()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build a triage-feature JSON from the eICU database for a given patient stay."
    )
    parser.add_argument(
        "patientunitstayid",
        type=int,
        help="The patientunitstayid to build the triage case for.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path. Defaults to stdout.",
    )
    args = parser.parse_args()

    case = build_triage_case(args.patientunitstayid)
    json_str = json.dumps(case, indent=2, ensure_ascii=False)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str + "\n")
        print(f"OK Triage case written to {args.output} ({len(json_str)} bytes)", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
