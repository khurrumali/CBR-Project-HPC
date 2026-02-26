#!/usr/bin/env python3
"""
build_case.py — eICU Case Construction Script

Accepts a patientunitstayid and builds a structured JSON case object
from the eICU SQLite database with demographics, clinical data,
semantic summaries, and triage context.

Usage:
    python3 build_case.py <patientunitstayid>
    python3 build_case.py <patientunitstayid> --output case.json
"""

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
# Extract path from sqlite:////path/to/db
DB_PATH = DATABASE_URL.replace("sqlite:///", "") if DATABASE_URL.startswith("sqlite:///") else DATABASE_URL

DATASET_TAG = "eicu-crd-demo-2.0.1"
CASE_TYPE = "icu_stay"
CREATED_FOR = "cbr_case_library"

# Chronic-condition keywords (lowercase)
CHRONIC_KEYWORDS = [
    "chronic", "diabetes", "copd", "hypertension", "renal failure",
    "heart failure", "cirrhosis", "asthma", "hiv", "hepatitis",
    "immunosuppression", "dialysis", "transplant", "cancer", "malignancy",
    "coronary artery disease", "atrial fibrillation", "hypothyroid",
    "hyperthyroid", "seizure", "epilepsy", "dementia", "alzheimer",
    "parkinson", "rheumatoid", "lupus", "scleroderma",
]

# Drug-category mapping (patterns → category label)
DRUG_CATEGORY_PATTERNS = [
    (r"warfarin|heparin|enoxaparin|rivaroxaban|apixaban|coumadin", "anticoagulant"),
    (r"aspirin|clopidogrel|plavix|ticagrelor", "antiplatelet"),
    (r"metoprolol|atenolol|propranolol|carvedilol|bisoprolol", "beta_blocker"),
    (r"diltiazem|verapamil|amlodipine|nifedipine", "calcium_channel_blocker"),
    (r"lisinopril|enalapril|ramipril|captopril|losartan|valsartan", "ace_arb"),
    (r"furosemide|bumetanide|hydrochlorothiazide|spironolactone|lasix", "diuretic"),
    (r"insulin|metformin|glipizide|glyburide", "antidiabetic"),
    (r"amiodarone|lidocaine|digoxin|adenosine", "antiarrhythmic"),
    (r"morphine|fentanyl|hydromorphone|oxycodone|acetaminophen|ibuprofen|ketorolac", "analgesic"),
    (r"vancomycin|ceftriaxone|meropenem|piperacillin|azithromycin|levofloxacin|ciprofloxacin|metronidazole", "antibiotic"),
    (r"omeprazole|pantoprazole|famotidine|ranitidine|lansoprazole", "gi_protective"),
    (r"propofol|midazolam|lorazepam|diazepam|dexmedetomidine", "sedative"),
    (r"norepinephrine|epinephrine|vasopressin|dopamine|dobutamine|phenylephrine", "vasopressor_inotrope"),
    (r"albuterol|ipratropium|budesonide|fluticasone|montelukast", "pulmonary"),
    (r"prednisone|prednisolone|methylprednisolone|dexamethasone|hydrocortisone", "corticosteroid"),
    (r"simvastatin|atorvastatin|rosuvastatin|pravastatin", "statin"),
]

# carePlanGeneral groups that map to physical-exam categories
EXAM_GROUP_MAP = {
    "Airway": "airway",
    "Ventilation": "ventilation",
    "Neurological": "neuro",
    "Cardiac": "cardiac",
    "GI / Nutrition": "gi_nutrition",
    "Renal": "renal",
    "Infectious Disease": "infectious_disease",
    "Skin / Wound": "skin_wound",
    "Activity": "activity",
    "Safety": "safety",
}

# carePlanGeneral groups that indicate care goals
GOAL_GROUPS = {
    "Care Limitation", "Code Status", "Advance Directive",
    "Goals of Care", "Discharge Planning", "Prognosis",
}


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def safe_int(val):
    """Parse to int, return None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def safe_float(val):
    """Parse to float, return None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_age(val):
    """Handle eICU age strings: '87' → 87, '> 89' → 90, '' → None."""
    if val is None:
        return None
    val = str(val).strip()
    if not val:
        return None
    if ">" in val:
        return 90  # eICU convention for ages > 89
    return safe_int(val)


def minutes_to_hours(minutes):
    """Convert offset in minutes to hours, rounded to 1 decimal."""
    if minutes is None:
        return None
    try:
        return round(float(minutes) / 60.0, 1)
    except (ValueError, TypeError):
        return None


def compute_bmi(height_cm, weight_kg):
    """BMI = weight(kg) / height(m)^2. Returns rounded float or None."""
    h = safe_float(height_cm)
    w = safe_float(weight_kg)
    if h and w and h > 0 and w > 0:
        h_m = h / 100.0
        bmi = w / (h_m * h_m)
        return round(bmi, 1)
    return None


def is_chronic_keyword(text):
    """Check if text contains chronic-condition markers."""
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in CHRONIC_KEYWORDS)


def classify_drug_category(drug_name):
    """Map a drug name to a therapeutic category, or 'other'."""
    if not drug_name:
        return "other"
    lower = drug_name.lower()
    for pattern, category in DRUG_CATEGORY_PATTERNS:
        if re.search(pattern, lower):
            return category
    return "other"


def age_group(age):
    """Bucket age into comparison groups."""
    if age is None:
        return "unknown"
    if age < 40:
        return "<40"
    if age < 65:
        return "40-64"
    if age < 80:
        return "65-79"
    return "80+"


# ──────────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────────
def fetch_rows(conn, table, patient_id):
    """SELECT * FROM table WHERE patientunitstayid = ?  →  list of dicts."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM [{table}] WHERE patientunitstayid = ?", (patient_id,))
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def fetch_patient(conn, patient_id):
    """Fetch single patient row or None."""
    rows = fetch_rows(conn, "patient", patient_id)
    return rows[0] if rows else None


# ──────────────────────────────────────────────
# Section Builders
# ──────────────────────────────────────────────
def build_metadata(patient_id):
    return {
        "dataset": DATASET_TAG,
        "case_type": CASE_TYPE,
        "created_for": CREATED_FOR,
        "patient_unit_stay_id": patient_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_patient(p):
    age_parsed = parse_age(p.get("age"))
    h = safe_float(p.get("admissionheight"))
    w = safe_float(p.get("admissionweight"))
    dw = safe_float(p.get("dischargeweight"))

    return {
        "demographics": {
            "unique_pid": p.get("uniquepid"),
            "patient_health_system_stay_id": safe_int(p.get("patienthealthsystemstayid")),
            "age": age_parsed,
            "age_raw": p.get("age"),
            "gender": p.get("gender"),
            "ethnicity": p.get("ethnicity"),
        },
        "physical_measurements": {
            "admission_height_cm": h,
            "admission_weight_kg": w,
            "discharge_weight_kg": dw,
            "bmi": compute_bmi(h, w),
        },
        "admission_context": {
            "hospital_id": safe_int(p.get("hospitalid")),
            "ward_id": safe_int(p.get("wardid")),
            "unit_type": p.get("unittype"),
            "unit_stay_type": p.get("unitstaytype"),
            "unit_visit_number": safe_int(p.get("unitvisitnumber")),
            "unit_admit_source": p.get("unitadmitsource"),
            "hospital_admit_source": p.get("hospitaladmitsource"),
            "hospital_admit_time": p.get("hospitaladmittime24"),
            "unit_admit_time": p.get("unitadmittime24"),
            "hospital_admit_offset_hrs": minutes_to_hours(p.get("hospitaladmitoffset")),
            "apache_admission_dx": p.get("apacheadmissiondx"),
        },
        "discharge_context": {
            "hospital_discharge_year": safe_int(p.get("hospitaldischargeyear")),
            "hospital_discharge_time": p.get("hospitaldischargetime24"),
            "hospital_discharge_offset_hrs": minutes_to_hours(p.get("hospitaldischargeoffset")),
            "hospital_discharge_location": p.get("hospitaldischargelocation"),
            "hospital_discharge_status": p.get("hospitaldischargestatus"),
            "unit_discharge_time": p.get("unitdischargetime24"),
            "unit_discharge_offset_hrs": minutes_to_hours(p.get("unitdischargeoffset")),
            "unit_discharge_location": p.get("unitdischargelocation"),
            "unit_discharge_status": p.get("unitdischargestatus"),
        },
    }


def build_clinical_data(conn, patient_id):
    """Fetch and transform rows from the 7 clinical tables."""

    # --- diagnosis ---
    dx_rows = fetch_rows(conn, "diagnosis", patient_id)
    diagnosis = []
    for r in dx_rows:
        diagnosis.append({
            "id": r.get("diagnosisid"),
            "diagnosis_string": r.get("diagnosisstring"),
            "icd9_code": r.get("icd9code"),
            "priority": r.get("diagnosispriority"),
            "offset_min": safe_int(r.get("diagnosisoffset")),
            "offset_hrs": minutes_to_hours(r.get("diagnosisoffset")),
            "active_upon_discharge": bool(r.get("activeupondischarge")),
            "is_chronic_condition": is_chronic_keyword(r.get("diagnosisstring")),
        })

    # --- admissionDx ---
    adx_rows = fetch_rows(conn, "admissionDx", patient_id)
    admission_dx = []
    for r in adx_rows:
        admission_dx.append({
            "id": r.get("admissiondxid"),
            "path": r.get("admitdxpath"),
            "name": r.get("admitdxname"),
            "text": r.get("admitdxtext"),
            "offset_min": safe_int(r.get("admitdxenteredoffset")),
            "offset_hrs": minutes_to_hours(r.get("admitdxenteredoffset")),
        })

    # --- carePlanGeneral ---
    cpg_rows = fetch_rows(conn, "carePlanGeneral", patient_id)
    care_plan_general = []
    for r in cpg_rows:
        care_plan_general.append({
            "id": r.get("cplgeneralid"),
            "group": r.get("cplgroup"),
            "value": r.get("cplitemvalue"),
            "offset_min": safe_int(r.get("cplitemoffset")),
            "offset_hrs": minutes_to_hours(r.get("cplitemoffset")),
            "active_upon_discharge": bool(r.get("activeupondischarge")),
        })

    # --- carePlanEOL ---
    eol_rows = fetch_rows(conn, "carePlanEOL", patient_id)
    care_plan_eol = []
    for r in eol_rows:
        care_plan_eol.append({
            "id": r.get("cpleolid"),
            "save_offset_min": safe_int(r.get("cpleolsaveoffset")),
            "discussion_offset_min": safe_int(r.get("cpleoldiscussionoffset")),
            "save_offset_hrs": minutes_to_hours(r.get("cpleolsaveoffset")),
            "discussion_offset_hrs": minutes_to_hours(r.get("cpleoldiscussionoffset")),
            "active_upon_discharge": bool(r.get("activeupondischarge")),
        })

    # --- pastHistory ---
    ph_rows = fetch_rows(conn, "pastHistory", patient_id)
    past_history = []
    for r in ph_rows:
        past_history.append({
            "id": r.get("pasthistoryid"),
            "note_type": r.get("pasthistorynotetype"),
            "path": r.get("pasthistorypath"),
            "value": r.get("pasthistoryvalue"),
            "value_text": r.get("pasthistoryvaluetext"),
            "offset_min": safe_int(r.get("pasthistoryoffset")),
            "offset_hrs": minutes_to_hours(r.get("pasthistoryoffset")),
            "entered_offset_hrs": minutes_to_hours(r.get("pasthistoryenteredoffset")),
        })

    # --- allergy ---
    alg_rows = fetch_rows(conn, "allergy", patient_id)
    allergy = []
    for r in alg_rows:
        allergy.append({
            "id": r.get("allergyid"),
            "drug_name": r.get("drugname"),
            "allergy_type": r.get("allergytype"),
            "allergy_name": r.get("allergyname"),
            "is_drug_allergy": (str(r.get("allergytype", "")).lower() == "drug"),
            "rx_included": bool(r.get("rxincluded")),
            "written_in_eicu": bool(r.get("writtenineicu")),
            "offset_min": safe_int(r.get("allergyoffset")),
            "offset_hrs": minutes_to_hours(r.get("allergyoffset")),
        })

    # --- medication ---
    med_rows = fetch_rows(conn, "medication", patient_id)
    medication = []
    for r in med_rows:
        prn_val = str(r.get("prn", "")).strip()
        medication.append({
            "id": r.get("medicationid"),
            "drug_name": r.get("drugname"),
            "dosage": r.get("dosage"),
            "route": r.get("routeadmin"),
            "frequency": r.get("frequency"),
            "prn": prn_val,
            "is_prn": prn_val.lower() == "yes",
            "loading_dose": r.get("loadingdose"),
            "iv_admixture": r.get("drugivadmixture"),
            "order_cancelled": r.get("drugordercancelled"),
            "drug_category": classify_drug_category(r.get("drugname")),
            "order_offset_hrs": minutes_to_hours(r.get("drugorderoffset")),
            "start_offset_hrs": minutes_to_hours(r.get("drugstartoffset")),
            "stop_offset_hrs": minutes_to_hours(r.get("drugstopoffset")),
        })

    return {
        "diagnosis": diagnosis,
        "admission_dx": admission_dx,
        "care_plan_general": care_plan_general,
        "care_plan_eol": care_plan_eol,
        "past_history": past_history,
        "allergy": allergy,
        "medication": medication,
    }


def build_semantic_summary(clinical_data):
    """Compute semantic_clinical_summary from fetched clinical rows."""

    # --- history ---
    history = []
    # Past history items
    for item in clinical_data.get("past_history", []):
        val = item.get("value")
        if val and val.lower() not in ("", "no health problems"):
            path = item.get("path", "")
            # Extract concise label from hierarchical path
            parts = path.split("/") if path else []
            label = parts[-1] if parts else val
            history.append(label)
    # Chronic diagnoses
    for item in clinical_data.get("diagnosis", []):
        if item.get("is_chronic_condition"):
            ds = item.get("diagnosis_string", "")
            # Use last segment of the hierarchical string
            parts = ds.split("|")
            label = parts[-1].strip() if parts else ds
            history.append(f"[chronic] {label}")

    # --- physical_exam ---
    physical_exam = {}
    for item in clinical_data.get("care_plan_general", []):
        grp = item.get("group", "")
        mapped = EXAM_GROUP_MAP.get(grp)
        if mapped:
            val = item.get("value", "")
            # Collect latest or append
            if mapped not in physical_exam:
                physical_exam[mapped] = val
            else:
                physical_exam[mapped] += f"; {val}"

    # --- treatments ---
    # Group medications by category
    category_drugs = {}
    for item in clinical_data.get("medication", []):
        if str(item.get("order_cancelled", "")).lower() == "yes":
            continue  # skip cancelled orders
        cat = item.get("drug_category", "other")
        name = item.get("drug_name", "unknown")
        # Shorten long drug names: take first meaningful token
        short_name = name.split(" ")[0] if name else name
        category_drugs.setdefault(cat, set()).add(short_name)

    treatments = []
    for cat in sorted(category_drugs):
        drugs = ", ".join(sorted(d for d in category_drugs[cat] if d))
        treatments.append(f"{cat}: {drugs}")

    # --- care_goals ---
    care_goals = []
    for item in clinical_data.get("care_plan_general", []):
        grp = item.get("group", "")
        if grp in GOAL_GROUPS:
            val = item.get("value", "")
            care_goals.append(f"{grp}: {val}")
    # Flag EOL discussions
    if clinical_data.get("care_plan_eol"):
        care_goals.append("End-of-life discussion documented")

    return {
        "history": history if history else ["No significant past history documented"],
        "physical_exam": physical_exam if physical_exam else {"status": "No exam data available"},
        "treatments": treatments if treatments else ["No active medications"],
        "care_goals": care_goals if care_goals else ["No specific care goals documented"],
    }


def build_triage_context(patient_section, clinical_data):
    """Build triage_context with severity indicators and comparison fields."""
    adm = patient_section.get("admission_context", {})
    dis = patient_section.get("discharge_context", {})
    demo = patient_section.get("demographics", {})

    # LOS calculations
    los_icu_hrs = dis.get("unit_discharge_offset_hrs")
    los_hospital_hrs = dis.get("hospital_discharge_offset_hrs")

    # Primary diagnosis: first admission_dx name, or apache dx
    primary_dx = None
    adx_list = clinical_data.get("admission_dx", [])
    # Pick the most specific admission dx (longest path)
    if adx_list:
        specific = max(adx_list, key=lambda x: len(x.get("path", "")))
        primary_dx = specific.get("name")
    if not primary_dx:
        primary_dx = adm.get("apache_admission_dx")

    return {
        "severity_indicators": {
            "apache_admission_dx": adm.get("apache_admission_dx"),
            "unit_type": adm.get("unit_type"),
            "unit_stay_type": adm.get("unit_stay_type"),
            "hospital_discharge_status": dis.get("hospital_discharge_status"),
            "unit_discharge_status": dis.get("unit_discharge_status"),
            "los_icu_hrs": los_icu_hrs,
            "los_hospital_hrs": los_hospital_hrs,
        },
        "multi_institute_comparison_fields": {
            "age_group": age_group(demo.get("age")),
            "outcome": dis.get("hospital_discharge_status"),
            "hospital_id": adm.get("hospital_id"),
            "primary_diagnosis": primary_dx,
        },
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def build_case(patient_id):
    """Construct the full case JSON for a given patientunitstayid."""
    if not DB_PATH or not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at '{DB_PATH}'", file=sys.stderr)
        print("       Check DATABASE_URL in your .env file.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        # Validate patient exists
        patient_row = fetch_patient(conn, patient_id)
        if not patient_row:
            print(f"ERROR: patientunitstayid {patient_id} not found in database.", file=sys.stderr)
            sys.exit(1)

        # Build sections
        metadata = build_metadata(patient_id)
        patient_section = build_patient(patient_row)
        clinical_data = build_clinical_data(conn, patient_id)
        semantic_summary = build_semantic_summary(clinical_data)
        triage_context = build_triage_context(patient_section, clinical_data)

        case = {
            "metadata": metadata,
            "patient": patient_section,
            "clinical_data": clinical_data,
            "semantic_clinical_summary": semantic_summary,
            "triage_context": triage_context,
        }

        # Table row counts for quick reference
        counts = {k: len(v) for k, v in clinical_data.items()}
        metadata["clinical_table_counts"] = counts

        return case

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build a structured JSON case from the eICU database for a given patient stay."
    )
    parser.add_argument(
        "patientunitstayid",
        type=int,
        help="The patientunitstayid to build the case for.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path. Defaults to stdout.",
    )
    args = parser.parse_args()

    case = build_case(args.patientunitstayid)
    json_str = json.dumps(case, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str + "\n")
        print(f"✓ Case written to {args.output} ({len(json_str)} bytes)", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
