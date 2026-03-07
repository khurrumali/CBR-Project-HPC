#!/usr/bin/env python3
"""
explore_triage_fields.py — R&D Exploration of eICU Tables for Triage Features

Connects to the eICU SQLite database and produces a formatted console report
for each table needed by the triage feature set.  For every table it prints:

  1. Schema (column names + SQLite types)
  2. Row counts (total, for sample patient, first-24h window)
  3. Sample rows for the patient
  4. Per-field profiling (distinct count, top values, NULLs, min/max, delimiters)

Usage:
    python scripts/explore_triage_fields.py [--patient_id 141765]
"""

import argparse
import os
import re
import sqlite3
import sys
import textwrap
from collections import Counter
from statistics import median

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_PATH = (
    DATABASE_URL.replace("sqlite:///", "")
    if DATABASE_URL.startswith("sqlite:///")
    else DATABASE_URL
)

# ── First-24h window (minutes) ──────────────────────────────────────
WINDOW_MIN = 0
WINDOW_MAX = 1440  # 24 h

# ── Colour helpers (ANSI) ───────────────────────────────────────────
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def hdr(text):
    return f"\n{BOLD}{CYAN}{'═' * 72}{RESET}\n{BOLD}{CYAN}  {text}{RESET}\n{BOLD}{CYAN}{'═' * 72}{RESET}"


def sub(text):
    return f"\n  {BOLD}{YELLOW}── {text}{RESET}"


# ── Generic profiling helpers ───────────────────────────────────────
def table_exists(conn, table):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=? COLLATE NOCASE",
        (table,),
    )
    return cur.fetchone() is not None


def get_schema(conn, table):
    cur = conn.execute(f"PRAGMA table_info([{table}])")
    return [(r[1], r[2]) for r in cur.fetchall()]  # (name, type)


def total_rows(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]


def patient_rows(conn, table, pid):
    try:
        return conn.execute(
            f"SELECT COUNT(*) FROM [{table}] WHERE patientunitstayid = ?", (pid,)
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return None  # table has no patientunitstayid


def patient_rows_24h(conn, table, pid, offset_col):
    if not offset_col:
        return None
    try:
        return conn.execute(
            f"SELECT COUNT(*) FROM [{table}] "
            f"WHERE patientunitstayid = ? AND [{offset_col}] BETWEEN ? AND ?",
            (pid, WINDOW_MIN, WINDOW_MAX),
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return None


def sample_rows(conn, table, pid, limit=5):
    try:
        cur = conn.execute(
            f"SELECT * FROM [{table}] WHERE patientunitstayid = ? LIMIT ?",
            (pid, limit),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return cols, rows
    except sqlite3.OperationalError:
        return [], []


def profile_column(conn, table, col, pid=None):
    """Return dict with null_count, distinct_count, top10, min, max, delimiters."""
    where = f"WHERE patientunitstayid = {pid}" if pid else ""

    n_total = conn.execute(f"SELECT COUNT(*) FROM [{table}] {where}").fetchone()[0]
    n_null = conn.execute(
        f"SELECT COUNT(*) FROM [{table}] {where} {'AND' if pid else 'WHERE'} "
        f"([{col}] IS NULL OR TRIM([{col}]) = '')"
    ).fetchone()[0] if n_total else 0

    n_distinct = conn.execute(
        f"SELECT COUNT(DISTINCT [{col}]) FROM [{table}] {where}"
    ).fetchone()[0]

    top10 = conn.execute(
        f"SELECT [{col}], COUNT(*) AS cnt FROM [{table}] {where} "
        f"GROUP BY [{col}] ORDER BY cnt DESC LIMIT 10"
    ).fetchall()

    # Min / Max (try numeric then text)
    try:
        mn = conn.execute(
            f"SELECT MIN(CAST([{col}] AS REAL)) FROM [{table}] {where} "
            f"{'AND' if pid else 'WHERE'} [{col}] IS NOT NULL AND [{col}] != ''"
        ).fetchone()[0]
        mx = conn.execute(
            f"SELECT MAX(CAST([{col}] AS REAL)) FROM [{table}] {where} "
            f"{'AND' if pid else 'WHERE'} [{col}] IS NOT NULL AND [{col}] != ''"
        ).fetchone()[0]
    except Exception:
        mn, mx = None, None

    # Delimiter scan (sample up to 200 non-null values)
    sample_vals = conn.execute(
        f"SELECT [{col}] FROM [{table}] {where} "
        f"{'AND' if pid else 'WHERE'} [{col}] IS NOT NULL AND [{col}] != '' LIMIT 200"
    ).fetchall()
    delimiters = Counter()
    str_lens = []
    for (v,) in sample_vals:
        s = str(v)
        str_lens.append(len(s))
        for d in ("|", "/", ",", ";", "\\", "→"):
            if d in s:
                delimiters[d] += 1

    return {
        "total": n_total,
        "null_count": n_null,
        "null_pct": round(100 * n_null / n_total, 1) if n_total else 0,
        "distinct": n_distinct,
        "top10": top10,
        "min": mn,
        "max": mx,
        "delimiters": dict(delimiters) if delimiters else None,
        "str_len_min": min(str_lens) if str_lens else None,
        "str_len_max": max(str_lens) if str_lens else None,
        "str_len_median": round(median(str_lens), 1) if str_lens else None,
    }


def print_profile(prof, col_name):
    print(f"      {'Distinct:':<16} {prof['distinct']}")
    print(f"      {'NULLs/empty:':<16} {prof['null_count']} / {prof['total']}  ({prof['null_pct']}%)")
    if prof["min"] is not None:
        print(f"      {'Min (numeric):':<16} {prof['min']}")
        print(f"      {'Max (numeric):':<16} {prof['max']}")
    if prof["str_len_min"] is not None:
        print(f"      {'Str len:':<16} min={prof['str_len_min']}  max={prof['str_len_max']}  median={prof['str_len_median']}")
    if prof["delimiters"]:
        print(f"      {'Delimiters:':<16} {prof['delimiters']}")
    print(f"      Top values:")
    for val, cnt in prof["top10"]:
        display = str(val)[:80] if val is not None else "<NULL>"
        print(f"        {cnt:>6}×  {display}")


# ── Table-specific exploration ─────────────────────────────────────

TABLE_SPECS = [
    # (table_name, offset_col, key_fields_to_profile)
    ("patient", None, ["age", "gender", "apacheadmissiondx", "unitdischargeoffset", "unittype", "unitstaytype"]),
    ("apacheApsVar", None, [
        "heartrate", "meanbp", "temperature", "respiratoryrate",
        "fio2", "pao2", "creatinine", "wbc", "bilirubin",
        "albumin", "glucose", "bun", "sodium", "hematocrit",
        "intubated", "ventilator", "eyes", "verbal", "motor", "urine",
    ]),
    ("apachePatientResult", None, [
        "apachescore", "predictedhospitalmortality",
        "actualicumortality", "actualhospitalmortality",
    ]),
    ("vitalPeriodic", "observationoffset", [
        "temperature", "sao2", "heartrate", "respiration",
        "systemicmean", "systemicsystolic", "systemicdiastolic",
        "observationoffset",
    ]),
    ("vitalAperiodic", "observationoffset", [
        "noninvasivemean", "noninvasivesystolic", "noninvasivediastolic",
        "observationoffset",
    ]),
    ("nurseCharting", "nursingchartoffset", [
        "nursingchartcelltypevalname", "nursingchartvalue",
        "nursingchartoffset",
    ]),
    ("lab", "labresultrevisedoffset", [
        "labname", "labresult", "labresulttext",
        "labmeasurenamesystem", "labresultrevisedoffset",
    ]),
    ("respiratoryCare", "respcarestartoffset", [
        "airwaytype", "airwaysize", "ventstartoffset",
        "respcarestartoffset",
    ]),
    ("infusionDrug", "infusionoffset", [
        "drugname", "infusionrate", "drugamount", "infusionoffset",
    ]),
    ("treatment", "treatmentoffset", [
        "treatmentstring", "treatmentoffset",
    ]),
    ("pastHistory", "pasthistoryoffset", [
        "pasthistorypath", "pasthistoryvalue", "pasthistoryvaluetext",
        "pasthistoryoffset",
    ]),
    ("carePlanGeneral", "cplitemoffset", [
        "cplgroup", "cplitemvalue", "cplitemoffset",
    ]),
]


def run_special_queries(conn, pid):
    """Run targeted discovery queries that help map raw values → triage features."""

    print(hdr("SPECIAL DISCOVERY QUERIES"))

    # 1. Distinct lab names (all patients) — need exact spellings
    print(sub("Distinct lab names (all patients, top 50)"))
    try:
        rows = conn.execute(
            "SELECT labname, COUNT(*) AS cnt FROM lab GROUP BY labname ORDER BY cnt DESC LIMIT 50"
        ).fetchall()
        for name, cnt in rows:
            flag = ""
            if name and any(
                kw in name.lower()
                for kw in ("lactate", "creatinine", "platelet", "bilirubin", "wbc", "white blood")
            ):
                flag = f"  {GREEN}◀ TARGET{RESET}"
            print(f"    {cnt:>8}×  {name}{flag}")
    except Exception as e:
        print(f"    {RED}ERROR: {e}{RESET}")

    # 2. GCS-related nurseCharting labels
    print(sub("GCS-related nurseCharting labels (global)"))
    try:
        rows = conn.execute(
            "SELECT nursingchartcelltypevalname, COUNT(*) AS cnt "
            "FROM nurseCharting "
            "WHERE lower(nursingchartcelltypevalname) LIKE '%glasgow%' "
            "   OR lower(nursingchartcelltypevalname) LIKE '%gcs%' "
            "   OR lower(nursingchartcelltypevalname) LIKE '%coma%' "
            "GROUP BY nursingchartcelltypevalname ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        if rows:
            for name, cnt in rows:
                print(f"    {cnt:>8}×  {name}")
        else:
            print("    (none found)")
    except Exception as e:
        print(f"    {RED}ERROR: {e}{RESET}")

    # 3. GCS sample values for the patient
    print(sub(f"GCS charting values for patient {pid} (first-24h)"))
    try:
        rows = conn.execute(
            "SELECT nursingchartcelltypevalname, nursingchartvalue, nursingchartoffset "
            "FROM nurseCharting "
            "WHERE patientunitstayid = ? "
            "  AND nursingchartoffset BETWEEN ? AND ? "
            "  AND (lower(nursingchartcelltypevalname) LIKE '%glasgow%' "
            "       OR lower(nursingchartcelltypevalname) LIKE '%gcs%' "
            "       OR lower(nursingchartcelltypevalname) LIKE '%coma%') "
            "ORDER BY nursingchartoffset LIMIT 20",
            (pid, WINDOW_MIN, WINDOW_MAX),
        ).fetchall()
        if rows:
            for name, val, off in rows:
                print(f"    offset={off:>6}  {name}: {val}")
        else:
            print("    (none found)")
    except Exception as e:
        print(f"    {RED}ERROR: {e}{RESET}")

    # 4. Vasopressor infusions (global distinct drug names)
    print(sub("Vasopressor-like drug names in infusionDrug (global)"))
    vaso_pattern = (
        r"norepinephrine|epinephrine|vasopressin|dopamine|dobutamine|"
        r"phenylephrine|milrinone|levophed|neosynephrine"
    )
    try:
        rows = conn.execute(
            "SELECT drugname, COUNT(*) AS cnt FROM infusionDrug "
            "WHERE drugname IS NOT NULL GROUP BY drugname ORDER BY cnt DESC"
        ).fetchall()
        matches = [(n, c) for n, c in rows if n and re.search(vaso_pattern, n, re.I)]
        if matches:
            for name, cnt in matches:
                print(f"    {cnt:>8}×  {name}")
        else:
            print("    (no vasopressor matches found)")
        print(f"    Total distinct infusion drug names: {len(rows)}")
    except Exception as e:
        print(f"    {RED}ERROR: {e}{RESET}")

    # 5. Dialysis-like treatment strings
    print(sub("Dialysis-related treatment strings (global)"))
    try:
        rows = conn.execute(
            "SELECT treatmentstring, COUNT(*) AS cnt FROM treatment "
            "WHERE lower(treatmentstring) LIKE '%dialysis%' "
            "   OR lower(treatmentstring) LIKE '%crrt%' "
            "   OR lower(treatmentstring) LIKE '%cvvh%' "
            "   OR lower(treatmentstring) LIKE '%hemodialysis%' "
            "   OR lower(treatmentstring) LIKE '%ultrafiltration%' "
            "   OR lower(treatmentstring) LIKE '%renal replacement%' "
            "GROUP BY treatmentstring ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        if rows:
            for ts, cnt in rows:
                print(f"    {cnt:>8}×  {ts}")
        else:
            print("    (none found)")
    except Exception as e:
        print(f"    {RED}ERROR: {e}{RESET}")

    # 6. Code status values from carePlanGeneral
    print(sub("Code Status values from carePlanGeneral (global)"))
    try:
        rows = conn.execute(
            "SELECT cplitemvalue, COUNT(*) AS cnt FROM carePlanGeneral "
            "WHERE lower(cplgroup) LIKE '%code status%' "
            "   OR lower(cplgroup) LIKE '%care limitation%' "
            "GROUP BY cplitemvalue ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        if rows:
            for val, cnt in rows:
                print(f"    {cnt:>8}×  {val}")
        else:
            print("    (none found)")
    except Exception as e:
        print(f"    {RED}ERROR: {e}{RESET}")

    # 7. FiO2-related fields in respiratoryCare
    print(sub("respiratoryCare columns (schema)"))
    try:
        schema = get_schema(conn, "respiratoryCare")
        for name, typ in schema:
            flag = ""
            if any(kw in name.lower() for kw in ("fio2", "oxygen", "vent", "airway", "peep")):
                flag = f"  {GREEN}◀ RELEVANT{RESET}"
            print(f"    {name:<40} {typ}{flag}")
    except Exception as e:
        print(f"    {RED}ERROR: {e}{RESET}")

    # 8. FiO2 values in nurseCharting
    print(sub("FiO2 / oxygen labels in nurseCharting (global)"))
    try:
        rows = conn.execute(
            "SELECT nursingchartcelltypevalname, COUNT(*) AS cnt "
            "FROM nurseCharting "
            "WHERE lower(nursingchartcelltypevalname) LIKE '%fio2%' "
            "   OR lower(nursingchartcelltypevalname) LIKE '%oxygen%' "
            "   OR lower(nursingchartcelltypevalname) LIKE '%o2%' "
            "GROUP BY nursingchartcelltypevalname ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        if rows:
            for name, cnt in rows:
                print(f"    {cnt:>8}×  {name}")
        else:
            print("    (none found)")
    except Exception as e:
        print(f"    {RED}ERROR: {e}{RESET}")

    # 9. Past history paths — look for comorbidity patterns
    print(sub("Past history paths containing comorbidity keywords (global top 30)"))
    try:
        rows = conn.execute(
            "SELECT pasthistorypath, COUNT(*) AS cnt FROM pastHistory "
            "WHERE lower(pasthistorypath) LIKE '%chf%' "
            "   OR lower(pasthistorypath) LIKE '%heart failure%' "
            "   OR lower(pasthistorypath) LIKE '%copd%' "
            "   OR lower(pasthistorypath) LIKE '%chronic kidney%' "
            "   OR lower(pasthistorypath) LIKE '%ckd%' "
            "   OR lower(pasthistorypath) LIKE '%renal%' "
            "   OR lower(pasthistorypath) LIKE '%cirrhosis%' "
            "   OR lower(pasthistorypath) LIKE '%liver%' "
            "   OR lower(pasthistorypath) LIKE '%immunosup%' "
            "   OR lower(pasthistorypath) LIKE '%immunodef%' "
            "   OR lower(pasthistorypath) LIKE '%cancer%' "
            "   OR lower(pasthistorypath) LIKE '%malignancy%' "
            "   OR lower(pasthistorypath) LIKE '%metastat%' "
            "   OR lower(pasthistorypath) LIKE '%lymphoma%' "
            "   OR lower(pasthistorypath) LIKE '%leukemia%' "
            "GROUP BY pasthistorypath ORDER BY cnt DESC LIMIT 30"
        ).fetchall()
        if rows:
            for path, cnt in rows:
                print(f"    {cnt:>8}×  {path}")
        else:
            print("    (none found)")
    except Exception as e:
        print(f"    {RED}ERROR: {e}{RESET}")

    # 10. apacheApsVar — check for chronic health fields
    print(sub("apacheApsVar full schema"))
    try:
        schema = get_schema(conn, "apacheApsVar")
        for name, typ in schema:
            print(f"    {name:<40} {typ}")
    except Exception as e:
        print(f"    {RED}Table apacheApsVar not available: {e}{RESET}")


def print_availability_summary(conn, pid, results):
    """Print a final checklist of all triage features vs data availability."""
    print(hdr("DATA AVAILABILITY SUMMARY FOR TRIAGE FEATURES"))
    print(f"  Patient: {pid}   |   Window: offset {WINDOW_MIN}–{WINDOW_MAX} min (first 24 h)\n")

    def check(label, ok):
        mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {mark}  {label}")

    # Demographics
    check("Age (patient.age)", results.get("patient", {}).get("exists", False))
    check("Gender (patient.gender)", results.get("patient", {}).get("exists", False))
    check("Admitting Dx (patient.apacheadmissiondx)", results.get("patient", {}).get("exists", False))

    # Comorbidities
    check("Past history (pastHistory)", results.get("pastHistory", {}).get("patient_rows", 0) > 0)

    # Acute physiology
    check("MAP — vitalPeriodic.systemicmean", results.get("vitalPeriodic", {}).get("rows_24h", 0) > 0)
    check("MAP — vitalAperiodic.noninvasivemean", results.get("vitalAperiodic", {}).get("rows_24h", 0) > 0)
    check("HR — vitalPeriodic.heartrate", results.get("vitalPeriodic", {}).get("rows_24h", 0) > 0)
    check("RR — vitalPeriodic.respiration", results.get("vitalPeriodic", {}).get("rows_24h", 0) > 0)
    check("SpO₂ — vitalPeriodic.sao2", results.get("vitalPeriodic", {}).get("rows_24h", 0) > 0)
    check("Temp — vitalPeriodic.temperature", results.get("vitalPeriodic", {}).get("rows_24h", 0) > 0)
    check("GCS — nurseCharting (Glasgow)", results.get("nurseCharting", {}).get("rows_24h", 0) > 0)

    # Labs
    check("Labs — lab table (first 24h)", results.get("lab", {}).get("rows_24h", 0) > 0)

    # Organ support
    check("Ventilation — respiratoryCare", results.get("respiratoryCare", {}).get("exists", False))
    check("Vasopressors — infusionDrug", results.get("infusionDrug", {}).get("rows_24h", 0) > 0)
    check("Dialysis — treatment", results.get("treatment", {}).get("exists", False))

    # Code status
    check("Code status — carePlanGeneral", results.get("carePlanGeneral", {}).get("patient_rows", 0) > 0)

    # APACHE precomputed
    check("APACHE APS — apacheApsVar", results.get("apacheApsVar", {}).get("exists", False))
    check("APACHE Result — apachePatientResult", results.get("apachePatientResult", {}).get("exists", False))

    # ICU LOS
    check("ICU LOS — patient.unitdischargeoffset", results.get("patient", {}).get("exists", False))
    print()


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Explore eICU tables for triage feature extraction")
    parser.add_argument("--patient_id", type=int, default=141765, help="Sample patientunitstayid")
    args = parser.parse_args()
    pid = args.patient_id

    if not DB_PATH or not os.path.exists(DB_PATH):
        print(f"{RED}ERROR: DB not found at '{DB_PATH}'. Check DATABASE_URL in .env{RESET}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    print(f"\n  DB: {DB_PATH}")
    print(f"  Sample patient: {pid}")

    # Enumerate all existing tables
    all_tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    print(f"  Tables in DB ({len(all_tables)}): {', '.join(all_tables)}\n")

    results = {}

    for table, offset_col, key_fields in TABLE_SPECS:
        exists = table_exists(conn, table)
        info = {"exists": exists}

        print(hdr(f"TABLE: {table}"))

        if not exists:
            # Try case-insensitive match
            match = [t for t in all_tables if t.lower() == table.lower()]
            if match:
                print(f"  {YELLOW}Note: table found as '{match[0]}' (case mismatch){RESET}")
                table = match[0]
                exists = True
                info["exists"] = True
            else:
                print(f"  {RED}TABLE NOT FOUND IN DATABASE{RESET}")
                results[table] = info
                continue

        # Schema
        print(sub("Schema"))
        schema = get_schema(conn, table)
        for col_name, col_type in schema:
            print(f"    {col_name:<45} {col_type}")

        # Row counts
        print(sub("Row Counts"))
        n_total = total_rows(conn, table)
        n_patient = patient_rows(conn, table, pid)
        n_24h = patient_rows_24h(conn, table, pid, offset_col)
        print(f"    Total rows:          {n_total:>10,}")
        if n_patient is not None:
            print(f"    Patient {pid}: {n_patient:>10,}")
        else:
            print(f"    (no patientunitstayid column)")
        if n_24h is not None:
            print(f"    Patient first-24h:   {n_24h:>10,}  (offset {WINDOW_MIN}–{WINDOW_MAX})")
        info["total_rows"] = n_total
        info["patient_rows"] = n_patient
        info["rows_24h"] = n_24h

        # Sample rows
        print(sub(f"Sample Rows (patient {pid}, limit 5)"))
        cols, rows = sample_rows(conn, table, pid, limit=5)
        if rows:
            for i, row in enumerate(rows):
                print(f"    Row {i}:")
                for c, v in zip(cols, row):
                    display = str(v)[:100] if v is not None else "<NULL>"
                    print(f"      {c:<40} {display}")
        else:
            print("    (no rows for this patient)")

        # Profile key fields
        available_cols = {name.lower(): name for name, _ in schema}
        for field in key_fields:
            actual_col = available_cols.get(field.lower())
            if not actual_col:
                print(sub(f"Field: {field}"))
                print(f"      {RED}COLUMN NOT FOUND{RESET}")
                continue
            print(sub(f"Field: {actual_col}"))
            # Profile globally (no pid filter) for distinct/top values
            prof = profile_column(conn, table, actual_col, pid=None)
            print_profile(prof, actual_col)

            # Also profile for the specific patient if rows exist
            if n_patient and n_patient > 0:
                print(f"      {BOLD}(patient-specific):{RESET}")
                prof_p = profile_column(conn, table, actual_col, pid=pid)
                print(f"      Distinct: {prof_p['distinct']}  NULLs: {prof_p['null_count']}/{prof_p['total']}")
                if prof_p["min"] is not None:
                    print(f"      Range: {prof_p['min']} → {prof_p['max']}")

        results[table] = info

    # Special queries
    run_special_queries(conn, pid)

    # Summary
    print_availability_summary(conn, pid, results)

    conn.close()
    print(f"  {GREEN}Exploration complete.{RESET}\n")


if __name__ == "__main__":
    main()
