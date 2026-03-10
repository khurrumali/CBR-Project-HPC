# 🚀 Todo (Execution Queue)
> **Executor Agent:** Work through these sequentially. When finished, change `[ ]` to `[x]` and move the item to the Completed section.

- [ ] **[TASK]** Enrich new-patient triage case with vitals and severity flags before MedGemma prompt construction
  - **Plan/Context:**
    - **Problem:** When a new patient is built via `--age`/`--gender`/`--problem` CLI args (direct input path in `build_triage_case.py`), the resulting case has empty vitals and no severity flags. MedGemma sees "Key Vitals: (none)" and cannot perform a meaningful comparison or triage.
    - **Goal:** Populate `vitals` (HR, MAP, GCS, SpO2, RR, Temp, Lactate, Creatinine) and `severity_flags` for the new patient before the prompt is assembled.
    - **Approach options (pick one or combine):**
      1. **Prompt the user for vitals at CLI invocation** — add `--vitals` JSON arg or individual `--hr`, `--map`, etc. flags to `run_cbr_medgemma.py` and pass them through to `build_triage_case.py`.
      2. **Lookup from KG/SQLite by patient ID** — if the patient exists in the DB (`--id` path), pull charted vitals from `vitalperiodic`/`vitalaperiodic` and lab values from `lab` tables, then run through the existing severity-flag logic.
      3. **Hybrid** — accept `--id` for DB lookup with a fallback to manual `--vitals` override.
    - **Files to change:** `src/inference/build_triage_case.py`, `scripts/run_cbr_medgemma.py` (arg parsing + case assembly), possibly `src/cbr/retrieval.py` (severity flag computation is already there).
    - **Acceptance criterion:** Slurm job log shows non-empty `Key Vitals` and `Severity Flags` in the GENERATED PROMPT PREVIEW for a new patient.


# 🧠 Planning Backlog
> **Planner Agent:** Read bugs from the Bug Inbox, formulate a step-by-step fix, and move them up to the Todo queue formatted as a `[TASK]`.

- [ ] **Retrieval gap — no structured query case:** `retrieve()` takes scalar `age`/`problem` but cannot compare acute physiology (MAP, GCS, lactate) between query and candidate. Requires building a structured query-patient case first (dependent on the vitals enrichment task above), then expanding `compute_similarity` to score vitals alignment.
- [ ] **Retrieval gap — SPARQL LIMIT arbitrary ordering:** Candidate pool (now 200) is returned in KG traversal order, not relevance order. For high-prevalence diagnoses (e.g., hypertension, sepsis) the pool may miss better-matched patients beyond rank 200. Consider a two-pass approach: broad SPARQL without age filter → rank → top-200 into hydration.

# 🐞 Bug Inbox
> **Bug Tracker Agent:** Append new bugs here.
> **Executor Agent:** IGNORE THIS SECTION.

- [ ] (Empty - waiting for bug reports)

# ✅ Completed
- [x] Architecture Design
- [x] CBR Inference Pipeline
- [x] Medgemma Download
- [x] Configure and successfully execute sbatch for `run_cbr_medgemma.py` (MedGemma 4B) — job 6615948 succeeded 2026-03-09 after fixing `float16` → `bfloat16` CUDA assertion crash
- [x] **Retrieval system gap analysis** — identified 10 gaps across scoring depth, candidate discovery, case library integration, and efficiency (2026-03-10)
- [x] **Wire `gender`/`comorbidity_flags` through retrieval call chain** (2026-03-10):
  - `cbr_retrieve.py`: added `--gender` and `--comorbidities` (comma-separated) CLI args, wired to `retrieve()`
  - `run_cbr_medgemma.py`: extracts `query_gender` and `active_comorbs` from `target_case`, passes to subprocess
  - For the `--age`/`--gender`/`--problem` direct-input path, gender is now passed immediately; comorbidities pass when the enrichment task (above) populates them
- [x] **Retrieval system v2 improvements** — implemented in `src/cbr/retrieval.py` (2026-03-10):
  - Replaced binary diagnosis match (1.0/0.5) with Jaccard token-overlap scoring
  - Added comorbidity Jaccard dimension (requires `query_comorbidity_flags` kwarg)
  - Added gender match dimension (requires `query_gender` kwarg)
  - Added APACHE score proximity dimension
  - Replaced raw organ-support count with triage severity flag richness signal
  - Rewired weight distribution: diag(0.28) + age(0.25) + comorb(0.20) + severity(0.15) + gender(0.07) + apache(0.05)
  - Connected CaseLibrary as secondary candidate source (`_retrieve_from_case_library`)
  - Increased SPARQL candidate pool from 50 → 200
  - `retrieve()` now accepts `gender` and `comorbidity_flags` kwargs
  - All 33 tests passing
