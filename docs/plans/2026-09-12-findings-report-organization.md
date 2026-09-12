# Findings Report Organization Implementation Plan

**Goal:** Organize current 2023 and 2026 evidence into reproducible, action-oriented reports modeled on the reference telemetry pipeline's reporting structure.

**Architecture:** Add one deterministic reporting module that reads existing experiment artifacts and produces a machine-readable evidence ledger, cohort profiles, a narrative findings report, and a curated report index. Keep raw data, experiment artifacts, and notebooks in place; reports summarize them without creating a database or API layer.

**Tech Stack:** Python 3.11+, pandas, NumPy, Markdown, JSON, unittest

---

### Task 1: Define report contract

**Files:**
- Create: `tests/test_reporting.py`
- Create: `lap_estimation/reporting.py`

**Step 1: Write failing tests**

Require a two-cohort evidence summary containing dataset profile, recurrence evidence, interpolation sensitivity, learned-model baseline checks, clustering diagnostics, method correlations, fusion dispositions, limitations, and next actions.

**Step 2: Verify failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_reporting -v`

Expected: import failure because `lap_estimation.reporting` does not exist.

**Step 3: Implement minimal artifact readers**

Read only committed CSV, JSON, and NPY artifacts. Preserve denominators and metric provenance. Reject missing required artifacts instead of silently fabricating values.

**Step 4: Verify pass**

Run: `.venv\Scripts\python.exe -m unittest tests.test_reporting -v`

Expected: all reporting contract tests pass.

### Task 2: Generate human and machine reports

**Files:**
- Modify: `tests/test_reporting.py`
- Modify: `lap_estimation/reporting.py`
- Create: `tools/build_reports.py`
- Generate: `reports/README.md`
- Generate: `reports/findings.md`
- Generate: `reports/findings.json`
- Generate: `reports/profile-2023.md`
- Generate: `reports/profile-2026.md`
- Generate: `reports/figures/*.png`

**Step 1: Write failing output tests**

Require deterministic JSON, cohort-specific profiles, action and limitation sections, reproducibility commands, and curated figures copied from validated artifacts.

**Step 2: Verify failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_reporting -v`

Expected: renderer or output files missing.

**Step 3: Implement report renderer and entry point**

Generate concise narrative from exact artifact values. Label recurrence and fusion rows as review candidates, never verified laps. Link every claim to its source artifact or notebook.

**Step 4: Generate reports**

Run: `.venv\Scripts\python.exe tools\build_reports.py`

Expected: report tree written without modifying raw CSVs.

**Step 5: Verify pass and determinism**

Run report generation twice, compare output hashes, then run focused tests.

### Task 3: Make repository navigation match report structure

**Files:**
- Modify: `README.md`

**Step 1: Add report-first navigation**

Lead with evidence status, report links, architecture flow, project structure, reproduction commands, and explicit limits.

**Step 2: Verify links and full suite**

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass and every local Markdown link resolves.

### Task 4: Cleanup and final audit

**Files:**
- Review all changed reporting files

**Step 1: Run simplifier once**

Preserve report schema and generated prose.

**Step 2: Rebuild reports after cleanup**

Run: `.venv\Scripts\python.exe tools\build_reports.py`

**Step 3: Final verification**

Confirm report outputs are deterministic, 18 notebooks remain executed, 92 curated diagnostic PNG artifacts remain available, and raw data hashes remain unchanged.
