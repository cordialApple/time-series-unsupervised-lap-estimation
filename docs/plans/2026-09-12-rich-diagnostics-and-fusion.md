# Rich Diagnostics and Fusion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn both cohort notebook suites into visual analyses with interpolation sensitivity, physical integration, and honest cross-method evidence fusion.

**Architecture:** Shared analysis functions prepare interpolation benchmarks, physical integral proxies, and method-family evidence tables. Notebook generator adds cohort-aware seaborn diagnostics to every method notebook and saves figure-ready diagnostic artifacts. Fusion combines comparable recurrence evidence by family and reports support, disagreement, coverage, and controls without calling scores lap probabilities.

**Tech Stack:** Python, Jupyter, pandas, NumPy, SciPy, seaborn, matplotlib, scikit-learn, PyTorch

---

### Task 1: Lock interpolation and integration contracts

**Files:**
- Create: `tests/test_integration.py`
- Create: `lap_estimation/integration.py`

**Steps:**
1. Write failing tests for time-linear and PCHIP hidden-sample reconstruction.
2. Require no extrapolation and explicit normalized RMSE.
3. Write failing tests for 2026 shaft-revolution/electrical-energy proxies and 2023 relative-heading/change-in-velocity proxies.
4. Implement minimal functions.
5. Run `python -m unittest tests.test_integration -v`; expect pass.

### Task 2: Lock family-aware fusion

**Files:**
- Modify: `tests/test_integration.py`
- Modify: `lap_estimation/integration.py`

**Steps:**
1. Write failing tests for recording-level recurrence evidence from signal, DTW, and channel correlation.
2. Require period agreement, method-family coverage, robust percentile support, dispersion, and `review`/`weak`/`insufficient_evidence` status.
3. Write failing tests for exact-window model typicality consensus kept separate from recurrence fusion.
4. Implement and verify tests.

### Task 3: Add method-specific notebook diagnostics

**Files:**
- Modify: `tools/build_notebooks.py`
- Modify: `tests/test_notebooks.py`

**Steps:**
1. Write failing notebook tests requiring seaborn setup and at least three plot calls in notebooks 1–8.
2. Add signal traces, coverage, sampling, recurrence, smoothing, interpolation, and physical-integral plots to notebook 1.
3. Add DTW matched/control, margin, period, and recording plots to notebook 2.
4. Add recurrence-lag, channel-period, cross-correlation, and pair heatmaps to notebook 3.
5. Add embedding projection, similarity, temporal-separation, and recording plots to notebook 4.
6. Add AE-versus-PCA, delta, timeline, latent, and motion plots to notebook 5.
7. Add contrastive projection, similarity, separation, and recording plots to notebook 6.
8. Add GRU-versus-persistence, excess-error, ratio, and timeline plots to notebook 7.
9. Add cluster-selection, occupancy, transition, distance, and regime-timeline plots to notebook 8.
10. Regenerate notebooks and verify code compilation.

### Task 4: Add fusion report diagnostics

**Files:**
- Modify: `tools/build_notebooks.py`
- Modify: `tests/test_notebooks.py`

**Steps:**
1. Add recurrence-family support, period agreement, method coverage, and disagreement plots.
2. Add exact-window typicality consensus and dispersion timelines.
3. Keep pairplots and Spearman heatmaps.
4. Save fusion and interpolation tables under each cohort artifact root.

### Task 5: Execute, inspect, and simplify

**Files:**
- Regenerate: `notebooks/drive_day_2026/*.ipynb`
- Regenerate: `notebooks/aim_2023/*.ipynb`
- Modify: `README.md`

**Steps:**
1. Execute all 18 notebooks in order.
2. Verify zero notebook errors and all code cells executed.
3. Inspect representative rendered plots from both cohorts.
4. Run full unit suite.
5. Run one simplifier pass.
6. Re-run final validation.
