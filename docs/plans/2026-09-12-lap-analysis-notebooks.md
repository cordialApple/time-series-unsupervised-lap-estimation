# Lap Analysis Notebooks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build nine reproducible notebooks comparing telemetry-based lap recurrence methods without overstating unlabeled data.

**Architecture:** Shared `lap_estimation` package owns ingestion, duplicate-safe recording IDs, preprocessing, model runners, recurrence metrics, and artifact contracts. Eight method notebooks consume same recording manifest and stratified recording-level splits; ninth reads saved results only. Dataset-scoped artifact paths keep 2026 and future 2023 cohorts separate.

**Tech Stack:** Python 3.11+, Jupyter, pandas, NumPy, SciPy, matplotlib, scikit-learn, PyTorch

---

### Task 1: Lock data and signal contracts

**Files:**
- Create: `tests/test_data.py`
- Create: `tests/test_signal.py`
- Create: `tests/test_notebooks.py`

**Steps:**
1. Test SHA256 deduplication keeps aliases.
2. Test timestamps become monotonic elapsed seconds.
3. Test constant channels are rejected.
4. Test autocorrelation recovers known periodicity.
5. Test exactly nine valid notebooks exist in required order.
6. Run tests and confirm failure because package and notebooks do not exist.

### Task 2: Build shared analysis package

**Files:**
- Create: `lap_estimation/__init__.py`
- Create: `lap_estimation/config.py`
- Create: `lap_estimation/data.py`
- Create: `lap_estimation/signal.py`
- Create: `lap_estimation/artifacts.py`

**Steps:**
1. Implement deterministic discovery and content-hash grouping.
2. Preserve source aliases and mark conflicting driver names.
3. Parse 10 Hz timestamps and report gaps.
4. Select informative numeric channels without crossing recording boundaries.
5. Implement smoothing, robust scaling, normalized autocorrelation, and local peak ranking.
6. Implement versioned manifest, predictions, and metrics outputs.
7. Run tests until green.

### Task 3: Build notebook suite

**Files:**
- Create: `notebooks/01_signal_processing.ipynb`
- Create: `notebooks/02_dtw.ipynb`
- Create: `notebooks/03_correlation.ipynb`
- Create: `notebooks/04_learned_embeddings.ipynb`
- Create: `notebooks/05_autoencoders.ipynb`
- Create: `notebooks/06_contrastive_learning.ipynb`
- Create: `notebooks/07_sequence_models.ipynb`
- Create: `notebooks/08_unsupervised_learning.ipynb`
- Create: `notebooks/09_comparison_report.ipynb`

**Steps:**
1. Give every notebook question, assumptions, split policy, method, outputs, and limitations.
2. Fully implement signal-processing inventory, controls, smoothing, autocorrelation, spectral check, candidate periods, and artifacts.
3. Implement and execute bounded DTW, correlation, embedding, autoencoder, contrastive, sequence, and clustering experiments.
4. Make comparison notebook load artifacts only and distinguish incompatible tasks.
5. Validate notebook JSON and compile all code cells.

### Task 4: Reproduce baseline and document environment

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `artifacts/.gitkeep`

**Steps:**
1. Record runtime dependencies and deterministic seeds.
2. Run signal-processing pipeline against all unique recordings.
3. Save data profile and provisional recurrence artifacts.
4. State missing sensors, duplicate aliases, stationary controls, and absent ground truth.
5. Run full test suite.

### Task 5: Simplify changed code

**Files:**
- Modify only recently created code when behavior stays identical.

**Steps:**
1. Run one simplifier review.
2. Apply useful reductions.
3. Re-run tests and notebook validation.
