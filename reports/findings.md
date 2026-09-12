# Findings

Evidence is separated from interpretation and action. `supported` means supported by current internal diagnostics, not verified lap accuracy.

## Recurrence evidence narrows manual review

**Status:** `supported`  
**Scope:** `both`

**Evidence:** Fusion marks 4 of 15 evidence-covered 2026 recordings and 3 of 10 evidence-covered 2023 recordings for review.

**Interpretation:** Signal, correlation, and DTW provide a smaller review queue, not verified lap labels.

**Action:** Annotate boundaries in review recordings before training a supervised detector.

**Sources:** [`recurrence_fusion.csv`](../artifacts/drive_day_2026/recurrence_fusion.csv), [`recurrence_fusion.csv`](../artifacts/aim_2023/recurrence_fusion.csv)

## DTW support is cohort-dependent and teacher-dependent

**Status:** `caution`  
**Scope:** `both`

**Evidence:** Held-out match rates are 50.0% for 2026 and 70.0% for 2023.

**Interpretation:** DTW uses signal-derived periods, so agreement cannot count as independent confirmation.

**Action:** Retest DTW against manually labeled lap pairs and unrelated hard negatives.

**Sources:** [`metrics.csv`](../artifacts/drive_day_2026/02_dtw/metrics.csv), [`metrics.csv`](../artifacts/aim_2023/02_dtw/metrics.csv)

## Interpolation repairs short gaps only

**Status:** `caution`  
**Scope:** `both`

**Evidence:** Best short-gap median normalized RMSE is 0.20 for 2026 and 0.54 for 2023; 0.5 s gaps rise to 0.30 and 2.45.

**Interpretation:** Interior interpolation can repair isolated samples but cannot create missing dynamics or timing truth.

**Action:** Use interpolation only below a validated gap limit and retain an imputation mask.

**Sources:** [`interpolation_benchmark.csv`](../artifacts/drive_day_2026/01_signal_processing/interpolation_benchmark.csv), [`interpolation_benchmark.csv`](../artifacts/aim_2023/01_signal_processing/interpolation_benchmark.csv)

## Masked-reconstruction embeddings collapse

**Status:** `negative`  
**Scope:** `both`

**Evidence:** Effective rank is 1.03 of 8 dimensions for 2026 and 1.03 of 8 for 2023.

**Interpretation:** Near-unity neighbor similarities do not demonstrate useful recurrence structure when almost all variance occupies one direction.

**Action:** Do not use these embeddings as pseudo-labels; redesign objectives and hard-negative sampling first.

**Sources:** [`embeddings.npy`](../artifacts/drive_day_2026/04_learned_embeddings/embeddings.npy), [`embeddings.npy`](../artifacts/aim_2023/04_learned_embeddings/embeddings.npy)

## Autoencoder fails to beat PCA

**Status:** `negative`  
**Scope:** `both`

**Evidence:** Test MSE ratios versus PCA are 1.28 for 2026 and 1.03 for 2023; values above 1 are worse.

**Interpretation:** Nonlinear reconstruction adds no demonstrated value at current model size and training setup.

**Action:** Keep PCA as baseline; revisit autoencoders only with a task-linked validation target.

**Sources:** [`metrics.csv`](../artifacts/drive_day_2026/05_autoencoders/metrics.csv), [`metrics.csv`](../artifacts/aim_2023/05_autoencoders/metrics.csv)

## GRU fails to beat persistence

**Status:** `negative`  
**Scope:** `both`

**Evidence:** Test MSE ratios versus persistence are 1.91 for 2026 and 7.81 for 2023.

**Interpretation:** Slowly sampled telemetry makes unchanged-value prediction a strong baseline; current GRU adds error.

**Action:** Do not promote GRU residuals to labels. Change horizon or objective before retraining.

**Sources:** [`metrics.csv`](../artifacts/drive_day_2026/07_sequence_models/metrics.csv), [`metrics.csv`](../artifacts/aim_2023/07_sequence_models/metrics.csv)

## Three stable operating regimes appear in both cohorts

**Status:** `supported`  
**Scope:** `both`

**Evidence:** Training selects k=3 with silhouette 0.862 for 2026 and 0.667 for 2023; seed ARI is 1.000 and 1.000.

**Interpretation:** Clusters are reproducible operating regimes, not lap phases or track sectors.

**Action:** Use regimes to stratify future labels and error analysis.

**Sources:** [`cluster_selection.csv`](../artifacts/drive_day_2026/08_unsupervised_learning/cluster_selection.csv), [`cluster_selection.csv`](../artifacts/aim_2023/08_unsupervised_learning/cluster_selection.csv)

## Strong method correlations mostly expose shared behavior

**Status:** `caution`  
**Scope:** `both`

**Evidence:** Strongest recording correlations are Autoencoder–Sequence at rho=0.97 for 2026 and Autoencoder–Sequence at rho=0.95 for 2023.

**Interpretation:** Correlated reconstruction and sequence errors reflect shared signal difficulty, not independent lap agreement.

**Action:** Fuse evidence by method family and preserve disagreement instead of averaging all scores.

**Sources:** [`recording_method_spearman.csv`](../artifacts/drive_day_2026/recording_method_spearman.csv), [`recording_method_spearman.csv`](../artifacts/aim_2023/recording_method_spearman.csv)

## What this data cannot tell us

- No independent lap boundaries, start-line beacon, GPS position, or official timing truth are present.
- 2023 elapsed time is inferred from validated 20 Hz sampling because source time cells are blank.
- 2026 timestamps are regular at 10 Hz; interpolation adds no higher-frequency information.
- DTW depends on signal-derived candidate periods, and learned-method windows overlap.
- Stable clusters identify operating regimes only.

## Next actions

1. Review and label boundaries in the seven fusion-selected recordings.
2. Record lap beacon or GPS truth during the next event.
3. Evaluate boundary error, lap-count error, precision, and recall on recording-level held-out splits.
4. Keep PCA and persistence as mandatory baselines.
5. Retain interpolation masks and reject gaps beyond validated limits.
