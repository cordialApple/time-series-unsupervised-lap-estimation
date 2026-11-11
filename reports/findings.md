# Findings

Evidence is separated from interpretation and action. `supported` means supported by current internal diagnostics, not verified lap accuracy. `negative` describes the tested configuration only.

## Fusion ranks recordings for review; it does not detect recurrence

**Status:** `caution`  
**Scope:** `both`

**Evidence:** Fusion marks 4 of 15 evidence-covered 2026 recordings and 3 of 10 evidence-covered 2023 recordings for review. Marks come from within-cohort percentile ranks cut at fused support >= 2/3, so roughly the top third is selected whatever the absolute evidence.

**Interpretation:** Signal and correlation run the same autocorrelation on the same channels, and DTW borrows the signal period. Fusion is effectively one evidence source and a priority ranking, not a count of recordings with recurrence.

**Action:** Label in ranked order. Do not quote review counts as detections.

**Sources:** [`recurrence_fusion.csv`](../artifacts/drive_day_2026/recurrence_fusion.csv), [`recurrence_fusion.csv`](../artifacts/aim_2023/recurrence_fusion.csv)

## Signal flag does not separate controls from driving

**Status:** `negative`  
**Scope:** `both`

**Evidence:** Review flag fires on 25.0% of 4 control recordings versus 44.4% of 9 non-control recordings in 2026, and on 50.0% of 2 control recordings versus 12.5% of 8 non-control recordings in 2023. Minimum rank-1 cycle count is 2.02 and 2.23: lag is capped at half the duration, so the `cycle_count >= 2` gate never rejects and the flag is correlation >= 0.25 alone.

**Interpretation:** Stationary or low-dynamics recordings pass about as often as driving at these sample sizes. The threshold has no demonstrated discriminative power.

**Action:** Calibrate recurrence scores against a surrogate null and report per-recording p-values before using the flag.

**Sources:** [`metrics.csv`](../artifacts/drive_day_2026/01_signal_processing/metrics.csv), [`metrics.csv`](../artifacts/aim_2023/01_signal_processing/metrics.csv)

## Many candidate periods sit at the search ceiling

**Status:** `caution`  
**Scope:** `both`

**Evidence:** Rank-1 signal lags at or above 80% of the search ceiling (half the duration, capped at 120 s): 9 of 13 (69%) in 2026 and 3 of 10 (30%) in 2023.

**Interpretation:** A peak at the ceiling compares roughly two halves of one recording and is edge-prone. It should not be read as a lap period.

**Action:** Require peaks well inside the search range, or record sessions spanning several expected laps.

**Sources:** [`periodicity_candidates.csv`](../artifacts/drive_day_2026/01_signal_processing/periodicity_candidates.csv), [`periodicity_candidates.csv`](../artifacts/aim_2023/01_signal_processing/periodicity_candidates.csv)

## 2023 recordings converge near 45 s; 2026 recordings do not converge on a usable period

**Status:** `caution`  
**Scope:** `both`

**Evidence:** 2023: 5 of 8 non-control recordings carry a signal or correlation period within 10% of 45.3 s; 2 more sit near half that period, a likely harmonic. 2026: 5 of 11 non-control recordings carry a signal or correlation period within 10% of 16.0 s; that period sits at the 15 s search floor, so it is a likely artifact.

**Interpretation:** Separate recordings agreeing on one period would be stronger evidence than any per-recording score, but these periods carry no null calibration and locate no boundary. The calibrated physics test does not confirm them.

**Action:** Do not quote the shared 2023 period as a lap time. Treat half-period values as harmonics.

**Sources:** [`recurrence_fusion.csv`](../artifacts/drive_day_2026/recurrence_fusion.csv), [`recurrence_fusion.csv`](../artifacts/aim_2023/recurrence_fusion.csv)

## Calibrated physics detector finds no recurrence beyond its false-positive rate

**Status:** `negative`  
**Scope:** `both`

**Evidence:** Block-shuffle null, 199 surrogates per recording. 2026 (motor speed): Distance domain: 1 of 8 driving recordings reach p < 0.05 (1 after false-discovery control, 3 at the search ceiling); benchmark recovers 98% of tiled laps and flags 25% of unrelated segments over 40 trials. Time domain: 3 of 9 driving recordings reach p < 0.05 (3 after false-discovery control, 3 at the search ceiling); benchmark recovers 98% of tiled laps and flags 20% of unrelated segments over 40 trials. 2023 (yaw rate): Time domain: 0 of 8 driving recordings reach p < 0.05 (0 after false-discovery control, 0 at the search ceiling); benchmark recovers 95% of tiled laps and flags 32% of unrelated segments over 40 trials.

**Interpretation:** The detector recovers planted laps, but its null is optimistic: unrelated real segments pass well above the nominal rate. No significant share in real recordings is distinguishable from that benchmark false-positive rate (smallest one-sided binomial p = 0.26), so neither cohort shows lap recurrence inside a recording. Resampling 2026 motor speed onto distance did not expose laps that time-domain analysis missed.

**Action:** Read these p-values as upper bounds on evidence. Tighten the null until benchmark false positives match nominal before trusting any single detection.

**Sources:** [`predictions.csv`](../artifacts/drive_day_2026/10_physics_laps/predictions.csv), [`predictions.csv`](../artifacts/aim_2023/10_physics_laps/predictions.csv)

## 2023 runs repeat one course: laps recur across recordings, not inside them

**Status:** `supported`  
**Scope:** `aim_2023`

**Evidence:** 8 driving runs segmented from 7 recordings trace one course: pairwise yaw-rate correlation after pace and offset alignment has median 0.83 and minimum 0.73, against a shuffled-run null with 95th percentile 0.42 and maximum 0.47 over 140 draws; 100% of 28 pairs beat the null. Aligned lap times span 80.1-108.5 s, median 86.6 s.

**Interpretation:** Each long 2023 recording holds one run of the same course, two in the longest, so the lap unit is the run. That is why no method found recurrence inside a recording. Lap times are relative to the reference run's span under one global pace factor; start and finish lines are not observed, and the null shares the optimism measured for the recurrence detector, though the margin here is wide.

**Action:** Use aligned runs as provisional lap labels. Replace the global pace factor with time-warped alignment for sector times, and log wheel speed or GPS to fix start and finish.

**Sources:** [`predictions.csv`](../artifacts/aim_2023/11_course_runs/predictions.csv)

## DTW held-out support is not distinguishable from chance

**Status:** `caution`  
**Scope:** `both`

**Evidence:** Held-out match rates are 50.0% (95% recording-level bootstrap 0%–100%; 16 pairs from 3 recordings; includes 50% chance) for 2026 and 70.0% (95% recording-level bootstrap 50%–90%; 20 pairs from 2 recordings; includes 50% chance) for 2023.

**Interpretation:** Held-out intervals include chance for: 2026 EV drive day, 2023 AiM event. Pairs are overlapping windows inside few recordings, and DTW uses signal-derived periods, so agreement cannot count as independent confirmation.

**Action:** Retest DTW against manually labeled lap pairs and unrelated hard negatives with leave-one-recording-out evaluation.

**Sources:** [`predictions.csv`](../artifacts/drive_day_2026/02_dtw/predictions.csv), [`predictions.csv`](../artifacts/aim_2023/02_dtw/predictions.csv)

## Interpolation repairs short gaps only

**Status:** `caution`  
**Scope:** `both`

**Evidence:** Best short-gap median normalized RMSE is 0.20 for 2026 and 0.54 for 2023; 0.5 s gaps rise to 0.30 and 2.45.

**Interpretation:** Interior interpolation can repair isolated samples but cannot create missing dynamics or timing truth.

**Action:** Use interpolation only below a validated gap limit and retain an imputation mask.

**Sources:** [`interpolation_benchmark.csv`](../artifacts/drive_day_2026/01_signal_processing/interpolation_benchmark.csv), [`interpolation_benchmark.csv`](../artifacts/aim_2023/01_signal_processing/interpolation_benchmark.csv)

## Masked-reconstruction encoder collapses by design

**Status:** `negative`  
**Scope:** `both`

**Evidence:** Effective rank: masked 1.03, contrastive 3.31, autoencoder 4.01 of 8 for 2026; masked 1.03, contrastive 4.34, autoencoder 6.28 of 8 for 2023.

**Interpretation:** The masked encoder ends in global average pooling and decodes through one linear layer, so a 10 s window cannot be rebuilt from the pooled code. Collapse is architectural, from one untuned configuration, and says nothing about the data. Contrastive and autoencoder codes do not collapse.

**Action:** Do not use these embeddings as pseudo-labels. Keep temporal resolution in the encoder and add hard negatives before retesting.

**Sources:** [`embeddings.npy`](../artifacts/drive_day_2026/04_learned_embeddings/embeddings.npy), [`embeddings.npy`](../artifacts/aim_2023/04_learned_embeddings/embeddings.npy)

## Untuned autoencoder does not beat PCA

**Status:** `negative`  
**Scope:** `both`

**Evidence:** Test MSE ratios versus PCA are 1.28 for 2026 and 1.03 for 2023; values above 1 are worse.

**Interpretation:** One configuration (8 latent dimensions, 25 epochs, no tuning). Ratios carry no interval, so values near 1 are not distinguishable from PCA.

**Action:** Keep PCA as baseline; revisit autoencoders only with a task-linked validation target.

**Sources:** [`metrics.csv`](../artifacts/drive_day_2026/05_autoencoders/metrics.csv), [`metrics.csv`](../artifacts/aim_2023/05_autoencoders/metrics.csv)

## Untuned GRU does not beat persistence

**Status:** `negative`  
**Scope:** `both`

**Evidence:** Test MSE ratios versus persistence are 1.91 for 2026 and 7.81 for 2023.

**Interpretation:** One configuration: 16 hidden units, 20 epochs, no tuning or early stopping. A model that loses to copying the last sample is undertrained; this is not evidence that sequence models fail on this data.

**Action:** Do not promote GRU residuals to labels. Predict deltas, evaluate multi-second horizons, and tune on the validation split before concluding.

**Sources:** [`metrics.csv`](../artifacts/drive_day_2026/07_sequence_models/metrics.csv), [`metrics.csv`](../artifacts/aim_2023/07_sequence_models/metrics.csv)

## Cluster structure is dominated by one regime

**Status:** `caution`  
**Scope:** `both`

**Evidence:** Training selects k=3 with silhouette 0.862 for 2026 and 0.667 for 2023; seed ARI is 1.000 and 1.000. 2026: largest cluster holds 94.6% of 500 windows, minority clusters draw on 3 of 21 recordings, and the largest cluster mixes 105 control windows with 368 non-control windows. 2023: largest cluster holds 66.9% of 936 windows, minority clusters draw on 7 of 18 recordings, and the largest cluster mixes 115 control windows with 511 non-control windows.

**Interpretation:** A dominant cluster that mixes controls with driving is not an operating regime, and high silhouette can come from a few outlying windows. Seed ARI tests initialization only, not resampling stability.

**Action:** Check stability with a recording-level bootstrap and drop redundant channels before using clusters as strata.

**Sources:** [`predictions.csv`](../artifacts/drive_day_2026/08_unsupervised_learning/predictions.csv), [`predictions.csv`](../artifacts/aim_2023/08_unsupervised_learning/predictions.csv)

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
- Methods 1-8 have no null calibration: scores are compared with fixed thresholds or within-cohort ranks, never with a surrogate distribution.
- The validation split is never used for model or threshold selection.
- Evaluation uses one fixed recording split; held-out sets hold two to five recordings per cohort.
- Clusters are not validated as regimes, lap phases, or track sectors.

## Next actions

1. Label boundaries in fusion-ranked order, starting with the 7 top-ranked recordings.
2. Record lap beacon or GPS truth during the next event.
3. Replace the global pace factor with time-warped alignment and derive sector times from aligned 2023 runs.
4. Tighten the physics null until benchmark false positives match nominal alpha.
5. Evaluate boundary error, lap-count error, precision, and recall with leave-one-recording-out splits and recording-level intervals.
6. Keep PCA and persistence as mandatory baselines.
7. Retain interpolation masks and reject gaps beyond validated limits.
