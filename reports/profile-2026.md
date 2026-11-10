# 2026 EV drive day profile

## Dataset

- Unique recordings: 21
- Source paths: 26
- Logged duration: 19.8 min
- Native sample rate: 10 Hz
- Clock provenance: `recorded_time_column`
- Control candidates: 5
- Long enough for recurrence evidence: 15

## Recurrence evidence

- Signal review flag: 25.0% of 4 control recordings versus 44.4% of 9 non-control recordings.
- Rank-1 signal lags at search ceiling: 9 of 13 (69%).
- Cross-recording period: 5 of 11 non-control recordings carry a signal or correlation period within 10% of 16.0 s; that period sits at the 15 s search floor, so it is a likely artifact.
- DTW period-shift match rate, all: 58.7%
- DTW period-shift match rate, test: 50.0% (95% recording-level bootstrap 0%–100%; 16 pairs from 3 recordings; includes 50% chance)
- Median absolute cross-correlation peak: 0.766
- Fusion dispositions: {"insufficient_evidence": 2, "review": 4, "weak": 9}

## Calibrated physics detector

- Distance domain: 1 of 8 driving recordings reach p < 0.05 (1 after false-discovery control, 3 at the search ceiling); benchmark recovers 98% of tiled laps and flags 25% of unrelated segments over 40 trials.
- Time domain: 3 of 9 driving recordings reach p < 0.05 (3 after false-discovery control, 3 at the search ceiling); benchmark recovers 98% of tiled laps and flags 20% of unrelated segments over 40 trials.

## Interpolation boundary

- 0.1 s gaps: best median normalized RMSE 0.205 using pchip.
- 0.5 s gaps: best median normalized RMSE 0.305 using linear.

## Learned-method checks

- Embedding effective rank: masked 1.03, contrastive 3.31, autoencoder 4.01 of 8.
- Autoencoder/PCA test-MSE ratio: 1.28.
- GRU/persistence test-MSE ratio: 1.91.
- Strongest recording-score correlation: Autoencoder–Sequence, rho=0.97.

## Unsupervised structure

- Selected clusters: 3
- Training silhouette: 0.862
- Seed adjusted Rand index: 1.000
- Composition: largest cluster holds 94.6% of 500 windows, minority clusters draw on 3 of 21 recordings, and the largest cluster mixes 105 control windows with 368 non-control windows.

Clusters are not validated regimes. No independent evidence maps them to laps or track sectors.
