# 2026 EV drive day profile

## Dataset

- Unique recordings: 21
- Source paths: 26
- Logged duration: 19.8 min
- Native sample rate: 10 Hz
- Clock provenance: `recorded_time_column`
- Control candidates: 5

## Recurrence evidence

- Signal review-candidate rate: 44.4%
- DTW period-shift match rate, all: 58.7%
- DTW period-shift match rate, test: 50.0%
- Median absolute cross-correlation peak: 0.766
- Fusion dispositions: {"insufficient_evidence": 2, "review": 4, "weak": 9}

## Interpolation boundary

- 0.1 s gaps: best median normalized RMSE 0.205 using pchip.
- 0.5 s gaps: best median normalized RMSE 0.305 using linear.

## Learned-method checks

- Embedding effective rank: 1.03 of 8.
- Autoencoder/PCA test-MSE ratio: 1.28.
- GRU/persistence test-MSE ratio: 1.91.
- Strongest recording-score correlation: Autoencoder–Sequence, rho=0.97.

## Unsupervised structure

- Selected clusters: 3
- Training silhouette: 0.862
- Seed adjusted Rand index: 1.000

These clusters are operating regimes. No independent evidence maps them to laps or track sectors.
