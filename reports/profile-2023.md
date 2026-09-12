# 2023 AiM event profile

## Dataset

- Unique recordings: 18
- Source paths: 18
- Logged duration: 33.7 min
- Native sample rate: 20 Hz
- Clock provenance: `inferred_uniform_from_metadata`
- Control candidates: 7

## Recurrence evidence

- Signal review-candidate rate: 12.5%
- DTW period-shift match rate, all: 73.9%
- DTW period-shift match rate, test: 70.0%
- Median absolute cross-correlation peak: 0.656
- Fusion dispositions: {"review": 3, "weak": 7}

## Interpolation boundary

- 0.05 s gaps: best median normalized RMSE 0.539 using pchip.
- 0.5 s gaps: best median normalized RMSE 2.447 using linear.

## Learned-method checks

- Embedding effective rank: 1.03 of 8.
- Autoencoder/PCA test-MSE ratio: 1.03.
- GRU/persistence test-MSE ratio: 7.81.
- Strongest recording-score correlation: Autoencoder–Sequence, rho=0.95.

## Unsupervised structure

- Selected clusters: 3
- Training silhouette: 0.667
- Seed adjusted Rand index: 1.000

These clusters are operating regimes. No independent evidence maps them to laps or track sectors.
