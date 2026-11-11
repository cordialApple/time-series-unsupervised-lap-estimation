# 2023 AiM event profile

## Dataset

- Unique recordings: 18
- Source paths: 18
- Logged duration: 33.7 min
- Native sample rate: 20 Hz
- Clock provenance: `inferred_uniform_from_metadata`
- Control candidates: 7
- Long enough for recurrence evidence: 10

## Recurrence evidence

- Signal review flag: 50.0% of 2 control recordings versus 12.5% of 8 non-control recordings.
- Rank-1 signal lags at search ceiling: 3 of 10 (30%).
- Cross-recording period: 5 of 8 non-control recordings carry a signal or correlation period within 10% of 45.3 s; 2 more sit near half that period, a likely harmonic.
- DTW period-shift match rate, all: 73.9%
- DTW period-shift match rate, test: 70.0% (95% recording-level bootstrap 50%–90%; 20 pairs from 2 recordings; includes 50% chance)
- Median absolute cross-correlation peak: 0.656
- Fusion dispositions: {"review": 3, "weak": 7}

## Calibrated physics detector

- Time domain: 0 of 8 driving recordings reach p < 0.05 (0 after false-discovery control, 0 at the search ceiling); benchmark recovers 95% of tiled laps and flags 32% of unrelated segments over 40 trials.
- Course runs: 8 driving runs segmented from 7 recordings trace one course: pairwise yaw-rate correlation after pace and offset alignment has median 0.83 and minimum 0.73, against a shuffled-run null with 95th percentile 0.42 and maximum 0.47 over 140 draws; 100% of 28 pairs beat the null. Aligned lap times span 80.1-108.5 s, median 86.6 s.

## Interpolation boundary

- 0.05 s gaps: best median normalized RMSE 0.539 using pchip.
- 0.5 s gaps: best median normalized RMSE 2.447 using linear.

## Learned-method checks

- Embedding effective rank: masked 1.03, contrastive 4.34, autoencoder 6.28 of 8.
- Autoencoder/PCA test-MSE ratio: 1.03.
- GRU/persistence test-MSE ratio: 7.81.
- Strongest recording-score correlation: Autoencoder–Sequence, rho=0.95.

## Unsupervised structure

- Selected clusters: 3
- Training silhouette: 0.667
- Seed adjusted Rand index: 1.000
- Composition: largest cluster holds 66.9% of 936 windows, minority clusters draw on 7 of 18 recordings, and the largest cluster mixes 115 control windows with 511 non-control windows.

Clusters are not validated regimes. No independent evidence maps them to laps or track sectors.
