# Experiment artifacts

Each cohort writes under `<dataset>/<method>/`. Every completed method contains `manifest.json`, `predictions.csv`, and `metrics.csv`. Signal baseline also contains data profile, sensor coverage, smoothing sensitivity, and ranked periodicity candidates.

Current cohorts:

- `drive_day_2026`: recorded 10 Hz clock, motor/electrical analysis channels
- `aim_2023`: inferred uniform 20 Hz clock, inertial analysis channels

Every manifest stores dataset key, native sample rate, clock provenance, selected channels, semantic mapping, motion policy, hashes, recording splits, versions, runtime, and label provenance. Comparison files stay inside each cohort root. Cross-cohort pooling is unsupported.
