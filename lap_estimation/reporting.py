import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


COHORT_LABELS = {
    "drive_day_2026": "2026 EV drive day",
    "aim_2023": "2023 AiM event",
}

FIGURE_SOURCES = {
    "2026-method-pairplot.png": "drive_day_2026/recording_method_pairplot.png",
    "2026-fusion-components.png": "drive_day_2026/family_component_heatmap.png",
    "2026-regime-timeline.png": "drive_day_2026/08_unsupervised_learning/regime_timelines.png",
    "2023-method-pairplot.png": "aim_2023/recording_method_pairplot.png",
    "2023-fusion-components.png": "aim_2023/family_component_heatmap.png",
    "2023-regime-timeline.png": "aim_2023/08_unsupervised_learning/regime_timelines.png",
}


def _metric(metrics, method, metric, split):
    selected = metrics.loc[
        metrics["method"].eq(method) & metrics["metric"].eq(metric) & metrics["split"].eq(split),
        "value",
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one metric row for {method}/{metric}/{split}, found {len(selected)}")
    return float(selected.iloc[0])


def _effective_rank(path):
    embeddings = np.asarray(np.load(path), dtype=float)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    fractions = singular**2
    fractions = fractions / fractions.sum()
    positive = fractions[fractions > 0]
    return float(np.exp(-(positive * np.log(positive)).sum()))


def _strongest_pair(path):
    correlation = pd.read_csv(path, index_col=0)
    pairs = (
        (str(first), str(second), float(correlation.loc[first, second]))
        for index, first in enumerate(correlation.columns)
        for second in correlation.columns[index + 1 :]
        if np.isfinite(correlation.loc[first, second])
    )
    strongest = max(pairs, key=lambda pair: abs(pair[2]), default=None)
    if strongest is None:
        raise ValueError(f"No finite method correlations in {path}")
    first, second, rho = strongest
    return {"methods": [first, second], "rho": rho}


def _interpolation_summary(path):
    interpolation = pd.read_csv(path)
    grouped = (
        interpolation.groupby(["gap_duration_s", "method"], as_index=False)["normalized_rmse"]
        .median()
        .sort_values(["gap_duration_s", "normalized_rmse"])
    )
    gaps = sorted(grouped["gap_duration_s"].unique())
    result = {}
    for label, gap in [("short_gap", gaps[0]), ("long_gap", gaps[-1])]:
        best = grouped.loc[grouped["gap_duration_s"].eq(gap)].iloc[0]
        result[label] = {
            "duration_s": float(gap),
            "best_method": str(best["method"]),
            "median_normalized_rmse": float(best["normalized_rmse"]),
        }
    return result


def build_cohort_summary(repo_root, cohort_key):
    repo_root = Path(repo_root)
    artifact_root = repo_root / "artifacts" / cohort_key
    metrics = pd.read_csv(artifact_root / "comparison_metrics.csv")
    profile = pd.read_csv(artifact_root / "01_signal_processing" / "data_profile.csv")
    fusion = pd.read_csv(artifact_root / "recurrence_fusion.csv")
    run_manifest = json.loads(
        (artifact_root / "01_signal_processing" / "manifest.json").read_text(encoding="utf-8")
    )
    autoencoder = _metric(metrics, "05_autoencoders", "autoencoder_mse", "test")
    pca = _metric(metrics, "05_autoencoders", "pca_mse", "test")
    gru = _metric(metrics, "07_sequence_models", "gru_mse", "test")
    persistence = _metric(metrics, "07_sequence_models", "persistence_mse", "test")
    return {
        "cohort_key": cohort_key,
        "label": COHORT_LABELS[cohort_key],
        "recordings": int(len(profile)),
        "source_paths": int(profile["alias_count"].sum()),
        "duration_min": float(profile["duration_s"].sum() / 60.0),
        "sample_rate_hz": float(run_manifest["data_provenance"]["native_sample_rate_hz"]),
        "clock_provenance": str(profile["clock_provenance"].iloc[0]),
        "control_candidates": int(profile["control_candidate"].sum()),
        "fusion_recordings": int(len(fusion)),
        "fusion_status_counts": {str(key): int(value) for key, value in fusion["evidence_status"].value_counts().items()},
        "signal_review_candidate_rate": _metric(
            metrics, "01_signal_processing", "non_control_review_candidate_rate", "all"
        ),
        "dtw_match_rate_all": _metric(metrics, "02_dtw", "matched_better_than_control_rate", "all"),
        "dtw_match_rate_test": _metric(metrics, "02_dtw", "matched_better_than_control_rate", "test"),
        "cross_correlation_median_absolute_peak": _metric(
            metrics, "03_correlation", "median_absolute_peak_correlation", "all"
        ),
        "embedding_effective_rank": _effective_rank(
            artifact_root / "04_learned_embeddings" / "embeddings.npy"
        ),
        "autoencoder_test_mse": autoencoder,
        "pca_test_mse": pca,
        "autoencoder_to_pca_test_mse_ratio": autoencoder / pca,
        "gru_test_mse": gru,
        "persistence_test_mse": persistence,
        "gru_to_persistence_test_mse_ratio": gru / persistence,
        "selected_cluster_count": int(
            _metric(metrics, "08_unsupervised_learning", "selected_cluster_count", "train")
        ),
        "cluster_silhouette": _metric(metrics, "08_unsupervised_learning", "silhouette", "train"),
        "cluster_seed_ari": _metric(
            metrics, "08_unsupervised_learning", "seed_adjusted_rand_index", "train"
        ),
        "strongest_recording_correlation": _strongest_pair(
            artifact_root / "recording_method_spearman.csv"
        ),
        "interpolation": _interpolation_summary(
            artifact_root / "01_signal_processing" / "interpolation_benchmark.csv"
        ),
    }


def _finding(identifier, title, status, scope, evidence, interpretation, action, sources):
    return {
        "id": identifier,
        "title": title,
        "status": status,
        "scope": scope,
        "evidence": evidence,
        "interpretation": interpretation,
        "action": action,
        "sources": sources,
    }


def build_report_bundle(repo_root):
    cohorts = {key: build_cohort_summary(repo_root, key) for key in COHORT_LABELS}
    y2026 = cohorts["drive_day_2026"]
    y2023 = cohorts["aim_2023"]
    findings = [
        _finding(
            "recurrence-review-set",
            "Recurrence evidence narrows manual review",
            "supported",
            "both",
            f"Fusion marks {y2026['fusion_status_counts'].get('review', 0)} of {y2026['fusion_recordings']} evidence-covered 2026 recordings and {y2023['fusion_status_counts'].get('review', 0)} of {y2023['fusion_recordings']} evidence-covered 2023 recordings for review.",
            "Signal, correlation, and DTW provide a smaller review queue, not verified lap labels.",
            "Annotate boundaries in review recordings before training a supervised detector.",
            ["artifacts/drive_day_2026/recurrence_fusion.csv", "artifacts/aim_2023/recurrence_fusion.csv"],
        ),
        _finding(
            "dtw-mixed-generalization",
            "DTW support is cohort-dependent and teacher-dependent",
            "caution",
            "both",
            f"Held-out match rates are {y2026['dtw_match_rate_test']:.1%} for 2026 and {y2023['dtw_match_rate_test']:.1%} for 2023.",
            "DTW uses signal-derived periods, so agreement cannot count as independent confirmation.",
            "Retest DTW against manually labeled lap pairs and unrelated hard negatives.",
            ["artifacts/drive_day_2026/02_dtw/metrics.csv", "artifacts/aim_2023/02_dtw/metrics.csv"],
        ),
        _finding(
            "interpolation-boundary",
            "Interpolation repairs short gaps only",
            "caution",
            "both",
            f"Best short-gap median normalized RMSE is {y2026['interpolation']['short_gap']['median_normalized_rmse']:.2f} for 2026 and {y2023['interpolation']['short_gap']['median_normalized_rmse']:.2f} for 2023; 0.5 s gaps rise to {y2026['interpolation']['long_gap']['median_normalized_rmse']:.2f} and {y2023['interpolation']['long_gap']['median_normalized_rmse']:.2f}.",
            "Interior interpolation can repair isolated samples but cannot create missing dynamics or timing truth.",
            "Use interpolation only below a validated gap limit and retain an imputation mask.",
            ["artifacts/drive_day_2026/01_signal_processing/interpolation_benchmark.csv", "artifacts/aim_2023/01_signal_processing/interpolation_benchmark.csv"],
        ),
        _finding(
            "embedding-collapse",
            "Masked-reconstruction embeddings collapse",
            "negative",
            "both",
            f"Effective rank is {y2026['embedding_effective_rank']:.2f} of 8 dimensions for 2026 and {y2023['embedding_effective_rank']:.2f} of 8 for 2023.",
            "Near-unity neighbor similarities do not demonstrate useful recurrence structure when almost all variance occupies one direction.",
            "Do not use these embeddings as pseudo-labels; redesign objectives and hard-negative sampling first.",
            ["artifacts/drive_day_2026/04_learned_embeddings/embeddings.npy", "artifacts/aim_2023/04_learned_embeddings/embeddings.npy"],
        ),
        _finding(
            "autoencoder-baseline",
            "Autoencoder fails to beat PCA",
            "negative",
            "both",
            f"Test MSE ratios versus PCA are {y2026['autoencoder_to_pca_test_mse_ratio']:.2f} for 2026 and {y2023['autoencoder_to_pca_test_mse_ratio']:.2f} for 2023; values above 1 are worse.",
            "Nonlinear reconstruction adds no demonstrated value at current model size and training setup.",
            "Keep PCA as baseline; revisit autoencoders only with a task-linked validation target.",
            ["artifacts/drive_day_2026/05_autoencoders/metrics.csv", "artifacts/aim_2023/05_autoencoders/metrics.csv"],
        ),
        _finding(
            "sequence-baseline",
            "GRU fails to beat persistence",
            "negative",
            "both",
            f"Test MSE ratios versus persistence are {y2026['gru_to_persistence_test_mse_ratio']:.2f} for 2026 and {y2023['gru_to_persistence_test_mse_ratio']:.2f} for 2023.",
            "Slowly sampled telemetry makes unchanged-value prediction a strong baseline; current GRU adds error.",
            "Do not promote GRU residuals to labels. Change horizon or objective before retraining.",
            ["artifacts/drive_day_2026/07_sequence_models/metrics.csv", "artifacts/aim_2023/07_sequence_models/metrics.csv"],
        ),
        _finding(
            "stable-operating-regimes",
            "Three stable operating regimes appear in both cohorts",
            "supported",
            "both",
            f"Training selects k=3 with silhouette {y2026['cluster_silhouette']:.3f} for 2026 and {y2023['cluster_silhouette']:.3f} for 2023; seed ARI is {y2026['cluster_seed_ari']:.3f} and {y2023['cluster_seed_ari']:.3f}.",
            "Clusters are reproducible operating regimes, not lap phases or track sectors.",
            "Use regimes to stratify future labels and error analysis.",
            ["artifacts/drive_day_2026/08_unsupervised_learning/cluster_selection.csv", "artifacts/aim_2023/08_unsupervised_learning/cluster_selection.csv"],
        ),
        _finding(
            "method-dependence",
            "Strong method correlations mostly expose shared behavior",
            "caution",
            "both",
            f"Strongest recording correlations are {'–'.join(y2026['strongest_recording_correlation']['methods'])} at rho={y2026['strongest_recording_correlation']['rho']:.2f} for 2026 and {'–'.join(y2023['strongest_recording_correlation']['methods'])} at rho={y2023['strongest_recording_correlation']['rho']:.2f} for 2023.",
            "Correlated reconstruction and sequence errors reflect shared signal difficulty, not independent lap agreement.",
            "Fuse evidence by method family and preserve disagreement instead of averaging all scores.",
            ["artifacts/drive_day_2026/recording_method_spearman.csv", "artifacts/aim_2023/recording_method_spearman.csv"],
        ),
    ]
    return {
        "schema_version": "1.0",
        "cohorts": cohorts,
        "findings": findings,
        "limitations": [
            "No independent lap boundaries, start-line beacon, GPS position, or official timing truth are present.",
            "2023 elapsed time is inferred from validated 20 Hz sampling because source time cells are blank.",
            "2026 timestamps are regular at 10 Hz; interpolation adds no higher-frequency information.",
            "DTW depends on signal-derived candidate periods, and learned-method windows overlap.",
            "Stable clusters identify operating regimes only.",
        ],
        "next_actions": [
            "Review and label boundaries in the seven fusion-selected recordings.",
            "Record lap beacon or GPS truth during the next event.",
            "Evaluate boundary error, lap-count error, precision, and recall on recording-level held-out splits.",
            "Keep PCA and persistence as mandatory baselines.",
            "Retain interpolation masks and reject gaps beyond validated limits.",
        ],
    }


def _cohort_table(cohorts):
    lines = [
        "| Cohort | Recordings | Fusion coverage | Source paths | Logged minutes | Clock | Review candidates |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for summary in cohorts.values():
        lines.append(
            f"| {summary['label']} | {summary['recordings']} | {summary['fusion_recordings']} | {summary['source_paths']} | {summary['duration_min']:.1f} | {summary['clock_provenance']} | {summary['fusion_status_counts'].get('review', 0)} |"
        )
    return "\n".join(lines)


def _render_index(bundle):
    cohorts = list(bundle["cohorts"].values())
    review_counts = [cohort["fusion_status_counts"].get("review", 0) for cohort in cohorts]
    review_count = sum(review_counts)
    recording_count = sum(cohort["recordings"] for cohort in cohorts)
    fusion_count = sum(cohort["fusion_recordings"] for cohort in cohorts)
    negative_count = sum(finding["status"] == "negative" for finding in bundle["findings"])
    variant_label = "variant" if negative_count == 1 else "variants"
    review_breakdown = " and ".join(
        f"{count} from {cohort['label']}"
        for cohort, count in zip(cohorts, review_counts)
    )
    ranks = " and ".join(f"{cohort['embedding_effective_rank']:.2f} of 8" for cohort in cohorts)
    long_gaps = sorted({cohort["interpolation"]["long_gap"]["duration_s"] for cohort in cohorts})
    gap_text = " and ".join(f"{gap:g} s" for gap in long_gaps)
    cluster_counts = sorted({cohort["selected_cluster_count"] for cohort in cohorts})
    cluster_text = "/".join(str(count) for count in cluster_counts)
    return f"""# Lap-estimation reports

> {review_count} recordings worth labeling. {negative_count} current model {variant_label} rejected. Gap sensitivity quantified. No verified lap truth yet.

## Immediate impact

1. **Label {review_count} review recordings.** Fusion covers {fusion_count} evidence-covered recordings from {recording_count} total and prioritizes {review_count}: {review_breakdown}.
2. **Stop promoting collapsed embeddings.** Effective rank is {ranks} across current cohorts.
3. **Keep PCA and persistence.** Current autoencoder and GRU lose to those simple baselines on held-out recordings.
4. **Repair isolated missing samples only.** {gap_text} gaps define current long-gap stress tests and remain unsuitable for inventing missing dynamics.
5. **Use {cluster_text}-cluster regimes for stratification.** They are stable operating states, not inferred laps.

![2026 fusion components](figures/2026-fusion-components.png)

## Evidence at a glance

{_cohort_table(bundle['cohorts'])}

## Lap findings

- **2026 EV drive day:** repeating behavior is plausible in {review_counts[0]} of {cohorts[0]['fusion_recordings']} evidence-covered recordings. Held-out DTW matches its signal-derived period only {cohorts[0]['dtw_match_rate_test']:.1%} of the time, so no 2026 lap is verified.
- **2023 AiM event:** inertial recurrence is plausible in {review_counts[1]} of {cohorts[1]['fusion_recordings']} evidence-covered recordings. Held-out DTW reaches {cohorts[1]['dtw_match_rate_test']:.1%}, but weak signal-candidate coverage and inferred timing prevent verified lap boundaries.

Bottom line: repeated driving structure exists in both cohorts; confirmed laps do not.

## 2026 EV drive day

![2026 method relationships](figures/2026-method-pairplot.png)

[Cohort profile](profile-2026.md) · [Executed comparison notebook](../notebooks/drive_day_2026/09_comparison_report.ipynb)

## 2023 AiM event

![2023 method relationships](figures/2023-method-pairplot.png)

![2023 fusion components](figures/2023-fusion-components.png)

[Cohort profile](profile-2023.md) · [Executed comparison notebook](../notebooks/aim_2023/09_comparison_report.ipynb)

## Findings

[Read evidence, interpretation, and actions](findings.md). Machine-readable copy: [`findings.json`](findings.json).

## Run

```powershell
.venv/Scripts/python.exe tools/build_notebooks.py
.venv/Scripts/python.exe tools/build_reports.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
```
"""


def _render_findings(bundle):
    sections = [
        "# Findings\n",
        "Evidence is separated from interpretation and action. `supported` means supported by current internal diagnostics, not verified lap accuracy.\n",
    ]
    for finding in bundle["findings"]:
        sources = ", ".join(f"[`{Path(source).name}`](../{source})" for source in finding["sources"])
        sections.append(
            f"## {finding['title']}\n\n"
            f"**Status:** `{finding['status']}`  \n"
            f"**Scope:** `{finding['scope']}`\n\n"
            f"**Evidence:** {finding['evidence']}\n\n"
            f"**Interpretation:** {finding['interpretation']}\n\n"
            f"**Action:** {finding['action']}\n\n"
            f"**Sources:** {sources}\n"
        )
    sections.append("## What this data cannot tell us\n\n" + "\n".join(f"- {item}" for item in bundle["limitations"]) + "\n")
    sections.append("## Next actions\n\n" + "\n".join(f"{index}. {item}" for index, item in enumerate(bundle["next_actions"], 1)) + "\n")
    return "\n".join(sections)


def _render_profile(summary):
    short = summary["interpolation"]["short_gap"]
    long = summary["interpolation"]["long_gap"]
    correlation = summary["strongest_recording_correlation"]
    return f"""# {summary['label']} profile

## Dataset

- Unique recordings: {summary['recordings']}
- Source paths: {summary['source_paths']}
- Logged duration: {summary['duration_min']:.1f} min
- Native sample rate: {summary['sample_rate_hz']:g} Hz
- Clock provenance: `{summary['clock_provenance']}`
- Control candidates: {summary['control_candidates']}

## Recurrence evidence

- Signal review-candidate rate: {summary['signal_review_candidate_rate']:.1%}
- DTW period-shift match rate, all: {summary['dtw_match_rate_all']:.1%}
- DTW period-shift match rate, test: {summary['dtw_match_rate_test']:.1%}
- Median absolute cross-correlation peak: {summary['cross_correlation_median_absolute_peak']:.3f}
- Fusion dispositions: {json.dumps(summary['fusion_status_counts'], sort_keys=True)}

## Interpolation boundary

- {short['duration_s']:g} s gaps: best median normalized RMSE {short['median_normalized_rmse']:.3f} using {short['best_method']}.
- {long['duration_s']:g} s gaps: best median normalized RMSE {long['median_normalized_rmse']:.3f} using {long['best_method']}.

## Learned-method checks

- Embedding effective rank: {summary['embedding_effective_rank']:.2f} of 8.
- Autoencoder/PCA test-MSE ratio: {summary['autoencoder_to_pca_test_mse_ratio']:.2f}.
- GRU/persistence test-MSE ratio: {summary['gru_to_persistence_test_mse_ratio']:.2f}.
- Strongest recording-score correlation: {'–'.join(correlation['methods'])}, rho={correlation['rho']:.2f}.

## Unsupervised structure

- Selected clusters: {summary['selected_cluster_count']}
- Training silhouette: {summary['cluster_silhouette']:.3f}
- Seed adjusted Rand index: {summary['cluster_seed_ari']:.3f}

These clusters are operating regimes. No independent evidence maps them to laps or track sectors.
"""


def write_reports(repo_root, output_dir=None):
    repo_root = Path(repo_root)
    output_dir = repo_root / "reports" if output_dir is None else Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_report_bundle(repo_root)
    documents = {
        "README.md": _render_index(bundle),
        "findings.md": _render_findings(bundle),
        "profile-2026.md": _render_profile(bundle["cohorts"]["drive_day_2026"]),
        "profile-2023.md": _render_profile(bundle["cohorts"]["aim_2023"]),
        "findings.json": json.dumps(bundle, indent=2, sort_keys=True) + "\n",
    }
    written = []
    for name, content in documents.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(path)
    for name, source in FIGURE_SOURCES.items():
        source_path = repo_root / "artifacts" / source
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        target = figure_dir / name
        shutil.copyfile(source_path, target)
        written.append(target)
    return written
