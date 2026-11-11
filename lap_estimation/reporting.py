import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .config import RANDOM_SEED


SIGNAL_MIN_LAG_S = 15.0
SIGNAL_MAX_LAG_S = 120.0
CEILING_FRACTION = 0.8
PERIOD_TOLERANCE = 0.10
BOOTSTRAP_DRAWS = 2000

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
    "2023-course-runs.png": "aim_2023/11_course_runs/course_runs.png",
    "physics-null-calibration.png": "physics_null_calibration.png",
}


def _metric_field(metrics, method, metric, split, field):
    selected = metrics.loc[
        metrics["method"].eq(method) & metrics["metric"].eq(metric) & metrics["split"].eq(split),
        field,
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one metric row for {method}/{metric}/{split}, found {len(selected)}")
    return selected.iloc[0]


def _metric(metrics, method, metric, split):
    return float(_metric_field(metrics, method, metric, split, "value"))


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


def _metric_count(metrics, method, metric, split):
    return int(_metric_field(metrics, method, metric, split, "sample_count"))


def _signal_ceiling_summary(artifact_root, profile):
    candidates = pd.read_csv(artifact_root / "01_signal_processing" / "periodicity_candidates.csv")
    top = candidates.loc[candidates["rank"].eq(1)].merge(profile[["recording_id", "duration_s"]], on="recording_id")
    ceiling = np.minimum(SIGNAL_MAX_LAG_S, top["duration_s"] / 2.0)
    near = top["lag_s"] >= CEILING_FRACTION * ceiling
    return {
        "recordings": int(len(top)),
        "near_ceiling": int(near.sum()),
        "share": float(near.mean()) if len(top) else float("nan"),
        "minimum_cycle_count": float(top["cycle_count"].min()) if len(top) else float("nan"),
    }


def _period_agreement_summary(fusion, profile):
    controls = profile.set_index("recording_id")["control_candidate"]
    periods = fusion.loc[~fusion["recording_id"].map(controls).astype(bool), ["signal_period_s", "correlation_period_s"]]
    values = periods.to_numpy(dtype=float)
    candidates = np.sort(values[np.isfinite(values)])

    def matches(target):
        return (np.abs(values / target - 1.0) <= PERIOD_TOLERANCE).any(axis=1)

    if not len(candidates):
        return {"recordings": int(len(periods)), "modal_period_s": float("nan"), "matching_recordings": 0, "half_period_recordings": 0, "near_search_floor": False}
    counts = [int(matches(candidate).sum()) for candidate in candidates]
    modal_candidate = candidates[int(np.argmax(counts))]
    matched = matches(modal_candidate)
    matched_values = values[matched]
    within = np.abs(matched_values / modal_candidate - 1.0) <= PERIOD_TOLERANCE
    modal_period = float(np.median(matched_values[within]))
    half = matches(modal_period / 2.0) & ~matched
    return {
        "recordings": int(len(periods)),
        "modal_period_s": modal_period,
        "matching_recordings": int(matched.sum()),
        "half_period_recordings": int(half.sum()),
        "near_search_floor": bool(modal_period <= SIGNAL_MIN_LAG_S * (1.0 + 2.0 * PERIOD_TOLERANCE)),
    }


def _dtw_test_interval(artifact_root):
    method_root = artifact_root / "02_dtw"
    pairs = pd.read_csv(method_root / "predictions.csv")
    splits = json.loads((method_root / "manifest.json").read_text(encoding="utf-8"))["split_assignments"]
    pairs = pairs.loc[pairs["recording_id"].map(splits).eq("test")]
    wins = (pairs["matched_distance"] < pairs["control_distance"]).groupby(pairs["recording_id"]).agg(["sum", "size"])
    totals = wins.to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    draws = totals[rng.integers(0, len(totals), size=(BOOTSTRAP_DRAWS, len(totals)))].sum(axis=1)
    low, high = np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975])
    return {
        "rate": float(totals[:, 0].sum() / totals[:, 1].sum()),
        "low": float(low),
        "high": float(high),
        "pairs": int(totals[:, 1].sum()),
        "recordings": int(len(totals)),
        "includes_chance": bool(low <= 0.5 <= high),
    }


def _cluster_composition(artifact_root, profile):
    windows = pd.read_csv(artifact_root / "08_unsupervised_learning" / "predictions.csv")
    largest = windows["cluster"].value_counts().idxmax()
    in_largest = windows["cluster"].eq(largest)
    controls = windows["recording_id"].map(profile.set_index("recording_id")["control_candidate"]).astype(bool)
    return {
        "largest_cluster_window_share": float(in_largest.mean()),
        "windows": int(len(windows)),
        "recordings": int(windows["recording_id"].nunique()),
        "minority_cluster_recordings": int(windows.loc[~in_largest, "recording_id"].nunique()),
        "largest_cluster_control_windows": int((in_largest & controls).sum()),
        "largest_cluster_non_control_windows": int((in_largest & ~controls).sum()),
    }


def _physics_summary(artifact_root):
    method_root = artifact_root / "10_physics_laps"
    predictions = pd.read_csv(method_root / "predictions.csv")
    metrics = pd.read_csv(method_root / "metrics.csv")
    config = json.loads((method_root / "manifest.json").read_text(encoding="utf-8"))["config"]
    alpha = float(config["alpha"])
    driving = predictions.loc[~predictions["control_candidate"].astype(bool)]
    domains = {}
    for domain in sorted(driving["domain"].unique(), key=lambda name: name != config["primary_domain"]):
        rows = driving.loc[driving["domain"].eq(domain)]
        benchmark = metrics.loc[metrics["task"].eq(f"{domain}_domain_recurrence")].set_index("metric")
        significant = int((rows["p_value"] < alpha).sum())
        false_positive_rate = float(benchmark.loc["synthetic_false_positive_rate", "value"])
        domains[domain] = {
            "driving_recordings": int(len(rows)),
            "significant": significant,
            "excess_p_value": float(binomtest(significant, len(rows), false_positive_rate, alternative="greater").pvalue),
            "fdr_significant": int((rows["q_value"] < 2 * alpha).sum()),
            "at_search_ceiling": int(rows["at_search_ceiling"].sum()),
            "minimum_p_value": float(rows["p_value"].min()),
            "synthetic_detection_rate": float(benchmark.loc["synthetic_detection_rate", "value"]),
            "synthetic_false_positive_rate": false_positive_rate,
            "synthetic_trials": int(benchmark.loc["synthetic_false_positive_rate", "sample_count"]),
        }
    summary = {
        "primary_domain": str(config["primary_domain"]),
        "alpha": alpha,
        "surrogate_count": int(config["surrogate_count"]),
        "domains": domains,
    }
    return summary


def _course_summary(artifact_root):
    method_root = artifact_root / "11_course_runs"
    if not method_root.exists():
        return None
    metrics = pd.read_csv(method_root / "metrics.csv").set_index("metric")
    counts = {"run_count": "runs", "recordings_with_runs": "recordings", "runs_matching_reference": "runs_matching_reference"}
    values = {
        "median_pair_correlation": "median_pair_correlation",
        "minimum_pair_correlation": "minimum_pair_correlation",
        "null_q95_correlation": "null_q95_correlation",
        "null_maximum_correlation": "null_maximum_correlation",
        "pair_significant_rate": "pair_significant_rate",
        "median_aligned_lap_time_s": "median_lap_time_s",
        "minimum_aligned_lap_time_s": "minimum_lap_time_s",
        "maximum_aligned_lap_time_s": "maximum_lap_time_s",
    }
    summary = {key: int(metrics.loc[metric, "value"]) for metric, key in counts.items()}
    summary.update({key: float(metrics.loc[metric, "value"]) for metric, key in values.items()})
    summary["pairs"] = int(metrics.loc["median_pair_correlation", "sample_count"])
    summary["null_draws"] = int(metrics.loc["null_q95_correlation", "sample_count"])
    return summary


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
        "autoencoder_effective_rank": _effective_rank(artifact_root / "05_autoencoders" / "embeddings.npy"),
        "contrastive_effective_rank": _effective_rank(artifact_root / "06_contrastive_learning" / "embeddings.npy"),
        "recurrence_eligible_recordings": int((profile["duration_s"] / 2.0 > SIGNAL_MIN_LAG_S).sum()),
        "signal_non_control_count": _metric_count(
            metrics, "01_signal_processing", "non_control_review_candidate_rate", "all"
        ),
        "signal_control_flag_rate": _metric(
            metrics, "01_signal_processing", "control_candidate_detection_rate", "all"
        ),
        "signal_control_count": _metric_count(
            metrics, "01_signal_processing", "control_candidate_detection_rate", "all"
        ),
        "signal_ceiling": _signal_ceiling_summary(artifact_root, profile),
        "period_agreement": _period_agreement_summary(fusion, profile),
        "dtw_test_interval": _dtw_test_interval(artifact_root),
        "cluster_composition": _cluster_composition(artifact_root, profile),
        "physics": _physics_summary(artifact_root),
        "course_runs": _course_summary(artifact_root),
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


def _interval_text(interval):
    chance = "includes" if interval["includes_chance"] else "excludes"
    return (
        f"{interval['rate']:.1%} (95% recording-level bootstrap {interval['low']:.0%}–{interval['high']:.0%}; "
        f"{interval['pairs']} pairs from {interval['recordings']} recordings; {chance} 50% chance)"
    )


def _period_text(agreement):
    text = (
        f"{agreement['matching_recordings']} of {agreement['recordings']} non-control recordings carry a signal or correlation period "
        f"within {PERIOD_TOLERANCE:.0%} of {agreement['modal_period_s']:.1f} s"
    )
    if agreement["half_period_recordings"]:
        text += f"; {agreement['half_period_recordings']} more sit near half that period, a likely harmonic"
    if agreement["near_search_floor"]:
        text += f"; that period sits at the {SIGNAL_MIN_LAG_S:g} s search floor, so it is a likely artifact"
    return text


def _control_text(summary):
    return (
        f"{summary['signal_control_flag_rate']:.1%} of {summary['signal_control_count']} control recordings versus "
        f"{summary['signal_review_candidate_rate']:.1%} of {summary['signal_non_control_count']} non-control recordings"
    )


def _ceiling_text(summary):
    ceiling = summary["signal_ceiling"]
    return f"{ceiling['near_ceiling']} of {ceiling['recordings']} ({ceiling['share']:.0%})"


def _cluster_text(summary):
    composition = summary["cluster_composition"]
    return (
        f"largest cluster holds {composition['largest_cluster_window_share']:.1%} of {composition['windows']} windows, "
        f"minority clusters draw on {composition['minority_cluster_recordings']} of {composition['recordings']} recordings, "
        f"and the largest cluster mixes {composition['largest_cluster_control_windows']} control windows with "
        f"{composition['largest_cluster_non_control_windows']} non-control windows"
    )


def _rank_text(summary):
    return (
        f"masked {summary['embedding_effective_rank']:.2f}, contrastive {summary['contrastive_effective_rank']:.2f}, "
        f"autoencoder {summary['autoencoder_effective_rank']:.2f} of 8"
    )


def _physics_domain_texts(summary):
    physics = summary["physics"]
    return [
        f"{domain.capitalize()} domain: {result['significant']} of {result['driving_recordings']} driving recordings reach p < {physics['alpha']:g} "
        f"({result['fdr_significant']} after false-discovery control, {result['at_search_ceiling']} at the search ceiling); "
        f"benchmark recovers {result['synthetic_detection_rate']:.0%} of tiled laps and flags {result['synthetic_false_positive_rate']:.0%} "
        f"of unrelated segments over {result['synthetic_trials']} trials"
        for domain, result in physics["domains"].items()
    ]


def _physics_text(summary):
    return ". ".join(_physics_domain_texts(summary))


def _course_text(summary):
    course = summary.get("course_runs")
    if not course:
        return ""
    return (
        f"{course['runs']} driving runs segmented from {course['recordings']} recordings trace one course: "
        f"pairwise yaw-rate correlation after pace and offset alignment has median {course['median_pair_correlation']:.2f} "
        f"and minimum {course['minimum_pair_correlation']:.2f}, against a shuffled-run null with 95th percentile "
        f"{course['null_q95_correlation']:.2f} and maximum {course['null_maximum_correlation']:.2f} over {course['null_draws']} draws; "
        f"{course['pair_significant_rate']:.0%} of {course['pairs']} pairs beat the null. "
        f"Aligned lap times span {course['minimum_lap_time_s']:.1f}-{course['maximum_lap_time_s']:.1f} s, median {course['median_lap_time_s']:.1f} s"
    )


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


def _sources(relative_path):
    return [f"artifacts/{cohort_key}/{relative_path}" for cohort_key in COHORT_LABELS]


def build_report_bundle(repo_root):
    cohorts = {key: build_cohort_summary(repo_root, key) for key in COHORT_LABELS}
    y2026 = cohorts["drive_day_2026"]
    y2023 = cohorts["aim_2023"]
    review_total = sum(cohort["fusion_status_counts"].get("review", 0) for cohort in cohorts.values())
    chance_cohorts = [cohort["label"] for cohort in cohorts.values() if cohort["dtw_test_interval"]["includes_chance"]]
    excess_p_value = min(result["excess_p_value"] for cohort in cohorts.values() for result in cohort["physics"]["domains"].values())
    chance_note = (
        f"Held-out intervals include chance for: {', '.join(chance_cohorts)}."
        if chance_cohorts
        else "Held-out intervals exclude chance in both cohorts."
    )
    findings = [
        _finding(
            "recurrence-review-set",
            "Fusion ranks recordings for review; it does not detect recurrence",
            "caution",
            "both",
            f"Fusion marks {y2026['fusion_status_counts'].get('review', 0)} of {y2026['fusion_recordings']} evidence-covered 2026 recordings and {y2023['fusion_status_counts'].get('review', 0)} of {y2023['fusion_recordings']} evidence-covered 2023 recordings for review. Marks come from within-cohort percentile ranks cut at fused support >= 2/3, so roughly the top third is selected whatever the absolute evidence.",
            "Signal and correlation run the same autocorrelation on the same channels, and DTW borrows the signal period. Fusion is effectively one evidence source and a priority ranking, not a count of recordings with recurrence.",
            "Label in ranked order. Do not quote review counts as detections.",
            _sources("recurrence_fusion.csv"),
        ),
        _finding(
            "signal-control-separation",
            "Signal flag does not separate controls from driving",
            "negative",
            "both",
            f"Review flag fires on {_control_text(y2026)} in 2026, and on {_control_text(y2023)} in 2023. Minimum rank-1 cycle count is {y2026['signal_ceiling']['minimum_cycle_count']:.2f} and {y2023['signal_ceiling']['minimum_cycle_count']:.2f}: lag is capped at half the duration, so the `cycle_count >= 2` gate never rejects and the flag is correlation >= 0.25 alone.",
            "Stationary or low-dynamics recordings pass about as often as driving at these sample sizes. The threshold has no demonstrated discriminative power.",
            "Calibrate recurrence scores against a surrogate null and report per-recording p-values before using the flag.",
            _sources("01_signal_processing/metrics.csv"),
        ),
        _finding(
            "signal-search-ceiling",
            "Many candidate periods sit at the search ceiling",
            "caution",
            "both",
            f"Rank-1 signal lags at or above {CEILING_FRACTION:.0%} of the search ceiling (half the duration, capped at {SIGNAL_MAX_LAG_S:g} s): {_ceiling_text(y2026)} in 2026 and {_ceiling_text(y2023)} in 2023.",
            "A peak at the ceiling compares roughly two halves of one recording and is edge-prone. It should not be read as a lap period.",
            "Require peaks well inside the search range, or record sessions spanning several expected laps.",
            _sources("01_signal_processing/periodicity_candidates.csv"),
        ),
        _finding(
            "cross-recording-period",
            f"2023 recordings converge near {y2023['period_agreement']['modal_period_s']:.0f} s; 2026 recordings do not converge on a usable period",
            "caution",
            "both",
            f"2023: {_period_text(y2023['period_agreement'])}. 2026: {_period_text(y2026['period_agreement'])}.",
            "Separate recordings agreeing on one period would be stronger evidence than any per-recording score, but these periods carry no null calibration and locate no boundary. The calibrated physics test does not confirm them.",
            "Do not quote the shared 2023 period as a lap time. Treat half-period values as harmonics.",
            _sources("recurrence_fusion.csv"),
        ),
        _finding(
            "physics-null-calibration",
            "Calibrated physics detector finds no recurrence beyond its false-positive rate",
            "negative",
            "both",
            f"Block-shuffle null, {y2026['physics']['surrogate_count']} surrogates per recording. 2026 (motor speed): {_physics_text(y2026)}. 2023 (yaw rate): {_physics_text(y2023)}.",
            f"The detector recovers planted laps, but its null is optimistic: unrelated real segments pass well above the nominal rate. No significant share in real recordings is distinguishable from that benchmark false-positive rate (smallest one-sided binomial p = {excess_p_value:.2f}), so neither cohort shows lap recurrence inside a recording. Resampling 2026 motor speed onto distance did not expose laps that time-domain analysis missed.",
            "Read these p-values as upper bounds on evidence. Tighten the null until benchmark false positives match nominal before trusting any single detection.",
            _sources("10_physics_laps/predictions.csv"),
        ),
        _finding(
            "course-runs",
            "2023 runs repeat one course: laps recur across recordings, not inside them",
            "supported",
            "aim_2023",
            f"{_course_text(y2023)}.",
            "Each long 2023 recording holds one run of the same course, two in the longest, so the lap unit is the run. That is why no method found recurrence inside a recording. Lap times are relative to the reference run's span under one global pace factor; start and finish lines are not observed, and the null shares the optimism measured for the recurrence detector, though the margin here is wide.",
            "Use aligned runs as provisional lap labels. Replace the global pace factor with time-warped alignment for sector times, and log wheel speed or GPS to fix start and finish.",
            ["artifacts/aim_2023/11_course_runs/predictions.csv"],
        ),
        _finding(
            "dtw-mixed-generalization",
            "DTW held-out support is not distinguishable from chance",
            "caution",
            "both",
            f"Held-out match rates are {_interval_text(y2026['dtw_test_interval'])} for 2026 and {_interval_text(y2023['dtw_test_interval'])} for 2023.",
            f"{chance_note} Pairs are overlapping windows inside few recordings, and DTW uses signal-derived periods, so agreement cannot count as independent confirmation.",
            "Retest DTW against manually labeled lap pairs and unrelated hard negatives with leave-one-recording-out evaluation.",
            _sources("02_dtw/predictions.csv"),
        ),
        _finding(
            "interpolation-boundary",
            "Interpolation repairs short gaps only",
            "caution",
            "both",
            f"Best short-gap median normalized RMSE is {y2026['interpolation']['short_gap']['median_normalized_rmse']:.2f} for 2026 and {y2023['interpolation']['short_gap']['median_normalized_rmse']:.2f} for 2023; 0.5 s gaps rise to {y2026['interpolation']['long_gap']['median_normalized_rmse']:.2f} and {y2023['interpolation']['long_gap']['median_normalized_rmse']:.2f}.",
            "Interior interpolation can repair isolated samples but cannot create missing dynamics or timing truth.",
            "Use interpolation only below a validated gap limit and retain an imputation mask.",
            _sources("01_signal_processing/interpolation_benchmark.csv"),
        ),
        _finding(
            "embedding-collapse",
            "Masked-reconstruction encoder collapses by design",
            "negative",
            "both",
            f"Effective rank: {_rank_text(y2026)} for 2026; {_rank_text(y2023)} for 2023.",
            "The masked encoder ends in global average pooling and decodes through one linear layer, so a 10 s window cannot be rebuilt from the pooled code. Collapse is architectural, from one untuned configuration, and says nothing about the data. Contrastive and autoencoder codes do not collapse.",
            "Do not use these embeddings as pseudo-labels. Keep temporal resolution in the encoder and add hard negatives before retesting.",
            _sources("04_learned_embeddings/embeddings.npy"),
        ),
        _finding(
            "autoencoder-baseline",
            "Untuned autoencoder does not beat PCA",
            "negative",
            "both",
            f"Test MSE ratios versus PCA are {y2026['autoencoder_to_pca_test_mse_ratio']:.2f} for 2026 and {y2023['autoencoder_to_pca_test_mse_ratio']:.2f} for 2023; values above 1 are worse.",
            "One configuration (8 latent dimensions, 25 epochs, no tuning). Ratios carry no interval, so values near 1 are not distinguishable from PCA.",
            "Keep PCA as baseline; revisit autoencoders only with a task-linked validation target.",
            _sources("05_autoencoders/metrics.csv"),
        ),
        _finding(
            "sequence-baseline",
            "Untuned GRU does not beat persistence",
            "negative",
            "both",
            f"Test MSE ratios versus persistence are {y2026['gru_to_persistence_test_mse_ratio']:.2f} for 2026 and {y2023['gru_to_persistence_test_mse_ratio']:.2f} for 2023.",
            "One configuration: 16 hidden units, 20 epochs, no tuning or early stopping. A model that loses to copying the last sample is undertrained; this is not evidence that sequence models fail on this data.",
            "Do not promote GRU residuals to labels. Predict deltas, evaluate multi-second horizons, and tune on the validation split before concluding.",
            _sources("07_sequence_models/metrics.csv"),
        ),
        _finding(
            "stable-operating-regimes",
            "Cluster structure is dominated by one regime",
            "caution",
            "both",
            f"Training selects k=3 with silhouette {y2026['cluster_silhouette']:.3f} for 2026 and {y2023['cluster_silhouette']:.3f} for 2023; seed ARI is {y2026['cluster_seed_ari']:.3f} and {y2023['cluster_seed_ari']:.3f}. 2026: {_cluster_text(y2026)}. 2023: {_cluster_text(y2023)}.",
            "A dominant cluster that mixes controls with driving is not an operating regime, and high silhouette can come from a few outlying windows. Seed ARI tests initialization only, not resampling stability.",
            "Check stability with a recording-level bootstrap and drop redundant channels before using clusters as strata.",
            _sources("08_unsupervised_learning/predictions.csv"),
        ),
        _finding(
            "method-dependence",
            "Strong method correlations mostly expose shared behavior",
            "caution",
            "both",
            f"Strongest recording correlations are {'–'.join(y2026['strongest_recording_correlation']['methods'])} at rho={y2026['strongest_recording_correlation']['rho']:.2f} for 2026 and {'–'.join(y2023['strongest_recording_correlation']['methods'])} at rho={y2023['strongest_recording_correlation']['rho']:.2f} for 2023.",
            "Correlated reconstruction and sequence errors reflect shared signal difficulty, not independent lap agreement.",
            "Fuse evidence by method family and preserve disagreement instead of averaging all scores.",
            _sources("recording_method_spearman.csv"),
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
            "Methods 1-8 have no null calibration: scores are compared with fixed thresholds or within-cohort ranks, never with a surrogate distribution.",
            "The validation split is never used for model or threshold selection.",
            "Evaluation uses one fixed recording split; held-out sets hold two to five recordings per cohort.",
            "Clusters are not validated as regimes, lap phases, or track sectors.",
        ],
        "next_actions": [
            f"Label boundaries in fusion-ranked order, starting with the {review_total} top-ranked recordings.",
            "Record lap beacon or GPS truth during the next event.",
            "Replace the global pace factor with time-warped alignment and derive sector times from aligned 2023 runs.",
            "Tighten the physics null until benchmark false positives match nominal alpha.",
            "Evaluate boundary error, lap-count error, precision, and recall with leave-one-recording-out splits and recording-level intervals.",
            "Keep PCA and persistence as mandatory baselines.",
            "Retain interpolation masks and reject gaps beyond validated limits.",
        ],
    }


def _cohort_table(cohorts):
    lines = [
        "| Cohort | Recordings | Long enough for recurrence | Fusion coverage | Source paths | Logged minutes | Clock | Top-ranked for review |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for summary in cohorts.values():
        lines.append(
            f"| {summary['label']} | {summary['recordings']} | {summary['recurrence_eligible_recordings']} | {summary['fusion_recordings']} | {summary['source_paths']} | {summary['duration_min']:.1f} | {summary['clock_provenance']} | {summary['fusion_status_counts'].get('review', 0)} |"
        )
    return "\n".join(lines)


def _course_sentence(cohort):
    text = _course_text(cohort)
    return f"Across recordings: {text}. " if text else ""


def _lap_finding_line(cohort):
    return (
        f"- **{cohort['label']}:** fusion ranks {cohort['fusion_status_counts'].get('review', 0)} of {cohort['fusion_recordings']} "
        f"evidence-covered recordings for review, about a third by construction. Held-out DTW match rate is "
        f"{_interval_text(cohort['dtw_test_interval'])}. {_period_text(cohort['period_agreement']).capitalize()}. "
        f"Signal flag fires on {_control_text(cohort)}. Calibrated physics detector: {_physics_text(cohort)}. {_course_sentence(cohort)}No lap is verified against timing truth."
    )


def _render_index(bundle):
    cohorts = list(bundle["cohorts"].values())
    review_counts = [cohort["fusion_status_counts"].get("review", 0) for cohort in cohorts]
    review_count = sum(review_counts)
    recording_count = sum(cohort["recordings"] for cohort in cohorts)
    fusion_count = sum(cohort["fusion_recordings"] for cohort in cohorts)
    eligible_count = sum(cohort["recurrence_eligible_recordings"] for cohort in cohorts)
    total_minutes = sum(cohort["duration_min"] for cohort in cohorts)
    eligible_breakdown = ", ".join(f"{cohort['recurrence_eligible_recordings']} in {cohort['label']}" for cohort in cohorts)
    negative_count = sum(finding["status"] == "negative" for finding in bundle["findings"])
    result_label = "result comes" if negative_count == 1 else "results come"
    review_breakdown = " and ".join(
        f"{count} from {cohort['label']}"
        for cohort, count in zip(cohorts, review_counts)
    )
    control_text = "; ".join(f"{cohort['label']}: {_control_text(cohort)}" for cohort in cohorts)
    ranks = "; ".join(f"{cohort['label']}: {_rank_text(cohort)}" for cohort in cohorts)
    long_gaps = sorted({cohort["interpolation"]["long_gap"]["duration_s"] for cohort in cohorts})
    gap_text = " and ".join(f"{gap:g} s" for gap in long_gaps)
    cluster_shares = " and ".join(f"{cohort['cluster_composition']['largest_cluster_window_share']:.0%}" for cohort in cohorts)
    lap_lines = "\n".join(_lap_finding_line(cohort) for cohort in cohorts)
    return f"""# Lap-estimation reports

> {total_minutes:.1f} logged minutes across {recording_count} recordings; {eligible_count} are long enough for recurrence evidence ({eligible_breakdown}). No lap is verified. Recurrence scores are uncalibrated rankings, and {negative_count} negative {result_label} from untuned or uncalibrated configurations.

## Immediate impact

1. **Label {review_count} top-ranked recordings first.** Fusion ranks {fusion_count} evidence-covered recordings from {recording_count} total and selects about the top third by construction: {review_breakdown}. This is a priority order, not a detection count.
2. **Do not trust the signal review flag yet.** It fires on controls about as often as on driving. {control_text}.
3. **Read learned-model negatives as untuned.** Effective rank — {ranks}. Only the masked encoder collapses, and its pooling makes that architectural.
4. **Keep PCA and persistence.** Current autoencoder and GRU configurations lose to those baselines on held-out recordings.
5. **Repair isolated missing samples only.** {gap_text} gaps define current long-gap stress tests and remain unsuitable for inventing missing dynamics.
6. **Do not stratify on clusters yet.** One cluster holds {cluster_shares} of windows across cohorts and mixes controls with driving.

![2023 course runs](figures/2023-course-runs.png)

*Eight 2023 driving runs, one per recording except the longest, trace the same heading profile once pace is aligned. Laps recur across recordings.*

![Null calibration](figures/physics-null-calibration.png)

*Inside a recording, real driving is not distinguishable from unrelated segments, and the block-shuffle null runs optimistic: unrelated segments sit above the calibrated diagonal.*

## Evidence at a glance

{_cohort_table(bundle['cohorts'])}

## Lap findings

{lap_lines}

Bottom line: 2023 laps are whole runs that repeat across recordings; no recording in either cohort contains repeated laps, and no lap is checked against timing truth.

## 2026 EV drive day

![2026 fusion components](figures/2026-fusion-components.png)

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
.venv/Scripts/python.exe tools/build_physics_laps.py
.venv/Scripts/python.exe tools/build_reports.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
```
"""


def _render_findings(bundle):
    sections = [
        "# Findings\n",
        "Evidence is separated from interpretation and action. `supported` means supported by current internal diagnostics, not verified lap accuracy. `negative` describes the tested configuration only.\n",
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


def _physics_profile_lines(summary):
    lines = _physics_domain_texts(summary)
    if summary.get("course_runs"):
        lines.append(f"Course runs: {_course_text(summary)}")
    return "\n".join(f"- {line}." for line in lines)


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
- Long enough for recurrence evidence: {summary['recurrence_eligible_recordings']}

## Recurrence evidence

- Signal review flag: {_control_text(summary)}.
- Rank-1 signal lags at search ceiling: {_ceiling_text(summary)}.
- Cross-recording period: {_period_text(summary['period_agreement'])}.
- DTW period-shift match rate, all: {summary['dtw_match_rate_all']:.1%}
- DTW period-shift match rate, test: {_interval_text(summary['dtw_test_interval'])}
- Median absolute cross-correlation peak: {summary['cross_correlation_median_absolute_peak']:.3f}
- Fusion dispositions: {json.dumps(summary['fusion_status_counts'], sort_keys=True)}

## Calibrated physics detector

{_physics_profile_lines(summary)}

## Interpolation boundary

- {short['duration_s']:g} s gaps: best median normalized RMSE {short['median_normalized_rmse']:.3f} using {short['best_method']}.
- {long['duration_s']:g} s gaps: best median normalized RMSE {long['median_normalized_rmse']:.3f} using {long['best_method']}.

## Learned-method checks

- Embedding effective rank: {_rank_text(summary)}.
- Autoencoder/PCA test-MSE ratio: {summary['autoencoder_to_pca_test_mse_ratio']:.2f}.
- GRU/persistence test-MSE ratio: {summary['gru_to_persistence_test_mse_ratio']:.2f}.
- Strongest recording-score correlation: {'–'.join(correlation['methods'])}, rho={correlation['rho']:.2f}.

## Unsupervised structure

- Selected clusters: {summary['selected_cluster_count']}
- Training silhouette: {summary['cluster_silhouette']:.3f}
- Seed adjusted Rand index: {summary['cluster_seed_ari']:.3f}
- Composition: {_cluster_text(summary)}.

Clusters are not validated regimes. No independent evidence maps them to laps or track sectors.
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
