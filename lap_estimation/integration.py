import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from .comparison import METHOD_LABELS, SCORE_DIRECTIONS, WINDOW_METHODS, build_window_score_matrix


def interpolate_at_times(observed_time, observed_values, target_time, method="linear"):
    observed_time = np.asarray(observed_time, dtype=float)
    observed_values = np.asarray(observed_values, dtype=float)
    target_time = np.asarray(target_time, dtype=float)
    finite = np.isfinite(observed_time) & np.isfinite(observed_values)
    observed_time = observed_time[finite]
    observed_values = observed_values[finite]
    order = np.argsort(observed_time)
    observed_time = observed_time[order]
    observed_values = observed_values[order]
    unique = np.concatenate(([True], np.diff(observed_time) > 0))
    observed_time = observed_time[unique]
    observed_values = observed_values[unique]
    if len(observed_time) < 2:
        raise ValueError("Interpolation requires two finite observed samples")
    if np.any(target_time < observed_time[0]) or np.any(target_time > observed_time[-1]):
        raise ValueError("Interpolation would require extrapolation")
    if method == "linear":
        return np.interp(target_time, observed_time, observed_values)
    if method == "pchip":
        return PchipInterpolator(observed_time, observed_values, extrapolate=False)(target_time)
    raise ValueError(f"Unknown interpolation method: {method}")


def benchmark_interpolation(frame, channels, methods=("linear", "pchip"), gap_samples=(1, 5)):
    time = pd.to_numeric(frame["elapsed_s"], errors="coerce").to_numpy(dtype=float)
    rows = []
    for channel in channels:
        values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(time) & np.isfinite(values)
        for gap in gap_samples:
            heldout = np.zeros(len(frame), dtype=bool)
            stride = max(5 * gap, gap + 2)
            for start in range(gap + 1, len(frame) - gap - 1, stride):
                heldout[start : start + gap] = True
            heldout &= finite
            observed = finite & ~heldout
            target_time = time[heldout]
            target_values = values[heldout]
            if observed.sum() < 2 or not len(target_time):
                continue
            lower_quartile, upper_quartile = np.nanquantile(values[observed], [0.25, 0.75])
            scale = upper_quartile - lower_quartile
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = np.nanstd(values[observed])
            if not np.isfinite(scale) or scale <= 1e-12:
                continue
            for method in methods:
                reconstructed = interpolate_at_times(time[observed], values[observed], target_time, method)
                error = reconstructed - target_values
                correlation = np.corrcoef(reconstructed, target_values)[0, 1] if len(target_values) > 1 else np.nan
                rows.append(
                    {
                        "channel": channel,
                        "method": method,
                        "gap_samples": int(gap),
                        "heldout_count": int(len(target_values)),
                        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / scale),
                        "normalized_mae": float(np.mean(np.abs(error)) / scale),
                        "correlation": float(correlation),
                    }
                )
    return pd.DataFrame(rows)


def _numeric(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)


def build_integrated_features(frame, dataset_key):
    time = pd.to_numeric(frame["elapsed_s"], errors="coerce").to_numpy(dtype=float)
    result = pd.DataFrame({"elapsed_s": time})
    if dataset_key == "drive_day_2026":
        rpm = _numeric(frame, "Motor RPM [Rpm]")
        voltage = _numeric(frame, "MC Volts [V]")
        current = _numeric(frame, "DC Current [A]")
        result["shaft_revolutions"] = cumulative_trapezoid(rpm / 60.0, time, initial=0.0)
        result["electrical_energy_wh"] = cumulative_trapezoid(voltage * current, time, initial=0.0) / 3600.0
        result["charge_throughput_ah"] = cumulative_trapezoid(current, time, initial=0.0) / 3600.0
        return result
    if dataset_key == "aim_2023":
        yaw = _numeric(frame, "YawRate")
        longitudinal = _numeric(frame, "InlineAcc") * 9.80665
        lateral = _numeric(frame, "LateralAcc") * 9.80665
        result["relative_heading_deg"] = cumulative_trapezoid(yaw, time, initial=0.0)
        result["centered_heading_deg"] = cumulative_trapezoid(yaw - np.median(yaw), time, initial=0.0)
        result["longitudinal_delta_v_m_s"] = cumulative_trapezoid(longitudinal, time, initial=0.0)
        result["lateral_delta_v_m_s"] = cumulative_trapezoid(lateral, time, initial=0.0)
        return result
    raise KeyError(f"Unknown dataset: {dataset_key}")


def _top_recording_scores(path, value_columns):
    frame = pd.read_csv(path)
    top = frame.sort_values("score").groupby("recording_id", as_index=False).tail(1)
    return top[["recording_id", *value_columns]].reset_index(drop=True)


def build_recurrence_fusion(artifact_root):
    artifact_root = Path(artifact_root)
    signal = _top_recording_scores(
        artifact_root / "01_signal_processing" / "predictions.csv",
        ["score", "predicted_period_s"],
    ).rename(columns={"score": "signal_score", "predicted_period_s": "signal_period_s"})
    dtw = pd.read_csv(artifact_root / "02_dtw" / "predictions.csv")
    dtw = dtw.groupby("recording_id", as_index=False).agg(
        dtw_margin=("score", "median"),
        dtw_pair_count=("score", "size"),
    )
    correlation = _top_recording_scores(
        artifact_root / "03_correlation" / "predictions.csv",
        ["score", "lag_s"],
    ).rename(columns={"score": "correlation_score", "lag_s": "correlation_period_s"})
    result = signal.merge(dtw, on="recording_id", how="outer").merge(correlation, on="recording_id", how="outer")
    valid_periods = result["signal_period_s"].gt(0) & result["correlation_period_s"].gt(0)
    result["period_agreement"] = np.nan
    period_ratio = result.loc[valid_periods, "signal_period_s"] / result.loc[valid_periods, "correlation_period_s"]
    result.loc[valid_periods, "period_agreement"] = np.exp(-np.abs(np.log(period_ratio)))
    for source, target in [
        ("signal_score", "signal_support"),
        ("dtw_margin", "dtw_support"),
        ("correlation_score", "correlation_support"),
        ("period_agreement", "period_support"),
    ]:
        result[target] = result[source].rank(pct=True)
    signal_columns = ["signal_support", "correlation_support", "period_support"]
    result["signal_family_support"] = result[signal_columns].median(axis=1, skipna=True)
    result["shape_family_support"] = result["dtw_support"]
    family_columns = ["signal_family_support", "shape_family_support"]
    result["fused_support"] = result[family_columns].median(axis=1, skipna=True)
    result["family_dispersion"] = result[family_columns].std(axis=1, ddof=0, skipna=True)
    method_columns = ["signal_score", "dtw_margin", "correlation_score"]
    result["method_count"] = result[method_columns].notna().sum(axis=1)
    result["evidence_status"] = "weak"
    result.loc[result["method_count"] < 2, "evidence_status"] = "insufficient_evidence"
    review = result["method_count"].eq(3) & result["fused_support"].ge(2.0 / 3.0) & result["period_agreement"].ge(0.5)
    result.loc[review, "evidence_status"] = "review"
    result["dependency_note"] = "dtw_uses_signal_period_teacher"
    return result.sort_values("fused_support", ascending=False).reset_index(drop=True)


def _training_percentile(values, training_values):
    training_values = np.sort(np.asarray(training_values, dtype=float))
    if not len(training_values):
        raise ValueError("Training scores required for consensus normalization")
    return np.searchsorted(training_values, np.asarray(values, dtype=float), side="right") / len(training_values)


def build_window_consensus(artifact_root, methods=WINDOW_METHODS):
    artifact_root = Path(artifact_root)
    matrix = build_window_score_matrix(artifact_root, methods=methods)
    manifest = json.loads((artifact_root / methods[0] / "manifest.json").read_text(encoding="utf-8"))
    split_assignments = manifest["split_assignments"]
    matrix["split"] = matrix["recording_id"].map(split_assignments)
    support_columns = []
    for method in methods:
        label = METHOD_LABELS[method]
        oriented = matrix[label] * SCORE_DIRECTIONS[label]
        training_values = oriented[matrix["split"].eq("train")]
        support_column = f"{label}_support"
        matrix[support_column] = _training_percentile(oriented, training_values)
        support_columns.append(support_column)
    matrix["consensus_support"] = matrix[support_columns].median(axis=1)
    matrix["support_iqr"] = matrix[support_columns].quantile(0.75, axis=1) - matrix[support_columns].quantile(0.25, axis=1)
    matrix["method_count"] = matrix[support_columns].notna().sum(axis=1)
    return matrix
