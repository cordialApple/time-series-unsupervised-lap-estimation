import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _load_predictions(predictions):
    if isinstance(predictions, pd.DataFrame):
        return predictions.copy(), None
    path = Path(predictions)
    return pd.read_csv(path), path


def _load_manifest(manifest):
    if isinstance(manifest, Mapping):
        return dict(manifest), None
    path = Path(manifest)
    return json.loads(path.read_text(encoding="utf-8")), path


def _attach_split(frame, manifest):
    result = frame.copy()
    if "split" not in result:
        result["split"] = result["recording_id"].map(manifest.get("split_assignments", {}))
    result["split"] = result["split"].fillna("unassigned")
    return result


def _find_motion_profile(predictions_path, manifest_path, output_dir):
    starts = [Path(output_dir)]
    if predictions_path is not None:
        starts.append(predictions_path.parent)
    if manifest_path is not None:
        starts.append(manifest_path.parent)
    candidates = []
    for start in starts:
        candidates.extend(
            [
                start / "data_profile.csv",
                start / "01_signal_processing" / "data_profile.csv",
                start.parent / "01_signal_processing" / "data_profile.csv",
                start.parent.parent / "01_signal_processing" / "data_profile.csv",
            ]
        )
    for path in candidates:
        if path.exists():
            profile = pd.read_csv(path)
            if {"recording_id", "motion_status"}.issubset(profile):
                return profile[["recording_id", "motion_status"]].drop_duplicates("recording_id")
    return pd.DataFrame(columns=["recording_id", "motion_status"])


def _attach_motion(frame, profile):
    result = frame.copy()
    if "motion_status" not in result:
        result = result.merge(profile, on="recording_id", how="left")
    result["motion_status"] = result["motion_status"].fillna("motion unavailable")
    return result


def _new_figure(figsize=(9, 6), panels=None):
    if panels is None:
        return plt.subplots(figsize=figsize, constrained_layout=True)
    return plt.subplots(*panels, figsize=figsize, constrained_layout=True)


def _finish_figure(fig, title, output_dir, name):
    fig.suptitle(f"{title} — unverified proxy; not lap accuracy", fontsize=13)
    fig.savefig(Path(output_dir) / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    return fig


def _short_recording_ids(frame):
    result = frame.copy()
    result["recording"] = result["recording_id"].astype(str).str.slice(0, 8)
    return result


def render_dtw_diagnostics(predictions, manifest, output_dir):
    frame, predictions_path = _load_predictions(predictions)
    metadata, manifest_path = _load_manifest(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"recording_id", "predicted_period_s", "matched_distance", "control_distance"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"DTW predictions missing columns: {sorted(missing)}")
    profile = _find_motion_profile(predictions_path, manifest_path, output_dir)
    frame = _attach_motion(_attach_split(frame, metadata), profile)
    frame = frame.dropna(subset=list(required - {"recording_id"})).copy()
    frame["margin"] = frame["control_distance"] - frame["matched_distance"]
    frame["matched_better"] = frame["margin"] > 0
    figures = {}

    fig, ax = _new_figure()
    sns.scatterplot(
        data=frame,
        x="matched_distance",
        y="control_distance",
        hue="split",
        style="motion_status",
        alpha=0.75,
        ax=ax,
    )
    limits = [frame[["matched_distance", "control_distance"]].min().min(), frame[["matched_distance", "control_distance"]].max().max()]
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1, label="equal distance")
    ax.set(xlabel="Period-shifted DTW distance", ylabel="Control-shift DTW distance")
    figures["dtw_distance_scatter"] = _finish_figure(fig, "DTW matched versus control distance", output_dir, "dtw_distance_scatter")

    fig, ax = _new_figure(figsize=(10, 6))
    sns.boxplot(data=frame, x="split", y="margin", hue="motion_status", showfliers=False, ax=ax)
    sns.stripplot(data=frame, x="split", y="margin", hue="motion_status", dodge=True, alpha=0.45, legend=False, ax=ax)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Recording split", ylabel="Control distance − matched distance")
    figures["dtw_margin_distribution"] = _finish_figure(fig, "DTW margin by split and motion status", output_dir, "dtw_margin_distribution")

    fig, ax = _new_figure()
    sns.scatterplot(
        data=frame,
        x="predicted_period_s",
        y="margin",
        hue="split",
        style="motion_status",
        alpha=0.75,
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Signal-derived period teacher (s)", ylabel="DTW margin; positive favors teacher shift")
    figures["dtw_period_margin"] = _finish_figure(fig, "Period teacher versus DTW margin", output_dir, "dtw_period_margin")

    rates = (
        frame.groupby(["recording_id", "split", "motion_status"], as_index=False, dropna=False)["matched_better"]
        .mean()
        .sort_values("matched_better")
    )
    rates = _short_recording_ids(rates)
    fig, ax = _new_figure(figsize=(10, max(5, len(rates) * 0.32)))
    sns.barplot(data=rates, x="matched_better", y="recording", hue="split", ax=ax)
    ax.set(xlim=(0, 1), xlabel="Share of pairs where period shift beats control", ylabel="Recording ID prefix")
    figures["dtw_recording_match_rate"] = _finish_figure(fig, "Per-recording DTW match rate", output_dir, "dtw_recording_match_rate")

    paired = frame.reset_index(drop=True).reset_index(names="pair_index").melt(
        id_vars=["pair_index", "split"],
        value_vars=["matched_distance", "control_distance"],
        var_name="comparison",
        value_name="distance",
    )
    labels = {"matched_distance": "period shift", "control_distance": "control shift"}
    paired["comparison"] = paired["comparison"].map(labels)
    fig, axes = _new_figure(figsize=(12, 5), panels=(1, 2))
    sns.lineplot(data=paired, x="comparison", y="distance", units="pair_index", estimator=None, alpha=0.18, color="0.35", ax=axes[0])
    sns.pointplot(data=paired, x="comparison", y="distance", estimator=np.median, errorbar=None, color="black", markers="D", ax=axes[0])
    axes[0].set(xlabel="Paired comparison", ylabel="DTW distance", title="Individual pairs and median")
    paired["comparison_split"] = paired["comparison"] + " · " + paired["split"]
    sns.ecdfplot(data=paired, x="distance", hue="comparison_split", ax=axes[1])
    axes[1].set(xlabel="DTW distance", ylabel="Empirical cumulative proportion", title="Distance ECDF")
    figures["dtw_paired_distance_ecdf"] = _finish_figure(fig, "Paired DTW distances and ECDF", output_dir, "dtw_paired_distance_ecdf")
    return figures


def _load_cross_correlation(predictions_path, manifest_path, output_dir):
    candidates = [Path(output_dir) / "cross_correlation.csv"]
    if predictions_path is not None:
        candidates.append(predictions_path.parent / "cross_correlation.csv")
    if manifest_path is not None:
        candidates.append(manifest_path.parent / "cross_correlation.csv")
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("cross_correlation.csv required for correlation diagnostics")


def render_correlation_diagnostics(predictions, manifest, output_dir):
    recurrence, predictions_path = _load_predictions(predictions)
    metadata, manifest_path = _load_manifest(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"recording_id", "channel", "lag_s", "correlation"}
    missing = required.difference(recurrence.columns)
    if missing:
        raise ValueError(f"Correlation predictions missing columns: {sorted(missing)}")
    recurrence = _attach_split(recurrence, metadata).dropna(subset=["lag_s", "correlation"]).copy()
    recurrence["absolute_correlation"] = recurrence["correlation"].abs()
    recurrence = _short_recording_ids(recurrence)
    cross = _load_cross_correlation(predictions_path, manifest_path, output_dir)
    cross_required = {"recording_id", "first_channel", "second_channel", "lag_samples", "correlation"}
    cross_missing = cross_required.difference(cross.columns)
    if cross_missing:
        raise ValueError(f"Cross-correlation data missing columns: {sorted(cross_missing)}")
    cross = _attach_split(cross, metadata).dropna(subset=["lag_samples", "correlation"]).copy()
    sample_rate_hz = float(metadata.get("data_provenance", {}).get("native_sample_rate_hz", 1.0))
    cross["lag_seconds"] = cross["lag_samples"] / sample_rate_hz
    cross["absolute_correlation"] = cross["correlation"].abs()
    cross["channel_pair"] = cross["first_channel"].astype(str) + " → " + cross["second_channel"].astype(str)
    cross = _short_recording_ids(cross)
    figures = {}

    channel_order = recurrence.groupby("channel")["lag_s"].median().sort_values().index
    fig, ax = _new_figure(figsize=(11, max(5, len(channel_order) * 0.5)))
    sns.boxplot(data=recurrence, x="lag_s", y="channel", order=channel_order, hue="split", showfliers=False, ax=ax)
    sns.stripplot(data=recurrence, x="lag_s", y="channel", order=channel_order, hue="split", dodge=True, alpha=0.45, legend=False, ax=ax)
    ax.set(xlabel="Peak recurrence lag (s)", ylabel="Sensor channel")
    figures["correlation_period_by_channel"] = _finish_figure(fig, "Recurrence period by channel", output_dir, "correlation_period_by_channel")

    fig, ax = _new_figure()
    sns.scatterplot(
        data=cross,
        x="lag_seconds",
        y="absolute_correlation",
        hue="split",
        style="channel_pair",
        alpha=0.75,
        ax=ax,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Signed peak cross-correlation lag (seconds)", ylabel="Absolute peak correlation")
    figures["correlation_lag_strength"] = _finish_figure(fig, "Cross-correlation lag versus strength", output_dir, "correlation_lag_strength")

    recurrence_matrix = recurrence.pivot_table(index="recording", columns="channel", values="lag_s", aggfunc="median")
    fig, ax = _new_figure(figsize=(max(8, recurrence_matrix.shape[1] * 1.3), max(5, recurrence_matrix.shape[0] * 0.45)))
    sns.heatmap(recurrence_matrix, cmap="viridis", linewidths=0.4, cbar_kws={"label": "Peak lag (s)"}, ax=ax)
    ax.set(xlabel="Sensor channel", ylabel="Recording ID prefix")
    ax.tick_params(axis="x", rotation=35)
    figures["correlation_recording_channel_heatmap"] = _finish_figure(fig, "Recording-channel recurrence map", output_dir, "correlation_recording_channel_heatmap")

    pair_matrix = cross.pivot_table(index="first_channel", columns="second_channel", values="correlation", aggfunc="median")
    fig, ax = _new_figure(figsize=(max(7, pair_matrix.shape[1] * 1.2), max(5, pair_matrix.shape[0] * 0.8)))
    sns.heatmap(pair_matrix, cmap="vlag", center=0, vmin=-1, vmax=1, annot=pair_matrix.size <= 64, fmt=".2f", cbar_kws={"label": "Median signed peak correlation"}, ax=ax)
    ax.set(xlabel="Second channel", ylabel="First channel")
    ax.tick_params(axis="x", rotation=35)
    figures["correlation_pair_heatmap"] = _finish_figure(fig, "Cross-correlation channel-pair map", output_dir, "correlation_pair_heatmap")

    fig, ax = _new_figure()
    sns.histplot(data=cross, x="lag_seconds", hue="split", element="step", fill=False, stat="density", common_norm=False, ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Signed peak lag (seconds); positive means second follows first", ylabel="Density")
    figures["correlation_signed_lag_distribution"] = _finish_figure(fig, "Signed sensor-lag distribution", output_dir, "correlation_signed_lag_distribution")
    return figures
