import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
DATASET_KEYS = ("drive_day_2026", "aim_2023")


METHODS = {
    "02_dtw": {
        "title": "Dynamic Time Warping",
        "question": "Do time-warped windows recur at signal-derived candidate periods?",
        "method": "Compare period-separated windows against non-period controls using one-second Sakoe-Chiba bands resolved at native sample rate.",
        "reading": "Positive held-out control margin supports teacher-period matching only. It does not validate laps.",
        "options": {},
        "renderer_module": "plotting_classical",
        "renderer": "render_dtw_diagnostics",
    },
    "03_correlation": {
        "title": "Correlation and cross-correlation",
        "question": "Which channels show consistent recurrence periods and sensor lags?",
        "method": "Compute overlap-normalized channel autocorrelation and pairwise cross-correlation within bounded lag ranges.",
        "reading": "Low lag disagreement means channel consistency. Coupled sensors are not independent lap evidence.",
        "options": {},
        "renderer_module": "plotting_classical",
        "renderer": "render_correlation_diagnostics",
    },
    "04_learned_embeddings": {
        "title": "Learned embeddings",
        "question": "Does masked reconstruction learn useful recurrence neighborhoods?",
        "method": "Train compact temporal encoder on training recordings only, then inspect held-out reconstruction and neighbors.",
        "reading": "Near-unity similarity may reflect overlapping windows or collapse. Reconstruction is not lap accuracy.",
        "options": {"epochs": 20},
        "renderer_module": "plotting_learned",
        "renderer": "render_embedding_diagnostics",
    },
    "05_autoencoders": {
        "title": "Autoencoders",
        "question": "Does nonlinear reconstruction beat equal-dimension PCA?",
        "method": "Train compact autoencoder on training windows. Compare held-out error with PCA fit on same split.",
        "reading": "Lower error than PCA supports representation efficiency only. Anomalies are not boundaries.",
        "options": {"epochs": 25},
        "renderer_module": "plotting_learned",
        "renderer": "render_autoencoder_diagnostics",
    },
    "06_contrastive_learning": {
        "title": "Contrastive learning",
        "question": "Do conservative signal augmentations improve recurrence retrieval?",
        "method": "Train compact encoder with mild amplitude, noise, and masking views on training recordings.",
        "reading": "View consistency measures augmentation invariance, not lap detection.",
        "options": {"epochs": 20},
        "renderer_module": "plotting_learned",
        "renderer": "render_contrastive_diagnostics",
    },
    "07_sequence_models": {
        "title": "Sequence models",
        "question": "Does compact GRU prediction beat persistence on held-out recordings?",
        "method": "Train next-sample GRU on training windows and compare against unchanged-value persistence.",
        "reading": "GRU must beat persistence before recurrent state deserves further use.",
        "options": {"epochs": 20},
        "renderer_module": "plotting_learned",
        "renderer": "render_sequence_diagnostics",
    },
    "08_unsupervised_learning": {
        "title": "Unsupervised learning",
        "question": "Which stable operating regimes appear without recurrence-derived labels?",
        "method": "Fit robust window features and K-means on training recordings. Select cluster count by internal structure and seed stability.",
        "reading": "Stable clusters describe regimes until independent lap labels exist.",
        "options": {},
        "renderer_module": "plotting_fusion",
        "renderer": "render_unsupervised_diagnostics",
    },
}


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def notebook(cells):
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def setup_cell(dataset_key, extra_imports=""):
    return code(
        f"""
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "lap_estimation").exists():
    ROOT = ROOT.parent
if not (ROOT / "lap_estimation").exists():
    raise FileNotFoundError("Repository root not found")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
{extra_imports}
from lap_estimation.artifacts import load_experiment_metrics
from lap_estimation.config import RANDOM_SEED
from lap_estimation.data import prepare_recording_manifest
from lap_estimation.datasets import resolve_dataset
from lap_estimation.runners import run_experiment

DATASET_KEY = "{dataset_key}"
dataset = resolve_dataset(DATASET_KEY, ROOT)
ARTIFACT_ROOT = dataset.artifact_root
np.random.seed(RANDOM_SEED)
sns.set_theme(style="ticks", palette="colorblind", context="notebook")
manifest = prepare_recording_manifest(dataset)
manifest[["recording_id", "duration_s", "motion_status", "control_candidate", "clock_provenance", "split"]]
"""
    )


def signal_notebook(dataset_key):
    clock_note = (
        "Source timestamps retained; elapsed time comes from recorded timestamp column."
        if dataset_key == "drive_day_2026"
        else "Source Time field is blank. Adapter generates elapsed_s = sample_index / 20 in memory after metadata and row-count validation. Raw CSV stays unchanged."
    )
    sensor_note = (
        "Motor and electrical channels drive recurrence analysis. Requested steering, wheel, and oil signals are absent."
        if dataset_key == "drive_day_2026"
        else "Inertial channels drive recurrence analysis. Wheel-speed channels are constant zero. VerticalAcc carries a large offset despite declared g units."
    )
    return notebook(
        [
            markdown(
                f"""
# 01 — Signal processing baseline

**Dataset:** `{dataset_key}`

**Question:** Does telemetry contain reviewable recurring periods?

{clock_note} {sensor_note} Recurrence candidates remain unverified.
"""
            ),
            setup_cell(
                dataset_key,
                "from lap_estimation.data import load_recording, select_informative_channels\nfrom lap_estimation.integration import benchmark_interpolation, build_integrated_features",
            ),
            code(
                """
overview = pd.Series(
    {
        "recordings": len(manifest),
        "source_paths": int(manifest["alias_count"].sum()),
        "duration_min": manifest["duration_s"].sum() / 60.0,
        "native_sample_rate_hz": dataset.native_sample_rate_hz,
        "control_candidates": int(manifest["control_candidate"].sum()),
        "clock_provenance": dataset.clock_provenance,
    },
    name="value",
)
overview.to_frame()
"""
            ),
            code(
                """
result = run_experiment("01_signal_processing", ROOT, dataset_key=DATASET_KEY)
profile = pd.read_csv(result.output_dir / "data_profile.csv")
coverage = pd.read_csv(result.output_dir / "sensor_coverage.csv")
candidates = pd.read_csv(result.output_dir / "periodicity_candidates.csv")
smoothing = pd.read_csv(result.output_dir / "smoothing_sensitivity.csv")
display(result.metrics)
display(result.predictions.sort_values("score", ascending=False))
display(profile.sort_values("duration_s", ascending=False))
"""
            ),
            markdown("## Recording inventory and sensor availability"),
            code(
                """
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(data=profile, x="duration_s", hue="motion_status", multiple="stack", ax=axes[0])
axes[0].set(title="Recording duration", xlabel="Duration (s)")
sns.boxplot(data=profile, x="split", y="motion_variation_score", hue="motion_status", ax=axes[1])
sns.stripplot(data=profile, x="split", y="motion_variation_score", color="black", alpha=0.55, ax=axes[1])
axes[1].set(title="Motion variation by split", ylabel="Median robust channel range")
fig.tight_layout()
"""
            ),
            code(
                """
coverage_matrix = coverage.assign(recording=coverage["recording_id"].str[:8], available=coverage["state"].map({"absent": 0, "constant": 0.5, "dynamic": 1.0})).pivot(index="channel", columns="recording", values="available")
fig, ax = plt.subplots(figsize=(max(9, coverage_matrix.shape[1] * 0.45), 5.5))
sns.heatmap(coverage_matrix, cmap="viridis", vmin=0, vmax=1, linewidths=0.2, ax=ax)
ax.set(title="Sensor coverage: 0 absent, 0.5 constant, 1 dynamic", xlabel="Recording SHA256 prefix", ylabel="Channel")
fig.tight_layout()
"""
            ),
            markdown("## Representative native sensor traces"),
            code(
                """
trace_rows = []
selected_records = manifest.loc[~manifest["control_candidate"]].nlargest(3, "duration_s")
for record in selected_records.itertuples(index=False):
    frame = load_recording(record.primary_path, dataset)
    channels = select_informative_channels(frame, dataset.analysis_channels)
    step = max(1, len(frame) // 1500)
    sampled = frame.iloc[::step]
    for channel in channels:
        values = pd.to_numeric(sampled[channel], errors="coerce")
        scale = values.quantile(0.75) - values.quantile(0.25)
        scaled = (values - values.median()) / (scale if scale > 1e-12 else 1.0)
        trace_rows.append(pd.DataFrame({"elapsed_s": sampled["elapsed_s"], "scaled_value": scaled, "channel": channel, "recording": record.recording_id[:8]}))
trace_data = pd.concat(trace_rows, ignore_index=True)
grid = sns.relplot(data=trace_data, x="elapsed_s", y="scaled_value", hue="channel", col="recording", col_wrap=1, kind="line", height=2.6, aspect=3.1, linewidth=0.8, facet_kws={"sharex": False})
grid.set_axis_labels("Elapsed time (s)", "Robust-scaled signal")
grid.figure.suptitle(f"{DATASET_KEY}: longest non-control recordings", y=1.01)
"""
            ),
            markdown("## Recurrence candidates and smoothing sensitivity"),
            code(
                """
top = candidates.loc[candidates["rank"] == 1].merge(manifest[["recording_id", "motion_status"]], on="recording_id", how="left")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sns.scatterplot(data=top, x="lag_s", y="correlation", hue="motion_status", size="cycle_count", sizes=(35, 180), ax=axes[0])
axes[0].axhline(0.25, color="black", linestyle="--", linewidth=1)
axes[0].set(title="Strongest period per recording", xlabel="Candidate period (s)", ylabel="Peak autocorrelation")
sns.histplot(data=candidates, x="lag_s", hue="control_candidate", multiple="stack", bins=20, ax=axes[1])
axes[1].set(title="All ranked candidate periods", xlabel="Candidate period (s)")
fig.tight_layout()
"""
            ),
            code(
                """
if len(smoothing):
    smoothing_plot = smoothing.assign(recording=smoothing["recording_id"].str[:8])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.lineplot(data=smoothing_plot, x="smooth_samples", y="lag_s", hue="recording", marker="o", legend=False, ax=axes[0])
    axes[0].set(title="Period stability under smoothing", xlabel="Smoothing samples", ylabel="Candidate period (s)")
    sns.lineplot(data=smoothing_plot, x="smooth_samples", y="correlation", hue="recording", marker="o", legend=False, ax=axes[1])
    axes[1].set(title="Peak-strength sensitivity", xlabel="Smoothing samples", ylabel="Autocorrelation")
    fig.tight_layout()
"""
            ),
            markdown(
                """
## Controlled interpolation sensitivity

Observed timestamps are already regular. Samples are deliberately hidden, then reconstructed from elapsed time. This tests gap sensitivity; it does not recover new timing information.
"""
            ),
            code(
                """
benchmark_rows = []
benchmark_records = pd.concat([manifest.nlargest(6, "duration_s"), manifest.loc[manifest["control_candidate"]].nlargest(2, "duration_s")]).drop_duplicates("recording_id")
long_gap_samples = max(2, round(dataset.native_sample_rate_hz * 0.5))
for record in benchmark_records.itertuples(index=False):
    frame = load_recording(record.primary_path, dataset)
    channels = select_informative_channels(frame, dataset.analysis_channels)
    benchmark = benchmark_interpolation(frame, channels, gap_samples=(1, long_gap_samples))
    benchmark["recording_id"] = record.recording_id
    benchmark["motion_status"] = record.motion_status
    benchmark_rows.append(benchmark)
interpolation = pd.concat(benchmark_rows, ignore_index=True)
interpolation["gap_duration_s"] = interpolation["gap_samples"] / dataset.native_sample_rate_hz
interpolation.to_csv(result.output_dir / "interpolation_benchmark.csv", index=False)
grid = sns.catplot(data=interpolation, x="channel", y="normalized_rmse", hue="method", col="gap_duration_s", kind="box", sharey=False, height=4.0, aspect=1.35)
grid.set_xticklabels(rotation=35, ha="right")
grid.set_axis_labels("Channel", "Held-out normalized RMSE")
grid.figure.suptitle(f"{DATASET_KEY}: interpolation sensitivity", y=1.03)
"""
            ),
            code(
                """
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.scatterplot(data=interpolation, x="normalized_rmse", y="correlation", hue="method", style="motion_status", size="gap_duration_s", sizes=(35, 150), ax=ax)
ax.set(title="Interpolation error versus retained shape", xlabel="Normalized RMSE", ylabel="Held-out correlation")
fig.tight_layout()
"""
            ),
            markdown("## Physically interpretable cumulative proxies"),
            code(
                """
example = manifest.loc[~manifest["control_candidate"]].nlargest(1, "duration_s").iloc[0]
example_frame = load_recording(example["primary_path"], dataset)
integrated = build_integrated_features(example_frame, DATASET_KEY)
integrated.to_csv(result.output_dir / "integrated_features_example.csv", index=False)
integrated_long = integrated.melt(id_vars="elapsed_s", var_name="feature", value_name="value")
grid = sns.relplot(data=integrated_long, x="elapsed_s", y="value", col="feature", col_wrap=2, kind="line", height=3.0, aspect=1.45, facet_kws={"sharey": False})
grid.set_axis_labels("Elapsed time (s)", "Cumulative proxy")
grid.figure.suptitle(f"{DATASET_KEY}: integrated sensor proxies, recording {example['recording_id'][:8]}", y=1.02)
"""
            ),
            code(
                """
run_manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
run_manifest["data_provenance"]
"""
            ),
            markdown(
                """
## Interpretation boundary

Positive recurrence means measured behavior repeats. It does not prove laps, locate a start line, identify a driver, or establish accuracy. Integrated RPM gives shaft revolutions, not distance. Electrical integration is an energy proxy. Yaw and acceleration integrals drift and provide only bounded relative-motion proxies.
"""
            ),
        ]
    )


def method_notebook(dataset_key, method, definition):
    renderer = definition["renderer"]
    renderer_import = f"from lap_estimation.{definition['renderer_module']} import {renderer}"
    return notebook(
        [
            markdown(
                f"""
# {method[:2]} — {definition['title']}

**Dataset:** `{dataset_key}`

**Question:** {definition['question']}

## Method

{definition['method']}
"""
            ),
            setup_cell(dataset_key, renderer_import),
            code(f"EXPERIMENT_OPTIONS = {definition['options']!r}\nEXPERIMENT_OPTIONS"),
            code(
                f"""
METHOD = "{method}"
result = run_experiment(METHOD, ROOT, dataset_key=DATASET_KEY, **EXPERIMENT_OPTIONS)
run_manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
display(result.metrics)
display(result.predictions.describe(include="all").T)
display(result.predictions.groupby("recording_id")["score"].agg(["count", "mean", "median", "std", "min", "max"]).sort_values("median", ascending=False))
diagnostic_figures = {renderer}(result.predictions, run_manifest, result.output_dir)
for figure in diagnostic_figures.values():
    display(figure)
    plt.close(figure)
"""
            ),
            code(
                """
pd.Series(
    {
        "status": run_manifest["status"],
        "runtime_s": run_manifest["runtime_s"],
        "clock_provenance": run_manifest["data_provenance"]["clock_provenance"],
        "sample_rate_hz": run_manifest["data_provenance"]["native_sample_rate_hz"],
        "channels": run_manifest["data_provenance"]["selected_channels"],
        "label_provenance": run_manifest["label_provenance"],
    },
    name="value",
).to_frame()
"""
            ),
            markdown(
                f"""
## Reading

{definition['reading']}

No lap precision, recall, count error, or duration error can be reported without independently reviewed boundaries.
"""
            ),
        ]
    )


def comparison_notebook(dataset_key):
    return notebook(
        [
            markdown(
                f"""
# 09 — Aggregate comparison report

**Dataset:** `{dataset_key}`

Reads saved artifacts only. Different tasks stay separate. No winner forced.
"""
            ),
            setup_cell(dataset_key, "from lap_estimation.comparison import METHOD_LABELS, RECORDING_METHODS, WINDOW_METHODS, build_recording_score_matrix, build_window_score_matrix, rank_method_scores\nfrom lap_estimation.integration import build_recurrence_fusion, build_window_consensus\nfrom lap_estimation.plotting_fusion import render_fusion_diagnostics"),
            code(
                """
METHODS = ["01_signal_processing", "02_dtw", "03_correlation", "04_learned_embeddings", "05_autoencoders", "06_contrastive_learning", "07_sequence_models", "08_unsupervised_learning"]
status_rows = []
for method in METHODS:
    path = ARTIFACT_ROOT / method / "manifest.json"
    if path.exists():
        run = json.loads(path.read_text(encoding="utf-8"))
        status_rows.append(
            {
                "method": method,
                "status": run.get("status"),
                "task": run.get("task"),
                "runtime_s": run.get("runtime_s"),
                "clock_provenance": run.get("data_provenance", {}).get("clock_provenance"),
                "sample_rate_hz": run.get("data_provenance", {}).get("native_sample_rate_hz"),
                "label_provenance": run.get("label_provenance"),
            }
        )
    else:
        status_rows.append({"method": method, "status": "not_run"})
run_status = pd.DataFrame(status_rows)
run_status
"""
            ),
            code(
                """
metrics = load_experiment_metrics(ARTIFACT_ROOT)
display(metrics.sort_values(["task", "metric", "method"]))
comparison_keys = ["task", "split", "metric", "reference_source"]
comparison_table = metrics.pivot_table(index=comparison_keys, columns="method", values="value", aggfunc="first")
display(comparison_table)
metrics.to_csv(ARTIFACT_ROOT / "comparison_metrics.csv", index=False)
"""
            ),
            code(
                """
def metric_value(method, metric, split="test"):
    selected = metrics.loc[(metrics["method"] == method) & (metrics["metric"] == metric) & (metrics["split"] == split), "value"]
    return float(selected.iloc[0]) if len(selected) else np.nan


findings = pd.DataFrame(
    [
        {"method": "Signal processing", "result": metric_value("01_signal_processing", "non_control_review_candidate_rate", "all"), "reading": "Review-candidate rate only; control detections remain visible."},
        {"method": "DTW", "result": metric_value("02_dtw", "matched_better_than_control_rate"), "reading": "Above 0.5 supports teacher-period pairs only."},
        {"method": "Correlation", "result": metric_value("03_correlation", "mean_lag_mad_s"), "reading": "Lower channel lag spread means consistency, not accuracy."},
        {"method": "Learned embeddings", "result": metric_value("04_learned_embeddings", "mean_neighbor_cosine_similarity"), "reading": "Near-unity values may mean overlap or collapse."},
        {"method": "Autoencoder", "result": metric_value("05_autoencoders", "autoencoder_mse") - metric_value("05_autoencoders", "pca_mse"), "reading": "Negative delta beats PCA."},
        {"method": "Contrastive learning", "result": metric_value("06_contrastive_learning", "mean_view_cosine_similarity"), "reading": "Measures augmentation invariance only."},
        {"method": "Sequence model", "result": metric_value("07_sequence_models", "gru_mse") - metric_value("07_sequence_models", "persistence_mse"), "reading": "Negative delta beats persistence."},
        {"method": "Unsupervised learning", "result": metric_value("08_unsupervised_learning", "seed_adjusted_rand_index", "train"), "reading": "Stable clusters remain operating regimes."},
    ]
)
findings.to_csv(ARTIFACT_ROOT / "report_findings.csv", index=False)
findings
"""
            ),
            markdown(
                """
## Method-score pairplots

Recording view aggregates each method to median score per recording. Percentile ranks make unlike scales comparable. Reconstruction and sequence errors are reversed so higher rank means lower error. Complete-case count is small.

Window view uses exact shared windows from embeddings, autoencoder, contrastive, sequence, and unsupervised methods. Overlapping windows are dependent observations. Correlations describe score behavior, not agreement on true laps.
"""
            ),
            code(
                """
sns.set_theme(style="ticks", palette="colorblind")
recording_scores = build_recording_score_matrix(ARTIFACT_ROOT)
recording_ranks = rank_method_scores(recording_scores, ["recording_id"])
recording_plot = recording_ranks.merge(manifest[["recording_id", "split", "motion_status"]], on="recording_id", how="left")
recording_columns = [METHOD_LABELS[method] for method in RECORDING_METHODS]
recording_complete = recording_plot.dropna(subset=recording_columns)
display(pd.Series({"all_recordings": len(recording_plot), "complete_recordings": len(recording_complete)}, name="count").to_frame())
grid = sns.pairplot(
    recording_complete,
    vars=recording_columns,
    hue="split",
    corner=True,
    diag_kind="hist",
    height=1.45,
    plot_kws={"s": 32, "alpha": 0.8},
)
grid.figure.suptitle(f"{DATASET_KEY}: recording-level method score ranks", y=1.01)
grid.savefig(ARTIFACT_ROOT / "recording_method_pairplot.png", dpi=180, bbox_inches="tight")
recording_plot.to_csv(ARTIFACT_ROOT / "recording_method_scores.csv", index=False)
"""
            ),
            code(
                """
recording_correlation = recording_ranks[recording_columns].corr(method="spearman", min_periods=5)
recording_correlation.to_csv(ARTIFACT_ROOT / "recording_method_spearman.csv")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(recording_correlation, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title(f"{DATASET_KEY}: recording-level Spearman correlation")
fig.tight_layout()
fig.savefig(ARTIFACT_ROOT / "recording_method_spearman.png", dpi=180, bbox_inches="tight")
"""
            ),
            code(
                """
window_scores = build_window_score_matrix(ARTIFACT_ROOT)
window_keys = ["recording_id", "start_s", "end_s"]
window_ranks = rank_method_scores(window_scores, window_keys)
window_plot = window_ranks.merge(manifest[["recording_id", "split"]], on="recording_id", how="left")
window_columns = [METHOD_LABELS[method] for method in WINDOW_METHODS]
grid = sns.pairplot(
    window_plot,
    vars=window_columns,
    hue="split",
    corner=True,
    diag_kind="hist",
    height=1.7,
    plot_kws={"s": 10, "alpha": 0.35},
)
grid.figure.suptitle(f"{DATASET_KEY}: exact-window method score ranks", y=1.01)
grid.savefig(ARTIFACT_ROOT / "window_method_pairplot.png", dpi=180, bbox_inches="tight")
window_plot.to_csv(ARTIFACT_ROOT / "window_method_scores.csv", index=False)
window_correlation = window_ranks[window_columns].corr(method="spearman")
window_correlation.to_csv(ARTIFACT_ROOT / "window_method_spearman.csv")
window_correlation
"""
            ),
            markdown(
                """
## Family-aware integration

Recurrence fusion combines signal/correlation period evidence with DTW shape support. DTW depends on signal-derived teacher periods, so those two are not independent votes. Window consensus normalizes learned-method scores against training distributions only. Autoencoder and sequence errors contribute typicality context, not lap votes.
"""
            ),
            code(
                """
recurrence_fusion = build_recurrence_fusion(ARTIFACT_ROOT)
window_consensus = build_window_consensus(ARTIFACT_ROOT)
recurrence_fusion.to_csv(ARTIFACT_ROOT / "recurrence_fusion.csv", index=False)
window_consensus.to_csv(ARTIFACT_ROOT / "window_consensus.csv", index=False)
display(recurrence_fusion)
display(window_consensus.sort_values(["recording_id", "start_s"]))

fusion_figures = render_fusion_diagnostics(recurrence_fusion, window_consensus, {"dataset_key": DATASET_KEY}, ARTIFACT_ROOT)
for figure in fusion_figures.values():
    display(figure)
    plt.close(figure)
"""
            ),
            markdown(
                """
## Report boundary

No independent lap boundaries or GPS exist. Report compares engineering evidence, baselines, stability, and failure behavior. It cannot name an accurate lap detector.
"""
            ),
        ]
    )


def build_dataset_notebooks(dataset_key):
    notebooks = {"01_signal_processing.ipynb": signal_notebook(dataset_key)}
    notebooks.update(
        {
            f"{method}.ipynb": method_notebook(dataset_key, method, definition)
            for method, definition in METHODS.items()
        }
    )
    notebooks["09_comparison_report.ipynb"] = comparison_notebook(dataset_key)
    return notebooks


def main():
    for dataset_key in DATASET_KEYS:
        output_dir = NOTEBOOK_ROOT / dataset_key
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in build_dataset_notebooks(dataset_key).items():
            (output_dir / name).write_text(json.dumps(content, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
