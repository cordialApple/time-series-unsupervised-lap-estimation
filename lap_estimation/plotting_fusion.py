from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIGURE_DPI = 160
PALETTE = "colorblind"
PRIMARY_COLOR = sns.color_palette(PALETTE)[0]


def _dataset_label(manifest):
    if isinstance(manifest, dict):
        config = manifest.get("config", {})
        dataset = manifest.get("dataset_key", manifest.get("dataset", config.get("dataset_key", "dataset")))
        return str(dataset)
    return "dataset"


def _save_figures(figures, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(output_dir / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    return figures


def _short_ids(values, length=10):
    return pd.Series(values, dtype="string").str.slice(0, length)


def _new_figure(size):
    return plt.subplots(figsize=size, constrained_layout=True)


def _timeline_data(data, recording_ids):
    selected = data[data["recording_id"].isin(recording_ids)].copy()
    selected["recording"] = _short_ids(selected["recording_id"])
    selected["midpoint_s"] = (selected["start_s"] + selected["end_s"]) / 2.0
    return selected


def _cluster_selection_figure(selection, dataset_label):
    fig, ax = _new_figure((8, 4.5))
    if selection.empty:
        ax.text(0.5, 0.5, "Cluster-selection data unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.lineplot(
            data=selection,
            x="clusters",
            y="silhouette",
            marker="o",
            color=PRIMARY_COLOR,
            ax=ax,
            label="Silhouette",
        )
        sns.lineplot(
            data=selection,
            x="clusters",
            y="seed_ari",
            marker="s",
            color=sns.color_palette(PALETTE)[1],
            ax=ax,
            label="Seed ARI",
        )
        ax.set(xlabel="Cluster count", ylabel="Internal score", ylim=(0, 1.05))
        ax.legend(title="Diagnostic")
    ax.set_title(f"{dataset_label}: cluster selection")
    return fig


def _occupancy_figure(predictions, dataset_label):
    occupancy = pd.crosstab(predictions["recording_id"], predictions["cluster"], normalize="index")
    occupancy.index = _short_ids(occupancy.index)
    fig, ax = _new_figure((8, max(4.5, 0.35 * len(occupancy))))
    sns.heatmap(occupancy, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set(xlabel="Regime cluster", ylabel="Recording", title=f"{dataset_label}: regime occupancy share")
    return fig


def _transition_table(predictions):
    ordered = predictions.sort_values(["recording_id", "start_s"])
    source = ordered.groupby("recording_id", sort=False)["cluster"].shift()
    transitions = pd.DataFrame({"source": source, "target": ordered["cluster"]}).dropna()
    clusters = sorted(pd.unique(predictions["cluster"]))
    counts = pd.crosstab(transitions["source"], transitions["target"])
    return counts.reindex(index=clusters, columns=clusters, fill_value=0)


def _transition_figure(predictions, dataset_label):
    transitions = _transition_table(predictions)
    fig, ax = _new_figure((6.5, 5.5))
    sns.heatmap(transitions, cmap="mako", annot=True, fmt="g", square=True, cbar_kws={"label": "Count"}, ax=ax)
    ax.set(xlabel="Next cluster", ylabel="Current cluster", title=f"{dataset_label}: within-recording transitions")
    return fig


def _distance_figure(predictions, dataset_label):
    fig, ax = _new_figure((8, 4.8))
    sns.violinplot(
        data=predictions,
        x="cluster",
        y="distance_to_centroid",
        inner="quart",
        palette=PALETTE,
        hue="cluster",
        legend=False,
        ax=ax,
    )
    ax.set(xlabel="Regime cluster", ylabel="Distance to centroid", title=f"{dataset_label}: regime fit distributions")
    return fig


def _timeline_figure(predictions, dataset_label, max_recordings=8):
    counts = predictions.groupby("recording_id").size().nlargest(max_recordings)
    selected = _timeline_data(predictions, counts.index)
    fig, ax = _new_figure((12, max(4.5, 0.55 * len(counts))))
    sns.scatterplot(
        data=selected,
        x="midpoint_s",
        y="recording",
        hue="cluster",
        palette="colorblind",
        marker="s",
        s=75,
        linewidth=0,
        ax=ax,
    )
    ax.set(xlabel="Window midpoint (s)", ylabel="Recording", title=f"{dataset_label}: regime timelines")
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
    return fig


def _profile_figure(predictions, dataset_label):
    ordered = predictions.sort_values(["recording_id", "start_s"]).copy()
    ordered["transition"] = ordered.groupby("recording_id")["cluster"].diff().ne(0)
    ordered.loc[ordered.groupby("recording_id").head(1).index, "transition"] = False
    profiles = ordered.groupby("recording_id", as_index=False).agg(
        regime_count=("cluster", "nunique"),
        transitions=("transition", "sum"),
        mean_distance=("distance_to_centroid", "mean"),
    )
    profiles["recording"] = _short_ids(profiles["recording_id"])
    long = profiles.melt(
        id_vars="recording",
        value_vars=["regime_count", "transitions", "mean_distance"],
        var_name="profile",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4.5, 0.35 * len(profiles))), constrained_layout=True)
    for ax, metric in zip(axes, ["regime_count", "transitions", "mean_distance"]):
        subset = long[long["profile"].eq(metric)].sort_values("value", ascending=False)
        sns.barplot(data=subset, x="value", y="recording", color=PRIMARY_COLOR, ax=ax)
        ax.set(xlabel=metric.replace("_", " ").title(), ylabel="Recording")
    fig.suptitle(f"{dataset_label}: per-recording regime profiles")
    return fig


def render_unsupervised_diagnostics(predictions, manifest, output_dir):
    predictions = pd.DataFrame(predictions).copy()
    output_dir = Path(output_dir)
    selection_path = output_dir / "cluster_selection.csv"
    selection = pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()
    dataset_label = _dataset_label(manifest)
    figures = {
        "cluster_selection": _cluster_selection_figure(selection, dataset_label),
        "occupancy_heatmap": _occupancy_figure(predictions, dataset_label),
        "transition_heatmap": _transition_figure(predictions, dataset_label),
        "distance_distributions": _distance_figure(predictions, dataset_label),
        "regime_timelines": _timeline_figure(predictions, dataset_label),
        "recording_profiles": _profile_figure(predictions, dataset_label),
    }
    return _save_figures(figures, output_dir)


def _period_agreement_figure(recurrence, dataset_label):
    data = recurrence.dropna(subset=["signal_period_s", "correlation_period_s"]).copy()
    fig, ax = _new_figure((7, 6))
    sns.scatterplot(
        data=data,
        x="signal_period_s",
        y="correlation_period_s",
        hue="period_agreement",
        size="fused_support",
        sizes=(50, 220),
        palette="viridis",
        ax=ax,
    )
    if not data.empty:
        low = min(data["signal_period_s"].min(), data["correlation_period_s"].min())
        high = max(data["signal_period_s"].max(), data["correlation_period_s"].max())
        ax.plot([low, high], [low, high], linestyle="--", color="0.35", label="Equal period")
    ax.set(
        xlabel="Signal recurrence period (s)",
        ylabel="Cross-correlation period (s)",
        title=f"{dataset_label}: recurrence-period agreement",
    )
    return fig


def _support_dispersion_figure(recurrence, dataset_label):
    fig, ax = _new_figure((8, 5))
    sns.scatterplot(
        data=recurrence,
        x="family_dispersion",
        y="fused_support",
        hue="evidence_status",
        size="method_count",
        sizes=(50, 220),
        palette=PALETTE,
        ax=ax,
    )
    ax.axhline(2.0 / 3.0, linestyle="--", color="0.45", linewidth=1)
    ax.set(
        xlabel="Family dispersion",
        ylabel="Fused support",
        title=f"{dataset_label}: support versus family disagreement",
    )
    return fig


def _coverage_figure(recurrence, dataset_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    sns.countplot(data=recurrence, x="method_count", color=PRIMARY_COLOR, ax=axes[0])
    sns.countplot(data=recurrence, x="evidence_status", palette=PALETTE, hue="evidence_status", legend=False, ax=axes[1])
    axes[0].set(xlabel="Available methods", ylabel="Recordings", title="Method coverage")
    axes[1].set(xlabel="Evidence status", ylabel="Recordings", title="Fusion disposition")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle(f"{dataset_label}: recurrence evidence coverage")
    return fig


def _family_heatmap_figure(recurrence, dataset_label):
    columns = [
        "signal_support",
        "dtw_support",
        "correlation_support",
        "period_support",
        "signal_family_support",
        "shape_family_support",
        "fused_support",
    ]
    available = [column for column in columns if column in recurrence]
    matrix = recurrence.set_index("recording_id")[available].sort_values("fused_support", ascending=False)
    matrix.index = _short_ids(matrix.index)
    fig, ax = _new_figure((11, max(4.5, 0.35 * len(matrix))))
    sns.heatmap(matrix, cmap="viridis", vmin=0, vmax=1, annot=len(matrix) <= 20, fmt=".2f", ax=ax)
    ax.set(xlabel="Component", ylabel="Recording", title=f"{dataset_label}: fusion components")
    ax.tick_params(axis="x", rotation=35)
    return fig


def _consensus_distribution_figure(consensus, dataset_label):
    fig, ax = _new_figure((8, 4.8))
    sns.histplot(
        data=consensus,
        x="consensus_support",
        hue="split",
        bins=20,
        stat="density",
        common_norm=False,
        element="step",
        palette=PALETTE,
        ax=ax,
    )
    ax.set(xlabel="Window consensus support", ylabel="Density", title=f"{dataset_label}: consensus distributions")
    return fig


def _dispersion_distribution_figure(consensus, dataset_label):
    fig, ax = _new_figure((8, 4.8))
    sns.violinplot(
        data=consensus,
        x="split",
        y="support_iqr",
        hue="split",
        palette=PALETTE,
        legend=False,
        inner="quart",
        ax=ax,
    )
    ax.set(xlabel="Split", ylabel="Method-support IQR", title=f"{dataset_label}: window disagreement")
    return fig


def _consensus_timeline_figure(consensus, dataset_label, max_recordings=8):
    variability = consensus.groupby("recording_id")["consensus_support"].std().fillna(0).nlargest(max_recordings)
    selected = _timeline_data(consensus, variability.index)
    fig, ax = _new_figure((12, 5.5))
    sns.lineplot(
        data=selected,
        x="midpoint_s",
        y="consensus_support",
        hue="recording",
        palette=PALETTE,
        estimator=None,
        linewidth=1.5,
        ax=ax,
    )
    ax.set(
        xlabel="Window midpoint (s)",
        ylabel="Consensus support",
        ylim=(0, 1.05),
        title=f"{dataset_label}: high-variability consensus timelines",
    )
    ax.legend(title="Recording", bbox_to_anchor=(1.01, 1), loc="upper left")
    return fig


def render_fusion_diagnostics(recurrence_fusion, window_consensus, manifest, output_dir):
    recurrence = pd.DataFrame(recurrence_fusion).copy()
    consensus = pd.DataFrame(window_consensus).copy()
    dataset_label = _dataset_label(manifest)
    figures = {
        "recurrence_period_agreement": _period_agreement_figure(recurrence, dataset_label),
        "fused_support_dispersion": _support_dispersion_figure(recurrence, dataset_label),
        "recurrence_coverage": _coverage_figure(recurrence, dataset_label),
        "family_component_heatmap": _family_heatmap_figure(recurrence, dataset_label),
        "window_consensus_distribution": _consensus_distribution_figure(consensus, dataset_label),
        "window_dispersion_distribution": _dispersion_distribution_figure(consensus, dataset_label),
        "window_consensus_timelines": _consensus_timeline_figure(consensus, dataset_label),
    }
    return _save_figures(figures, output_dir)
