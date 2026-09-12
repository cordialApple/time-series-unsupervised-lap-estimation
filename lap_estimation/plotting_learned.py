from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LEARNED_METHOD_PLOT_COUNTS = {
    "04_learned_embeddings": 5,
    "05_autoencoders": 5,
    "06_contrastive_learning": 5,
    "07_sequence_models": 5,
}


def _dataset_label(manifest):
    return manifest.get("data_provenance", {}).get(
        "dataset_key",
        manifest.get("dataset_key", manifest.get("config", {}).get("dataset_key", "dataset")),
    )


def _prepare_predictions(predictions, manifest):
    frame = pd.DataFrame(predictions).copy()
    assignments = manifest.get("split_assignments", {})
    frame["split"] = frame["recording_id"].map(assignments).fillna("unassigned")
    frame["recording"] = frame["recording_id"].astype(str).str[:8]
    return frame


def _load_embeddings(output_dir, row_count):
    path = Path(output_dir) / "embeddings.npy"
    if not path.exists():
        raise ValueError(f"embeddings file missing: {path}")
    embeddings = np.asarray(np.load(path), dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] != row_count:
        raise ValueError("embeddings rows must align with predictions")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    return embeddings


def _new_figure(size=(9, 5)):
    return plt.subplots(figsize=size, constrained_layout=True)


def _save_figures(figures, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    return figures


def _latent_projection_figure(frame, embeddings, title):
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vectors[:2].T
    plotted = frame.assign(pc1=projection[:, 0], pc2=projection[:, 1])
    fig, ax = _new_figure((8, 6))
    sns.scatterplot(
        data=plotted,
        x="pc1",
        y="pc2",
        hue="split",
        size="start_s",
        sizes=(18, 90),
        alpha=0.72,
        palette="colorblind",
        ax=ax,
    )
    ax.set(title=title, xlabel="Latent PC1", ylabel="Latent PC2")
    return fig


def _latent_spectrum_figure(embeddings, title):
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    fractions = variance / variance.sum() if variance.sum() else np.zeros_like(variance)
    positive = fractions[fractions > 0]
    effective_rank = float(np.exp(-(positive * np.log(positive)).sum())) if len(positive) else 0.0
    spectrum = pd.DataFrame(
        {
            "component": np.arange(1, len(fractions) + 1),
            "variance_fraction": fractions,
            "cumulative_variance": np.cumsum(fractions),
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    sns.barplot(data=spectrum, x="component", y="variance_fraction", color="#4C72B0", ax=axes[0])
    sns.lineplot(
        data=spectrum,
        x="component",
        y="cumulative_variance",
        marker="o",
        color="#DD8452",
        ax=axes[1],
    )
    axes[0].set(title=f"{title}\nEffective rank {effective_rank:.2f}", ylabel="Variance fraction")
    axes[1].set(title="Cumulative latent variance", ylabel="Cumulative fraction", ylim=(0, 1.03))
    return fig


def _score_distribution(frame, value, title, xlabel):
    fig, ax = _new_figure()
    sns.histplot(
        data=frame,
        x=value,
        hue="split",
        bins=24,
        element="step",
        stat="density",
        common_norm=False,
        palette="colorblind",
        ax=ax,
    )
    ax.set(title=title, xlabel=xlabel, ylabel="Density")
    return fig


def _similarity_vs_separation(frame, title):
    plotted = frame.assign(neighbor_scope=_neighbor_scope(frame))
    fig, ax = _new_figure()
    sns.scatterplot(
        data=plotted,
        x="temporal_separation_s",
        y="score",
        hue="neighbor_scope",
        style="split",
        alpha=0.68,
        palette="colorblind",
        ax=ax,
    )
    ax.set(title=title, xlabel="Neighbor temporal separation (s)", ylabel="Cosine similarity")
    return fig


def _neighbor_scope(frame):
    return np.where(
        frame["recording_id"] == frame["neighbor_recording_id"],
        "same recording",
        "cross recording",
    )


def _baseline_scatter(frame, x, y, title, xlabel, ylabel):
    fig, ax = _new_figure((6.5, 6))
    sns.scatterplot(
        data=frame,
        x=x,
        y=y,
        hue="split",
        alpha=0.7,
        palette="colorblind",
        ax=ax,
    )
    bounds = np.asarray(ax.get_xlim() + ax.get_ylim())
    lower, upper = float(np.nanmin(bounds)), float(np.nanmax(bounds))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(lower, upper), ylim=(lower, upper))
    return fig


def _excess_error_figure(frame, value, title, xlabel):
    fig, ax = _new_figure()
    sns.histplot(
        data=frame,
        x=value,
        hue="split",
        bins=25,
        element="step",
        palette="colorblind",
        ax=ax,
    )
    ax.axvline(0, linestyle="--", color="black", linewidth=1)
    ax.set(title=title, xlabel=xlabel, ylabel="Windows")
    return fig


def _error_timeline_figure(frame, value, title):
    fig, ax = _new_figure((11, 5))
    sns.lineplot(
        data=frame,
        x="start_s",
        y=value,
        hue="model",
        style="recording",
        estimator="median",
        errorbar=None,
        palette="colorblind",
        ax=ax,
    )
    ax.set(title=title, xlabel="Window start (s)", ylabel="Median MSE")
    return fig


def render_embedding_diagnostics(predictions, manifest, output_dir):
    frame = _prepare_predictions(predictions, manifest)
    embeddings = _load_embeddings(output_dir, len(frame))
    dataset = _dataset_label(manifest)
    separation = frame.assign(neighbor_scope=_neighbor_scope(frame))
    fig, ax = _new_figure()
    sns.ecdfplot(
        data=separation,
        x="temporal_separation_s",
        hue="neighbor_scope",
        palette="colorblind",
        ax=ax,
    )
    ax.set(
        title=f"{dataset}: embedding neighbor temporal reach",
        xlabel="Temporal separation (s)",
        ylabel="Cumulative fraction",
    )
    figures = {
        "latent_pca": _latent_projection_figure(
            frame, embeddings, f"{dataset}: learned embedding PCA"
        ),
        "similarity_distribution": _score_distribution(
            frame,
            "score",
            f"{dataset}: embedding neighbor similarity",
            "Cosine similarity",
        ),
        "similarity_vs_separation": _similarity_vs_separation(
            frame, f"{dataset}: similarity versus temporal separation"
        ),
        "separation_distribution": fig,
        "latent_spectrum": _latent_spectrum_figure(
            embeddings, f"{dataset}: learned embedding spectrum"
        ),
    }
    return _save_figures(figures, output_dir)


def render_autoencoder_diagnostics(predictions, manifest, output_dir):
    frame = _prepare_predictions(predictions, manifest)
    embeddings = _load_embeddings(output_dir, len(frame))
    dataset = _dataset_label(manifest)
    compared = frame.assign(
        error_delta=frame["score"] - frame["pca_reconstruction_error"]
    )
    long_errors = compared.melt(
        id_vars=["recording_id", "recording", "start_s", "split"],
        value_vars=["score", "pca_reconstruction_error"],
        var_name="model",
        value_name="reconstruction_error",
    ).replace({"score": "Autoencoder", "pca_reconstruction_error": "PCA"})
    fig_distribution, ax = _new_figure()
    sns.ecdfplot(
        data=long_errors,
        x="reconstruction_error",
        hue="model",
        palette="colorblind",
        ax=ax,
    )
    ax.set(title=f"{dataset}: reconstruction error ECDF", xlabel="Window MSE")
    fig_scatter = _baseline_scatter(
        compared,
        "pca_reconstruction_error",
        "score",
        f"{dataset}: autoencoder versus PCA",
        "PCA reconstruction MSE",
        "Autoencoder reconstruction MSE",
    )
    fig_delta = _excess_error_figure(
        compared,
        "error_delta",
        f"{dataset}: autoencoder excess error",
        "Autoencoder MSE − PCA MSE",
    )
    fig_timeline = _error_timeline_figure(
        long_errors,
        "reconstruction_error",
        f"{dataset}: reconstruction error over session time",
    )
    figures = {
        "latent_pca": _latent_projection_figure(
            frame, embeddings, f"{dataset}: autoencoder latent PCA"
        ),
        "error_distribution": fig_distribution,
        "autoencoder_vs_pca": fig_scatter,
        "error_delta": fig_delta,
        "error_timeline": fig_timeline,
    }
    return _save_figures(figures, output_dir)


def render_contrastive_diagnostics(predictions, manifest, output_dir):
    frame = _prepare_predictions(predictions, manifest)
    embeddings = _load_embeddings(output_dir, len(frame))
    dataset = _dataset_label(manifest)
    compared = frame.assign(neighbor_scope=_neighbor_scope(frame))
    fig_scope, ax = _new_figure()
    sns.violinplot(
        data=compared,
        x="neighbor_scope",
        y="score",
        hue="split",
        inner="quart",
        cut=0,
        palette="colorblind",
        ax=ax,
    )
    ax.set(
        title=f"{dataset}: contrastive neighbor scope",
        xlabel="Nearest-neighbor scope",
        ylabel="Cosine similarity",
    )
    figures = {
        "latent_pca": _latent_projection_figure(
            frame, embeddings, f"{dataset}: contrastive embedding PCA"
        ),
        "similarity_distribution": _score_distribution(
            frame,
            "score",
            f"{dataset}: contrastive neighbor similarity",
            "Cosine similarity",
        ),
        "similarity_vs_separation": _similarity_vs_separation(
            frame, f"{dataset}: contrastive temporal neighbor structure"
        ),
        "cross_recording_neighbors": fig_scope,
        "latent_spectrum": _latent_spectrum_figure(
            embeddings, f"{dataset}: contrastive embedding spectrum"
        ),
    }
    return _save_figures(figures, output_dir)


def render_sequence_diagnostics(predictions, manifest, output_dir):
    frame = _prepare_predictions(predictions, manifest)
    dataset = _dataset_label(manifest)
    compared = frame.assign(
        excess_error=frame["score"] - frame["persistence_error"],
        relative_error=frame["score"] / frame["persistence_error"].clip(lower=1e-12),
    )
    long_errors = compared.melt(
        id_vars=["recording_id", "recording", "start_s", "split"],
        value_vars=["score", "persistence_error"],
        var_name="model",
        value_name="prediction_error",
    ).replace({"score": "GRU", "persistence_error": "Persistence"})
    fig_distribution, ax = _new_figure()
    sns.ecdfplot(
        data=long_errors,
        x="prediction_error",
        hue="model",
        palette="colorblind",
        ax=ax,
    )
    ax.set(title=f"{dataset}: next-sample error ECDF", xlabel="Window MSE")
    fig_scatter = _baseline_scatter(
        compared,
        "persistence_error",
        "score",
        f"{dataset}: GRU versus persistence",
        "Persistence MSE",
        "GRU MSE",
    )
    fig_excess = _excess_error_figure(
        compared,
        "excess_error",
        f"{dataset}: GRU excess error",
        "GRU MSE − persistence MSE",
    )
    fig_timeline = _error_timeline_figure(
        long_errors,
        "prediction_error",
        f"{dataset}: next-sample error over session time",
    )
    fig_relative, ax = _new_figure((9, 5.5))
    sns.violinplot(
        data=compared,
        x="relative_error",
        y="recording",
        hue="split",
        inner="quart",
        cut=0,
        palette="colorblind",
        ax=ax,
    )
    ax.axvline(1, linestyle="--", color="black", linewidth=1)
    ax.set(
        title=f"{dataset}: GRU error relative to persistence",
        xlabel="GRU MSE / persistence MSE",
        ylabel="Recording",
    )
    figures = {
        "error_distribution": fig_distribution,
        "gru_vs_persistence": fig_scatter,
        "excess_error_distribution": fig_excess,
        "error_timeline": fig_timeline,
        "relative_error_by_recording": fig_relative,
    }
    return _save_figures(figures, output_dir)
