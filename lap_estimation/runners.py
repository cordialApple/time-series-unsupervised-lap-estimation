import json
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import write_experiment_artifacts
from .config import RANDOM_SEED
from .data import load_recording, prepare_recording_manifest, select_informative_channels
from .datasets import DatasetSpec, resolve_dataset
from .experiments import MLPAutoencoder, SequencePredictor, TemporalEncoder, build_window_dataset, dtw_distance, engineered_features, fit_channel_scaler, seconds_to_samples, transform_windows
from .signal import build_multivariate_trace, lagged_cross_correlation, normalized_autocorrelation, rank_autocorrelation_peaks, recurrence_candidates


EVALUATION_SPLITS = ("train", "validation", "test", "all")


@dataclass
class ExperimentResult:
    method: str
    output_dir: Path
    predictions: pd.DataFrame
    metrics: pd.DataFrame


@dataclass
class ExperimentContext:
    dataset_key: str
    specification: DatasetSpec
    artifact_root: Path
    sample_rate_hz: float
    channels: list[str]
    manifest: pd.DataFrame
    windows: np.ndarray
    metadata: pd.DataFrame
    features: np.ndarray
    train_mask: np.ndarray


def _resolve_enabled_dataset(dataset_key, project_root):
    specification = resolve_dataset(dataset_key, project_root)
    if not specification.enabled:
        raise ValueError(f"Dataset is not enabled: {dataset_key}")
    return specification


def _build_context(project_root, dataset_key="drive_day_2026", window_s=10.0, step_s=2.0):
    specification = _resolve_enabled_dataset(dataset_key, project_root)
    manifest = prepare_recording_manifest(specification)
    window_data = build_window_dataset(
        manifest,
        specification,
        window_s,
        step_s,
    )
    train_mask = window_data.metadata["split"].eq("train").to_numpy()
    center, scale = fit_channel_scaler(window_data.values[train_mask])
    windows = transform_windows(window_data.values, center, scale)
    features = engineered_features(windows)
    feature_scaler = RobustScaler().fit(features[train_mask])
    features = feature_scaler.transform(features)
    return ExperimentContext(
        dataset_key=dataset_key,
        specification=specification,
        artifact_root=specification.artifact_root,
        sample_rate_hz=specification.native_sample_rate_hz,
        channels=window_data.channels,
        manifest=manifest,
        windows=windows,
        metadata=window_data.metadata,
        features=features,
        train_mask=train_mask,
    )


def _write_result(context, method, task, config, predictions, metrics, label_provenance, started_at):
    output_dir = write_experiment_artifacts(
        context.artifact_root,
        method=method,
        task=task,
        config={"dataset_key": context.dataset_key, **config},
        source_hashes=context.manifest["recording_id"],
        split_assignments=context.manifest.set_index("recording_id")["split"],
        predictions=predictions,
        metrics=metrics,
        label_provenance=label_provenance,
        data_provenance={
            "dataset_key": context.dataset_key,
            "native_sample_rate_hz": context.sample_rate_hz,
            "clock_provenance": context.specification.clock_provenance,
            "selected_channels": context.channels,
            "semantic_mapping": context.specification.semantic_mapping,
            "semantic_mapping_version": context.specification.semantic_mapping_version,
            "motion_policy": context.specification.motion_policy,
        },
        started_at=started_at,
    )
    return ExperimentResult(method, output_dir, predictions, metrics)


def _split_metric_rows(metadata, values, task, metric, reference_source):
    rows = []
    for split in EVALUATION_SPLITS:
        mask = np.ones(len(metadata), dtype=bool) if split == "all" else metadata["split"].eq(split).to_numpy()
        selected = np.asarray(values)[mask]
        rows.append(
            {
                "task": task,
                "split": split,
                "metric": metric,
                "value": float(np.mean(selected)) if len(selected) else np.nan,
                "sample_count": int(len(selected)),
                "reference_source": reference_source,
            }
        )
    return rows


def _window_predictions(metadata, kind, score, **extra):
    predictions = metadata.copy().reset_index(drop=True)
    predictions["segment_id"] = [f"window_{index}" for index in range(len(predictions))]
    predictions["prediction_kind"] = kind
    predictions["score"] = np.asarray(score)
    predictions["label_status"] = "provisional_unverified"
    for key, values in extra.items():
        predictions[key] = values
    required = ["recording_id", "segment_id", "start_s", "end_s", "prediction_kind", "score", "label_status"]
    optional = [column for column in predictions.columns if column not in required + ["split", "control_candidate"]]
    return predictions[required + optional]


def _nearest_neighbor_predictions(embeddings, metadata, kind):
    values = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.where(norms > 1e-12, norms, 1.0)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    recording_ids = metadata["recording_id"].to_numpy()
    starts = metadata["start_s"].to_numpy()
    for index in range(len(metadata)):
        overlapping = (recording_ids == recording_ids[index]) & (np.abs(starts - starts[index]) < 15.0)
        similarities[index, overlapping] = -np.inf
    neighbors = np.argmax(similarities, axis=1)
    scores = similarities[np.arange(len(similarities)), neighbors]
    separation = np.abs(starts - starts[neighbors])
    return _window_predictions(
        metadata,
        kind,
        scores,
        neighbor_recording_id=recording_ids[neighbors],
        neighbor_start_s=starts[neighbors],
        temporal_separation_s=separation,
    )


def run_signal_processing(project_root, dataset_key="drive_day_2026", **_):
    started_at = time.perf_counter()
    specification = _resolve_enabled_dataset(dataset_key, project_root)
    manifest = prepare_recording_manifest(specification)
    rows = []
    coverage = []
    sensitivity_rows = []
    for record in manifest.itertuples(index=False):
        frame = load_recording(record.primary_path, specification)
        for channel in specification.trajectory_channels:
            if channel not in frame:
                state = "absent"
            else:
                values = pd.to_numeric(frame[channel], errors="coerce")
                state = "dynamic" if values.nunique(dropna=True) > 1 else "constant"
            coverage.append({"recording_id": record.recording_id, "channel": channel, "state": state})
        channels = select_informative_channels(frame, specification.analysis_channels)
        if not channels:
            continue
        peaks = recurrence_candidates(frame, channels, specification.native_sample_rate_hz)
        for rank, peak in peaks.iterrows():
            rows.append(
                {
                    "recording_id": record.recording_id,
                    "split": record.split,
                    "control_candidate": record.control_candidate,
                    "rank": rank + 1,
                    "channels": json.dumps(channels),
                    **peak.to_dict(),
                }
            )
        if not record.control_candidate:
            for smooth_samples in [3, 5, 10]:
                trace = build_multivariate_trace(frame, channels, smooth_samples)
                max_lag_s = min(120.0, frame["elapsed_s"].max() / 2.0)
                if max_lag_s <= 15.0:
                    continue
                correlation = normalized_autocorrelation(trace, round(max_lag_s * specification.native_sample_rate_hz))
                top = rank_autocorrelation_peaks(correlation, specification.native_sample_rate_hz, 15.0, max_lag_s, 1)
                if not top.empty:
                    sensitivity_rows.append({"recording_id": record.recording_id, "smooth_samples": smooth_samples, **top.iloc[0].to_dict()})
    candidates = pd.DataFrame(
        rows,
        columns=[
            "recording_id",
            "split",
            "control_candidate",
            "rank",
            "channels",
            "lag_samples",
            "lag_s",
            "correlation",
            "prominence",
            "cycle_count",
        ],
    )
    candidates["review_candidate"] = (candidates["correlation"] >= 0.25) & (candidates["cycle_count"] >= 2.0)
    top = candidates.query("rank == 1").sort_values("correlation", ascending=False).reset_index(drop=True)
    durations = manifest.set_index("recording_id")["duration_s"]
    predictions = top.assign(
        segment_id=lambda frame: "period_candidate_" + frame["rank"].astype(str),
        start_s=0.0,
        end_s=lambda frame: frame["recording_id"].map(durations),
        prediction_kind="recurrence_period",
        score=lambda frame: frame["correlation"],
        predicted_period_s=lambda frame: frame["lag_s"],
        label_status="provisional_unverified",
    )[["recording_id", "segment_id", "start_s", "end_s", "prediction_kind", "score", "predicted_period_s", "label_status"]]
    non_controls = top.loc[~top["control_candidate"]]
    controls = top.loc[top["control_candidate"]]
    metrics = pd.DataFrame(
        [
            {"task": "recurrence", "split": "all", "metric": "non_control_recording_coverage", "value": len(non_controls) / max(1, int((~manifest["control_candidate"]).sum())), "sample_count": int((~manifest["control_candidate"]).sum()), "reference_source": "none"},
            {"task": "recurrence", "split": "all", "metric": "non_control_review_candidate_rate", "value": non_controls["review_candidate"].mean() if len(non_controls) else np.nan, "sample_count": len(non_controls), "reference_source": "unverified_threshold"},
            {"task": "recurrence", "split": "all", "metric": "control_candidate_detection_rate", "value": controls["review_candidate"].mean() if len(controls) else np.nan, "sample_count": len(controls), "reference_source": specification.motion_policy},
            {"task": "data_quality", "split": "all", "metric": "duplicate_alias_count", "value": int((manifest["alias_count"] - 1).sum()), "sample_count": int(manifest["alias_count"].sum()), "reference_source": "sha256"},
        ]
    )
    context = ExperimentContext(
        dataset_key,
        specification,
        specification.artifact_root,
        specification.native_sample_rate_hz,
        list(specification.analysis_channels),
        manifest,
        np.empty(0),
        pd.DataFrame(),
        np.empty(0),
        np.empty(0, dtype=bool),
    )
    result = _write_result(context, "01_signal_processing", "unverified_recurrence_period_detection", {"min_lag_s": 15.0, "max_lag_s": 120.0, "minimum_correlation_for_review": 0.25, "minimum_cycle_count": 2.0}, predictions, metrics, "signal_processing_teacher_unverified", started_at)
    profile_columns = ["recording_id", "alias_count", "driver_labels", "driver_ambiguous", "row_count", "duration_s", "declared_duration_s", "sampled_elapsed_span_s", "median_interval_s", "max_gap_s", "dynamic_numeric_count", "max_abs_rpm", "motion_variation_score", "motion_status", "motion_status_source", "control_candidate", "clock_provenance", "vertical_accel_offset_flag", "split"]
    manifest[profile_columns].sort_values("duration_s", ascending=False).to_csv(result.output_dir / "data_profile.csv", index=False)
    pd.DataFrame(coverage).to_csv(result.output_dir / "sensor_coverage.csv", index=False)
    candidates.to_csv(result.output_dir / "periodicity_candidates.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(result.output_dir / "smoothing_sensitivity.csv", index=False)
    return result


def run_dtw(project_root, dataset_key="drive_day_2026", **_):
    started_at = time.perf_counter()
    context = _build_context(project_root, dataset_key)
    signal_path = context.artifact_root / "01_signal_processing" / "predictions.csv"
    if not signal_path.exists():
        run_signal_processing(project_root, dataset_key)
    periods = pd.read_csv(signal_path).set_index("recording_id")["predicted_period_s"]
    rng = np.random.default_rng(RANDOM_SEED)
    warping_band_s = 1.0
    warping_band_samples = seconds_to_samples(warping_band_s, context.sample_rate_hz)
    rows = []
    for recording_id, group in context.metadata.groupby("recording_id"):
        if recording_id not in periods:
            continue
        period = periods[recording_id]
        group = group.sort_values("start_s")
        positions = group.index.to_numpy()
        starts = group["start_s"].to_numpy()
        pair_count = 0
        for local_index, position in enumerate(positions):
            target = starts[local_index] + period
            nearest_local = int(np.argmin(np.abs(starts - target)))
            if abs(starts[nearest_local] - target) > 2.0 or nearest_local == local_index:
                continue
            controls = np.flatnonzero((np.abs(starts - starts[local_index]) >= 15.0) & (np.abs(np.abs(starts - starts[local_index]) - period) >= 5.0))
            if not len(controls):
                continue
            control_local = int(rng.choice(controls))
            matched_distance = dtw_distance(context.windows[position], context.windows[positions[nearest_local]], band=warping_band_samples)
            control_distance = dtw_distance(context.windows[position], context.windows[positions[control_local]], band=warping_band_samples)
            rows.append(
                {
                    "recording_id": recording_id,
                    "segment_id": f"pair_{pair_count}",
                    "start_s": starts[local_index],
                    "end_s": starts[local_index] + 10.0,
                    "prediction_kind": "dtw_period_match",
                    "score": control_distance - matched_distance,
                    "label_status": "provisional_unverified",
                    "predicted_period_s": period,
                    "matched_distance": matched_distance,
                    "control_distance": control_distance,
                    "split": group.iloc[0]["split"],
                }
            )
            pair_count += 1
            if pair_count >= 10:
                break
    full = pd.DataFrame(rows)
    predictions = full.drop(columns=["split"])
    metrics = []
    for split in EVALUATION_SPLITS:
        selected = full if split == "all" else full.loc[full["split"] == split]
        metrics.extend(
            [
                {"task": "period_pair_retrieval", "split": split, "metric": "median_matched_dtw", "value": selected["matched_distance"].median(), "sample_count": len(selected), "reference_source": "signal_period_teacher"},
                {"task": "period_pair_retrieval", "split": split, "metric": "matched_better_than_control_rate", "value": (selected["matched_distance"] < selected["control_distance"]).mean(), "sample_count": len(selected), "reference_source": "within_recording_control"},
            ]
        )
    return _write_result(context, "02_dtw", "time_warped_period_pair_retrieval", {"window_s": 10.0, "step_s": 2.0, "warping_band_s": warping_band_s, "warping_band_samples": warping_band_samples}, predictions, pd.DataFrame(metrics), "signal_period_teacher_unverified", started_at)


def run_correlation(project_root, dataset_key="drive_day_2026", **_):
    started_at = time.perf_counter()
    context = _build_context(project_root, dataset_key)
    period_rows = []
    response_rows = []
    for record in context.manifest.itertuples(index=False):
        frame = load_recording(record.primary_path, context.specification)
        channels = select_informative_channels(frame, context.channels)
        max_lag_s = min(120.0, frame["elapsed_s"].max() / 2.0)
        if max_lag_s > 15.0:
            for channel in channels:
                values = pd.to_numeric(frame[channel], errors="coerce").interpolate(limit_direction="both").to_numpy()
                correlation = normalized_autocorrelation(values, round(max_lag_s * context.sample_rate_hz))
                peaks = rank_autocorrelation_peaks(correlation, context.sample_rate_hz, 15.0, max_lag_s, 1)
                if not peaks.empty:
                    period_rows.append({"recording_id": record.recording_id, "split": record.split, "channel": channel, **peaks.iloc[0].to_dict()})
        for first, second in combinations(channels, 2):
            first_values = pd.to_numeric(frame[first], errors="coerce").interpolate(limit_direction="both").to_numpy()
            second_values = pd.to_numeric(frame[second], errors="coerce").interpolate(limit_direction="both").to_numpy()
            cross = lagged_cross_correlation(first_values, second_values, round(5.0 * context.sample_rate_hz))
            peak = cross.iloc[cross["correlation"].abs().argmax()]
            response_rows.append({"recording_id": record.recording_id, "split": record.split, "first_channel": first, "second_channel": second, **peak.to_dict()})
    period_frame = pd.DataFrame(period_rows)
    period_frame["segment_id"] = period_frame.groupby("recording_id").cumcount().map(lambda value: f"channel_{value}")
    durations = context.manifest.set_index("recording_id")["duration_s"]
    predictions = period_frame.assign(
        start_s=0.0,
        end_s=lambda frame: frame["recording_id"].map(durations),
        prediction_kind="channel_recurrence_period",
        score=lambda frame: frame["correlation"],
        label_status="provisional_unverified",
    ).drop(columns=["split"])
    spread = period_frame.groupby(["recording_id", "split"])["lag_s"].agg(lambda values: np.median(np.abs(values - np.median(values)))).reset_index(name="lag_mad_s")
    metrics = _split_metric_rows(spread, spread["lag_mad_s"], "channel_lag_consensus", "mean_lag_mad_s", "none")
    response_frame = pd.DataFrame(response_rows)
    for split in EVALUATION_SPLITS:
        selected = response_frame if split == "all" else response_frame.loc[response_frame["split"] == split]
        metrics.append({"task": "sensor_cross_correlation", "split": split, "metric": "median_absolute_peak_correlation", "value": selected["correlation"].abs().median(), "sample_count": len(selected), "reference_source": "none"})
    result = _write_result(context, "03_correlation", "channel_recurrence_and_sensor_lag", {"recurrence_lag_s": [15.0, 120.0], "sensor_lag_s": [-5.0, 5.0], "positive_lag_definition": "second_channel_follows_first_channel"}, predictions, pd.DataFrame(metrics), "unverified_channel_recurrence", started_at)
    response_frame.to_csv(result.output_dir / "cross_correlation.csv", index=False)
    return result


class _MaskedPredictionModel(nn.Module):
    def __init__(self, window_size, channel_count, embedding_dim):
        super().__init__()
        self.encoder = TemporalEncoder(channel_count, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, window_size * channel_count)
        self.window_size = window_size
        self.channel_count = channel_count

    def forward(self, values):
        embedding = self.encoder(values)
        reconstruction = self.decoder(embedding).reshape(-1, self.window_size, self.channel_count)
        return reconstruction, embedding


def _loader(values, batch_size=64, shuffle=True):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor), batch_size=min(batch_size, len(tensor)), shuffle=shuffle)


def _train_reconstruction(model, train_values, epochs, masked=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    for _ in range(epochs):
        for (batch,) in _loader(train_values):
            inputs = batch
            if masked:
                mask = torch.rand(batch.shape, generator=generator) < 0.2
                inputs = batch.masked_fill(mask, 0.0)
            reconstruction, _ = model(inputs)
            loss = nn.functional.mse_loss(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def _reconstruction_outputs(model, values):
    model.eval()
    with torch.no_grad():
        reconstruction, embeddings = model(torch.as_tensor(values, dtype=torch.float32))
    errors = ((reconstruction.numpy() - values) ** 2).mean(axis=(1, 2))
    return errors, embeddings.numpy()


def run_learned_embeddings(project_root, dataset_key="drive_day_2026", epochs=20, **_):
    started_at = time.perf_counter()
    torch.manual_seed(RANDOM_SEED)
    context = _build_context(project_root, dataset_key)
    model = _MaskedPredictionModel(context.windows.shape[1], context.windows.shape[2], 8)
    _train_reconstruction(model, context.windows[context.train_mask], epochs, masked=True)
    errors, embeddings = _reconstruction_outputs(model, context.windows)
    predictions = _nearest_neighbor_predictions(embeddings, context.metadata, "masked_prediction_embedding_neighbor")
    metrics = _split_metric_rows(context.metadata, errors, "masked_signal_prediction", "mean_reconstruction_mse", "none")
    metrics.extend(_split_metric_rows(context.metadata, predictions["score"], "embedding_retrieval", "mean_neighbor_cosine_similarity", "none"))
    result = _write_result(context, "04_learned_embeddings", "masked_signal_representation_learning", {"window_s": 10.0, "step_s": 2.0, "embedding_dim": 8, "mask_fraction": 0.2, "epochs": epochs, "seed": RANDOM_SEED}, predictions, pd.DataFrame(metrics), "self_supervised_unverified", started_at)
    np.save(result.output_dir / "embeddings.npy", embeddings)
    return result


def run_autoencoders(project_root, dataset_key="drive_day_2026", epochs=25, **_):
    started_at = time.perf_counter()
    torch.manual_seed(RANDOM_SEED)
    context = _build_context(project_root, dataset_key)
    model = MLPAutoencoder(context.windows.shape[1], context.windows.shape[2], 8)
    _train_reconstruction(model, context.windows[context.train_mask], epochs)
    errors, embeddings = _reconstruction_outputs(model, context.windows)
    flat = context.windows.reshape(len(context.windows), -1)
    pca = PCA(n_components=8, random_state=RANDOM_SEED).fit(flat[context.train_mask])
    pca_errors = ((pca.inverse_transform(pca.transform(flat)) - flat) ** 2).mean(axis=1)
    predictions = _window_predictions(context.metadata, "autoencoder_reconstruction_error", errors, pca_reconstruction_error=pca_errors)
    metrics = _split_metric_rows(context.metadata, errors, "window_reconstruction", "autoencoder_mse", "none")
    metrics.extend(_split_metric_rows(context.metadata, pca_errors, "window_reconstruction", "pca_mse", "none"))
    result = _write_result(context, "05_autoencoders", "window_reconstruction_and_latent_neighbors", {"window_s": 10.0, "step_s": 2.0, "latent_dim": 8, "epochs": epochs, "seed": RANDOM_SEED}, predictions, pd.DataFrame(metrics), "reconstruction_score_unverified", started_at)
    np.save(result.output_dir / "embeddings.npy", embeddings)
    return result


def _augment(batch):
    scale = 0.95 + 0.10 * torch.rand((len(batch), 1, batch.shape[2]))
    noise = 0.02 * torch.randn_like(batch)
    mask = torch.rand_like(batch) < 0.05
    return (batch * scale + noise).masked_fill(mask, 0.0)


def _contrastive_loss(first, second, temperature=0.2):
    first = nn.functional.normalize(first, dim=1)
    second = nn.functional.normalize(second, dim=1)
    logits = first @ second.T / temperature
    labels = torch.arange(len(first))
    return (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels)) / 2.0


def run_contrastive_learning(project_root, dataset_key="drive_day_2026", epochs=20, **_):
    started_at = time.perf_counter()
    torch.manual_seed(RANDOM_SEED)
    context = _build_context(project_root, dataset_key)
    model = TemporalEncoder(context.windows.shape[2], 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for (batch,) in _loader(context.windows[context.train_mask]):
            loss = _contrastive_loss(model(_augment(batch)), model(_augment(batch)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    values = torch.as_tensor(context.windows, dtype=torch.float32)
    with torch.no_grad():
        embeddings = model(values).numpy()
        first_view = model(_augment(values)).numpy()
        second_view = model(_augment(values)).numpy()
    consistency = np.sum(first_view * second_view, axis=1) / np.maximum(np.linalg.norm(first_view, axis=1) * np.linalg.norm(second_view, axis=1), 1e-12)
    predictions = _nearest_neighbor_predictions(embeddings, context.metadata, "contrastive_embedding_neighbor")
    metrics = _split_metric_rows(context.metadata, consistency, "augmentation_retrieval", "mean_view_cosine_similarity", "same_window_augmentations")
    metrics.extend(_split_metric_rows(context.metadata, predictions["score"], "embedding_retrieval", "mean_neighbor_cosine_similarity", "none"))
    result = _write_result(context, "06_contrastive_learning", "self_supervised_contrastive_retrieval", {"window_s": 10.0, "step_s": 2.0, "embedding_dim": 8, "temperature": 0.2, "epochs": epochs, "seed": RANDOM_SEED}, predictions, pd.DataFrame(metrics), "same_window_augmentations_unverified", started_at)
    np.save(result.output_dir / "embeddings.npy", embeddings)
    return result


def run_sequence_models(project_root, dataset_key="drive_day_2026", epochs=20, **_):
    started_at = time.perf_counter()
    torch.manual_seed(RANDOM_SEED)
    context = _build_context(project_root, dataset_key)
    model = SequencePredictor(context.windows.shape[2], 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for (batch,) in _loader(context.windows[context.train_mask]):
            prediction = model(batch[:, :-1])
            loss = nn.functional.mse_loss(prediction, batch[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    values = torch.as_tensor(context.windows, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(values[:, :-1]).numpy()
    targets = context.windows[:, 1:]
    errors = ((prediction - targets) ** 2).mean(axis=(1, 2))
    persistence = ((context.windows[:, :-1] - targets) ** 2).mean(axis=(1, 2))
    predictions = _window_predictions(context.metadata, "sequence_prediction_error", errors, persistence_error=persistence)
    metrics = _split_metric_rows(context.metadata, errors, "next_sample_prediction", "gru_mse", "none")
    metrics.extend(_split_metric_rows(context.metadata, persistence, "next_sample_prediction", "persistence_mse", "persistence_baseline"))
    return _write_result(context, "07_sequence_models", "next_sample_prediction", {"window_s": 10.0, "step_s": 2.0, "hidden_dim": 16, "layers": 1, "epochs": epochs, "seed": RANDOM_SEED}, predictions, pd.DataFrame(metrics), "prediction_error_unverified", started_at)


def run_unsupervised_learning(project_root, dataset_key="drive_day_2026", **_):
    started_at = time.perf_counter()
    context = _build_context(project_root, dataset_key)
    train_features = context.features[context.train_mask]
    evaluations = []
    for clusters in range(2, 7):
        first = KMeans(n_clusters=clusters, random_state=RANDOM_SEED, n_init=20).fit(train_features)
        second = KMeans(n_clusters=clusters, random_state=RANDOM_SEED + 1, n_init=20).fit(train_features)
        evaluations.append(
            {
                "clusters": clusters,
                "silhouette": silhouette_score(train_features, first.labels_),
                "seed_ari": adjusted_rand_score(first.labels_, second.labels_),
            }
        )
    evaluation = pd.DataFrame(evaluations)
    selected_clusters = int(evaluation.sort_values(["silhouette", "seed_ari"], ascending=False).iloc[0]["clusters"])
    model = KMeans(n_clusters=selected_clusters, random_state=RANDOM_SEED, n_init=20).fit(train_features)
    labels = model.predict(context.features)
    distances = model.transform(context.features).min(axis=1)
    predictions = _window_predictions(context.metadata, "driving_regime_cluster", -distances, cluster=labels, distance_to_centroid=distances)
    metrics = [
        {"task": "driving_regime_clustering", "split": "train", "metric": "selected_cluster_count", "value": selected_clusters, "sample_count": len(train_features), "reference_source": "internal_selection"},
        {"task": "driving_regime_clustering", "split": "train", "metric": "silhouette", "value": evaluation.loc[evaluation["clusters"] == selected_clusters, "silhouette"].iloc[0], "sample_count": len(train_features), "reference_source": "internal_only"},
        {"task": "driving_regime_clustering", "split": "train", "metric": "seed_adjusted_rand_index", "value": evaluation.loc[evaluation["clusters"] == selected_clusters, "seed_ari"].iloc[0], "sample_count": len(train_features), "reference_source": "internal_only"},
    ]
    metrics.extend(_split_metric_rows(context.metadata, distances, "driving_regime_clustering", "mean_centroid_distance", "none"))
    result = _write_result(context, "08_unsupervised_learning", "driving_regime_clustering", {"window_s": 10.0, "step_s": 2.0, "cluster_candidates": [2, 3, 4, 5, 6], "seed": RANDOM_SEED}, predictions, pd.DataFrame(metrics), "cluster_regime_not_lap", started_at)
    evaluation.to_csv(result.output_dir / "cluster_selection.csv", index=False)
    return result


EXPERIMENT_RUNNERS = {
    "01_signal_processing": run_signal_processing,
    "02_dtw": run_dtw,
    "03_correlation": run_correlation,
    "04_learned_embeddings": run_learned_embeddings,
    "05_autoencoders": run_autoencoders,
    "06_contrastive_learning": run_contrastive_learning,
    "07_sequence_models": run_sequence_models,
    "08_unsupervised_learning": run_unsupervised_learning,
}


def run_experiment(method, project_root, dataset_key="drive_day_2026", **options):
    if method not in EXPERIMENT_RUNNERS:
        raise KeyError(f"Unknown experiment: {method}")
    return EXPERIMENT_RUNNERS[method](project_root, dataset_key=dataset_key, **options)
