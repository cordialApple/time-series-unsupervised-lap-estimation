from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import load_recording, make_windows


@dataclass
class WindowDataset:
    values: np.ndarray
    metadata: pd.DataFrame
    channels: list[str]
    sample_rate_hz: float


def seconds_to_samples(seconds, sample_rate_hz):
    if seconds <= 0 or sample_rate_hz <= 0:
        raise ValueError("Seconds and sample rate must be positive")
    return int(round(seconds * sample_rate_hz))


def build_window_dataset(manifest, specification, window_s, step_s, channels=None):
    channels = list(specification.analysis_channels if channels is None else channels)
    sample_rate_hz = specification.native_sample_rate_hz
    window_size = seconds_to_samples(window_s, sample_rate_hz)
    step_size = seconds_to_samples(step_s, sample_rate_hz)
    windows = []
    metadata = []
    for record in manifest.itertuples(index=False):
        frame = load_recording(record.primary_path, specification)
        numeric = pd.DataFrame(index=frame.index)
        for channel in channels:
            if channel in frame:
                numeric[channel] = pd.to_numeric(frame[channel], errors="coerce")
            else:
                numeric[channel] = np.nan
        numeric = numeric.interpolate(limit=max(1, round(sample_rate_hz)), limit_direction="both")
        recording_windows, starts = make_windows(numeric.to_numpy(dtype=float), window_size, step_size)
        finite = np.isfinite(recording_windows).all(axis=(1, 2))
        windows.extend(recording_windows[finite])
        for start in starts[finite]:
            metadata.append(
                {
                    "recording_id": record.recording_id,
                    "split": record.split,
                    "control_candidate": record.control_candidate,
                    "start_s": start / sample_rate_hz,
                    "end_s": (start + window_size) / sample_rate_hz,
                }
            )
    shape = (0, window_size, len(channels))
    values = np.stack(windows) if windows else np.empty(shape, dtype=float)
    if not len(values):
        raise ValueError(f"No finite windows for dataset: {specification.key}")
    return WindowDataset(values, pd.DataFrame(metadata), channels, sample_rate_hz)


def fit_channel_scaler(values):
    flat = np.asarray(values, dtype=float).reshape(-1, values.shape[-1])
    center = np.nanmedian(flat, axis=0)
    q25 = np.nanquantile(flat, 0.25, axis=0)
    q75 = np.nanquantile(flat, 0.75, axis=0)
    center = np.nan_to_num(center)
    scale = np.nan_to_num(q75 - q25, nan=1.0)
    scale[scale <= 1e-12] = 1.0
    return center, scale


def transform_windows(values, center, scale):
    transformed = (np.asarray(values, dtype=float) - center) / scale
    return np.nan_to_num(transformed)


def engineered_features(windows):
    values = np.asarray(windows, dtype=float)
    difference = np.diff(values, axis=1)
    summaries = [
        np.mean(values, axis=1),
        np.std(values, axis=1),
        np.min(values, axis=1),
        np.max(values, axis=1),
        np.quantile(values, 0.25, axis=1),
        np.quantile(values, 0.75, axis=1),
        np.std(difference, axis=1),
    ]
    return np.concatenate(summaries, axis=1)


def dtw_distance(first, second, band):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    band = max(int(band), abs(len(first) - len(second)))
    costs = np.full((len(first) + 1, len(second) + 1), np.inf)
    lengths = np.zeros((len(first) + 1, len(second) + 1), dtype=int)
    costs[0, 0] = 0.0
    for row in range(1, len(first) + 1):
        lower = max(1, row - band)
        upper = min(len(second), row + band)
        for column in range(lower, upper + 1):
            options = [
                (costs[row - 1, column], lengths[row - 1, column]),
                (costs[row, column - 1], lengths[row, column - 1]),
                (costs[row - 1, column - 1], lengths[row - 1, column - 1]),
            ]
            previous_cost, previous_length = min(options, key=lambda item: item[0])
            point_cost = float(np.linalg.norm(first[row - 1] - second[column - 1]))
            costs[row, column] = previous_cost + point_cost
            lengths[row, column] = previous_length + 1
    return costs[-1, -1] / max(1, lengths[-1, -1])


class TemporalEncoder(nn.Module):
    def __init__(self, channel_count, embedding_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channel_count, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(16, embedding_dim)

    def forward(self, values):
        hidden = self.network(values.transpose(1, 2)).squeeze(-1)
        return self.projection(hidden)


class MLPAutoencoder(nn.Module):
    def __init__(self, window_size, channel_count, latent_dim):
        super().__init__()
        input_dim = window_size * channel_count
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, input_dim))
        self.window_size = window_size
        self.channel_count = channel_count

    def forward(self, values):
        flat = values.flatten(start_dim=1)
        latent = self.encoder(flat)
        reconstruction = self.decoder(latent).reshape(-1, self.window_size, self.channel_count)
        return reconstruction, latent


class SequencePredictor(nn.Module):
    def __init__(self, channel_count, hidden_dim):
        super().__init__()
        self.recurrent = nn.GRU(channel_count, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, channel_count)

    def forward(self, values):
        hidden, _ = self.recurrent(values)
        return self.output(hidden)
