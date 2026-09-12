import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import STATIONARY_RPM_THRESHOLD


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_csvs(root):
    return sorted(Path(root).rglob("*.csv"))


def infer_driver(path, root):
    relative = Path(path).relative_to(root)
    if len(relative.parts) < 2:
        return None
    parent = relative.parts[-2]
    if parent.lower().endswith(" drive"):
        return parent[:-6].strip() or None
    return parent or None


def _elapsed_seconds(values):
    raw = pd.to_timedelta(values, errors="coerce").dt.total_seconds().to_numpy(dtype=float)
    adjusted = raw.copy()
    offset = 0.0
    previous = np.nan
    for index, value in enumerate(raw):
        if np.isfinite(previous) and np.isfinite(value) and value + offset < previous:
            offset += 86400.0
        if np.isfinite(value):
            adjusted[index] = value + offset
            previous = adjusted[index]
    finite = adjusted[np.isfinite(adjusted)]
    if not finite.size:
        raise ValueError("Time contains no parseable timestamps")
    return np.round(adjusted - finite[0], 9)


def _set_recording_attributes(
    frame,
    metadata,
    units,
    clock_provenance,
    sample_rate_hz,
    declared_duration_s,
):
    frame.attrs.update(
        {
            "metadata": metadata,
            "units": units,
            "clock_provenance": clock_provenance,
            "sample_rate_hz": sample_rate_hz,
            "declared_duration_s": declared_duration_s,
        }
    )
    return frame


def _load_flat_recording(path, specification=None):
    frame = pd.read_csv(path)
    if "Time" not in frame.columns:
        raise ValueError(f"Missing Time column: {path}")
    frame = frame.copy()
    frame.insert(1, "elapsed_s", _elapsed_seconds(frame["Time"]))
    return _set_recording_attributes(
        frame,
        metadata={},
        units={},
        clock_provenance="recorded_time_column",
        sample_rate_hz=None if specification is None else specification.native_sample_rate_hz,
        declared_duration_s=None,
    )


def _load_aim_recording(path, specification):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    header_index = next(
        (index for index, row in enumerate(rows) if len(row) > 2 and row[0] == "Time"),
        None,
    )
    if header_index is None:
        raise ValueError(f"Missing AiM channel header: {path}")
    metadata = {row[0]: row[1] for row in rows[:header_index] if len(row) >= 2 and row[0]}
    try:
        metadata_rate = float(metadata["Sample Rate"])
        declared_duration = float(metadata["Duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid AiM timing metadata: {path}") from error
    if not np.isfinite(metadata_rate) or metadata_rate <= 0 or not np.isfinite(declared_duration) or declared_duration <= 0:
        raise ValueError(f"Invalid AiM timing metadata: {path}")
    if not np.isclose(metadata_rate, specification.native_sample_rate_hz):
        raise ValueError(f"AiM sample rate mismatch: {path}")
    columns = rows[header_index]
    if header_index + 1 >= len(rows):
        raise ValueError(f"Missing AiM units row: {path}")
    unit_values = rows[header_index + 1]
    if len(unit_values) != len(columns):
        raise ValueError(f"AiM units row width mismatch: {path}")
    sample_rows = [row for row in rows[header_index + 2 :] if any(value.strip() for value in row)]
    if not sample_rows:
        raise ValueError(f"AiM file has no samples: {path}")
    if any(len(row) != len(columns) for row in sample_rows):
        raise ValueError(f"AiM sample row width mismatch: {path}")
    sampled_duration = len(sample_rows) / metadata_rate
    if not np.isclose(sampled_duration, declared_duration, atol=0.5 / metadata_rate):
        raise ValueError(f"AiM duration mismatch: {path}")
    frame = pd.DataFrame(sample_rows, columns=columns)
    frame.insert(1, "elapsed_s", np.arange(len(frame), dtype=float) / metadata_rate)
    return _set_recording_attributes(
        frame,
        metadata=metadata,
        units=dict(zip(columns, unit_values)),
        clock_provenance=specification.clock_provenance,
        sample_rate_hz=metadata_rate,
        declared_duration_s=declared_duration,
    )


def load_recording(path, specification=None):
    if specification is not None and specification.csv_format == "aim_csv_missing_sample_time":
        return _load_aim_recording(path, specification)
    return _load_flat_recording(path, specification)


def select_informative_channels(frame, candidates, minimum_non_null_rate=0.95):
    selected = []
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().mean() < minimum_non_null_rate:
            continue
        if values.nunique(dropna=True) > 1:
            selected.append(column)
    return selected


def summarize_recording(path, specification=None):
    frame = load_recording(path, specification)
    delta = frame["elapsed_s"].diff().dropna()
    rpm_column = "Motor RPM [Rpm]" if specification is None else specification.rpm_column
    rpm = pd.to_numeric(frame.get(rpm_column, pd.Series(np.nan, index=frame.index)), errors="coerce")
    numeric = frame.drop(columns=["Time", "elapsed_s"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    motion_columns = [] if specification is None else specification.analysis_channels
    variation = []
    for column in motion_columns:
        values = pd.to_numeric(frame.get(column), errors="coerce") if column in frame else pd.Series(dtype=float)
        if values.notna().any():
            variation.append(float(values.quantile(0.95) - values.quantile(0.05)))
    sampled_elapsed_span = float(frame["elapsed_s"].max())
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns) - 1),
        "duration_s": sampled_elapsed_span,
        "sampled_elapsed_span_s": sampled_elapsed_span,
        "declared_duration_s": frame.attrs.get("declared_duration_s"),
        "median_interval_s": float(delta.median()) if not delta.empty else np.nan,
        "max_gap_s": float(delta.max()) if not delta.empty else np.nan,
        "missing_value_count": int(numeric.isna().sum().sum()),
        "dynamic_numeric_count": int((numeric.nunique(dropna=True) > 1).sum()),
        "max_abs_rpm": float(rpm.abs().max()) if rpm.notna().any() else np.nan,
        "motion_variation_score": float(np.median(variation)) if variation else np.nan,
        "clock_provenance": frame.attrs.get("clock_provenance", "unknown"),
        "sample_rate_hz": frame.attrs.get("sample_rate_hz"),
        "source_metadata": json.dumps(frame.attrs.get("metadata", {}), sort_keys=True),
        "vertical_accel_offset_flag": bool(
            "VerticalAcc" in frame
            and abs(pd.to_numeric(frame["VerticalAcc"], errors="coerce").median()) > 2.0
        ),
    }


def build_recording_manifest(root, specification=None):
    if hasattr(root, "raw_root"):
        specification = root
        root = specification.raw_root
    root = Path(root)
    grouped = {}
    for path in discover_csvs(root):
        recording_id = file_sha256(path)
        grouped.setdefault(recording_id, []).append(path)

    records = []
    for recording_id, aliases in sorted(grouped.items()):
        drivers = sorted({driver for path in aliases if (driver := infer_driver(path, root))})
        summary = summarize_recording(aliases[0], specification)
        relative_aliases = [str(path.relative_to(root)) for path in aliases]
        max_abs_rpm = summary["max_abs_rpm"]
        rpm_control = bool(np.isfinite(max_abs_rpm) and max_abs_rpm < STATIONARY_RPM_THRESHOLD)
        records.append(
            {
                "recording_id": recording_id,
                "primary_path": str(aliases[0]),
                "aliases": json.dumps(relative_aliases),
                "alias_count": len(aliases),
                "driver_labels": json.dumps(drivers),
                "driver_ambiguous": len(drivers) > 1,
                "control_candidate": rpm_control,
                "stationary_control": rpm_control,
                **summary,
            }
        )
    return pd.DataFrame(records)


def assign_recording_splits(manifest, train_fraction=0.65, validation_fraction=0.15, stratify_column="auto"):
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must leave a positive test fraction")
    result = manifest.copy()
    result["split"] = "train"
    if stratify_column == "auto":
        stratify_column = "control_candidate" if "control_candidate" in result else "stationary_control"
    strata = [False, True] if stratify_column and stratify_column in result else [None]
    for stratum in strata:
        mask = np.ones(len(result), dtype=bool) if stratum is None else result[stratify_column].eq(stratum).to_numpy()
        group = result.loc[mask].copy()
        group["split_key"] = group["recording_id"].map(lambda value: int(value[:16], 16))
        group = group.sort_values("split_key")
        count = len(group)
        if count < 3:
            labels = ["train"] * count
        else:
            validation_count = max(1, int(round(count * validation_fraction)))
            test_count = max(1, int(round(count * (1.0 - train_fraction - validation_fraction))))
            train_count = count - validation_count - test_count
            if train_count < 1:
                train_count = 1
                test_count = count - train_count - validation_count
            labels = ["train"] * train_count + ["validation"] * validation_count + ["test"] * test_count
        result.loc[group.index, "split"] = labels
    return result


def prepare_recording_manifest(specification):
    manifest = build_recording_manifest(specification)
    if specification.motion_policy == "training_low_dynamics_quantile":
        manifest = assign_recording_splits(manifest, stratify_column=None)
        train_scores = manifest.loc[manifest["split"] == "train", "motion_variation_score"].dropna()
        if train_scores.empty:
            raise ValueError(f"No training motion scores for dataset: {specification.key}")
        threshold = float(train_scores.quantile(1.0 / 3.0))
        manifest["control_candidate"] = manifest["motion_variation_score"].le(threshold)
        manifest["stationary_control"] = False
        manifest["motion_status"] = np.where(
            manifest["control_candidate"],
            "low_dynamics_candidate",
            "dynamic_candidate",
        )
        manifest["motion_threshold"] = threshold
    else:
        manifest["motion_status"] = np.where(
            manifest["control_candidate"],
            "stationary_candidate",
            "moving_candidate",
        )
        manifest["motion_threshold"] = STATIONARY_RPM_THRESHOLD
        manifest = assign_recording_splits(manifest)
    manifest["motion_status_source"] = specification.motion_policy
    return manifest


def make_windows(values, window_size, step_size):
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window and step sizes must be positive")
    starts = np.arange(0, max(len(array) - window_size + 1, 0), step_size, dtype=int)
    if not len(starts):
        return np.empty((0, window_size, array.shape[1])), starts
    return np.stack([array[start : start + window_size] for start in starts]), starts
