import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch


SCHEMA_VERSION = "1.0"
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "method",
    "task",
    "config",
    "source_hashes",
    "split_assignments",
    "label_provenance",
    "status",
    "runtime_s",
    "package_versions",
    "data_provenance",
}
REQUIRED_PREDICTION_COLUMNS = {
    "recording_id",
    "segment_id",
    "start_s",
    "end_s",
    "prediction_kind",
    "score",
    "label_status",
}
REQUIRED_METRIC_COLUMNS = {
    "task",
    "split",
    "metric",
    "value",
    "sample_count",
    "reference_source",
}


def package_versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }


def write_experiment_artifacts(
    output_root,
    method,
    task,
    config,
    source_hashes,
    split_assignments,
    predictions,
    metrics,
    label_provenance,
    data_provenance,
    status="complete",
    started_at=None,
):
    output_dir = Path(output_root) / method
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(predictions)
    metrics = pd.DataFrame(metrics)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "task": task,
        "config": config,
        "source_hashes": list(source_hashes),
        "split_assignments": dict(split_assignments),
        "label_provenance": label_provenance,
        "status": status,
        "runtime_s": None if started_at is None else round(time.perf_counter() - started_at, 3),
        "package_versions": package_versions(),
        "data_provenance": data_provenance,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def load_experiment_metrics(output_root):
    records = []
    for path in sorted(Path(output_root).glob("*/metrics.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "method", path.parent.name)
        records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def validate_experiment_artifacts(output_dir):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output_dir / "predictions.csv")
    metrics = pd.read_csv(output_dir / "metrics.csv")
    missing_manifest = REQUIRED_MANIFEST_FIELDS - manifest.keys()
    missing_predictions = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    missing_metrics = REQUIRED_METRIC_COLUMNS - set(metrics.columns)
    if missing_manifest or missing_predictions or missing_metrics:
        raise ValueError(
            f"Artifact contract failure: manifest={sorted(missing_manifest)}, "
            f"predictions={sorted(missing_predictions)}, metrics={sorted(missing_metrics)}"
        )
    if (predictions["end_s"] < predictions["start_s"]).any():
        raise ValueError("Prediction end_s precedes start_s")
    return {
        "method": manifest["method"],
        "prediction_count": len(predictions),
        "metric_count": len(metrics),
    }
