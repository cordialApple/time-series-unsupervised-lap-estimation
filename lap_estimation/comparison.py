from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "01_signal_processing": "Signal",
    "02_dtw": "DTW",
    "03_correlation": "Correlation",
    "04_learned_embeddings": "Embeddings",
    "05_autoencoders": "Autoencoder",
    "06_contrastive_learning": "Contrastive",
    "07_sequence_models": "Sequence",
    "08_unsupervised_learning": "Unsupervised",
}

SCORE_DIRECTIONS = {
    "Signal": 1.0,
    "DTW": 1.0,
    "Correlation": 1.0,
    "Embeddings": 1.0,
    "Autoencoder": -1.0,
    "Contrastive": 1.0,
    "Sequence": -1.0,
    "Unsupervised": 1.0,
}

RECORDING_METHODS = tuple(METHOD_LABELS)
WINDOW_METHODS = RECORDING_METHODS[3:]


def _read_scores(artifact_root, method):
    path = Path(artifact_root) / method / "predictions.csv"
    columns = ["recording_id", "start_s", "end_s", "score"]
    return pd.read_csv(path, usecols=columns)


def _build_score_matrix(artifact_root, methods, keys, merge_how):
    matrix = None
    for method in methods:
        scores = _read_scores(artifact_root, method)
        scores = scores.groupby(keys, as_index=False)["score"].median()
        scores = scores.rename(columns={"score": METHOD_LABELS[method]})
        matrix = scores if matrix is None else matrix.merge(scores, on=keys, how=merge_how)
    return matrix.sort_values(keys).reset_index(drop=True)


def build_recording_score_matrix(artifact_root, methods=RECORDING_METHODS):
    return _build_score_matrix(artifact_root, methods, "recording_id", "outer")


def build_window_score_matrix(artifact_root, methods=WINDOW_METHODS):
    keys = ["recording_id", "start_s", "end_s"]
    return _build_score_matrix(artifact_root, methods, keys, "inner")


def rank_method_scores(matrix, id_columns):
    ranked = matrix.copy()
    for column in ranked.columns:
        if column in id_columns:
            continue
        ranked[column] = (ranked[column] * SCORE_DIRECTIONS[column]).rank(pct=True)
    return ranked
