import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class ExperimentTests(unittest.TestCase):
    def test_seconds_to_samples_respects_native_rate(self):
        from lap_estimation.experiments import seconds_to_samples

        self.assertEqual(seconds_to_samples(1.0, 10.0), 10)
        self.assertEqual(seconds_to_samples(1.0, 20.0), 20)

    def test_window_dataset_preserves_recording_splits(self):
        from lap_estimation.datasets import resolve_dataset
        from lap_estimation.experiments import build_window_dataset

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            rows = ["Time,Motor RPM [Rpm],MC Duty Cycle [%]"]
            rows.extend(f"12:00:{index // 10:02d}.{(index % 10) * 100:03d},{index},{index % 5}" for index in range(40))
            path.write_text("\n".join(rows), encoding="utf-8")
            manifest = pd.DataFrame(
                [{"recording_id": "abc", "primary_path": str(path), "split": "test", "control_candidate": False}]
            )
            specification = resolve_dataset("drive_day_2026", Path(directory))

            dataset = build_window_dataset(
                manifest,
                specification,
                window_s=2.0,
                step_s=1.0,
                channels=["Motor RPM [Rpm]", "MC Duty Cycle [%]"],
            )

        self.assertEqual(dataset.values.shape, (3, 20, 2))
        self.assertEqual(dataset.metadata["split"].unique().tolist(), ["test"])
        self.assertEqual(dataset.metadata["recording_id"].unique().tolist(), ["abc"])

    def test_2023_windows_use_native_rate_and_dynamic_inertial_values(self):
        from lap_estimation.datasets import resolve_dataset
        from lap_estimation.experiments import build_window_dataset

        specification = resolve_dataset("aim_2023", Path.cwd())
        manifest = pd.DataFrame(
            [
                {
                    "recording_id": "aim-1",
                    "primary_path": str(specification.raw_root / "1.csv"),
                    "split": "train",
                    "control_candidate": False,
                }
            ]
        )

        dataset = build_window_dataset(manifest, specification, window_s=2.0, step_s=1.0)

        self.assertEqual(dataset.sample_rate_hz, 20.0)
        self.assertEqual(dataset.values.shape[1:], (40, 5))
        self.assertTrue(np.isfinite(dataset.values).all())
        self.assertGreater(np.abs(dataset.values).sum(), 0.0)

    def test_dtw_distance_is_zero_for_identical_sequences(self):
        from lap_estimation.experiments import dtw_distance

        sequence = np.arange(8.0)[:, None]

        distance = dtw_distance(sequence, sequence, band=2)

        self.assertAlmostEqual(distance, 0.0)

    def test_engineered_features_have_fixed_finite_shape(self):
        from lap_estimation.experiments import engineered_features

        windows = np.arange(60.0).reshape(3, 10, 2)

        features = engineered_features(windows)

        self.assertEqual(features.shape, (3, 14))
        self.assertTrue(np.isfinite(features).all())

    def test_temporal_encoder_returns_requested_embedding_size(self):
        import torch

        from lap_estimation.experiments import TemporalEncoder

        model = TemporalEncoder(channel_count=3, embedding_dim=7)
        values = torch.zeros((4, 20, 3), dtype=torch.float32)

        embeddings = model(values)

        self.assertEqual(tuple(embeddings.shape), (4, 7))

    def test_runner_registry_contains_all_methods(self):
        from lap_estimation.runners import EXPERIMENT_RUNNERS

        self.assertEqual(
            set(EXPERIMENT_RUNNERS),
            {
                "01_signal_processing",
                "02_dtw",
                "03_correlation",
                "04_learned_embeddings",
                "05_autoencoders",
                "06_contrastive_learning",
                "07_sequence_models",
                "08_unsupervised_learning",
            },
        )


if __name__ == "__main__":
    unittest.main()
