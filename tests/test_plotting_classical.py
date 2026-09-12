import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")


class ClassicalPlottingTests(unittest.TestCase):
    def test_dtw_diagnostics_render_five_honest_figures(self):
        from lap_estimation.plotting_classical import render_dtw_diagnostics

        predictions = pd.DataFrame(
            [
                {
                    "recording_id": "moving-a",
                    "predicted_period_s": 20.0,
                    "matched_distance": 1.0,
                    "control_distance": 2.0,
                    "motion_status": "moving",
                },
                {
                    "recording_id": "moving-a",
                    "predicted_period_s": 20.0,
                    "matched_distance": 1.5,
                    "control_distance": 1.0,
                    "motion_status": "moving",
                },
                {
                    "recording_id": "control-b",
                    "predicted_period_s": 35.0,
                    "matched_distance": 2.5,
                    "control_distance": 2.0,
                    "motion_status": "low_dynamics_control_candidate",
                },
            ]
        )
        manifest = {"split_assignments": {"moving-a": "train", "control-b": "test"}}

        with tempfile.TemporaryDirectory() as directory:
            figures = render_dtw_diagnostics(predictions, manifest, directory)

            self.assertEqual(
                set(figures),
                {
                    "dtw_distance_scatter",
                    "dtw_margin_distribution",
                    "dtw_period_margin",
                    "dtw_recording_match_rate",
                    "dtw_paired_distance_ecdf",
                },
            )
            self.assertTrue(all((Path(directory) / f"{name}.png").exists() for name in figures))
            self.assertTrue(all("unverified" in figure._suptitle.get_text().lower() for figure in figures.values()))

    def test_correlation_diagnostics_use_cross_correlation_and_seconds(self):
        from lap_estimation.plotting_classical import render_correlation_diagnostics

        recurrence = pd.DataFrame(
            [
                {"recording_id": "a", "channel": "speed", "lag_s": 20.0, "correlation": 0.8},
                {"recording_id": "a", "channel": "current", "lag_s": 21.0, "correlation": 0.6},
                {"recording_id": "b", "channel": "speed", "lag_s": 40.0, "correlation": -0.4},
                {"recording_id": "b", "channel": "current", "lag_s": 38.0, "correlation": 0.5},
            ]
        )
        cross_correlation = pd.DataFrame(
            [
                {"recording_id": "a", "split": "train", "first_channel": "speed", "second_channel": "current", "lag_samples": 2.0, "correlation": 0.9},
                {"recording_id": "b", "split": "test", "first_channel": "speed", "second_channel": "current", "lag_samples": -4.0, "correlation": -0.7},
            ]
        )
        manifest = {
            "split_assignments": {"a": "train", "b": "test"},
            "data_provenance": {"native_sample_rate_hz": 20.0},
        }

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            recurrence_path = artifact / "predictions.csv"
            manifest_path = artifact / "manifest.json"
            recurrence.to_csv(recurrence_path, index=False)
            cross_correlation.to_csv(artifact / "cross_correlation.csv", index=False)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            figures = render_correlation_diagnostics(recurrence_path, manifest_path, artifact)

            self.assertEqual(
                set(figures),
                {
                    "correlation_period_by_channel",
                    "correlation_lag_strength",
                    "correlation_recording_channel_heatmap",
                    "correlation_pair_heatmap",
                    "correlation_signed_lag_distribution",
                },
            )
            self.assertTrue(all((artifact / f"{name}.png").exists() for name in figures))
            signed_lag_axis = figures["correlation_signed_lag_distribution"].axes[0]
            self.assertIn("seconds", signed_lag_axis.get_xlabel().lower())
            self.assertTrue(all("unverified" in figure._suptitle.get_text().lower() for figure in figures.values()))


if __name__ == "__main__":
    unittest.main()
