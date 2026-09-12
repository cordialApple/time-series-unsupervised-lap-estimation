import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class IntegrationTests(unittest.TestCase):
    def test_interpolation_reconstructs_interior_targets_and_rejects_extrapolation(self):
        from lap_estimation.integration import interpolate_at_times

        time = np.array([0.0, 1.0, 2.0, 3.0])
        values = 2.0 * time + 1.0

        reconstructed = interpolate_at_times(time, values, np.array([0.5, 1.5, 2.5]), "linear")

        np.testing.assert_allclose(reconstructed, [2.0, 4.0, 6.0])
        with self.assertRaisesRegex(ValueError, "extrapolation"):
            interpolate_at_times(time, values, np.array([-0.1]), "pchip")

    def test_interpolation_benchmark_reports_gap_and_normalized_error(self):
        from lap_estimation.integration import benchmark_interpolation

        time = np.arange(30, dtype=float) * 0.1
        frame = pd.DataFrame({"elapsed_s": time, "signal": np.sin(time)})

        result = benchmark_interpolation(
            frame,
            ["signal"],
            methods=("linear", "pchip"),
            gap_samples=(1, 3),
        )

        self.assertEqual(set(result["method"]), {"linear", "pchip"})
        self.assertEqual(set(result["gap_samples"]), {1, 3})
        self.assertTrue(np.isfinite(result["normalized_rmse"]).all())
        self.assertTrue((result["heldout_count"] > 0).all())

    def test_physical_integrals_use_elapsed_seconds(self):
        from lap_estimation.integration import build_integrated_features

        time = np.array([0.0, 1.0, 2.0])
        electric = pd.DataFrame(
            {
                "elapsed_s": time,
                "Motor RPM [Rpm]": [60.0, 60.0, 60.0],
                "MC Volts [V]": [10.0, 10.0, 10.0],
                "DC Current [A]": [2.0, 2.0, 2.0],
            }
        )
        inertial = pd.DataFrame(
            {
                "elapsed_s": time,
                "YawRate": [10.0, 10.0, 10.0],
                "InlineAcc": [1.0, 1.0, 1.0],
                "LateralAcc": [0.5, 0.5, 0.5],
            }
        )

        electric_result = build_integrated_features(electric, "drive_day_2026")
        inertial_result = build_integrated_features(inertial, "aim_2023")

        self.assertAlmostEqual(electric_result["shaft_revolutions"].iloc[-1], 2.0)
        self.assertAlmostEqual(electric_result["electrical_energy_wh"].iloc[-1], 40.0 / 3600.0)
        self.assertAlmostEqual(inertial_result["relative_heading_deg"].iloc[-1], 20.0)
        self.assertAlmostEqual(inertial_result["longitudinal_delta_v_m_s"].iloc[-1], 19.6133, places=4)

    @staticmethod
    def _write_predictions(root, method, rows, split_assignments=None):
        output = Path(root) / method
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
        if split_assignments is not None:
            (output / "manifest.json").write_text(
                json.dumps({"split_assignments": split_assignments}),
                encoding="utf-8",
            )

    def test_recurrence_fusion_keeps_dependencies_and_missing_evidence_visible(self):
        from lap_estimation.integration import build_recurrence_fusion

        with tempfile.TemporaryDirectory() as directory:
            self._write_predictions(
                directory,
                "01_signal_processing",
                [
                    {"recording_id": "a", "score": 0.9, "predicted_period_s": 20.0},
                    {"recording_id": "b", "score": 0.2, "predicted_period_s": 40.0},
                    {"recording_id": "c", "score": 0.5, "predicted_period_s": 30.0},
                ],
            )
            self._write_predictions(
                directory,
                "02_dtw",
                [
                    {"recording_id": "a", "score": 0.8, "predicted_period_s": 20.0},
                    {"recording_id": "b", "score": -0.3, "predicted_period_s": 40.0},
                ],
            )
            self._write_predictions(
                directory,
                "03_correlation",
                [
                    {"recording_id": "a", "score": 0.8, "lag_s": 21.0},
                    {"recording_id": "b", "score": 0.2, "lag_s": 80.0},
                ],
            )

            result = build_recurrence_fusion(directory).set_index("recording_id")

        self.assertGreater(result.loc["a", "fused_support"], result.loc["b", "fused_support"])
        self.assertGreater(result.loc["a", "period_agreement"], result.loc["b", "period_agreement"])
        self.assertEqual(result.loc["c", "evidence_status"], "insufficient_evidence")
        self.assertEqual(result.loc["a", "dependency_note"], "dtw_uses_signal_period_teacher")

    def test_window_consensus_normalizes_against_training_split(self):
        from lap_estimation.integration import build_window_consensus

        methods = ["04_learned_embeddings", "05_autoencoders"]
        with tempfile.TemporaryDirectory() as directory:
            splits = {"train-a": "train", "test-b": "test"}
            for method, scores in zip(methods, [[0.1, 0.9], [2.0, 1.0]]):
                self._write_predictions(
                    directory,
                    method,
                    [
                        {"recording_id": "train-a", "start_s": 0.0, "end_s": 10.0, "score": scores[0]},
                        {"recording_id": "test-b", "start_s": 0.0, "end_s": 10.0, "score": scores[1]},
                    ],
                    splits,
                )

            result = build_window_consensus(directory, methods=methods)

        self.assertEqual(set(result["split"]), {"train", "test"})
        self.assertIn("consensus_support", result)
        self.assertIn("support_iqr", result)
        self.assertTrue(result["consensus_support"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
