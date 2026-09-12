import tempfile
import unittest
from pathlib import Path

import pandas as pd


class ComparisonTests(unittest.TestCase):
    @staticmethod
    def _write_predictions(root, method, rows):
        output = Path(root) / method
        output.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)

    def test_recording_matrix_aggregates_and_orients_method_scores(self):
        from lap_estimation.comparison import build_recording_score_matrix, rank_method_scores

        with tempfile.TemporaryDirectory() as directory:
            self._write_predictions(
                directory,
                "01_signal_processing",
                [
                    {"recording_id": "a", "start_s": 0.0, "end_s": 10.0, "score": 0.2},
                    {"recording_id": "b", "start_s": 0.0, "end_s": 10.0, "score": 0.8},
                ],
            )
            self._write_predictions(
                directory,
                "05_autoencoders",
                [
                    {"recording_id": "a", "start_s": 0.0, "end_s": 10.0, "score": 1.0},
                    {"recording_id": "b", "start_s": 0.0, "end_s": 10.0, "score": 2.0},
                ],
            )

            matrix = build_recording_score_matrix(
                directory,
                methods=["01_signal_processing", "05_autoencoders"],
            )
            ranked = rank_method_scores(matrix, ["recording_id"])

        self.assertEqual(matrix.columns.tolist(), ["recording_id", "Signal", "Autoencoder"])
        self.assertEqual(ranked.set_index("recording_id").loc["b", "Signal"], 1.0)
        self.assertEqual(ranked.set_index("recording_id").loc["a", "Autoencoder"], 1.0)

    def test_window_matrix_keeps_only_exactly_aligned_windows(self):
        from lap_estimation.comparison import build_window_score_matrix

        with tempfile.TemporaryDirectory() as directory:
            self._write_predictions(
                directory,
                "04_learned_embeddings",
                [
                    {"recording_id": "a", "start_s": 0.0, "end_s": 10.0, "score": 0.2},
                    {"recording_id": "a", "start_s": 2.0, "end_s": 12.0, "score": 0.3},
                ],
            )
            self._write_predictions(
                directory,
                "05_autoencoders",
                [
                    {"recording_id": "a", "start_s": 0.0, "end_s": 10.0, "score": 1.0},
                    {"recording_id": "a", "start_s": 4.0, "end_s": 14.0, "score": 2.0},
                ],
            )

            matrix = build_window_score_matrix(
                directory,
                methods=["04_learned_embeddings", "05_autoencoders"],
            )

        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix.loc[0, "start_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
