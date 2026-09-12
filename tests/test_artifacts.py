import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class ArtifactTests(unittest.TestCase):
    def test_written_artifacts_pass_contract_validation(self):
        from lap_estimation.artifacts import validate_experiment_artifacts, write_experiment_artifacts

        predictions = pd.DataFrame(
            [
                {
                    "recording_id": "abc",
                    "segment_id": "candidate_1",
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "prediction_kind": "period",
                    "score": 0.5,
                    "label_status": "provisional_unverified",
                }
            ]
        )
        metrics = pd.DataFrame(
            [
                {
                    "task": "recurrence",
                    "split": "test",
                    "metric": "coverage",
                    "value": 1.0,
                    "sample_count": 1,
                    "reference_source": "none",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            output = write_experiment_artifacts(
                directory,
                method="method",
                task="task",
                config={},
                source_hashes=["abc"],
                split_assignments={"abc": "test"},
                predictions=predictions,
                metrics=metrics,
                label_provenance="unverified",
                data_provenance={
                    "dataset_key": "test",
                    "native_sample_rate_hz": 10.0,
                    "clock_provenance": "recorded_time_column",
                    "selected_channels": ["signal"],
                    "semantic_mapping": {"signal": "signal"},
                    "semantic_mapping_version": "1.0",
                    "motion_policy": "test_policy",
                },
            )

            result = validate_experiment_artifacts(output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(result["method"], "method")
        self.assertEqual(result["prediction_count"], 1)
        self.assertEqual(result["metric_count"], 1)
        self.assertEqual(manifest["data_provenance"]["dataset_key"], "test")
        self.assertEqual(manifest["data_provenance"]["selected_channels"], ["signal"])


if __name__ == "__main__":
    unittest.main()
