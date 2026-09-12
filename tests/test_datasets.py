import unittest
from pathlib import Path


class DatasetConfigTests(unittest.TestCase):
    def test_2026_dataset_resolves_scoped_artifact_directory(self):
        from lap_estimation.datasets import resolve_dataset

        root = Path("C:/project")

        dataset = resolve_dataset("drive_day_2026", root)

        self.assertEqual(dataset.raw_root, root / "2026_drive_data")
        self.assertEqual(dataset.artifact_root, root / "artifacts" / "drive_day_2026")
        self.assertEqual(dataset.native_sample_rate_hz, 10.0)

    def test_2023_dataset_is_enabled_with_inferred_clock_and_inertial_channels(self):
        from lap_estimation.datasets import resolve_dataset

        root = Path("C:/project")

        dataset = resolve_dataset("aim_2023", root)

        self.assertTrue(dataset.enabled)
        self.assertEqual(dataset.artifact_root, root / "artifacts" / "aim_2023")
        self.assertEqual(dataset.raw_root, root / "2023_drive_data")
        self.assertEqual(dataset.native_sample_rate_hz, 20.0)
        self.assertEqual(dataset.clock_provenance, "inferred_uniform_from_metadata")
        self.assertIn("InlineAcc", dataset.analysis_channels)
        self.assertEqual(dataset.motion_policy, "training_low_dynamics_quantile")


if __name__ == "__main__":
    unittest.main()
