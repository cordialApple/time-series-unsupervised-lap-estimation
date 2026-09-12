import tempfile
import unittest
from pathlib import Path

import pandas as pd


class DataTests(unittest.TestCase):
    @staticmethod
    def _aim_payload(duration="0.1", second_row_width=4):
        rows = [
            '"Format","AiM CSV File"',
            '"Sample Rate","20"',
            f'"Duration","{duration}"',
            '',
            '"Time","InlineAcc","LateralAcc","ECU RPM"',
            '"s","g","g","rpm"',
            '',
            ',"0.1","0.2","1000"',
        ]
        second = ['', '0.2', '0.3', '1100'][:second_row_width]
        rows.append(','.join(f'"{value}"' if value else '' for value in second))
        return "\n".join(rows) + "\n"

    def test_aim_loader_generates_elapsed_time_and_preserves_source_metadata(self):
        from lap_estimation.data import load_recording
        from lap_estimation.datasets import resolve_dataset

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            path.write_text(self._aim_payload(), encoding="utf-8")
            dataset = resolve_dataset("aim_2023", Path(directory))

            frame = load_recording(path, dataset)

        self.assertEqual(frame["elapsed_s"].tolist(), [0.0, 0.05])
        self.assertEqual(frame["Time"].tolist(), ["", ""])
        self.assertEqual(frame.attrs["clock_provenance"], "inferred_uniform_from_metadata")
        self.assertEqual(frame.attrs["metadata"]["Sample Rate"], "20")
        self.assertEqual(frame.attrs["units"]["InlineAcc"], "g")

    def test_aim_loader_rejects_malformed_sample_width(self):
        from lap_estimation.data import load_recording
        from lap_estimation.datasets import resolve_dataset

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            path.write_text(self._aim_payload(second_row_width=3), encoding="utf-8")
            dataset = resolve_dataset("aim_2023", Path(directory))

            with self.assertRaisesRegex(ValueError, "sample row width"):
                load_recording(path, dataset)

    def test_aim_loader_rejects_duration_mismatch(self):
        from lap_estimation.data import load_recording
        from lap_estimation.datasets import resolve_dataset

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            path.write_text(self._aim_payload(duration="1.0"), encoding="utf-8")
            dataset = resolve_dataset("aim_2023", Path(directory))

            with self.assertRaisesRegex(ValueError, "duration mismatch"):
                load_recording(path, dataset)

    def test_manifest_groups_identical_files_and_keeps_aliases(self):
        from lap_estimation.data import build_recording_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "Driver A" / "run.csv"
            second = root / "Driver B" / "run.csv"
            first.parent.mkdir()
            second.parent.mkdir()
            payload = "Time,Motor RPM [Rpm]\n12:00:00.000,0\n12:00:00.100,100\n"
            first.write_text(payload, encoding="utf-8")
            second.write_text(payload, encoding="utf-8")

            manifest = build_recording_manifest(root)

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest.loc[0, "alias_count"], 2)
        self.assertTrue(manifest.loc[0, "driver_ambiguous"])

    def test_load_recording_adds_elapsed_seconds(self):
        from lap_estimation.data import load_recording

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            path.write_text(
                "Time,Motor RPM [Rpm]\n12:00:00.000,0\n12:00:00.100,100\n",
                encoding="utf-8",
            )

            frame = load_recording(path)

        self.assertEqual(frame["elapsed_s"].tolist(), [0.0, 0.1])
        self.assertTrue(frame["elapsed_s"].is_monotonic_increasing)

    def test_select_channels_rejects_constant_values(self):
        from lap_estimation.data import select_informative_channels

        frame = pd.DataFrame({"varying": [0.0, 1.0, 2.0], "constant": [1.0, 1.0, 1.0]})

        selected = select_informative_channels(frame, ["varying", "constant"])

        self.assertEqual(selected, ["varying"])

    def test_recording_summary_excludes_derived_time_from_dynamic_channels(self):
        from lap_estimation.data import summarize_recording

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            path.write_text(
                "Time,Motor RPM [Rpm],constant\n12:00:00.000,0,1\n12:00:00.100,100,1\n",
                encoding="utf-8",
            )

            summary = summarize_recording(path)

        self.assertEqual(summary["dynamic_numeric_count"], 1)

    def test_split_assignment_balances_each_control_stratum(self):
        from lap_estimation.data import assign_recording_splits

        manifest = pd.DataFrame(
            {
                "recording_id": [f"{index:064x}" for index in range(20)],
                "stationary_control": [False] * 16 + [True] * 4,
                "duration_s": list(range(20)),
            }
        )

        assigned = assign_recording_splits(manifest)
        counts = assigned.groupby(["stationary_control", "split"]).size()

        for stationary_control in [False, True]:
            for split in ["train", "validation", "test"]:
                self.assertGreater(counts.loc[(stationary_control, split)], 0)
        self.assertEqual(assigned["split"].tolist(), assign_recording_splits(manifest)["split"].tolist())

    def test_2023_manifest_records_clock_duration_and_motion_provenance(self):
        from lap_estimation.data import prepare_recording_manifest
        from lap_estimation.datasets import resolve_dataset

        dataset = resolve_dataset("aim_2023", Path.cwd())

        manifest = prepare_recording_manifest(dataset)
        first = manifest.loc[manifest["primary_path"].map(lambda value: Path(value).name == "1.csv")].iloc[0]

        self.assertEqual(len(manifest), 18)
        self.assertEqual(first["row_count"], 1800)
        self.assertAlmostEqual(first["declared_duration_s"], 90.0)
        self.assertAlmostEqual(first["sampled_elapsed_span_s"], 89.95)
        self.assertEqual(first["clock_provenance"], "inferred_uniform_from_metadata")
        self.assertIn(first["motion_status"], {"low_dynamics_candidate", "dynamic_candidate"})
        self.assertEqual(first["motion_status_source"], "training_low_dynamics_quantile")
        self.assertFalse(manifest["motion_status"].eq("stationary").any())
        self.assertGreater(first["motion_variation_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
