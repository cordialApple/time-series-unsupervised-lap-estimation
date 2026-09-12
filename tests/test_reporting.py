import json
import tempfile
import unittest
from pathlib import Path


def _snapshot(paths, root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_cohort_summary_preserves_evidence_and_baselines(self):
        from lap_estimation.reporting import build_cohort_summary

        summary = build_cohort_summary(self.root, "drive_day_2026")

        self.assertEqual(summary["recordings"], 21)
        self.assertEqual(summary["source_paths"], 26)
        self.assertEqual(summary["fusion_recordings"], 15)
        self.assertEqual(summary["fusion_status_counts"]["review"], 4)
        self.assertAlmostEqual(summary["signal_review_candidate_rate"], 4 / 9)
        self.assertGreater(summary["autoencoder_to_pca_test_mse_ratio"], 1)
        self.assertGreater(summary["gru_to_persistence_test_mse_ratio"], 1)
        self.assertLess(summary["embedding_effective_rank"], 2)
        self.assertEqual(summary["selected_cluster_count"], 3)
        self.assertIn("methods", summary["strongest_recording_correlation"])
        self.assertIn("rho", summary["strongest_recording_correlation"])

    def test_report_bundle_keeps_claims_actions_and_limits_separate(self):
        from lap_estimation.reporting import build_report_bundle

        bundle = build_report_bundle(self.root)

        self.assertEqual(bundle["schema_version"], "1.0")
        self.assertEqual(set(bundle["cohorts"]), {"drive_day_2026", "aim_2023"})
        self.assertGreaterEqual(len(bundle["findings"]), 6)
        for finding in bundle["findings"]:
            self.assertIn(finding["status"], {"supported", "caution", "negative"})
            self.assertTrue(finding["evidence"])
            self.assertTrue(finding["interpretation"])
            self.assertTrue(finding["action"])
        self.assertTrue(bundle["limitations"])
        self.assertTrue(bundle["next_actions"])

    def test_write_reports_creates_deterministic_index_profiles_and_figures(self):
        from lap_estimation.reporting import write_reports

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            written = write_reports(self.root, output)
            first = _snapshot(written, output)
            written_again = write_reports(self.root, output)
            second = _snapshot(written_again, output)

            self.assertEqual(first, second)
            self.assertEqual(
                {path.name for path in output.glob("*.md")},
                {"README.md", "findings.md", "profile-2023.md", "profile-2026.md"},
            )
            self.assertTrue((output / "findings.json").exists())
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 6)
            index = (output / "README.md").read_text(encoding="utf-8")
            findings = (output / "findings.md").read_text(encoding="utf-8")
            self.assertLess(index.index("## Immediate impact"), index.index("## Evidence at a glance"))
            self.assertIn("Label 7 review recordings", index)
            self.assertIn("Stop promoting collapsed embeddings", index)
            self.assertIn("## Evidence at a glance", index)
            self.assertIn("## Lap findings", index)
            self.assertIn("2026 EV drive day", index)
            self.assertIn("2023 AiM event", index)
            self.assertIn("## What this data cannot tell us", findings)
            self.assertIn("## Next actions", findings)
            parsed = json.loads((output / "findings.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema_version"], "1.0")

    def test_root_readme_leads_with_impact_before_architecture(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")

        self.assertLess(readme.index("## Immediate impact"), readme.index("## Architecture"))
        self.assertIn("reports/README.md", readme)
        self.assertIn("seven priority recordings", readme)
        self.assertIn("39 recordings", readme)
        self.assertIn("## Lap findings", readme)
        self.assertIn("### 2026", readme)
        self.assertIn("### 2023", readme)
        self.assertIn("## Notebook order", readme)
        self.assertIn("## Run", readme)

    def test_generated_impact_language_uses_bundle_values(self):
        from lap_estimation.reporting import _render_index

        cohort = {
            "label": "Synthetic",
            "recordings": 5,
            "source_paths": 5,
            "duration_min": 2.0,
            "clock_provenance": "synthetic",
            "fusion_status_counts": {"review": 3},
            "fusion_recordings": 4,
            "dtw_match_rate_test": 0.6,
            "embedding_effective_rank": 2.25,
            "selected_cluster_count": 4,
            "interpolation": {"long_gap": {"duration_s": 0.25}},
        }
        bundle = {
            "cohorts": {"drive_day_2026": cohort, "aim_2023": cohort},
            "findings": [{"status": "negative"}, {"status": "caution"}],
        }

        index = _render_index(bundle)

        self.assertIn("6 recordings worth labeling", index)
        self.assertIn("1 current model variant rejected", index)
        self.assertIn("Label 6 review recordings", index)
        self.assertIn("8 evidence-covered recordings from 10 total", index)
        self.assertIn("2.25 of 8", index)
        self.assertIn("0.25 s gaps", index)
        self.assertIn("4-cluster regimes", index)


if __name__ == "__main__":
    unittest.main()
