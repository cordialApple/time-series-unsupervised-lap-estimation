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

    def test_cohort_summary_exposes_calibration_gaps(self):
        from lap_estimation.reporting import build_cohort_summary

        summary = build_cohort_summary(self.root, "aim_2023")

        self.assertEqual(summary["recurrence_eligible_recordings"], 10)
        self.assertEqual(summary["signal_control_count"], 2)
        self.assertEqual(summary["signal_non_control_count"], 8)
        self.assertGreater(summary["signal_control_flag_rate"], summary["signal_review_candidate_rate"])
        self.assertGreaterEqual(summary["signal_ceiling"]["minimum_cycle_count"], 2.0)
        self.assertEqual(summary["signal_ceiling"]["recordings"], 10)
        self.assertEqual(summary["signal_ceiling"]["near_ceiling"], 3)
        agreement = summary["period_agreement"]
        self.assertAlmostEqual(agreement["modal_period_s"], 45.0, delta=2.0)
        self.assertEqual(agreement["matching_recordings"], 5)
        self.assertEqual(agreement["half_period_recordings"], 2)
        self.assertFalse(agreement["near_search_floor"])
        interval = summary["dtw_test_interval"]
        self.assertAlmostEqual(interval["rate"], summary["dtw_match_rate_test"])
        self.assertLessEqual(interval["low"], interval["rate"])
        self.assertGreaterEqual(interval["high"], interval["rate"])
        self.assertEqual(interval["pairs"], 20)
        self.assertEqual(interval["recordings"], 2)
        self.assertTrue(interval["includes_chance"])
        self.assertGreater(summary["contrastive_effective_rank"], summary["embedding_effective_rank"])
        self.assertGreater(summary["autoencoder_effective_rank"], summary["embedding_effective_rank"])
        composition = summary["cluster_composition"]
        self.assertGreater(composition["largest_cluster_window_share"], 0.5)
        self.assertEqual(composition["minority_cluster_recordings"], 7)
        self.assertGreater(composition["largest_cluster_control_windows"], 0)

    def test_cohort_summary_reports_calibrated_physics_detector(self):
        from lap_estimation.reporting import build_cohort_summary

        physics_2026 = build_cohort_summary(self.root, "drive_day_2026")["physics"]
        physics_2023 = build_cohort_summary(self.root, "aim_2023")["physics"]

        self.assertEqual(list(physics_2026["domains"]), ["distance", "time"])
        self.assertEqual(list(physics_2023["domains"]), ["time"])
        for physics in (physics_2026, physics_2023):
            for result in physics["domains"].values():
                self.assertLessEqual(result["fdr_significant"], result["significant"])
                self.assertLessEqual(result["significant"], result["driving_recordings"])
                self.assertGreater(result["synthetic_detection_rate"], 0.8)
                self.assertGreater(result["excess_p_value"], 0.05)
                self.assertGreaterEqual(result["minimum_p_value"], 1.0 / (1 + physics["surrogate_count"]))
        self.assertEqual(physics_2023["domains"]["time"]["significant"], 0)

    def test_cohort_summary_reports_cross_recording_course_runs(self):
        from lap_estimation.reporting import build_cohort_summary

        course = build_cohort_summary(self.root, "aim_2023")["course_runs"]

        self.assertIsNone(build_cohort_summary(self.root, "drive_day_2026")["course_runs"])
        self.assertEqual(course["runs"], 8)
        self.assertEqual(course["recordings"], 7)
        self.assertEqual(course["pairs"], 28)
        self.assertGreater(course["minimum_pair_correlation"], course["null_maximum_correlation"])
        self.assertEqual(course["pair_significant_rate"], 1.0)
        self.assertLessEqual(course["minimum_lap_time_s"], course["median_lap_time_s"])
        self.assertLessEqual(course["median_lap_time_s"], course["maximum_lap_time_s"])

    def test_2026_period_agreement_is_flagged_at_search_floor(self):
        from lap_estimation.reporting import build_cohort_summary

        summary = build_cohort_summary(self.root, "drive_day_2026")

        self.assertTrue(summary["period_agreement"]["near_search_floor"])
        self.assertGreater(summary["signal_ceiling"]["share"], 0.5)
        self.assertGreater(summary["cluster_composition"]["largest_cluster_window_share"], 0.9)
        self.assertTrue(summary["dtw_test_interval"]["includes_chance"])

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
        statuses = {finding["id"]: finding["status"] for finding in bundle["findings"]}
        self.assertEqual(statuses["recurrence-review-set"], "caution")
        self.assertEqual(statuses["stable-operating-regimes"], "caution")
        self.assertEqual(statuses["signal-control-separation"], "negative")
        self.assertIn("cross-recording-period", statuses)
        self.assertEqual(statuses["physics-null-calibration"], "negative")
        self.assertEqual(statuses["course-runs"], "supported")
        self.assertIn("signal-search-ceiling", statuses)
        self.assertTrue(any("null calibration" in item for item in bundle["limitations"]))
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
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 8)
            index = (output / "README.md").read_text(encoding="utf-8")
            findings = (output / "findings.md").read_text(encoding="utf-8")
            self.assertLess(index.index("## Immediate impact"), index.index("## Evidence at a glance"))
            self.assertIn("Label 7 top-ranked recordings first", index)
            self.assertIn("figures/2023-course-runs.png", index)
            self.assertIn("8 driving runs segmented from 7 recordings", index)
            self.assertIn("by construction", index)
            self.assertIn("includes 50% chance", index)
            self.assertIn("long enough for recurrence evidence", index)
            self.assertNotIn("worth labeling", index)
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
        self.assertIn("seven top-ranked recordings", readme)
        self.assertIn("reports/figures/2023-course-runs.png", readme)
        self.assertIn("Eight driving runs", readme)
        self.assertNotIn("We found", readme)
        self.assertIn("proportional to distance", readme)
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
            "dtw_test_interval": {"rate": 0.6, "low": 0.55, "high": 0.65, "pairs": 12, "recordings": 4, "includes_chance": False},
            "recurrence_eligible_recordings": 4,
            "signal_review_candidate_rate": 0.5,
            "signal_non_control_count": 2,
            "signal_control_flag_rate": 0.0,
            "signal_control_count": 1,
            "period_agreement": {"recordings": 2, "modal_period_s": 30.0, "matching_recordings": 2, "half_period_recordings": 0, "near_search_floor": False},
            "cluster_composition": {"largest_cluster_window_share": 0.4},
            "physics": {
                "primary_domain": "time",
                "alpha": 0.05,
                "surrogate_count": 9,
                "domains": {"time": {"driving_recordings": 4, "significant": 1, "fdr_significant": 0, "at_search_ceiling": 1, "minimum_p_value": 0.02, "synthetic_detection_rate": 0.9, "synthetic_false_positive_rate": 0.25, "synthetic_trials": 8}},
            },
            "embedding_effective_rank": 2.25,
            "contrastive_effective_rank": 3.5,
            "autoencoder_effective_rank": 4.5,
            "selected_cluster_count": 4,
            "interpolation": {"long_gap": {"duration_s": 0.25}},
        }
        bundle = {
            "cohorts": {"drive_day_2026": cohort, "aim_2023": cohort},
            "findings": [{"status": "negative"}, {"status": "caution"}],
        }

        index = _render_index(bundle)

        self.assertIn("4.0 logged minutes across 10 recordings; 8 are long enough", index)
        self.assertIn("1 negative result comes", index)
        self.assertIn("Label 6 top-ranked recordings first", index)
        self.assertIn("8 evidence-covered recordings from 10 total", index)
        self.assertIn("masked 2.25, contrastive 3.50, autoencoder 4.50 of 8", index)
        self.assertIn("excludes 50% chance", index)
        self.assertIn("Time domain: 1 of 4 driving recordings reach p < 0.05", index)
        self.assertIn("flags 25% of unrelated segments over 8 trials", index)
        self.assertIn("40% and 40% of windows", index)
        self.assertIn("0.25 s gaps", index)


if __name__ == "__main__":
    unittest.main()
