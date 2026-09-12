import tempfile
import unittest
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


matplotlib.use("Agg")


class PlottingFusionTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_unsupervised_renderer_returns_named_figures_and_excludes_recording_boundaries(self):
        from lap_estimation.plotting_fusion import render_unsupervised_diagnostics

        predictions = pd.DataFrame(
            {
                "recording_id": ["recording-a"] * 3 + ["recording-b"] * 3,
                "start_s": [0.0, 2.0, 4.0, 0.0, 2.0, 4.0],
                "end_s": [10.0, 12.0, 14.0, 10.0, 12.0, 14.0],
                "cluster": [0, 1, 0, 1, 0, 1],
                "distance_to_centroid": [0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            }
        )
        selection = pd.DataFrame(
            {
                "clusters": [2, 3, 4],
                "silhouette": [0.4, 0.6, 0.5],
                "seed_ari": [0.9, 1.0, 0.8],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection.to_csv(root / "cluster_selection.csv", index=False)
            figures = render_unsupervised_diagnostics(
                predictions,
                {"config": {"dataset_key": "synthetic"}},
                root,
            )
            saved = {name: (root / f"{name}.png").exists() for name in figures}

        expected = {
            "cluster_selection",
            "occupancy_heatmap",
            "transition_heatmap",
            "distance_distributions",
            "regime_timelines",
            "recording_profiles",
        }
        self.assertEqual(set(figures), expected)
        self.assertTrue(all(isinstance(figure, Figure) for figure in figures.values()))
        self.assertTrue(all(saved.values()))
        self.assertIn("synthetic", figures["cluster_selection"].axes[0].get_title())
        transition_matrix = figures["transition_heatmap"].axes[0].collections[0].get_array()
        self.assertEqual(float(transition_matrix[0, 0]), 0.0)
        self.assertEqual(float(transition_matrix[1, 1]), 0.0)

    def test_fusion_renderer_returns_support_coverage_and_timeline_figures(self):
        from lap_estimation.plotting_fusion import render_fusion_diagnostics

        recurrence = pd.DataFrame(
            {
                "recording_id": ["a", "b", "c"],
                "signal_period_s": [20.0, 40.0, 30.0],
                "correlation_period_s": [21.0, 80.0, 29.0],
                "period_agreement": [0.95, 0.5, 0.97],
                "signal_support": [0.9, 0.2, 0.7],
                "dtw_support": [0.8, 0.1, 0.6],
                "correlation_support": [0.85, 0.3, 0.65],
                "period_support": [0.8, 0.2, 1.0],
                "signal_family_support": [0.85, 0.2, 0.7],
                "shape_family_support": [0.8, 0.1, 0.6],
                "fused_support": [0.825, 0.15, 0.65],
                "family_dispersion": [0.025, 0.05, 0.05],
                "method_count": [3, 2, 3],
                "evidence_status": ["review", "weak", "weak"],
            }
        )
        consensus = pd.DataFrame(
            {
                "recording_id": ["a", "a", "b", "b", "c", "c"],
                "start_s": [0.0, 2.0, 0.0, 2.0, 0.0, 2.0],
                "end_s": [10.0, 12.0, 10.0, 12.0, 10.0, 12.0],
                "split": ["train", "train", "validation", "validation", "test", "test"],
                "Embeddings_support": [0.8, 0.7, 0.4, 0.5, 0.2, 0.3],
                "Autoencoder_support": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                "consensus_support": [0.75, 0.65, 0.45, 0.45, 0.25, 0.25],
                "support_iqr": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                "method_count": [2, 2, 2, 2, 2, 2],
            }
        )
        expected = {
            "recurrence_period_agreement",
            "fused_support_dispersion",
            "recurrence_coverage",
            "family_component_heatmap",
            "window_consensus_distribution",
            "window_dispersion_distribution",
            "window_consensus_timelines",
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figures = render_fusion_diagnostics(
                recurrence,
                consensus,
                {"dataset": "synthetic"},
                root,
            )

            self.assertTrue(all((root / f"{name}.png").exists() for name in expected))

        self.assertEqual(set(figures), expected)
        self.assertTrue(all(isinstance(figure, Figure) for figure in figures.values()))
        self.assertGreaterEqual(len(figures["window_consensus_timelines"].axes[0].lines), 3)


if __name__ == "__main__":
    unittest.main()
