import unittest
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")


class LearnedPlottingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        rows = 36
        self.embeddings = rng.normal(size=(rows, 8))
        self.neighbor_predictions = pd.DataFrame(
            {
                "recording_id": np.repeat(["a", "b", "c"], 12),
                "start_s": np.tile(np.arange(12) * 2.0, 3),
                "end_s": np.tile(np.arange(12) * 2.0 + 10.0, 3),
                "score": np.linspace(0.82, 0.999, rows),
                "neighbor_recording_id": np.roll(np.repeat(["a", "b", "c"], 12), 3),
                "neighbor_start_s": np.tile(np.arange(12)[::-1] * 2.0, 3),
                "temporal_separation_s": np.linspace(2.0, 120.0, rows),
            }
        )
        self.autoencoder_predictions = self.neighbor_predictions[
            ["recording_id", "start_s", "end_s"]
        ].assign(
            score=np.linspace(0.08, 0.7, rows),
            pca_reconstruction_error=np.linspace(0.12, 0.55, rows)[::-1],
        )
        self.sequence_predictions = self.neighbor_predictions[
            ["recording_id", "start_s", "end_s"]
        ].assign(
            score=np.linspace(0.05, 0.8, rows),
            persistence_error=np.linspace(0.1, 0.65, rows)[::-1],
        )
        self.directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.directory.name)
        np.save(self.output_dir / "embeddings.npy", self.embeddings)
        self.manifest = {"config": {"dataset_key": "synthetic"}}

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")
        self.directory.cleanup()

    def assert_figure_collection(self, figures, expected_names):
        from matplotlib.figure import Figure

        self.assertEqual(set(figures), set(expected_names))
        self.assertTrue(all(isinstance(figure, Figure) for figure in figures.values()))
        self.assertTrue(all(figure.axes for figure in figures.values()))

    def test_learned_embedding_diagnostics_expose_five_named_figures(self):
        from lap_estimation.plotting_learned import render_embedding_diagnostics

        figures = render_embedding_diagnostics(
            self.neighbor_predictions, self.manifest, self.output_dir
        )

        self.assert_figure_collection(
            figures,
            {
                "latent_pca",
                "similarity_distribution",
                "similarity_vs_separation",
                "separation_distribution",
                "latent_spectrum",
            },
        )
        self.assertIn("synthetic", figures["latent_pca"].axes[0].get_title())

    def test_autoencoder_diagnostics_compare_model_and_pca(self):
        from lap_estimation.plotting_learned import render_autoencoder_diagnostics

        figures = render_autoencoder_diagnostics(
            self.autoencoder_predictions, self.manifest, self.output_dir
        )

        self.assert_figure_collection(
            figures,
            {
                "latent_pca",
                "error_distribution",
                "autoencoder_vs_pca",
                "error_delta",
                "error_timeline",
            },
        )

    def test_contrastive_diagnostics_show_neighbor_structure(self):
        from lap_estimation.plotting_learned import render_contrastive_diagnostics

        figures = render_contrastive_diagnostics(
            self.neighbor_predictions, self.manifest, self.output_dir
        )

        self.assert_figure_collection(
            figures,
            {
                "latent_pca",
                "similarity_distribution",
                "similarity_vs_separation",
                "cross_recording_neighbors",
                "latent_spectrum",
            },
        )

    def test_sequence_diagnostics_compare_gru_and_persistence(self):
        from lap_estimation.plotting_learned import render_sequence_diagnostics

        figures = render_sequence_diagnostics(
            self.sequence_predictions, self.manifest, self.output_dir
        )

        self.assert_figure_collection(
            figures,
            {
                "error_distribution",
                "gru_vs_persistence",
                "excess_error_distribution",
                "error_timeline",
                "relative_error_by_recording",
            },
        )

    def test_public_plot_counts_are_exposed(self):
        from lap_estimation.plotting_learned import LEARNED_METHOD_PLOT_COUNTS

        self.assertEqual(
            LEARNED_METHOD_PLOT_COUNTS,
            {
                "04_learned_embeddings": 5,
                "05_autoencoders": 5,
                "06_contrastive_learning": 5,
                "07_sequence_models": 5,
            },
        )


if __name__ == "__main__":
    unittest.main()
