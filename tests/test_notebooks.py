import ast
import json
import unittest
from pathlib import Path


EXPECTED_NOTEBOOKS = [
    "01_signal_processing.ipynb",
    "02_dtw.ipynb",
    "03_correlation.ipynb",
    "04_learned_embeddings.ipynb",
    "05_autoencoders.ipynb",
    "06_contrastive_learning.ipynb",
    "07_sequence_models.ipynb",
    "08_unsupervised_learning.ipynb",
    "09_comparison_report.ipynb",
]


class NotebookTests(unittest.TestCase):
    def test_each_dataset_has_nine_valid_notebooks_with_matching_key(self):
        notebook_root = Path("notebooks")
        self.assertEqual(list(notebook_root.glob("*.ipynb")), [])

        for dataset_key in ["drive_day_2026", "aim_2023"]:
            notebook_dir = notebook_root / dataset_key
            self.assertEqual(sorted(path.name for path in notebook_dir.glob("*.ipynb")), EXPECTED_NOTEBOOKS)
            for name in EXPECTED_NOTEBOOKS:
                notebook = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                source = "".join("".join(cell["source"]) for cell in notebook["cells"])
                self.assertIn(f'DATASET_KEY = "{dataset_key}"', source)
                self.assertNotIn(".head(20)", source)
                self.assertIn("import seaborn as sns", source)
                if name == "01_signal_processing.ipynb":
                    self.assertIn("benchmark_interpolation", source)
                    self.assertIn("build_integrated_features", source)
                    self.assertGreaterEqual(source.count("sns."), 4)
                if name[0:2] in {"02", "03", "04", "05", "06", "07", "08"}:
                    self.assertIn("render_", source)
                    self.assertIn("diagnostic_figures", source)
                if name == "09_comparison_report.ipynb":
                    self.assertIn("import seaborn as sns", source)
                    self.assertIn("sns.pairplot", source)
                    self.assertIn("build_recording_score_matrix", source)
                    self.assertIn("build_window_score_matrix", source)
                    self.assertIn("build_recurrence_fusion", source)
                    self.assertIn("build_window_consensus", source)
                    self.assertIn("render_fusion_diagnostics", source)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]), filename=f"{dataset_key}/{name}")


if __name__ == "__main__":
    unittest.main()
