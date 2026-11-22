import unittest

import numpy as np


class AnalysisToolsTest(unittest.TestCase):
    def test_roc_table_contains_auc_and_points(self):
        from analysis.roc_analysis import compute_roc_table

        table, auc = compute_roc_table(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))

        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)
        self.assertIn("fpr", table.dtype.names)
        self.assertIn("tpr", table.dtype.names)

    def test_tsne_embedding_has_two_columns(self):
        from analysis.tsne_analysis import compute_tsne_embedding

        features = np.arange(48, dtype=float).reshape(8, 6)
        embedding = compute_tsne_embedding(features, perplexity=3, random_state=7)

        self.assertEqual(embedding.shape, (8, 2))

    def test_parameter_count_reports_trainable_parameters(self):
        from analysis.parameter_count import count_glgan_parameters

        count = count_glgan_parameters(n_regions=6, hidden_dim=4)

        self.assertGreater(count, 0)

    def test_sensitivity_statistics_groups_metric_values(self):
        from analysis.sensitivity_statistics import compute_sensitivity_statistics
        import csv
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.csv"
            output_path = Path(tmp_dir) / "output.csv"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["learning_rate", "accuracy"])
                writer.writerows([["5e-5", "0.8"], ["5e-5", "0.9"]])

            compute_sensitivity_statistics(input_path, output_path, "learning_rate", "accuracy")

            self.assertIn("accuracy_std", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
