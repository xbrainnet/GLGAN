import numpy as np
from tempfile import TemporaryDirectory
import unittest

from config.config import Config
from training.trainer import run_cross_validation


class TrainingCheckTest(unittest.TestCase):
    def test_run_cross_validation_on_small_arrays(self):
        with TemporaryDirectory() as tmp_dir:
            rng = np.random.default_rng(7)
            fmri = rng.normal(size=(12, 6, 6)).astype("float32")
            dti = np.tile(np.eye(6, dtype="float32"), (12, 1, 1))
            labels = np.array([0, 1] * 6)

            config = Config(
                n_regions=6,
                hidden_dim=4,
                batch_size=4,
                epochs=1,
                k_folds=3,
                output_dir=tmp_dir,
                dropout=0.0,
            )

            result = run_cross_validation(fmri, dti, labels, config)

            self.assertEqual(len(result.fold_metrics), 3)
            self.assertTrue((__import__("pathlib").Path(tmp_dir) / "fold_metrics.csv").exists())
            self.assertIn("accuracy_mean", result.aggregate_metrics)


if __name__ == "__main__":
    unittest.main()
