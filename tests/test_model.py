import torch
import unittest

from config.config import Config
from models.model import GLGAN


class GLGANModelTest(unittest.TestCase):
    def test_glgan_forward_returns_logits_and_features(self):
        config = Config(n_regions=6, hidden_dim=4, n_classes=2, dropout=0.0)
        model = GLGAN(config)
        fmri = torch.randn(3, 6, 6)
        dti = torch.eye(6).repeat(3, 1, 1)

        logits, features = model(fmri, dti)

        self.assertEqual(logits.shape, (3, 2))
        self.assertEqual(features.shape[0], 3)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(features).all())

    def test_glgan_can_disable_global_branch_for_ablation(self):
        config = Config(n_regions=6, hidden_dim=4, n_classes=2, dropout=0.0, use_global_branch=False)
        model = GLGAN(config)
        fmri = torch.randn(2, 6, 6)
        dti = torch.eye(6).repeat(2, 1, 1)

        logits, features = model(fmri, dti)

        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(features.shape, (2, 24))


if __name__ == "__main__":
    unittest.main()
