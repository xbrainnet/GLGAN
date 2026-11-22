import math
import unittest

import torch
from torch import nn

from config.config import Config
from models.model import GLGAN, GlobalGraphAttentionLayer, LocalGraphAttentionLayer, pagerank_centrality


class LocalGraphAttentionLayerTest(unittest.TestCase):
    def test_adds_dti_attention_and_gat_weights_before_neighbor_aggregation(self):
        features = torch.tensor([[[2.0, -1.0], [1.0, 2.0], [-3.0, 1.0]]])
        adjacency = torch.tensor([[[0, 1, 2], [0, 0, 3], [0, 0, 0]]])
        structural_scores = torch.tensor([[5.0, 6.0, 0.0], [6.0, 9.0, 0.0], [0.0, 0.0, 0.0]])
        structural_attention = (structural_scores / math.sqrt(2)).softmax(dim=-1)
        gat_weights = torch.tensor([4.0, 3.0]).softmax(dim=-1)

        for use_self_attention in (True, False):
            with self.subTest(use_self_attention=use_self_attention):
                layer = LocalGraphAttentionLayer(
                    2, 2, dropout=0.0, alpha=0.2, use_self_attention=use_self_attention, n_regions=3
                )
                with torch.no_grad():
                    layer.value_proj.weight.copy_(torch.eye(2))
                    layer.attention.copy_(torch.tensor([[1.0], [0.0], [0.0], [1.0]]))
                    if use_self_attention:
                        for projection in (layer.dti_q_proj, layer.dti_k_proj):
                            projection.weight.copy_(torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
                            projection.bias.zero_()

                nd = structural_attention if use_self_attention else torch.zeros_like(structural_attention)
                expected = torch.zeros_like(features)
                expected[0, 0] = (gat_weights + nd[0, 1:]) @ features[0, 1:]
                expected[0, 1] = (1.0 + nd[1, 2]) * features[0, 2]
                output = layer(features, adjacency)
                torch.testing.assert_close(output, nn.functional.elu(expected))

                output.sum().backward()
                for parameter in layer.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())


class GlobalGraphAttentionLayerTest(unittest.TestCase):
    def test_combines_pagerank_and_functional_scores_on_structural_edges(self):
        layer = GlobalGraphAttentionLayer(2, 2, dropout=0.0, use_self_attention=True)
        with torch.no_grad():
            for projection in (layer.q_proj, layer.k_proj, layer.value_proj):
                projection.weight.copy_(torch.eye(2))
                projection.bias.zero_()

        features = torch.tensor([[[2.0, -1.0], [1.0, 2.0], [-3.0, 1.0]]])
        adjacency = torch.tensor([[[-2.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
        scores = torch.tensor([[5.0, 0.0, -7.0], [0.0, 5.0, -1.0], [-7.0, -1.0, 10.0]])
        functional_attention = (scores / math.sqrt(2)).softmax(dim=-1)
        # One PageRank iteration with alpha=0.5 and dangling mass redistribution.
        pagerank_weights = (torch.tensor([11.0, 17.0]) / 36).softmax(dim=-1)

        for use_pagerank, neighbor_weights in ((True, pagerank_weights), (False, torch.tensor([0.5, 0.5]))):
            with self.subTest(use_pagerank=use_pagerank):
                output = layer(features, adjacency, alpha=0.5, iterations=1, use_pagerank=use_pagerank)
                expected = torch.zeros_like(features)
                expected[0, 0] = (neighbor_weights + functional_attention[0, 1:]) @ features[0, 1:]
                expected[0, 1] = (1.0 + functional_attention[1, 2]) * features[0, 2]
                torch.testing.assert_close(output, nn.functional.elu(expected))

                layer.zero_grad()
                output.sum().backward()
                for parameter in layer.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_uses_uniform_structural_weights_when_both_scores_are_disabled(self):
        layer = GlobalGraphAttentionLayer(1, 1, dropout=0.0, use_self_attention=False)
        with torch.no_grad():
            layer.value_proj.weight.fill_(1.0)
            layer.value_proj.bias.zero_()

        features = torch.tensor([[[1.0], [2.0], [4.0]]])
        adjacency = torch.tensor([[[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        output = layer(features, adjacency, alpha=0.5, iterations=1, use_pagerank=False)

        torch.testing.assert_close(output, torch.tensor([[[3.0], [1.0], [0.0]]]))

    def test_pagerank_binary_and_weighted_transitions_preserve_mass(self):
        adjacency = torch.tensor(
            [[[0, 1, 3], [0, 0, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
        )
        binary = pagerank_centrality(adjacency, alpha=0.5, iterations=1)
        weighted = pagerank_centrality(adjacency, alpha=0.5, iterations=1, weighted=True)
        torch.testing.assert_close(binary[0], torch.tensor([8.0, 11.0, 17.0]) / 36)
        torch.testing.assert_close(weighted[0], torch.tensor([16.0, 19.0, 37.0]) / 72)
        torch.testing.assert_close(binary, pagerank_centrality(adjacency > 0, alpha=0.5, iterations=1))
        torch.testing.assert_close(binary[1], torch.full((3,), 1 / 3))
        torch.testing.assert_close(weighted.sum(dim=-1), torch.ones(2))


class GLGANModelTest(unittest.TestCase):
    def test_glgan_forward_returns_logits_and_contrastive_loss(self):
        config = Config(n_regions=6, hidden_dim=4, n_classes=2, dropout=0.0)
        model = GLGAN(config)
        fmri = torch.randn(3, 6, 6)
        dti = torch.eye(6).repeat(3, 1, 1)
        dti[:, 0, 1] = 0.5
        dti[:, -1, :] = 0.0

        logits, contrastive_loss = model(fmri, dti)

        self.assertEqual(logits.shape, (3, 2))
        self.assertEqual(model.classifier[0].in_features, 24)
        self.assertEqual(contrastive_loss.ndim, 0)
        self.assertGreater(contrastive_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(contrastive_loss))

        (logits.mean() + contrastive_loss).backward()
        for parameter in (
            model.local_layer.dti_q_proj.weight,
            model.global_layer.q_proj.weight,
            model.projector[0].weight,
            model.roi_attention[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_fusion_uses_both_match_directions_and_roi_attention(self):
        model = GLGAN(Config(n_regions=3, hidden_dim=2))
        model.projector = nn.Identity()
        with torch.no_grad():
            model.roi_attention[0].weight.copy_(torch.eye(2))
            model.roi_attention[0].bias.zero_()
            model.roi_attention[2].weight.copy_(torch.tensor([[1.0, 0.0]]))

        z_global = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]])
        z_local = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, -1.0]]])
        z_final, contrastive_loss = model._fuse_branches(z_global, z_local)

        scores = torch.tensor([[2.5, 2.5, 0.0], [0.0, 0.0, -2.5], [0.0, 0.0, 2.5]])
        p_global = scores.softmax(dim=1).diagonal()
        p_local = scores.softmax(dim=0).diagonal()
        delta = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]).tanh()
        attention = delta.softmax(dim=1)
        expected_features = (attention[:, 0] + p_global).unsqueeze(-1) * z_global
        expected_features += (attention[:, 1] + p_local).unsqueeze(-1) * z_local

        torch.testing.assert_close(z_final, expected_features)
        torch.testing.assert_close(contrastive_loss, -((p_global + p_local).mean() / 2).log())

    def test_contrastive_loss_averages_subject_losses_over_the_batch(self):
        model = GLGAN(Config(n_regions=2, hidden_dim=2))
        model.projector = nn.Identity()
        z_global = torch.eye(2).repeat(2, 1, 1)
        z_local = torch.stack((z_global[0], z_global[1].flip(dims=(0,))))

        _, contrastive_loss = model._fuse_branches(z_global, z_local)

        # The two subjects have aligned and reversed ROI matches, respectively.
        match_probabilities = (torch.tensor([1.0, -1.0]) / model.tau).sigmoid()
        expected_loss = -match_probabilities.log().mean()
        torch.testing.assert_close(contrastive_loss, expected_loss)

    def test_contrastive_loss_clamps_zero_probability_before_log(self):
        model = GLGAN(Config(n_regions=2, hidden_dim=2, tau=1e-3))
        model.projector = nn.Identity()
        z_global = torch.eye(2).unsqueeze(0)
        z_local = z_global.flip(dims=(1,))

        _, contrastive_loss = model._fuse_branches(z_global, z_local)

        torch.testing.assert_close(contrastive_loss, torch.tensor(-math.log(1e-8)))

    def test_glgan_can_use_either_branch_alone(self):
        config = Config(n_regions=6, hidden_dim=4, n_classes=2, dropout=0.0)
        fmri = torch.randn(2, 6, 6)
        dti = torch.eye(6).repeat(2, 1, 1)

        for disabled_branch in ("use_global_branch", "use_local_branch"):
            with self.subTest(disabled_branch=disabled_branch):
                model = GLGAN(config.with_updates(**{disabled_branch: False}))
                logits, contrastive_loss = model(fmri, dti)

                self.assertEqual(logits.shape, (2, 2))
                self.assertEqual(model.classifier[0].in_features, 24)
                self.assertEqual(contrastive_loss.ndim, 0)
                self.assertEqual(contrastive_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
