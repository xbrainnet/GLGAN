import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from analysis._elastic_net import fit_elastic_net
from analysis.elastic_net_roi_stability import summarize_roi_stability
from analysis.paired_fold_test import compare_runs
from config.config import Config
from data.abide_loader import load_abide_data
from data.data_loader import BrainConnectivityData, build_dynamic_connectivity
from models.model import BidirectionalAdjacencyRefiner, GLGAN, class_conditional_view_loss


class SupplementaryTest(unittest.TestCase):
    def test_dynamic_connectivity_averages_complete_windows(self):
        base = np.arange(60, dtype=float)
        piecewise = np.concatenate((np.zeros(40), np.arange(1, 21)))
        series = np.stack((base, -base, piecewise))[None]
        actual = build_dynamic_connectivity(series)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = sum(
                np.nan_to_num(np.abs(np.corrcoef(series[0, :, start : start + 40]))) for start in (0, 20)
            ) / 2
        np.testing.assert_allclose(actual[0], expected, rtol=1e-6)
        self.assertGreater(actual[0, 0, 2], 0)
        with self.assertRaisesRegex(ValueError, "40 time points"):
            build_dynamic_connectivity(series[..., :39])

    def test_embedding_elastic_net_uses_nonnegative_coefficients(self):
        labels = np.tile([0, 1], 10)
        features = np.column_stack((labels, -labels, np.arange(len(labels))))
        fit = fit_elastic_net(features, labels, positive=True, folds=5, seed=7)
        coefficients = fit.best_estimator_.named_steps["elasticnet"].coef_
        self.assertGreater(coefficients[0], 0)
        self.assertTrue((coefficients >= 0).all())

    def test_refiner_preserves_edges_and_normalizes_rows(self):
        refiner = BidirectionalAdjacencyRefiner(3, 2)
        x = torch.eye(3).unsqueeze(0)
        dti = torch.tensor([[[0.0, 2.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        for parameter in refiner.parameters():
            torch.nn.init.zeros_(parameter)
        actual = refiner(x, dti)
        torch.testing.assert_close(actual, torch.tensor([[[0, 2.05 / 3.1, 1.05 / 3.1], [1, 0, 0], [0, 0, 0]]]))
        torch.testing.assert_close(actual > 0, dti > 0)

    def test_class_conditional_loss_is_symmetric_and_uses_labels_only_in_training(self):
        global_view = torch.eye(3)
        local_view = torch.tensor([[1.0, 0, 0], [1.0, 1.0, 0], [0, 0, 1.0]])
        labels = torch.tensor([0, 0, 1])
        loss = class_conditional_view_loss(global_view, local_view, labels)
        self.assertGreater(float(loss), 0)
        torch.testing.assert_close(loss, class_conditional_view_loss(local_view, global_view, labels))
        torch.testing.assert_close(
            class_conditional_view_loss(global_view, local_view, torch.zeros(3, dtype=torch.long)), torch.tensor(0.0)
        )

        model = GLGAN(Config(n_regions=3, hidden_dim=3, contrastive_mode="class_conditional"))
        x = torch.stack((torch.eye(3), torch.ones(3, 3), torch.zeros(3, 3)))
        dti = torch.ones_like(x)
        with self.assertRaisesRegex(ValueError, "one label per subject"):
            model(x, dti)
        self.assertTrue(torch.isfinite(model(x, dti, labels=labels)[1]))
        model.eval()
        with torch.no_grad():
            logits, evaluation_loss = model(x, dti)
            torch.testing.assert_close(logits, model(x, dti, labels=1 - labels)[0])
        self.assertEqual(float(evaluation_loss), 0)

    def test_roi_weights_use_both_modalities_and_only_outer_training_rows(self):
        n_regions = 12
        n_subjects = 20
        rows, cols = np.triu_indices(n_regions, k=1)
        fmri = np.broadcast_to(np.arange(n_subjects)[:, None, None], (n_subjects, n_regions, n_regions))
        data = BrainConnectivityData(fmri, fmri + 100, np.arange(n_subjects) % 2, np.arange(n_subjects))
        splits = [np.delete(np.arange(n_subjects), [2 * fold, 2 * fold + 1]) for fold in range(10)]
        coefficients = np.zeros(2 * len(rows))
        coefficients[0] = -2.0  # fMRI edge (0, 1)
        coefficients[len(rows) + 1] = 3.0  # DTI edge (0, 2)
        fit = SimpleNamespace(best_estimator_=SimpleNamespace(named_steps={"elasticnet": SimpleNamespace(coef_=coefficients)}))
        with patch("analysis.elastic_net_roi_stability.fit_elastic_net", return_value=fit) as fitter:
            records = summarize_roi_stability(data, splits, [str(i) for i in range(n_regions)], inner_folds=5, seed=7)
        for train, call in zip(splits, fitter.call_args_list):
            np.testing.assert_array_equal(call.args[0][:, 0], train)
            np.testing.assert_array_equal(call.args[0][:, len(rows)], train + 100)
            np.testing.assert_array_equal(call.args[1], data.labels[train])
        self.assertEqual([record["roi"] for record in records[:3]], ["0", "2", "1"])
        np.testing.assert_allclose([record["mean_prominence"] for record in records[:3]], [1.0, 0.6, 0.4])
        self.assertEqual(records[0]["stable"], "Yes")

    def test_fold_test_matches_members_by_fold_and_rejects_different_validation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, variant = root / "baseline", root / "variant"
            indices = np.arange(20)
            for run in (baseline, variant):
                run.mkdir()
                np.savez(run / "cohort.npz", subject_ids=indices.astype(str), labels=indices % 2)
                with (run / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("fold", "accuracy"))
                    for fold in reversed(range(1, 11)):
                        remainder = np.delete(indices, [2 * fold - 2, 2 * fold - 1])
                        np.savez(
                            run / f"fold_{fold}_split.npz",
                            train_idx=remainder[4:], val_idx=remainder[:4], test_idx=indices[2 * fold - 2 : 2 * fold],
                        )
                        writer.writerow((fold, 0.6 + (fold / 100 if run == variant else 0)))
            report = compare_runs(baseline, variant)
            self.assertEqual(report["n_folds"], 10)
            self.assertAlmostEqual(report["mean_accuracy_difference_variant_minus_baseline"], 0.055)
            self.assertLess(report["p_value"], 0.05)
            np.savez(
                variant / "fold_1_split.npz",
                train_idx=np.concatenate(([2], indices[7:])), val_idx=indices[3:7], test_idx=indices[:2],
            )
            with self.assertRaisesRegex(ValueError, "different train, validation or test membership"):
                compare_runs(baseline, variant)

    def test_abide_loads_five_largest_sites_and_shared_template(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "cohort.csv"
            template = np.ones((90, 90), dtype=np.float32) - np.eye(90, dtype=np.float32)
            np.save(root / "template.npy", template)
            time_series = np.repeat(np.arange(4.0)[None, :], 90, axis=0)
            time_series[0] *= -1
            np.save(root / "series.npy", time_series)
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("subject_id", "site", "label", "timeseries"))
                subject = 0
                for site, count in enumerate((130, 125, 120, 115, 113, 50)):
                    for _ in range(count):
                        writer.writerow((subject, site, int(subject < 296), "series.npy" if site < 5 else "unused.npy"))
                        subject += 1
            data = load_abide_data(manifest, root / "template.npy")
            self.assertEqual(data.fmri.shape, (603, 90, 90))
            self.assertEqual(int(data.labels.sum()), 296)
            np.testing.assert_array_equal(data.dti[-1], template)
            np.testing.assert_allclose(data.fmri[0, 0, 1:], -1)


if __name__ == "__main__":
    unittest.main()
