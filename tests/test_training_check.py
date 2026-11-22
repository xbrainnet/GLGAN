from itertools import cycle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

from config.config import Config
from training.trainer import _train_one_epoch, run_cross_validation


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
                n_repeats=2,
                output_dir=tmp_dir,
                dropout=0.0,
            )

            result = run_cross_validation(fmri, dti, labels, config)

            self.assertEqual(len(result.fold_metrics), 6)
            for run in (1, 2):
                run_dir = Path(tmp_dir) / f"run_{run:02d}"
                self.assertTrue((run_dir / "fold_metrics.csv").exists())
                self.assertTrue((run_dir / "checkpoints" / "fold_1_best.pt").exists())
                self.assertEqual(len(list(run_dir.glob("fold_*_split.npz"))), config.k_folds)
            self.assertIn("accuracy_mean", result.aggregate_metrics)

    def test_repeat_protocol_controls_split_and_training_seeds(self):
        features = np.zeros((12, 2, 2), dtype="float32")
        labels = np.array([0, 1] * 6)
        for mode, aggregate_unit, expected_std in (
            ("resplit", "run", 0.25),
            ("fixed", "fold", np.sqrt(0.125)),
        ):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp_dir:
                config = Config(
                    k_folds=2, n_repeats=2, repeat_split_mode=mode, aggregate_unit=aggregate_unit, output_dir=tmp_dir
                )
                split_seeds = []
                test_partitions = []

                def record_folds(dataset, labels, splits, run_dir, split_seed, config):
                    split_seeds.append(split_seed)
                    test_partitions.append([tuple(test) for _, test in splits])
                    start = (len(split_seeds) - 1) * 0.5
                    return [{"fold": 1.0, "accuracy": start}, {"fold": 2.0, "accuracy": start + 0.5}]

                with patch("training.trainer._run_folds", side_effect=record_folds), patch(
                    "training.trainer.set_seed"
                ) as set_seed:
                    result = run_cross_validation(features, features, labels, config)

                expected_splits = [7, 8] if mode == "resplit" else [7, 7]
                expected_training = [7, 7] if mode == "resplit" else [7, 8]
                self.assertEqual(split_seeds, expected_splits)
                self.assertEqual([call.args[0] for call in set_seed.call_args_list], expected_training)
                self.assertEqual(test_partitions[0] == test_partitions[1], mode == "fixed")
                self.assertAlmostEqual(result.aggregate_metrics["accuracy_mean"], 0.5)
                self.assertAlmostEqual(result.aggregate_metrics["accuracy_std"], expected_std)

    def test_train_one_epoch_uses_configured_contrastive_weight(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        config = Config(device="cpu", lambda_contrast=0.25)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        inputs = torch.zeros(2, 1, 1)
        batch = (inputs, inputs, torch.tensor([0, 1]))

        with patch.object(model, "forward", return_value=(torch.zeros(2, 2), model.weight.square().sum())):
            _train_one_epoch(model, [batch], nn.CrossEntropyLoss(), optimizer, config)

        torch.testing.assert_close(model.weight.detach(), torch.tensor([[0.9]]))

    def test_each_test_fold_is_evaluated_once_with_the_best_validation_state(self):
        fmri = np.zeros((12, 2, 2), dtype="float32")
        dti = np.zeros_like(fmri)
        labels = np.array([0, 1] * 6)

        with TemporaryDirectory() as tmp_dir:
            config = Config(k_folds=3, n_repeats=1, epochs=2, output_dir=tmp_dir, device="cpu")
            splitter = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.random_seed)
            splits = list(splitter.split(fmri, labels))
            test_partitions = {frozenset(test_idx) for _, test_idx in splits}
            epochs = cycle(range(1, config.epochs + 1))
            trace = []

            def record_training(model, loader, *_):
                epoch = next(epochs)
                with torch.no_grad():
                    model.weight.fill_(epoch)
                trace.append(("train", frozenset(loader.dataset.indices), epoch))

            def record_evaluation(model, loader, *_):
                indices = frozenset(loader.dataset.indices)
                epoch = int(model.weight.item())
                trace.append(("evaluate", indices, epoch))
                accuracy = 0.25 if indices in test_partitions else 1.0 / epoch
                return {"accuracy": accuracy}

            with patch("training.trainer.GLGAN", side_effect=lambda _: nn.Linear(1, 1, bias=False)), patch(
                "training.trainer._train_one_epoch", side_effect=record_training
            ), patch("training.trainer.evaluate", side_effect=record_evaluation):
                result = run_cross_validation(fmri, dti, labels, config)

            expected_trace = []
            for fold, (outer_train_idx, test_idx) in enumerate(splits, start=1):
                train_idx, val_idx = train_test_split(
                    outer_train_idx, test_size=0.2, random_state=config.random_seed, stratify=labels[outer_train_idx]
                )
                split_path = Path(tmp_dir) / "run_01" / f"fold_{fold}_split.npz"
                with np.load(split_path, allow_pickle=False) as split:
                    self.assertEqual(set(split.files), {"train_idx", "val_idx", "test_idx"})
                    np.testing.assert_array_equal(split["train_idx"], train_idx)
                    np.testing.assert_array_equal(split["val_idx"], val_idx)
                    np.testing.assert_array_equal(split["test_idx"], test_idx)
                expected_trace.extend(
                    [
                        ("train", frozenset(train_idx), 1),
                        ("evaluate", frozenset(val_idx), 1),
                        ("train", frozenset(train_idx), 2),
                        ("evaluate", frozenset(val_idx), 2),
                        ("evaluate", frozenset(test_idx), 1),
                    ]
                )
                checkpoint = Path(tmp_dir) / "run_01" / "checkpoints" / f"fold_{fold}_best.pt"
                torch.testing.assert_close(torch.load(checkpoint, map_location="cpu")["weight"], torch.ones(1, 1))

            self.assertEqual(trace, expected_trace)
            self.assertEqual(result.aggregate_metrics["accuracy_mean"], 0.25)


if __name__ == "__main__":
    unittest.main()
