from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.io import savemat

from config.config import Config
from data.data_loader import load_brain_connectivity_data


class DataLoaderTest(unittest.TestCase):
    def test_load_brain_connectivity_data_from_mat_files(self):
        with TemporaryDirectory() as tmp_dir:
            fmri = np.stack([np.eye(4), np.ones((4, 4)), np.tril(np.ones((4, 4)))])
            dti = np.stack([np.eye(4), np.eye(4) * 2, np.eye(4) * 3])
            labels = np.array([1, 2, 1])

            ids = np.array(["001_S_001", "001_S_002", "001_S_003"], dtype=object)
            label_order = [1, 2, 0]
            dti_order = [2, 0, 1]
            savemat(f"{tmp_dir}/ADNI_fmri.mat", {
                "fmri": fmri, "subject_ids": ids, "label": labels[label_order], "label_subject_ids": ids[label_order],
            })
            savemat(f"{tmp_dir}/ADNI_DTI.mat", {"dti": dti[dti_order], "subject_ids": ids[dti_order]})

            config = Config(
                data_dir=tmp_dir,
                fmri_file="ADNI_fmri.mat",
                dti_file="ADNI_DTI.mat",
                fmri_key="fmri",
                dti_key="dti",
                label_key="label",
                n_regions=4,
            )

            dataset = load_brain_connectivity_data(config)

            self.assertEqual(dataset.fmri.shape, (3, 4, 4))
            self.assertEqual(dataset.dti.shape, (3, 4, 4))
            self.assertEqual(dataset.labels.tolist(), [0, 1, 0])
            self.assertEqual(dataset.subject_ids.tolist(), ids.tolist())
            np.testing.assert_array_equal(dataset.dti, dti)

    def test_class_groups_merge_raw_codes_and_preserve_subject_alignment(self):
        with TemporaryDirectory() as tmp_dir:
            config = Config(n_regions=2, data_dir=tmp_dir, negative_classes=[42], positive_classes=[7, 19])
            ids = np.arange(100, 105)
            matrices = np.arange(20).reshape(5, 2, 2)
            order = [1, 0, 3, 4, 2]
            labels = np.array([42, 19, 7, 99, 42])
            savemat(config.data_path / config.fmri_file, {
                "fmri": matrices, "subject_ids": ids, "labels": labels[order], "label_subject_ids": ids[order],
            })
            savemat(config.data_path / config.dti_file, {"dti": matrices[::-1], "subject_ids": ids[::-1]})

            dataset = load_brain_connectivity_data(config)

            kept = [0, 1, 2, 4]
            self.assertEqual(dataset.labels.tolist(), [0, 1, 1, 0])
            np.testing.assert_array_equal(dataset.fmri, matrices[kept])
            np.testing.assert_array_equal(dataset.dti, matrices[kept])
            self.assertEqual(dataset.subject_ids.tolist(), ids[kept].astype(str).tolist())
            with self.assertRaisesRegex(ValueError, "missing from the data"):
                load_brain_connectivity_data(config.with_updates(positive_classes=[7, 88]))

    def test_binary_class_values_use_original_codes_before_normalization(self):
        with TemporaryDirectory() as tmp_dir:
            config = Config(n_regions=2, data_dir=tmp_dir, class_values=[20, 10])
            ids = np.arange(100, 104)
            labels = np.array([10, 20, 10, 20])
            matrices = np.zeros((4, 2, 2))
            embedded_labels = np.column_stack((matrices.reshape(4, 4), labels))
            savemat(config.data_path / config.fmri_file, {"fmri": embedded_labels, "subject_ids": ids})
            savemat(config.data_path / config.dti_file, {"dti": matrices, "subject_ids": ids})

            self.assertEqual(load_brain_connectivity_data(config).labels.tolist(), [1, 0, 1, 0])

    def test_rejects_subject_count_and_id_problems_before_filtering(self):
        with TemporaryDirectory() as tmp_dir:
            config = Config(n_regions=2, data_dir=tmp_dir, class_values=[20, 10])
            ids = np.arange(100, 104)
            matrices = np.zeros((4, 2, 2))
            fmri_mat = {"fmri": matrices, "subject_ids": ids, "labels": [10, 20, 10, 99]}
            dti_path = Path(tmp_dir) / config.dti_file
            fmri_path = Path(tmp_dir) / config.fmri_file

            cases = (
                ({"dti": matrices[:3], "subject_ids": ids[:3]}, ids, "Subject counts differ"),
                ({"dti": matrices}, ids, "Missing subject ID key.*DTI"),
                ({"dti": matrices, "subject_ids": ids[:3]}, ids, "DTI has 3 subject IDs for 4 rows"),
                ({"dti": matrices, "subject_ids": [100, 101, 102, 102]}, ids, "Duplicate subject IDs in DTI"),
                ({"dti": matrices, "subject_ids": [100, 101, 102, 104]}, ids, "DTI subject IDs do not match"),
                ({"dti": matrices, "subject_ids": ids}, [100, 101, 102, 104], "labels subject IDs do not match"),
            )
            for dti_mat, label_ids, message in cases:
                with self.subTest(message=message):
                    savemat(fmri_path, {**fmri_mat, "label_subject_ids": label_ids})
                    savemat(dti_path, dti_mat)
                    with self.assertRaisesRegex(ValueError, message):
                        load_brain_connectivity_data(config)


if __name__ == "__main__":
    unittest.main()
