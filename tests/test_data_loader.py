import numpy as np
from scipy.io import savemat
from tempfile import TemporaryDirectory
import unittest

from config.config import Config
from data.data_loader import load_brain_connectivity_data


class DataLoaderTest(unittest.TestCase):
    def test_load_brain_connectivity_data_from_mat_files(self):
        with TemporaryDirectory() as tmp_dir:
            fmri = np.stack([np.eye(4), np.ones((4, 4)), np.tril(np.ones((4, 4)))])
            dti = np.stack([np.eye(4), np.eye(4) * 2, np.eye(4) * 3])
            labels = np.array([1, 2, 1])

            savemat(f"{tmp_dir}/ADNI_fmri.mat", {"fmri": fmri, "label": labels})
            savemat(f"{tmp_dir}/ADNI_DTI.mat", {"dti": dti})

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

if __name__ == "__main__":
    unittest.main()
