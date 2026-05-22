from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from config.config import Config


@dataclass(frozen=True)
class BrainConnectivityData:
    fmri: np.ndarray
    dti: np.ndarray
    labels: np.ndarray


class BrainConnectivityDataset(Dataset):
    def __init__(self, fmri: np.ndarray, dti: np.ndarray, labels: np.ndarray):
        self.fmri = torch.as_tensor(fmri, dtype=torch.float32)
        self.dti = torch.as_tensor(dti, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.fmri[index], self.dti[index], self.labels[index]


def load_brain_connectivity_data(config: Config) -> BrainConnectivityData:
    fmri_path = config.data_path / config.fmri_file
    dti_path = config.data_path / config.dti_file
    if not fmri_path.exists():
        raise FileNotFoundError(f"Missing fMRI file: {fmri_path}")
    if not dti_path.exists():
        raise FileNotFoundError(f"Missing DTI file: {dti_path}")

    fmri_mat = sio.loadmat(fmri_path)
    dti_mat = sio.loadmat(dti_path)

    fmri_raw = _read_array(fmri_mat, config.fmri_key, preferred=("fmri", "X_data_gnd", "X_data"))
    dti_raw = _read_array(dti_mat, config.dti_key, preferred=("dti", "G_all", "ADNI_DTI"))
    labels = _read_labels(fmri_mat, config.label_key, fmri_raw)

    fmri = _to_connectivity_matrices(fmri_raw, config.n_regions)
    dti = _to_dti_matrices(dti_raw, config.n_regions)
    labels = _normalize_binary_labels(labels)

    n = min(len(fmri), len(dti), len(labels))
    fmri, dti, labels = fmri[:n], dti[:n], labels[:n]

    if config.class_values is not None:
        fmri, dti, labels = _filter_classes(fmri, dti, labels, config.class_values)

    return BrainConnectivityData(fmri=fmri.astype("float32"), dti=dti.astype("float32"), labels=labels.astype("int64"))


def _read_array(mat: dict, key: Optional[str], preferred: tuple[str, ...]) -> np.ndarray:
    if key:
        if key not in mat:
            raise KeyError(f"Key '{key}' not found. Available keys: {_public_keys(mat)}")
        return np.asarray(mat[key])

    for candidate in preferred:
        if candidate in mat:
            return np.asarray(mat[candidate])

    arrays = [(name, value) for name, value in mat.items() if not name.startswith("__") and isinstance(value, np.ndarray)]
    if not arrays:
        raise ValueError("No MATLAB arrays found in file.")
    arrays.sort(key=lambda item: item[1].size, reverse=True)
    return np.asarray(arrays[0][1])


def _read_labels(mat: dict, key: Optional[str], fmri_raw: np.ndarray) -> np.ndarray:
    if key:
        if key not in mat:
            raise KeyError(f"Label key '{key}' not found. Available keys: {_public_keys(mat)}")
        return np.asarray(mat[key]).reshape(-1)

    for candidate in ("label", "labels", "y", "Y", "gnd", "target"):
        if candidate in mat:
            return np.asarray(mat[candidate]).reshape(-1)

    public_arrays = [np.asarray(value) for name, value in mat.items() if not name.startswith("__")]
    one_dimensional = [value.reshape(-1) for value in public_arrays if value.ndim <= 2 and min(value.shape) == 1]
    if one_dimensional:
        return max(one_dimensional, key=len)

    if fmri_raw.ndim == 2 and fmri_raw.shape[1] > 1:
        return fmri_raw[:, -1].reshape(-1)

    raise ValueError("Could not infer labels. Set label_key in the config or CLI.")


def _to_connectivity_matrices(data: np.ndarray, n_regions: int) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 3 and data.shape[1] == n_regions and data.shape[2] == n_regions:
        return _clean_matrix_batch(data)

    if data.ndim == 3 and data.shape[0] == n_regions and data.shape[1] == n_regions:
        return _clean_matrix_batch(np.transpose(data, (2, 0, 1)))

    if data.ndim == 3 and data.shape[1] == n_regions:
        matrices = [np.corrcoef(subject) for subject in data]
        return _clean_matrix_batch(np.asarray(matrices))

    if data.ndim == 2 and data.shape[1] == n_regions * n_regions:
        return _clean_matrix_batch(data.reshape(-1, n_regions, n_regions))

    if data.ndim == 2 and data.shape[1] == n_regions * n_regions + 1:
        return _clean_matrix_batch(data[:, :-1].reshape(-1, n_regions, n_regions))

    raise ValueError(f"Unsupported fMRI shape {data.shape}; expected connectivity matrices or 90-region series.")


def _to_dti_matrices(data: np.ndarray, n_regions: int) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 3 and data.shape[1] == n_regions and data.shape[2] == n_regions:
        return _clean_matrix_batch(data)
    if data.ndim == 3 and data.shape[0] == n_regions and data.shape[1] == n_regions:
        return _clean_matrix_batch(np.transpose(data, (2, 0, 1)))
    if data.ndim == 2 and data.shape[1] == n_regions * n_regions:
        return _clean_matrix_batch(data.reshape(-1, n_regions, n_regions))
    raise ValueError(f"Unsupported DTI shape {data.shape}; expected [N,{n_regions},{n_regions}].")


def _clean_matrix_batch(data: np.ndarray) -> np.ndarray:
    cleaned = np.nan_to_num(data.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned


def _normalize_binary_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype("int64")
    unique = np.unique(labels)
    if unique.size == 2:
        return np.searchsorted(unique, labels).astype("int64")
    return labels


def _filter_classes(
    fmri: np.ndarray, dti: np.ndarray, labels: np.ndarray, class_values: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(class_values) != 2:
        raise ValueError("class_values must contain exactly two labels for binary classification.")
    mask = np.isin(labels, np.asarray(class_values))
    filtered_labels = np.where(labels[mask] == class_values[0], 0, 1)
    return fmri[mask], dti[mask], filtered_labels.astype("int64")


def _public_keys(mat: dict) -> list[str]:
    return [key for key in mat if not key.startswith("__")]
