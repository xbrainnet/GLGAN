from dataclasses import dataclass
import math
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
    subject_ids: np.ndarray


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
    labels, embedded_labels = _read_labels(fmri_mat, config.label_key, fmri_raw, config.n_regions)

    if config.connectivity_mode == "window_absmean" or config.fmri_time_series:
        if (
            fmri_raw.ndim != 3
            or fmri_raw.shape[1] != config.n_regions
            or (fmri_raw.shape[2] == config.n_regions and not config.fmri_time_series)
        ):
            raise ValueError(
                "Expected fMRI time series [subjects, regions, time]. "
                "For time series with as many time points as regions, set --fmri-time-series."
            )
        if config.connectivity_mode == "window_absmean":
            fmri = build_dynamic_connectivity(fmri_raw)
        else:
            fmri = _pearson_connectivity(fmri_raw)
    else:
        fmri = _to_connectivity_matrices(fmri_raw, config.n_regions)
    dti = _to_dti_matrices(dti_raw, config.n_regions)
    if not (len(fmri) == len(dti) == len(labels)):
        raise ValueError(
            f"Subject counts differ across fMRI, DTI and labels ({len(fmri)}, {len(dti)}, {len(labels)})."
        )

    subject_ids = _read_subject_ids(fmri_mat, config.fmri_id_key, len(fmri), "fMRI")
    dti_ids = _read_subject_ids(dti_mat, config.dti_id_key, len(dti), "DTI")
    dti = dti[_subject_order(subject_ids, dti_ids, "DTI")]
    if not embedded_labels:
        label_ids = _read_subject_ids(fmri_mat, config.label_id_key, len(labels), "labels")
        labels = labels[_subject_order(subject_ids, label_ids, "labels")]

    mask, labels = _select_binary_task(labels, config)
    labels = _normalize_binary_labels(labels)
    return BrainConnectivityData(fmri[mask], dti[mask], labels, subject_ids[mask])


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


def _read_labels(
    mat: dict, key: Optional[str], fmri_raw: np.ndarray, n_regions: int
) -> tuple[np.ndarray, bool]:
    if key:
        if key not in mat:
            raise KeyError(f"Label key '{key}' not found. Available keys: {_public_keys(mat)}")
        labels = _as_vector(mat[key], f"label key '{key}'")
        return _integer_labels(labels), False

    for candidate in ("label", "labels", "y", "Y", "gnd", "target"):
        if candidate in mat:
            labels = _as_vector(mat[candidate], f"label key '{candidate}'")
            return _integer_labels(labels), False

    if fmri_raw.ndim == 2 and fmri_raw.shape[1] == n_regions * n_regions + 1:
        return _integer_labels(fmri_raw[:, -1]), True

    raise ValueError("Could not infer labels. Set --label-key to the label vector.")


def _as_vector(data: np.ndarray, name: str) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 1 and (data.ndim != 2 or 1 not in data.shape):
        raise ValueError(f"{name} must be a vector.")
    return data.reshape(-1)


def _integer_labels(labels: np.ndarray) -> np.ndarray:
    if labels.dtype.kind not in "biuf" or not np.isfinite(labels).all():
        raise ValueError("Labels must contain finite integer codes.")
    if labels.dtype.kind == "f" and np.any(labels != np.trunc(labels)):
        raise ValueError("Labels must contain finite integer codes.")
    return labels


def _read_subject_ids(mat: dict, key: str, count: int, source: str) -> np.ndarray:
    if key not in mat:
        raise ValueError(f"Missing subject ID key '{key}' for {source}. Available keys: {_public_keys(mat)}")
    raw_ids = _as_vector(mat[key], f"{source} subject IDs")
    if len(raw_ids) != count:
        raise ValueError(f"{source} has {len(raw_ids)} subject IDs for {count} rows.")

    subject_ids: list[str] = []
    for value in raw_ids:
        cell = np.asarray(value)
        if cell.size != 1:
            raise ValueError(f"{source} subject IDs must be nonempty strings or integers.")
        item = cell.item()
        if isinstance(item, str) and item.strip():
            subject_ids.append(item.strip())
        elif isinstance(item, int) and not isinstance(item, bool):
            subject_ids.append(str(item))
        elif isinstance(item, float) and math.isfinite(item) and item.is_integer():
            subject_ids.append(str(int(item)))
        else:
            raise ValueError(f"{source} subject IDs must be nonempty strings or integers.")

    if len(set(subject_ids)) != count:
        raise ValueError(f"Duplicate subject IDs in {source}.")
    return np.asarray(subject_ids, dtype=str)


def _subject_order(reference: np.ndarray, subject_ids: np.ndarray, source: str) -> np.ndarray:
    by_id = {subject_id: index for index, subject_id in enumerate(subject_ids)}
    if set(reference) != by_id.keys():
        raise ValueError(f"{source} subject IDs do not match the fMRI subject ID set.")
    return np.asarray([by_id[subject_id] for subject_id in reference], dtype=np.intp)


def _to_connectivity_matrices(data: np.ndarray, n_regions: int) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 3 and data.shape[1] == n_regions and data.shape[2] == n_regions:
        return _clean_matrix_batch(data)

    if data.ndim == 3 and data.shape[0] == n_regions and data.shape[1] == n_regions:
        return _clean_matrix_batch(np.transpose(data, (2, 0, 1)))

    if data.ndim == 3 and data.shape[1] == n_regions:
        return _pearson_connectivity(data)

    if data.ndim == 2 and data.shape[1] == n_regions * n_regions:
        return _clean_matrix_batch(data.reshape(-1, n_regions, n_regions))

    if data.ndim == 2 and data.shape[1] == n_regions * n_regions + 1:
        return _clean_matrix_batch(data[:, :-1].reshape(-1, n_regions, n_regions))

    raise ValueError(f"Unsupported fMRI shape {data.shape}; expected connectivity matrices or 90-region series.")


def _pearson_connectivity(data: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return _clean_matrix_batch(np.asarray([np.corrcoef(subject) for subject in data]))


def build_dynamic_connectivity(time_series: np.ndarray) -> np.ndarray:
    """Mean absolute Pearson connectivity over complete 40-point windows stepped by 20."""
    time_series = np.asarray(time_series)
    window, step = 40, 20
    if time_series.ndim != 3 or time_series.shape[0] == 0 or time_series.shape[1] < 2:
        raise ValueError("Expected nonempty fMRI time series [subjects, regions, time].")
    if time_series.shape[-1] < window:
        raise ValueError("Dynamic functional connectivity needs at least 40 time points.")

    starts = range(0, time_series.shape[-1] - window + 1, step)
    # Apply the static Pearson convention to each window before averaging.
    total = np.zeros((time_series.shape[0], time_series.shape[1], time_series.shape[1]), dtype=np.float32)
    for start in starts:
        total += np.abs(_pearson_connectivity(time_series[:, :, start : start + window]))
    return total / len(starts)


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
    unique = np.unique(labels)
    if unique.size != 2:
        raise ValueError("A binary task requires two classes. Set negative_classes and positive_classes for multiclass data.")
    return np.searchsorted(unique, labels).astype("int64")


def _select_binary_task(labels: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    negative_classes: Optional[Sequence[int]] = config.negative_classes
    positive_classes: Optional[Sequence[int]] = config.positive_classes
    if config.class_values is not None:
        if negative_classes is not None or positive_classes is not None:
            raise ValueError("class_values cannot be combined with negative_classes or positive_classes.")
        if len(config.class_values) != 2:
            raise ValueError("class_values must contain exactly two labels.")
        negative_classes, positive_classes = config.class_values[:1], config.class_values[1:]
    if negative_classes is None and positive_classes is None:
        return np.ones(len(labels), dtype=bool), labels
    if negative_classes is None or positive_classes is None or not len(negative_classes) or not len(positive_classes):
        raise ValueError("Both negative_classes and positive_classes must be nonempty.")

    negative_codes, positive_codes = set(negative_classes), set(positive_classes)
    if negative_codes & positive_codes:
        raise ValueError("negative_classes and positive_classes must not overlap.")
    missing = (negative_codes | positive_codes) - set(labels)
    if missing:
        raise ValueError(f"Requested label codes are missing from the data: {sorted(missing)}.")

    positive = np.isin(labels, positive_classes)
    mask = np.isin(labels, negative_classes) | positive
    return mask, positive[mask]


def _public_keys(mat: dict) -> list[str]:
    return [key for key in mat if not key.startswith("__")]
