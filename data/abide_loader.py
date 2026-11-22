"""Load the ABIDE AAL-90 experiment cohort and its shared structural template."""

import csv
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from data.data_loader import BrainConnectivityData, _pearson_connectivity


logger = logging.getLogger("GLGAN.data.abide")


def load_abide_data(manifest_path: Path, template_path: Path) -> BrainConnectivityData:
    """Select the five largest eligible sites from a manifest of AAL-90 time series."""
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "site", "label", "timeseries"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"ABIDE manifest requires columns {', '.join(sorted(required))}.")
        rows = [{key: (row[key] or "").strip() for key in required} for row in reader]

    if any(not all(row.values()) or row["label"] not in ("0", "1") for row in rows):
        raise ValueError("ABIDE rows require an ID, site, time-series path and label (0=control, 1=ASD).")
    if len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate subject IDs in the ABIDE manifest.")

    sites = Counter(row["site"] for row in rows).most_common()
    if len(sites) < 5 or sites[4][1] < 50:
        raise ValueError("ABIDE requires five sites with at least 50 eligible subjects each.")
    if len(sites) > 5 and sites[4][1] == sites[5][1]:
        raise ValueError("The site cutoff is tied. Supply the original five-site cohort manifest.")
    selected_sites = {site for site, _ in sites[:5]}
    cohort = [row for row in rows if row["site"] in selected_sites]
    labels = np.asarray([int(row["label"]) for row in cohort], dtype=np.int64)
    if len(labels) != 603 or labels.sum() != 296:
        raise ValueError(
            f"Selected ABIDE sites {sorted(selected_sites)} yield {len(labels)} subjects "
            f"({labels.sum()} ASD, {len(labels) - labels.sum()} controls); "
            "expected 603 subjects (296 ASD, 307 controls)."
        )

    logger.info("ABIDE selected site counts: %s", dict(sites[:5]))
    logger.info("ABIDE selected %d subjects: %d ASD and %d controls", len(labels), labels.sum(), len(labels) - labels.sum())
    logger.info("ABIDE manifest=%s structural_template=%s", manifest_path, template_path)
    template = np.load(template_path, allow_pickle=False)
    if (
        template.shape != (90, 90)
        or template.dtype.kind not in "iuf"
        or not np.isfinite(template).all()
        or (template < 0).any()
        or not (template > 0).any()
        or not np.allclose(template, template.T)
    ):
        raise ValueError("Expected a finite, nonnegative, symmetric AAL-90 structural matrix.")

    matrices = []
    for row in cohort:
        path = manifest_path.parent / row["timeseries"]
        series = np.load(path, allow_pickle=False)
        if (
            series.ndim != 2
            or series.shape[0] != 90
            or series.shape[1] < 2
            or series.dtype.kind not in "iuf"
            or not np.isfinite(series).all()
        ):
            raise ValueError(f"Expected finite AAL-90 time series [90, time] in {path}.")
        matrices.append(_pearson_connectivity(series[None])[0])
    dti = np.broadcast_to(template.astype(np.float32), (len(labels), 90, 90)).copy()
    return BrainConnectivityData(
        np.stack(matrices), dti, labels, np.asarray([row["subject_id"] for row in cohort], dtype=str)
    )
