from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class Config:
    """Experiment configuration aligned with the revised manuscript."""

    # Data
    data_dir: str = "data/raw"
    fmri_file: str = "ADNI_fmri.mat"
    dti_file: str = "ADNI_DTI.mat"
    fmri_key: Optional[str] = None
    dti_key: Optional[str] = None
    label_key: Optional[str] = None
    n_regions: int = 90
    n_classes: int = 2
    class_values: Optional[Sequence[int]] = None

    # Model
    hidden_dim: int = 90
    dropout: float = 0.5
    alpha: float = 0.2
    pagerank_alpha: float = 0.85
    pagerank_iters: int = 50
    classifier_hidden_1: int = 1024
    classifier_hidden_2: int = 128
    use_local_branch: bool = True
    use_global_branch: bool = True
    use_pagerank: bool = True
    use_self_attention: bool = True

    # Training
    batch_size: int = 20
    epochs: int = 300
    learning_rate: float = 5e-5
    weight_decay: float = 5e-6
    k_folds: int = 10
    random_seed: int = 7

    # Output
    output_dir: str = "results"
    save_checkpoints: bool = True
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    def with_updates(self, **kwargs: object) -> "Config":
        return replace(self, **kwargs)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")

    @property
    def flattened_feature_dim(self) -> int:
        enabled_branches = int(self.use_local_branch) + int(self.use_global_branch)
        if enabled_branches == 0:
            raise ValueError("At least one of use_local_branch or use_global_branch must be enabled.")
        return self.n_regions * self.hidden_dim * enabled_branches


DEFAULT_CONFIG = Config()
