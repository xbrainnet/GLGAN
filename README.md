# GLGAN: Global-Local Graph Attention Network

GLGAN classifies subjects from paired fMRI and DTI connectivity. It uses fMRI connectivity profiles as node features and DTI connectivity to define the edges for graph propagation. The global branch combines structural PageRank scores with functional self-attention. The local branch combines fMRI-based neighbor attention with DTI self-attention. ROI-level contrastive alignment and attention fuse the two representations, and an MLP classifies the resulting subject embedding.

## Quick Start

From the repository root, set up the Python environment.

```bash
conda create -n glgan python=3.10
conda activate glgan
python -m pip install -r requirements.txt
```

Place the processed files `ADNI_fmri.mat` and `ADNI_DTI.mat` in `data/raw/`. The loader matches fMRI, DTI, and labels by subject ID. See the [data format](data/README.md) for array shapes, MATLAB keys, and label codes.

Run the default configuration on a binary cohort.

```bash
python main.py
```

For a multiclass input file, select the negative and positive groups with `--negative-classes` and `--positive-classes`. Use the original codes from the dataset.

Please cite the corresponding manuscript when using this code.

