# Experiment Commands

The original working directory contained separate scripts for each task and ablation. This public repository consolidates the repeated code into one command-line runner while keeping the same experimental ingredients: fMRI/DTI inputs, 10-fold cross-validation, Adam optimization, and GLGAN global-local branches.

## Main ADNI Tasks

Adjust `--class-values` to match the label encoding in your processed ADNI files. The commands below use the unified default configuration reported in the manuscript response.

```bash
python main.py --class-values 0 1 --output-dir results/NC_SMC
python main.py --class-values 0 2 --output-dir results/NC_EMCI
python main.py --class-values 0 3 --output-dir results/NC_iLL
python main.py --class-values 1 2 --output-dir results/SMC_EMCI
```

## Ablation

```bash
python main.py --disable-local --output-dir results/ablation_no_local
python main.py --disable-global --output-dir results/ablation_no_global
python main.py --disable-pagerank --output-dir results/ablation_no_pagerank
python main.py --disable-self-attention --output-dir results/ablation_no_self_attention
```

## Reproducibility Notes

- Default seed: `7`.
- Default epochs: `300`.
- Default batch size: `20`.
- Default learning rate: `5e-5`.
- Default weight decay: `5e-6`.
- Fold-level metrics are saved in `fold_metrics.csv`.
- Mean and standard deviation metrics across folds are saved in `aggregate_metrics.csv`.

## Analysis Scripts

Post-processing scripts are available under `analysis/`:

```bash
python -m analysis.parameter_count --output results/analysis/parameter_count.csv
python -m analysis.roc_analysis --input predictions.csv --output results/analysis/roc_points.csv
python -m analysis.tsne_analysis --features features.npy --labels labels.npy --output results/analysis/tsne_coordinates.csv
python -m analysis.hubness_analysis --adjacency adjacency.npy --output results/analysis/hubness.csv
python -m analysis.roi_ranking --input results/analysis/hubness.csv --output results/analysis/top_rois.csv
python -m analysis.sensitivity_statistics --input sensitivity_results.csv --parameter-column learning_rate --metric-column accuracy
```
