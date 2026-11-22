# Experiment Commands

The original working directory contained separate scripts for each task and ablation. This public repository consolidates the repeated code into one command-line runner while keeping the same experimental ingredients: fMRI/DTI inputs, 10-fold cross-validation, Adam optimization, and GLGAN global-local branches.

## Classification Tasks

Set the shell variables to the integer labels in the codebook for each dataset. Set `EPILEPSY_DATA_DIR`, `EPILEPSY_FMRI_FILE`, and `EPILEPSY_DTI_FILE` to the corresponding paths or file names. Each command maps the negative group to `0` and the positive group to `1`.

```bash
python main.py --negative-classes "${ADNI_NC:?}" --positive-classes "${ADNI_SMC:?}" --output-dir results/NC_SMC
python main.py --negative-classes "${ADNI_NC:?}" --positive-classes "${ADNI_EMCI:?}" --output-dir results/NC_EMCI
python main.py --negative-classes "${ADNI_NC:?}" --positive-classes "${ADNI_SMC:?}" "${ADNI_EMCI:?}" --output-dir results/NC_SMC_EMCI
python main.py --negative-classes "${ADNI_SMC:?}" --positive-classes "${ADNI_EMCI:?}" --output-dir results/SMC_EMCI

run_epilepsy() {
    python main.py --data-dir "${EPILEPSY_DATA_DIR:?}" \
        --fmri-file "${EPILEPSY_FMRI_FILE:?}" --dti-file "${EPILEPSY_DTI_FILE:?}" "$@"
}
run_epilepsy --negative-classes "${EPILEPSY_NC:?}" --positive-classes "${EPILEPSY_FLE:?}" --output-dir results/NC_FLE
run_epilepsy --negative-classes "${EPILEPSY_NC:?}" --positive-classes "${EPILEPSY_TLE:?}" --output-dir results/NC_TLE
run_epilepsy --negative-classes "${EPILEPSY_NC:?}" --positive-classes "${EPILEPSY_FLE:?}" "${EPILEPSY_TLE:?}" --output-dir results/NC_FLE_TLE
run_epilepsy --negative-classes "${EPILEPSY_TLE:?}" --positive-classes "${EPILEPSY_FLE:?}" --output-dir results/TLE_FLE
```

## Ablation

The provided ablation switches support illustrative component-removal experiments. They do not constitute a complete reproduction of every ablation configuration reported in the manuscript.

`--disable-local` and `--disable-global` turn off complete branches. Table IV reports individual attention-module configurations. The public CLI expects paired fMRI and DTI inputs, while Table V reports separate fMRI-only and DTI-only experiments.

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
- Paired experiments default to 10 repetitions of 10 folds each. `--repeat-split-mode resplit` increments the split seed and resets training to the base seed. `--repeat-split-mode fixed` uses fixed splits and increments the training seed.
- Each `run_XX/` directory contains fold metrics, per-fold split files, and checkpoints when enabled. The split files store `train_idx`, `val_idx`, and `test_idx` as row indices into the aligned, task-filtered dataset.
- `training.log` records the split and training seeds for each run, aggregation unit, experiment configuration, and Git commit when available. A Git commit with local working-tree changes is marked in the log.
- For paired experiments, `aggregate_metrics.csv` summarizes run means by default. ABIDE defaults to fold-level aggregation, which is also available for paired experiments with `--aggregate-unit fold`. Both modes report means and population standard deviations.
- PageRank defaults to binary DTI edges with damping factor `0.85`. Use `--weighted-pagerank` when the experiment calls for strength-weighted transitions.

## Supplementary Experiments

Use the classification task arguments above to select NC vs. pooled FLE/TLE or NC vs. pooled SMC/EMCI.

- **Table S4** uses `--connectivity-mode window_absmean` with a time-series input. For the baseline, use `--connectivity-mode static` with the same series. Pass `--fmri-time-series` for square time-series arrays.
- **Table S5** uses `--coupling-mode bidirectional`. Each subject has one functional-attention refinement of existing positive DTI edges per forward pass. The refined rows are normalized and passed to both branches. Use matching cohort and split settings, then compare the resulting run directories with the [paired fold test](../analysis/README.md#table-s5-paired-test). The default propagation keeps the original binary PageRank transitions.
- **Table S6** uses `--contrastive-mode class_conditional`. During training, same-class cross-view pairs are positives, including a subject's own pair, and different-class cross-view pairs are negatives. The loss averages both directions at temperature 0.4. Inference uses the same subject-level fusion without labels.

For the ABIDE experiment, prepare the [manifest and structural template](../data/README.md#abide) and run:

```bash
python main.py --dataset abide --abide-manifest data/raw/abide_manifest.csv \
    --abide-template data/raw/abide_dti_template.npy --output-dir results/abide
```

By default, ABIDE trains and tests within the selected five-site cohort in a single 10-fold stratified run. Read `accuracy_mean`, `accuracy_std`, `f1_mean`, and `f1_std` from `aggregate_metrics.csv`. The values are proportions with population standard deviations across the ten test folds.

## Analysis Scripts

Post-processing scripts are available under `analysis/`:

```bash
python -m analysis.roc_analysis --input predictions.csv --output results/analysis/roc_points.csv
python -m analysis.tsne_analysis --features features.npy --labels labels.npy --output results/analysis/tsne_coordinates.csv
python -m analysis.hubness_analysis --adjacency adjacency.npy --output results/analysis/hubness.csv
python -m analysis.roi_ranking --input results/analysis/hubness.csv --output results/analysis/top_rois.csv
python -m analysis.sensitivity_statistics --input sensitivity_results.csv --parameter-column learning_rate --metric-column accuracy
```
