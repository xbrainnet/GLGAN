# Analysis Scripts

Run these commands from the repository root. Use the inputs described below. The training CLI saves cohort and fold summaries. Prepare prediction CSVs, feature arrays, and sensitivity sweep tables separately.

## ROC

Supply a CSV with a binary `label` (0 or 1) and a prediction `score` for each subject.

```bash
python -m analysis.roc_analysis --input predictions.csv --output results/analysis/roc_points.csv
```

## t-SNE

Supply a NumPy feature matrix of shape `[n_samples, n_features]` and, optionally, a label vector in the same order.

```bash
python -m analysis.tsne_analysis --features features.npy --labels labels.npy --output results/analysis/tsne_coordinates.csv
```

## Hubness and ROI Ranking

Supply a NumPy adjacency array of shape `[n_regions, n_regions]` or `[n_subjects, n_regions, n_regions]`. The ranking script sorts the output by PageRank by default. For the Elastic Net analysis of Table S3, use the workflow below.

```bash
python -m analysis.hubness_analysis --adjacency adjacency.npy --output results/analysis/hubness.csv
python -m analysis.roi_ranking --input results/analysis/hubness.csv --output results/analysis/top_rois.csv
```

## Table S3 ROI Stability

Run separately for NC vs. FLE, NC vs. TLE, and FLE vs. TLE. Supply the raw label codes and a text file containing the 90 ROI abbreviations in the exact order of the connectivity matrices.

```bash
python -m analysis.elastic_net_roi_stability \
    --data-dir "$EPILEPSY_DATA_DIR" --fmri-file "$EPILEPSY_FMRI_FILE" --dti-file "$EPILEPSY_DTI_FILE" \
    --negative-classes "$EPILEPSY_NC" --positive-classes "$EPILEPSY_FLE" \
    --roi-labels aal90_labels.txt --run-dir results/NC_FLE/run_01 \
    --output results/analysis/roi_NC_FLE.csv
```

The script fits Elastic Net to upper-triangular fMRI and DTI edges within each outer training fold. It fits the scaler within each inner split, selects hyperparameters by stratified cross-validation, and refits on the full outer training partition. It sums absolute coefficients by incident ROI, normalizes by the fold maximum, and reports frequency, rank, and prominence across selected folds. A region is stable if it ranks in the top ten in at least seven folds. Use `--run-dir` to match a saved cohort and outer folds, or omit it to create a seeded ten-fold split.

## Figure 6 Embedding Elastic Net

```bash
python -m analysis.embedding_elastic_net --features pre_softmax_embeddings.npy \
    --labels labels.npy --output results/analysis/embedding_elastic_net.csv
```

Supply the embedding matrix `[subjects, features]` and binary labels in the same order. The script fits a non-negative Elastic Net and writes coefficients for those embedding features. Reconstructing the anatomical plots requires the original feature-to-edge mapping.

## Table S5 Paired Test

```bash
python -m analysis.paired_fold_test --baseline results/s5_unidirectional/run_01 \
    --variant results/s5_bidirectional/run_01
```

Use one pair of completed ten-fold run directories. The script verifies cohort IDs, labels, and split membership before running a two-sided paired t-test on fold accuracies. It reports a `null` p-value if the paired differences have zero variance.

## Sensitivity Statistics

Supply a CSV with a parameter column and a metric column. The script summarizes metrics for each parameter value.

```bash
python -m analysis.sensitivity_statistics --input sensitivity_results.csv --parameter-column learning_rate --metric-column accuracy
```
