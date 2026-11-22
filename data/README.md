# Data Format

The repository expects processed fMRI and DTI connectivity data. ADNI raw or processed data are not redistributed because they are subject to ADNI access and data-use rules.

## Files

Place files here:

```text
data/raw/
|-- ADNI_fmri.mat
`-- ADNI_DTI.mat
```

If your structural connectivity file is named `ADNI_dti.mat`, pass that exact filename.

```bash
python main.py --dti-file ADNI_dti.mat
```

## Required Variables

The fMRI file should contain:

- fMRI connectivity matrices with shape `[n_subjects, 90, 90]`, or regional time series with shape `[n_subjects, 90, time_points]`. Use `--fmri-time-series` to explicitly select time series when the subject count or time-point count is 90.
- A label vector. If the label key is not automatically detected, pass `--label-key`.
- `subject_ids` in fMRI row order and `label_subject_ids` in label-vector order.

The DTI file should contain:

- DTI structural connectivity matrices with shape `[n_subjects, 90, 90]`.
- `subject_ids` in DTI row order.

If MATLAB variable names are different, pass them explicitly:

```bash
python main.py --fmri-key X_data_gnd --dti-key G_all --label-key labels
```

Use `--fmri-id-key`, `--dti-id-key`, and `--label-id-key` for different ID variable names. Store IDs as a vector of integers, strings, or MATLAB cells containing one ID each. The loader verifies matching counts and unique ID sets, then reorders DTI and labels to match the fMRI rows. Export the ID vector together with each source during preprocessing. If preprocessing has verified that the fMRI and label vectors already share an ID vector, pass its key to both relevant options. Labels embedded as the final column of a flattened fMRI matrix use the fMRI row IDs.

## Labels

For an already binary dataset, the smaller original code is mapped to `0` and the larger to `1`. To select a task, pass the original label codes in task order:

```bash
python main.py --negative-classes "$NC" --positive-classes "$SMC" "$EMCI"
```

The example assigns NC to `0` and merges SMC and EMCI into `1`. Set the variables using the codebook for the input files. For one code per group, `--class-values NEGATIVE POSITIVE` is also supported.

## Dynamic Functional Connectivity

`--connectivity-mode window_absmean` requires regional fMRI time series with at least 40 time points and uses complete windows of 40 time points stepped by 20. It averages the absolute Pearson correlation matrices over those windows for each subject. Undefined correlations, such as those involving a constant region within a window, are set to zero for that window. For square time-series arrays of shape `[subjects, 90, 90]`, pass `--fmri-time-series` to distinguish them from precomputed connectivity.

## ABIDE

Use a CSV with the columns `subject_id,site,label,timeseries`. Each row is one eligible subject with a unique ID, a site identifier, a label (`0` for control, `1` for ASD), and a path to a NumPy array of AAL-90 time series in `[90, time]` order. Relative paths resolve from the manifest directory. Provide a manifest of eligible subjects after the original preprocessing and inclusion steps. The loader selects the five largest sites with at least 50 subjects each and checks for 603 subjects, including 296 ASD and 307 controls.

Supply the population-average DTI template derived from 88 healthy subjects as a finite, symmetric, nonnegative `[90, 90]` NumPy array with at least one positive entry. The template and every time-series array must use the same AAL-90 region order. Template provenance and individual subject inclusion depend on the source files supplied for the run.
