# Data Format

The repository expects processed fMRI and DTI connectivity data. ADNI raw or processed data are not redistributed because they are subject to ADNI access and data-use rules.

## Files

Place files here:

```text
data/raw/
|-- ADNI_fmri.mat
`-- ADNI_DTI.mat
```

The default loader can also read `ADNI_fmri.mat` and `ADNI_dti.mat` if you pass the file name explicitly:

```bash
python main.py --dti-file ADNI_dti.mat
```

## Required Variables

The fMRI file should contain:

- fMRI connectivity matrices with shape `[n_subjects, 90, 90]`, or regional time series with shape `[n_subjects, 90, time_points]`.
- A label vector. If the label key is not automatically detected, pass `--label-key`.

The DTI file should contain:

- DTI structural connectivity matrices with shape `[n_subjects, 90, 90]`.

If MATLAB variable names are different, pass them explicitly:

```bash
python main.py --fmri-key X_data_gnd --dti-key G_all --label-key labels
```

## Labels

Binary labels are normalized to `0` and `1`. For multi-class ADNI labels, use:

```bash
python main.py --class-values 0 1
```

The first value is mapped to class `0`; the second value is mapped to class `1`.
