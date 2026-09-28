# Image Features for Bluesky Repost Prediction

This repository contains the code for a dissertation examining whether image
features provide additional information for Bluesky repost prediction.

## Structure

```text
data/       Local input data used by the analyses (not uploaded)
scripts/    Feature preparation, experiments, and analyses
```

## Local data required to run the analyses

The `data/` folder is intentionally excluded from GitHub. It remains in the
local project and is needed when running the scripts:

- `data/jonas_original/` contains the Jonas hybrid 1:5 analytical dataset.
- `data/study_generated/` contains this study's feature tables:
  39 non-semantic image features, post-level CLIP512 embeddings, two additional
  image features, and recognised OCR text used to construct the final OCR
  cross-content features at runtime.

These large input tables, as well as the downloaded image archive, are not
uploaded. Scripts for image download and feature construction are in
`scripts/00_data_and_image_features/`.

## Scripts

The numbered script folders follow the dissertation workflow:

- `00_data_and_image_features/`: image download and feature construction
- `01_parameter_selection/`: XGBoost parameter and CLIP PCA selection
- `02_formal_evaluation/`: three evaluation settings and six feature sets
- `03_feature_importance/`: gain, sequential-removal, and permutation analyses
- `04_ocr_analysis/`: OCR complementarity analysis

Each analysis folder contains a short README describing its retained scripts.

## Requirements

`requirements.txt` lists the packages needed for the analyses.
`requirements-image-features.txt` lists the additional packages required only
when rebuilding image or CLIP feature inputs.
