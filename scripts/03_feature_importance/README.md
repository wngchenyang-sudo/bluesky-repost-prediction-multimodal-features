# Feature Importance Analysis

This folder contains the scripts used for feature-importance
analyses.

- `run_final_umi_feature_gain.py` generates Table 7.1: the Top 20 UMI research
  features ranked by mean normalised XGBoost gain across the three Mixed folds.
  It does not perform feature removal.
- `run_umi_sequential_removal_mixed_ood.py` performs the full sequential UMI
  feature-removal analysis and produces the Mixed and OOD F1 curves in Figure
  7.1. The first stage also writes the same Top-20 gain ranking.
- `run_final_image_group_permutation.py` performs the MI image-group
  permutation analysis reported in Table 7.2. It tests whether predictive
  performance declines when the complete image-feature group is randomly
  exchanged between test samples.

All analyses use the selected XGBoost parameters, the three fixed Mixed folds,
39 non-semantic image features, and train-fold CLIP PCA with 64 components.
