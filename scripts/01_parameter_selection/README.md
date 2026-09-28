# Parameter and CLIP PCA Selection

This folder contains the selection procedures used before formal evaluation.

- `select_training_parameters_3split_mi.py` evaluates candidate XGBoost
  learning rates, tree depths, and positive-class weights using the MI feature
  configuration. It produces the ranked parameter results reported in Table
  5.1.
- `select_clip_pca_dimension.py` compares CLIP PCA dimensions of 16, 32, 64,
  and 128 under the selected MI model. It produces the results reported in
  Table 5.2.
- `selection_common.py` is a shared utility module used by both scripts.

Both procedures use three fixed, jointly stratified development splits rather
than formal three-fold cross-validation. Each split uses 63\% of the data for
training, 7\% for validation, and leaves the remaining 30\% unused during
selection. The selected configuration is learning rate 0.1, maximum depth 3,
positive-class weight 5, and CLIP PCA with 64 components.
