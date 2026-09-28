# Formal Evaluation

`run_final_evaluation_settings.py` reproduces the dissertation's formal model
evaluation.

It evaluates six feature configurations:

- M: message features
- I: image features
- MI: message plus image features
- U: user features
- UM: user plus message features
- UMI: user, message, and image features

The script supports three evaluation settings:

- Mixed Hashtag Prediction
- Per Hashtag Prediction
- Hashtag Out-of-Distribution Prediction

For each setting, it writes fold-level results and summary tables containing
F1, precision, recall, and the change in F1 between feature configurations.
Image features consist of 39 non-semantic image features and train-fold CLIP
PCA with 64 components. The selected XGBoost parameters are used throughout.
