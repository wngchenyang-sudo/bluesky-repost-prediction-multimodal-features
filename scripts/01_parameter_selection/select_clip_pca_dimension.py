"""Compare CLIP PCA dimensions on three mixed-data development splits.

The script keeps the selected XGBoost parameters and 39 non-semantic image
features fixed. It compares 16, 32, 64, and 128 CLIP principal components.
Both I and MI are reported, but the final dimension is selected only by the
mean MI validation F1 across the three predefined development splits.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from selection_common import BASE, EXTRA2, add_hashtag, load_data, metric_row, split_indices


SPLIT_SEEDS = (42, 101, 202)
PCA_DIMENSIONS = (16, 32, 64, 128)
TRAINING_SELECTED_PARAMETERS = {
    "learning_rate": 0.1,
    "max_depth": 3,
    "scale_pos_weight": 5,
}
JONAS_PARAMETERS = {
    "learning_rate": 0.1,
    "max_depth": 8,
    "scale_pos_weight": 3,
}


def make_pcs(
    raw_clip: np.ndarray,
    clip_present: np.ndarray,
    train: np.ndarray,
    dimensions: int,
    seed: int,
) -> tuple[pd.DataFrame, float, int]:
    """Fit PCA only on training rows with CLIP and retain zeros for missing rows."""
    fit_rows = train[clip_present[train]]
    pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=seed)
    pca.fit(raw_clip[fit_rows])

    pcs = np.zeros((len(raw_clip), dimensions), dtype="float32")
    present_rows = np.flatnonzero(clip_present)
    pcs[present_rows] = pca.transform(raw_clip[present_rows]).astype("float32")
    columns = [f"I-clip-pca{dimensions}-{index:03d}" for index in range(dimensions)]
    return pd.DataFrame(pcs, columns=columns), float(pca.explained_variance_ratio_.sum()), len(fit_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parameter-setting",
        choices=("training_selected", "jonas"),
        default="training_selected",
        help="XGBoost parameter setting used for PCA-dimension comparison.",
    )
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--shallow37", type=Path, required=True)
    parser.add_argument("--extra2", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    parameter_sets = {
        "training_selected": TRAINING_SELECTED_PARAMETERS,
        "jonas": JONAS_PARAMETERS,
    }
    selected_parameters = parameter_sets[args.parameter_setting]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data, m_cols, shallow_cols, clip_cols = load_data(args.hybrid, args.shallow37, args.extra2, args.clip)
    y = data["label"].to_numpy(dtype=int)
    tags = data["hashtag"].fillna("missing").astype(str).astype("category")
    message = data[m_cols].fillna(0).astype("float32")
    nonsemantic = data[shallow_cols + EXTRA2].fillna(0).astype("float32")
    raw_clip = data[clip_cols].fillna(0).astype("float32").to_numpy()
    clip_present = data[clip_cols].notna().all(axis=1).to_numpy()

    run_rows: list[dict[str, float | int | str]] = []
    for seed in SPLIT_SEEDS:
        train, validation = split_indices(data, seed)
        print(f"seed={seed}, train={len(train)}, validation={len(validation)}", flush=True)

        for dimensions in PCA_DIMENSIONS:
            clip_pcs, explained_variance, fit_rows = make_pcs(
                raw_clip, clip_present, train, dimensions, seed
            )
            conditions = {
                "I": add_hashtag([nonsemantic, clip_pcs], tags),
                "MI": add_hashtag([message, nonsemantic, clip_pcs], tags),
            }
            for feature_set, features in conditions.items():
                model = XGBClassifier(
                    **BASE, **selected_parameters, random_state=seed
                )
                model.fit(features.iloc[train], y[train])
                metrics = metric_row(y[validation], model.predict(features.iloc[validation]))
                run_rows.append({
                    "split_seed": seed,
                    "pca_dimensions": dimensions,
                    "feature_set": feature_set,
                    **metrics,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "pca_fit_rows": fit_rows,
                    "explained_variance_ratio": explained_variance,
                })
                print(
                    f"seed={seed} PCA{dimensions} {feature_set}_F1={metrics['f1']:.6f} "
                    f"explained_variance={explained_variance:.6f}",
                    flush=True,
                )

    results = pd.DataFrame(run_rows)
    summary = (
        results.groupby(["pca_dimensions", "feature_set"], as_index=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            explained_variance_ratio_mean=("explained_variance_ratio", "mean"),
        )
        .sort_values(["feature_set", "f1_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    mi_summary = summary.loc[summary["feature_set"] == "MI"].sort_values(
        ["f1_mean", "f1_sd"], ascending=[False, True]
    )
    selected = mi_summary.iloc[0]

    results.to_csv(out / "run_level_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "dimension_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "selection_rule": "highest mean MI validation F1 across three development splits",
        "split_seeds": list(SPLIT_SEEDS),
        "split_protocol": "jointly stratified by hashtag and label, 63% train, 7% validation, 30% unused",
        "pca_dimensions_compared": list(PCA_DIMENSIONS),
        "parameter_setting": args.parameter_setting,
        "xgboost_parameters": {**BASE, **selected_parameters},
        "nonsemantic_feature_count": 39,
        "pca_protocol": "fitted only on CLIP-present training records within each split; missing-image rows retain zero vectors",
        "selected_pca_dimensions": int(selected["pca_dimensions"]),
        "selected_mi_scores": {
            "F1_mean": float(selected["f1_mean"]),
            "F1_sd": float(selected["f1_sd"]),
            "explained_variance_ratio_mean": float(selected["explained_variance_ratio_mean"]),
        },
    }
    (out / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nPCA dimension summary")
    print(summary.to_string(index=False))
    print("\nSelected PCA dimension")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
