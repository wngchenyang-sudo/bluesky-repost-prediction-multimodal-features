"""Test whether the final MI model relies on correctly matched image features.

For each jointly stratified mixed-data fold, the model is trained once with
normal post-to-image matching.  The complete 103-dimensional image block is
then jointly permuted across *all* test samples, including zero vectors for
posts without usable images.  The fitted model is not retrained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


PARAMETERS = dict(
    learning_rate=0.1,
    max_depth=3,
    scale_pos_weight=5,
    n_estimators=100,
    min_child_weight=1,
    subsample=1.0,
    reg_lambda=1,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    enable_categorical=True,
    n_jobs=4,
)
EXTRA_COLUMNS = [
    "I-nonsem-extra-image_pixel_count",
    "I-nonsem-extra-aspect_ratio",
]


def metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def add_hashtag(message: pd.DataFrame, image: pd.DataFrame, tags: pd.Series) -> pd.DataFrame:
    features = pd.concat([message, image], axis=1)
    features.insert(0, "hashtag", tags)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--shallow37", type=Path, required=True)
    parser.add_argument("--extra2", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hybrid_header = pd.read_csv(args.hybrid, nrows=0).columns.tolist()
    message_columns = [column for column in hybrid_header if column.startswith("M-")]
    shallow_columns = [column for column in pd.read_csv(args.shallow37, nrows=0).columns if column.startswith("I-nonsem-")]
    clip_columns = [column for column in pd.read_csv(args.clip, nrows=0).columns if column.startswith("I-clip-")]
    if (len(message_columns), len(shallow_columns), len(clip_columns)) != (66, 37, 512):
        raise ValueError("Unexpected message, non-semantic, or CLIP feature dimensions")

    base = pd.read_csv(args.hybrid, usecols=["P_id", "hashtag", "label"] + message_columns)
    base["_order"] = np.arange(len(base))
    shallow = pd.read_csv(args.shallow37, usecols=["post_uri"] + shallow_columns)
    extra = pd.read_csv(args.extra2, usecols=["post_uri"] + EXTRA_COLUMNS)
    clip = pd.read_csv(args.clip, usecols=["post_uri"] + clip_columns)
    for frame, name in ((shallow, "shallow"), (extra, "extra"), (clip, "CLIP")):
        if frame.post_uri.duplicated().any():
            raise ValueError(f"Duplicate post_uri in {name} table")
    data = base.merge(shallow, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one").drop(columns="post_uri")
    data = data.merge(extra, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one").drop(columns="post_uri")
    data = data.merge(clip, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one").drop(columns="post_uri")
    data = data.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    y = data.label.to_numpy(dtype=int)
    tags = data.hashtag.fillna("missing").astype(str).astype("category")
    strata = data.hashtag.fillna("missing").astype(str) + "__" + data.label.astype(str)
    message = data[message_columns].fillna(0).astype("float32")
    nonsemantic = data[shallow_columns + EXTRA_COLUMNS].fillna(0).astype("float32")
    raw_clip = data[clip_columns].fillna(0).astype("float32").to_numpy()
    clip_present = data[clip_columns].notna().all(axis=1).to_numpy()

    rows: list[dict[str, object]] = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(np.arange(len(data)), strata), start=1):
        fit_rows = train[clip_present[train]]
        pca = PCA(n_components=64, svd_solver="randomized", random_state=args.seed + fold)
        pca.fit(raw_clip[fit_rows])
        pcs = np.zeros((len(data), 64), dtype="float32")
        present_rows = np.flatnonzero(clip_present)
        pcs[present_rows] = pca.transform(raw_clip[present_rows]).astype("float32")
        pca_columns = [f"I-clip-pca64-{number:03d}" for number in range(64)]
        image = pd.concat([nonsemantic, pd.DataFrame(pcs, columns=pca_columns)], axis=1)
        features = add_hashtag(message, image, tags)

        model = XGBClassifier(**PARAMETERS, random_state=args.seed + fold)
        model.fit(features.iloc[train], y[train])
        original = metrics(y[test], model.predict(features.iloc[test]))
        rows.append({
            "fold": fold,
            "condition": "original_matching",
            "trial": 0,
            **original,
            "train_samples": len(train),
            "test_samples": len(test),
            "pca_fit_samples": len(fit_rows),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        })
        print(json.dumps(rows[-1]), flush=True)

        # A single row permutation moves all 103 image columns together.
        # This includes zero image vectors for samples without usable images.
        generator = np.random.default_rng(args.seed * 10_000 + fold)
        original_image_test = image.iloc[test].to_numpy()
        for trial in range(1, args.trials + 1):
            changed = features.iloc[test].copy()
            changed.loc[:, image.columns] = original_image_test[generator.permutation(len(test))]
            shuffled = metrics(y[test], model.predict(changed))
            rows.append({
                "fold": fold,
                "condition": "image_group_permuted",
                "trial": trial,
                **shuffled,
                "train_samples": len(train),
                "test_samples": len(test),
                "pca_fit_samples": len(fit_rows),
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "f1_drop": original["f1"] - shuffled["f1"],
            })
            print(json.dumps(rows[-1]), flush=True)

    results = pd.DataFrame(rows)
    summary = results.groupby("condition", as_index=False).agg(
        f1_mean=("f1", "mean"),
        f1_sd=("f1", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        runs=("f1", "count"),
    )
    drops = results.loc[results.condition.eq("image_group_permuted")].groupby("fold", as_index=False).agg(
        original_f1=("f1_drop", lambda values: np.nan),
        f1_drop_mean=("f1_drop", "mean"),
        f1_drop_sd=("f1_drop", "std"),
        trials=("f1_drop", "count"),
    )
    original_by_fold = results.loc[results.condition.eq("original_matching"), ["fold", "f1"]].rename(columns={"f1": "original_f1"})
    drops = drops.drop(columns="original_f1").merge(original_by_fold, on="fold", validate="one_to_one")
    overall = {
        "original_f1_mean": float(original_by_fold.original_f1.mean()),
        "permuted_f1_mean": float(results.loc[results.condition.eq("image_group_permuted"), "f1"].mean()),
        "f1_drop_mean": float(results.loc[results.condition.eq("image_group_permuted"), "f1_drop"].mean()),
        "f1_drop_sd": float(results.loc[results.condition.eq("image_group_permuted"), "f1_drop"].std()),
        "permutation_runs": int(args.trials * 3),
    }
    results.to_csv(args.output_dir / "image_group_permutation_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "image_group_permutation_summary.csv", index=False, encoding="utf-8-sig")
    drops.to_csv(args.output_dir / "image_group_permutation_fold_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "protocol": "Mixed ID three-fold cross-validation jointly stratified by hashtag and label",
        "model": "MI with M66, 39 non-semantic features and training-fold CLIP PCA64",
        "parameters": PARAMETERS,
        "permutation": "All 103 image features are jointly permuted across every test sample, including zero vectors. The fitted model is not retrained.",
        "trials_per_fold": args.trials,
        "overall": overall,
    }, indent=2), encoding="utf-8")
    print("\nSummary")
    print(summary.to_string(index=False))
    print("\nOverall F1 change")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
