"""Final thesis evaluation under the selected XGBoost and image settings.

Runs Mixed ID, Per-Hashtag ID, and leave-one-hashtag-out (OOD) evaluation.
All settings use one fixed XGBoost parameter set, 39 non-semantic image
features, and 64 CLIP principal components fitted within each training fold.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


PARAMETERS = {
    "learning_rate": 0.1,
    "max_depth": 3,
    "scale_pos_weight": 5,
    "n_estimators": 100,
    "min_child_weight": 1,
    "subsample": 1.0,
    "reg_lambda": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "enable_categorical": True,
    "n_jobs": 4,
}
EXTRA_COLUMNS = ["I-nonsem-extra-image_pixel_count", "I-nonsem-extra-aspect_ratio"]
PCA_DIMENSIONS = 64
SEED = 42


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def add_hashtag(parts: list[pd.DataFrame], hashtags: pd.Series) -> pd.DataFrame:
    result = pd.concat(parts, axis=1)
    result.insert(0, "hashtag", hashtags)
    return result


def joint_strata(frame: pd.DataFrame) -> pd.Series:
    return frame["hashtag"].astype(str) + "__" + frame["label"].astype(str)


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
    }


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    hybrid_header = pd.read_csv(args.hybrid, nrows=0).columns.tolist()
    user_columns = [column for column in hybrid_header if column.startswith("U-")]
    message_columns = [column for column in hybrid_header if column.startswith("M-")]
    if len(user_columns) != 172 or len(message_columns) != 66:
        raise ValueError(f"Expected U172 and M66, found U{len(user_columns)} and M{len(message_columns)}.")

    shallow_header = pd.read_csv(args.shallow37, nrows=0).columns.tolist()
    shallow_columns = [column for column in shallow_header if column.startswith("I-nonsem-")]
    if len(shallow_columns) != 37:
        raise ValueError(f"Expected 37 shallow features, found {len(shallow_columns)}.")

    extra_header = pd.read_csv(args.extra2, nrows=0).columns.tolist()
    missing_extra = set(EXTRA_COLUMNS) - set(extra_header)
    if missing_extra:
        raise ValueError(f"Missing extra image columns: {sorted(missing_extra)}")

    clip_header = pd.read_csv(args.clip, nrows=0).columns.tolist()
    clip_columns = [column for column in clip_header if column.startswith("I-clip-")]
    if len(clip_columns) != 512:
        raise ValueError(f"Expected CLIP512, found {len(clip_columns)} columns.")

    base = pd.read_csv(args.hybrid, usecols=["P_id", "hashtag", "label"] + user_columns + message_columns)
    base["_row_order"] = np.arange(len(base))
    shallow = pd.read_csv(args.shallow37, usecols=["post_uri"] + shallow_columns)
    extra = pd.read_csv(args.extra2, usecols=["post_uri"] + EXTRA_COLUMNS)
    clip = pd.read_csv(args.clip, usecols=["post_uri"] + clip_columns)
    for frame, name in ((shallow, "shallow"), (extra, "extra"), (clip, "CLIP")):
        if frame["post_uri"].duplicated().any():
            raise ValueError(f"Duplicate post_uri in {name} feature table.")

    data = base.merge(shallow, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one").drop(columns="post_uri")
    data = data.merge(extra, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one").drop(columns="post_uri")
    data = data.merge(clip, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one").drop(columns="post_uri")
    data = data.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
    data["hashtag"] = data["hashtag"].fillna("missing").astype(str).astype("category")
    return data, user_columns, message_columns, shallow_columns + EXTRA_COLUMNS, clip_columns


def make_feature_sets(
    data: pd.DataFrame,
    user_columns: list[str],
    message_columns: list[str],
    image_columns: list[str],
    clip_columns: list[str],
    train: np.ndarray,
    split_id: int,
) -> tuple[dict[str, pd.DataFrame], int, float]:
    """Fit PCA64 only on CLIP-present training samples and create six feature sets."""
    tags = data["hashtag"].astype("category")
    user = data[user_columns].fillna(0).astype("float32")
    message = data[message_columns].fillna(0).astype("float32")
    image_nonsemantic = data[image_columns].fillna(0).astype("float32")
    raw_clip = data[clip_columns].fillna(0).astype("float32").to_numpy()
    clip_present = data[clip_columns].notna().all(axis=1).to_numpy()
    pca_rows = train[clip_present[train]]
    if len(pca_rows) < PCA_DIMENSIONS:
        raise ValueError(f"Only {len(pca_rows)} CLIP-present training samples for PCA{PCA_DIMENSIONS}.")

    pca = PCA(n_components=PCA_DIMENSIONS, svd_solver="randomized", random_state=SEED + split_id)
    pca.fit(raw_clip[pca_rows])
    transformed = np.zeros((len(data), PCA_DIMENSIONS), dtype="float32")
    present_indices = np.flatnonzero(clip_present)
    transformed[present_indices] = pca.transform(raw_clip[present_indices]).astype("float32")
    clip_pca = pd.DataFrame(
        transformed,
        columns=[f"I-clip-pca64-{number:03d}" for number in range(PCA_DIMENSIONS)],
    )
    image = add_hashtag([image_nonsemantic, clip_pca], tags)
    return {
        "M": add_hashtag([message], tags),
        "I": image,
        "MI": add_hashtag([message, image_nonsemantic, clip_pca], tags),
        "U": add_hashtag([user], tags),
        "UM": add_hashtag([user, message], tags),
        "UMI": add_hashtag([user, message, image_nonsemantic, clip_pca], tags),
    }, len(pca_rows), float(pca.explained_variance_ratio_.sum())


def fit_and_score(features: pd.DataFrame, labels: np.ndarray, train: np.ndarray, test: np.ndarray, split_id: int) -> dict[str, float]:
    model = XGBClassifier(**PARAMETERS, random_state=SEED + split_id)
    model.fit(features.iloc[train], labels[train])
    return metrics(labels[test], model.predict(features.iloc[test]))


def existing_rows(path: Path) -> tuple[list[dict[str, object]], set[tuple[object, ...]]]:
    if not path.exists():
        return [], set()
    rows = pd.read_csv(path).to_dict("records")
    return rows, set()


def run_mixed(data: pd.DataFrame, columns: tuple[list[str], list[str], list[str], list[str]], output_dir: Path) -> pd.DataFrame:
    user_columns, message_columns, image_columns, clip_columns = columns
    output = output_dir / "mixed_cv_results.csv"
    rows, _ = existing_rows(output)
    completed = {(int(row["fold"]), str(row["feature_set"])) for row in rows}
    labels = data.label.to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(splitter.split(np.arange(len(data)), joint_strata(data)), start=1):
        feature_sets, pca_fit_samples, explained_variance = make_feature_sets(
            data, user_columns, message_columns, image_columns, clip_columns, train, fold
        )
        for feature_set, features in feature_sets.items():
            if (fold, feature_set) in completed:
                continue
            row = {
                "fold": fold,
                "feature_set": feature_set,
                **fit_and_score(features, labels, train, test, fold),
                "n_samples": len(data),
                "n_positive": int(labels.sum()),
                "n_negative": int((1 - labels).sum()),
                "train_samples": len(train),
                "test_samples": len(test),
                "pca64_fit_samples": pca_fit_samples,
                "pca64_explained_variance": explained_variance,
            }
            rows.append(row)
            completed.add((fold, feature_set))
            atomic_csv(pd.DataFrame(rows), output)
            print(json.dumps({"setting": "mixed", **row}), flush=True)
    return pd.DataFrame(rows)


def run_per_hashtag(data: pd.DataFrame, columns: tuple[list[str], list[str], list[str], list[str]], output_dir: Path) -> pd.DataFrame:
    user_columns, message_columns, image_columns, clip_columns = columns
    output = output_dir / "per_hashtag_cv_results.csv"
    rows, _ = existing_rows(output)
    completed = {(str(row["hashtag"]), int(row["fold"]), str(row["feature_set"])) for row in rows}
    tags = sorted(data.hashtag.astype(str).unique())
    for tag_number, tag in enumerate(tags, start=1):
        subset = data.loc[data.hashtag.astype(str).eq(tag)].reset_index(drop=True)
        labels = subset.label.to_numpy(dtype=int)
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        print(f"per_hashtag={tag} ({tag_number}/{len(tags)}), n={len(subset)}", flush=True)
        for fold, (train, test) in enumerate(splitter.split(np.arange(len(subset)), labels), start=1):
            feature_sets, pca_fit_samples, explained_variance = make_feature_sets(
                subset, user_columns, message_columns, image_columns, clip_columns, train, fold
            )
            for feature_set, features in feature_sets.items():
                if (tag, fold, feature_set) in completed:
                    continue
                row = {
                    "hashtag": tag,
                    "fold": fold,
                    "feature_set": feature_set,
                    **fit_and_score(features, labels, train, test, fold),
                    "n_samples": len(subset),
                    "n_positive": int(labels.sum()),
                    "n_negative": int((1 - labels).sum()),
                    "train_samples": len(train),
                    "test_samples": len(test),
                    "pca64_fit_samples": pca_fit_samples,
                    "pca64_explained_variance": explained_variance,
                }
                rows.append(row)
                completed.add((tag, fold, feature_set))
                atomic_csv(pd.DataFrame(rows), output)
                print(json.dumps({"setting": "per_hashtag", **row}), flush=True)
    return pd.DataFrame(rows)


def run_ood(data: pd.DataFrame, columns: tuple[list[str], list[str], list[str], list[str]], output_dir: Path) -> pd.DataFrame:
    user_columns, message_columns, image_columns, clip_columns = columns
    output = output_dir / "ood_results.csv"
    rows, _ = existing_rows(output)
    completed = {(str(row["held_out_hashtag"]), int(row["fold"]), str(row["feature_set"])) for row in rows}
    all_tags = data.hashtag.astype(str)
    tags = sorted(all_tags.unique())
    for tag_number, held_tag in enumerate(tags, start=1):
        held_mask = all_tags.eq(held_tag).to_numpy()
        train_pool = data.loc[~held_mask].reset_index(drop=True)
        held_out = data.loc[held_mask].reset_index(drop=True)
        combined = pd.concat([train_pool, held_out], ignore_index=True)
        train_pool_indices = np.arange(len(train_pool))
        held_out_indices = np.arange(len(train_pool), len(combined))
        labels = combined.label.to_numpy(dtype=int)
        pool_strata = joint_strata(train_pool)
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        print(f"ood_held_out={held_tag} ({tag_number}/{len(tags)}), train_pool={len(train_pool)}, test={len(held_out)}", flush=True)
        for fold, (train, _unused) in enumerate(splitter.split(train_pool_indices, pool_strata), start=1):
            feature_sets, pca_fit_samples, explained_variance = make_feature_sets(
                combined, user_columns, message_columns, image_columns, clip_columns, train, fold
            )
            for feature_set, features in feature_sets.items():
                if (held_tag, fold, feature_set) in completed:
                    continue
                row = {
                    "held_out_hashtag": held_tag,
                    "fold": fold,
                    "feature_set": feature_set,
                    **fit_and_score(features, labels, train, held_out_indices, fold),
                    "train_pool_samples": len(train_pool),
                    "train_samples": len(train),
                    "test_samples": len(held_out),
                    "test_positive": int(held_out.label.sum()),
                    "test_negative": int((1 - held_out.label).sum()),
                    "pca64_fit_samples": pca_fit_samples,
                    "pca64_explained_variance": explained_variance,
                }
                rows.append(row)
                completed.add((held_tag, fold, feature_set))
                atomic_csv(pd.DataFrame(rows), output)
                print(json.dumps({"setting": "ood", **row}), flush=True)
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.groupby(keys, as_index=False).agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        runs=("fold", "count"),
    )


def write_metadata(output_dir: Path, data: pd.DataFrame) -> None:
    metadata = {
        "dataset_rows": len(data),
        "positive_samples": int(data.label.sum()),
        "negative_samples": int((1 - data.label).sum()),
        "hashtags": sorted(data.hashtag.astype(str).unique().tolist()),
        "xgboost_parameters": PARAMETERS,
        "feature_sets": {
            "M": "hashtag + M66",
            "I": "hashtag + 39 non-semantic image features + train-fold CLIP PCA64",
            "MI": "hashtag + M66 + 39 non-semantic image features + train-fold CLIP PCA64",
            "U": "hashtag + U172",
            "UM": "hashtag + U172 + M66",
            "UMI": "hashtag + U172 + M66 + 39 non-semantic image features + train-fold CLIP PCA64",
        },
        "protocols": {
            "mixed": "three-fold CV jointly stratified by hashtag and label",
            "per_hashtag": "three-fold CV within each hashtag, stratified by label",
            "ood": "leave one hashtag out as fixed test data; create three joint hashtag-label stratified training folds from the remaining nine hashtags, train on two folds each time",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mixed", "per_hashtag", "ood", "all"), default="all")
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--shallow37", type=Path, required=True)
    parser.add_argument("--extra2", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, user_columns, message_columns, image_columns, clip_columns = load_data(args)
    columns = (user_columns, message_columns, image_columns, clip_columns)
    write_metadata(args.output_dir, data)
    print(json.dumps({"n_samples": len(data), "n_positive": int(data.label.sum()), "mode": args.mode}), flush=True)

    if args.mode in {"mixed", "all"}:
        mixed = run_mixed(data, columns, args.output_dir)
        atomic_csv(summarise(mixed, ["feature_set"]), args.output_dir / "mixed_summary.csv")
    if args.mode in {"per_hashtag", "all"}:
        per_hashtag = run_per_hashtag(data, columns, args.output_dir)
        atomic_csv(summarise(per_hashtag, ["hashtag", "feature_set"]), args.output_dir / "per_hashtag_summary.csv")
    if args.mode in {"ood", "all"}:
        ood = run_ood(data, columns, args.output_dir)
        atomic_csv(summarise(ood, ["held_out_hashtag", "feature_set"]), args.output_dir / "ood_summary.csv")


if __name__ == "__main__":
    main()
