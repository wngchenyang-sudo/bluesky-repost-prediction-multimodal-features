"""Reproduce the OCR complementarity analysis reported in Table 8.1.

Test A compares MI with MI plus the OCR--post-text cross-content features.
Test B evaluates the same two trained models separately for test samples with
high versus low OCR new-word ratios.  The model uses the 14 pre-specified
OCR--post-text cross-content features below, calculated only for posts with
usable recognised OCR text.  The threshold is the median new-word ratio
calculated from that fold's training samples only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Kept here rather than imported from a local exploratory script so this
# GitHub release remains self-contained.
PARAMETERS = dict(
    learning_rate=0.1, max_depth=3, scale_pos_weight=5, n_estimators=100,
    min_child_weight=1, subsample=1.0, reg_lambda=1,
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    enable_categorical=True, n_jobs=4,
)
EXTRA_COLUMNS = ["I-nonsem-extra-image_pixel_count", "I-nonsem-extra-aspect_ratio"]
TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def words(value: object) -> set[str]:
    return {word.lower() for word in TOKEN.findall("" if pd.isna(value) else str(value))}


def cross_features(post_text: object, ocr_text: object, image_count: int) -> dict[str, float]:
    post, ocr = ("" if pd.isna(value) else str(value) for value in (post_text, ocr_text))
    post_words, ocr_words = words(post), words(ocr)
    shared_words = post_words & ocr_words
    post_word_count, ocr_word_count = len(TOKEN.findall(post)), len(TOKEN.findall(ocr))
    return {
        "I-ocr-cross-ocr_text_image_count": float(image_count),
        "I-ocr-cross-ocr_char_count": float(len(ocr)),
        "I-ocr-cross-ocr_word_count": float(ocr_word_count),
        "I-ocr-cross-ocr_digit_count": float(sum(character.isdigit() for character in ocr)),
        "I-ocr-cross-ocr_question_count": float(ocr.count("?")),
        "I-ocr-cross-ocr_hashtag_count": float(ocr.count("#")),
        "I-ocr-cross-ocr_mention_count": float(ocr.count("@")),
        "I-ocr-cross-abs_post_ocr_char_diff": float(abs(len(post) - len(ocr))),
        "I-ocr-cross-ocr_to_post_char_ratio": float(len(ocr) / (len(post) + 1)),
        "I-ocr-cross-abs_post_ocr_word_diff": float(abs(post_word_count - ocr_word_count)),
        "I-ocr-cross-ocr_to_post_word_ratio": float(ocr_word_count / (post_word_count + 1)),
        "I-ocr-cross-unique_word_jaccard": float(len(shared_words) / len(post_words | ocr_words)) if post_words | ocr_words else 0.0,
        "I-ocr-cross-ocr_new_word_ratio": float(len(ocr_words - post_words) / len(ocr_words)) if ocr_words else 0.0,
        "I-ocr-cross-exact_text_match": float(bool(post.strip()) and post.strip().lower() == ocr.strip().lower()),
    }


def score(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def add_hashtag(parts: list[pd.DataFrame], tags: pd.Series) -> pd.DataFrame:
    output = pd.concat(parts, axis=1)
    output.insert(0, "hashtag", tags)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--shallow37", type=Path, required=True)
    parser.add_argument("--extra2", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--ocr-by-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of XGBoost worker threads. Defaults to one for reproducible ablations.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters = {**PARAMETERS, "n_jobs": args.n_jobs}

    hybrid_header = pd.read_csv(args.hybrid, nrows=0).columns.tolist()
    message_columns = [column for column in hybrid_header if column.startswith("M-")]
    shallow_columns = [column for column in pd.read_csv(args.shallow37, nrows=0).columns if column.startswith("I-nonsem-")]
    clip_columns = [column for column in pd.read_csv(args.clip, nrows=0).columns if column.startswith("I-clip-")]
    if (len(message_columns), len(shallow_columns), len(clip_columns)) != (66, 37, 512):
        raise ValueError("Expected M66, 37 shallow image features, and CLIP512.")

    base = pd.read_csv(args.hybrid, usecols=["P_id", "hashtag", "label"] + message_columns)
    base["_row_order"] = np.arange(len(base))
    shallow = pd.read_csv(args.shallow37, usecols=["post_uri"] + shallow_columns)
    extra = pd.read_csv(args.extra2, usecols=["post_uri"] + EXTRA_COLUMNS)
    clip = pd.read_csv(args.clip, usecols=["post_uri"] + clip_columns)
    for frame, name in ((shallow, "shallow"), (extra, "extra"), (clip, "CLIP")):
        if frame.post_uri.duplicated().any():
            raise ValueError(f"Duplicate post_uri in {name} table.")
    data = base.merge(shallow, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one").drop(columns="post_uri")
    data = data.merge(extra, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one").drop(columns="post_uri")
    data = data.merge(clip, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one").drop(columns="post_uri")
    data = data.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)

    needed = ["post_uri", "image_number", "post_text", "ocr_status", "ocr_text"]
    ocr_header = pd.read_csv(args.ocr_by_image, nrows=0).columns.tolist()
    missing_ocr_columns = set(needed) - set(ocr_header)
    if missing_ocr_columns:
        raise ValueError(f"OCR input is missing required columns: {sorted(missing_ocr_columns)}")
    ocr = pd.read_csv(
        args.ocr_by_image,
        usecols=needed,
        low_memory=False,
    )
    ocr = ocr.loc[ocr.ocr_status.eq("ok")].copy()
    ocr = ocr.loc[ocr.ocr_text.fillna("").astype(str).str.strip().ne("")]
    ocr = ocr.sort_values(["post_uri", "image_number"], kind="mergesort")
    posts = ocr.groupby("post_uri", as_index=False).agg(
        post_text=("post_text", "first"),
        ocr_text=("ocr_text", lambda values: " ".join(values.astype(str))),
        ocr_image_count=("image_number", "size"),
    )
    cross = pd.DataFrame(
        [cross_features(row.post_text, row.ocr_text, int(row.ocr_image_count)) for row in posts.itertuples(index=False)],
        dtype="float32",
    )
    cross_columns = cross.columns.tolist()
    posts = pd.concat([posts.reset_index(drop=True), cross], axis=1)
    data = data.merge(
        posts[["post_uri"] + cross_columns],
        left_on="P_id", right_on="post_uri", how="left", validate="many_to_one",
    ).drop(columns="post_uri")

    # Retain only samples from posts with non-empty OCR-recognised text.
    selected = data[cross_columns[0]].notna().to_numpy()
    data = data.loc[selected].reset_index(drop=True)
    if data.empty:
        raise ValueError("No hybrid samples matched posts with usable OCR text.")

    y = data.label.to_numpy(dtype=int)
    hashtags = data.hashtag.fillna("missing").astype(str).astype("category")
    strata = data.hashtag.fillna("missing").astype(str) + "__" + data.label.astype(str)
    message = data[message_columns].fillna(0).astype("float32")
    nonsemantic = data[shallow_columns + EXTRA_COLUMNS].fillna(0).astype("float32")
    raw_clip = data[clip_columns].fillna(0).astype("float32").to_numpy()
    clip_present = data[clip_columns].notna().all(axis=1).to_numpy()
    ocr_cross = data[cross_columns].fillna(0).astype("float32")
    if len(ocr_cross.columns) != 14:
        raise ValueError(f"Expected the 14 final OCR cross-content features, found {len(ocr_cross.columns)}.")
    novelty = ocr_cross["I-ocr-cross-ocr_new_word_ratio"].to_numpy(dtype=float)
    print(json.dumps({"samples": len(data), "positive": int(y.sum()), "negative": int((1 - y).sum())}), flush=True)
    rows: list[dict[str, object]] = []
    thresholds: list[dict[str, object]] = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)

    for fold, (train, test) in enumerate(splitter.split(np.arange(len(data)), strata), start=1):
        threshold = float(np.median(novelty[train]))
        high_test = novelty[test] >= threshold
        groups = {
            "all_ocr_text": np.ones(len(test), dtype=bool),
            "lower_new_content": ~high_test,
            "higher_new_content": high_test,
        }
        thresholds.append({
            "fold": fold,
            "training_median_new_word_ratio": threshold,
            "test_lower_new_content_samples": int((~high_test).sum()),
            "test_higher_new_content_samples": int(high_test.sum()),
        })

        pca_rows = train[clip_present[train]]
        pca = PCA(n_components=64, svd_solver="randomized", random_state=args.seed + fold)
        pca.fit(raw_clip[pca_rows])
        pcs = np.zeros((len(data), 64), dtype="float32")
        present = np.flatnonzero(clip_present)
        pcs[present] = pca.transform(raw_clip[present]).astype("float32")
        clip_pcs = pd.DataFrame(pcs, columns=[f"I-clip-pca64-{number:03d}" for number in range(64)])
        feature_sets = {
            "MI": add_hashtag([message, nonsemantic, clip_pcs], hashtags),
            "MI_plus_OCR": add_hashtag([message, nonsemantic, clip_pcs, ocr_cross], hashtags),
        }

        for condition, features in feature_sets.items():
            model = XGBClassifier(**parameters, random_state=args.seed + fold)
            model.fit(features.iloc[train], y[train])
            prediction = model.predict(features.iloc[test])
            for group, mask in groups.items():
                group_y = y[test][mask]
                group_prediction = prediction[mask]
                result = {
                    "fold": fold,
                    "evaluation_group": group,
                    "condition": condition,
                    **score(group_y, group_prediction),
                    "test_samples": len(group_y),
                    "test_positive": int(group_y.sum()),
                    "training_median_new_word_ratio": threshold,
                    "pca_fit_samples": len(pca_rows),
                    "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                }
                rows.append(result)
                print(json.dumps(result), flush=True)

    results = pd.DataFrame(rows)
    summary = results.groupby(["evaluation_group", "condition"], as_index=False).agg(
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
        test_samples_mean=("test_samples", "mean"), runs=("f1", "count"),
    )
    pivot = results.pivot(index=["fold", "evaluation_group"], columns="condition", values="f1").reset_index()
    deltas = pivot[["fold", "evaluation_group"]].copy()
    comparison_columns = [condition for condition in feature_sets if condition != "MI"]
    for condition in comparison_columns:
        deltas[f"MI_to_{condition}_delta_f1"] = pivot[condition] - pivot["MI"]
    delta_summary = deltas.groupby("evaluation_group", as_index=False).agg(
        **{
            f"{column}_mean": (column, "mean")
            for column in deltas.columns if column.startswith("MI_to_")
        },
        **{
            f"{column}_sd": (column, "std")
            for column in deltas.columns if column.startswith("MI_to_")
        },
        folds=("fold", "count"),
    )

    results.to_csv(args.output_dir / "ocr_new_content_group_run_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "ocr_new_content_group_summary.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(args.output_dir / "ocr_new_content_group_fold_deltas.csv", index=False, encoding="utf-8-sig")
    delta_summary.to_csv(args.output_dir / "ocr_new_content_group_delta_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(thresholds).to_csv(args.output_dir / "ocr_new_content_group_thresholds.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "purpose_a": "overall MI versus MI plus OCR cross-content features among samples with usable OCR text",
        "purpose_b": "the same comparison in high and low new-content test groups",
        "high_low_rule": "within each fold, the training-sample median OCR new-word ratio is the threshold; high is greater than or equal to that threshold",
        "protocol": "three-fold mixed evaluation jointly stratified by hashtag and label",
        "base_features": "hashtag, M66, 39 non-semantic image features, and train-fold CLIP PCA64",
        "ocr_features": cross_columns,
        "ocr_feature_count": len(cross_columns),
        "excluded_feature": "I-ocr-cross-ocr_exclamation_count",
        "ocr_m66_features_used": False,
        "parameters": parameters,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nSummary")
    print(summary.to_string(index=False))
    print("\nMI to MI + OCR F1 changes")
    print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
