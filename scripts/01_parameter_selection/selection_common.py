"""Shared data preparation for the dissertation's selection scripts.

This module was written for this dissertation.  It joins Jonas's hybrid input
table to the visual feature tables and creates the fixed development splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE = {
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
EXTRA2 = ["I-nonsem-extra-image_pixel_count", "I-nonsem-extra-aspect_ratio"]


def metric_row(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def add_hashtag(parts: list[pd.DataFrame], tags: pd.Series) -> pd.DataFrame:
    frame = pd.concat(parts, axis=1)
    frame.insert(0, "hashtag", tags)
    return frame


def split_indices(data: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create one jointly stratified 63 percent training and 7 percent validation split."""
    index = np.arange(len(data))
    strata = data["hashtag"].astype(str) + "__" + data["label"].astype(str)
    train, remainder = train_test_split(
        index, train_size=0.63, random_state=seed, stratify=strata
    )
    validation, _ = train_test_split(
        remainder,
        train_size=7 / 37,
        random_state=seed,
        stratify=strata.iloc[remainder],
    )
    return np.sort(train), np.sort(validation)


def load_data(hybrid_path, shallow_path, extra_path, clip_path):
    """Merge Jonas M66 input with the dissertation's post-level visual inputs."""
    hybrid_columns = pd.read_csv(hybrid_path, nrows=0).columns.tolist()
    message_columns = [column for column in hybrid_columns if column.startswith("M-")]
    shallow_columns = [
        column for column in pd.read_csv(shallow_path, nrows=0).columns
        if column.startswith("I-nonsem-")
    ]
    clip_columns = [
        column for column in pd.read_csv(clip_path, nrows=0).columns
        if column.startswith("I-clip-")
    ]
    if (len(message_columns), len(shallow_columns), len(clip_columns)) != (66, 37, 512):
        raise ValueError("Expected M66, 37 non-semantic image features, and CLIP512.")

    base = pd.read_csv(hybrid_path, usecols=["P_id", "hashtag", "label"] + message_columns)
    base["_row"] = np.arange(len(base))
    shallow = pd.read_csv(shallow_path, usecols=["post_uri"] + shallow_columns)
    extra = pd.read_csv(extra_path, usecols=["post_uri"] + EXTRA2)
    clip = pd.read_csv(clip_path, usecols=["post_uri"] + clip_columns)
    for frame, name in ((shallow, "shallow"), (extra, "extra"), (clip, "CLIP")):
        if frame["post_uri"].duplicated().any():
            raise ValueError(f"Duplicate post_uri in {name} feature table")

    data = base.merge(shallow, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one", sort=False)
    data = data.drop(columns="post_uri").merge(extra, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one", sort=False)
    data = data.drop(columns="post_uri").merge(clip, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one", sort=False)
    data = data.drop(columns="post_uri").sort_values("_row").drop(columns="_row").reset_index(drop=True)
    return data, message_columns, shallow_columns, clip_columns
