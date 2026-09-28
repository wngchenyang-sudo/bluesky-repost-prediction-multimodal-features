"""Sequential removal for UMI features with Mixed-ID and OOD curves.

The model contains U172, M66, 39 non-semantic image features, CLIP PCA64,
and hashtag as a categorical control variable. Hashtag remains in every fitted
model, but is excluded from the UMI gain ranking and can never be removed.
At every step, the UMI feature with the highest mean XGBoost gain across the
three Mixed-ID folds is removed. The same cumulative removal list is then
evaluated under the formal OOD protocol.

The script supports an initial-gain-only mode and resumable 0--70 removal.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SEED = 42
PCA_DIMENSIONS = 64
EXTRA_COLUMNS = [
    "I-nonsem-extra-image_pixel_count",
    "I-nonsem-extra-aspect_ratio",
]
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
}
RUN_CONFIGURATION = {
    "gain_definition": "XGBClassifier.feature_importances_: normalised average split gain",
    "gain_aggregation": "mean across the three fixed Mixed-ID folds",
    "hashtag": "categorical control retained in all models; excluded from ranking and removal",
}


@dataclass
class PreparedSplit:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_hashtags: pd.Series
    test_hashtags: pd.Series
    train_clip_path: Path
    test_clip_path: Path
    pca_fit_samples: int
    explained_variance: float


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def joint_strata(frame: pd.DataFrame) -> pd.Series:
    tags = frame["hashtag"].astype("object").fillna("missing").astype(str)
    return tags + "__" + frame["label"].astype(str)


def load_data(args: argparse.Namespace):
    hybrid_header = pd.read_csv(args.hybrid, nrows=0).columns.tolist()
    user_columns = [column for column in hybrid_header if column.startswith("U-")]
    message_columns = [column for column in hybrid_header if column.startswith("M-")]
    shallow_columns = [
        column
        for column in pd.read_csv(args.shallow37, nrows=0).columns
        if column.startswith("I-nonsem-")
    ]
    clip_columns = [
        column
        for column in pd.read_csv(args.clip, nrows=0).columns
        if column.startswith("I-clip-")
    ]
    if (len(user_columns), len(message_columns), len(shallow_columns), len(clip_columns)) != (172, 66, 37, 512):
        raise ValueError(
            "Expected U172, M66, 37 shallow image features, and CLIP512; "
            f"found U{len(user_columns)}, M{len(message_columns)}, "
            f"shallow{len(shallow_columns)}, CLIP{len(clip_columns)}."
        )

    base = pd.read_csv(
        args.hybrid,
        usecols=["P_id", "hashtag", "label"] + user_columns + message_columns,
    )
    base["_row_order"] = np.arange(len(base))
    shallow = pd.read_csv(args.shallow37, usecols=["post_uri"] + shallow_columns)
    extra = pd.read_csv(args.extra2, usecols=["post_uri"] + EXTRA_COLUMNS)
    clip = pd.read_csv(args.clip, usecols=["post_uri"] + clip_columns)
    for frame, name in ((shallow, "shallow"), (extra, "extra"), (clip, "CLIP")):
        if frame["post_uri"].duplicated().any():
            raise ValueError(f"Duplicate post_uri in {name} table.")

    data = base.merge(
        shallow, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one"
    ).drop(columns="post_uri")
    data = data.merge(
        extra, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one"
    ).drop(columns="post_uri")
    data = data.merge(
        clip, left_on="P_id", right_on="post_uri", how="left", validate="many_to_one"
    ).drop(columns="post_uri")
    data = data.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
    # Keep one global category mapping, matching the formal UMI evaluation.
    data["hashtag"] = data["hashtag"].fillna("missing").astype(str).astype("category")

    nonsemantic_columns = shallow_columns + EXTRA_COLUMNS
    base_columns = user_columns + message_columns + nonsemantic_columns
    base_values = data[base_columns].fillna(0).astype("float32").to_numpy()
    raw_clip = data[clip_columns].fillna(0).astype("float32").to_numpy()
    clip_present = data[clip_columns].notna().all(axis=1).to_numpy()
    pca_columns = [f"I-clip-pca64-{number:03d}" for number in range(PCA_DIMENSIONS)]
    feature_groups = {
        **{name: "user" for name in user_columns},
        **{name: "message" for name in message_columns},
        **{name: "nonsemantic_image" for name in nonsemantic_columns},
        **{name: "clip_pca64" for name in pca_columns},
    }
    return data, base_values, raw_clip, clip_present, base_columns, pca_columns, feature_groups


def transform_clip(
    pca: PCA,
    raw_clip: np.ndarray,
    clip_present: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(indices), PCA_DIMENSIONS), dtype="float32")
    present_positions = np.flatnonzero(clip_present[indices])
    if len(present_positions):
        result[present_positions] = pca.transform(raw_clip[indices[present_positions]]).astype("float32")
    return result


def prepare_split(
    fold: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    raw_clip: np.ndarray,
    clip_present: np.ndarray,
    hashtags: pd.Series,
    cache_directory: Path,
    cache_key: str,
    held_out_hashtag: str | None = None,
) -> PreparedSplit:
    pca_rows = train_indices[clip_present[train_indices]]
    if len(pca_rows) < PCA_DIMENSIONS:
        raise ValueError(f"Fold {fold} has only {len(pca_rows)} CLIP-present PCA rows.")
    pca = PCA(n_components=PCA_DIMENSIONS, svd_solver="randomized", random_state=SEED + fold)
    pca.fit(raw_clip[pca_rows])
    train_hashtags = hashtags.iloc[train_indices].reset_index(drop=True)
    if held_out_hashtag is None:
        test_hashtags = hashtags.iloc[test_indices].reset_index(drop=True)
    else:
        # Jonas-style OOD treatment: the held-out category was never observed
        # in training, so encode it as an unknown/missing category at test time.
        test_hashtags = pd.Series(
            pd.Categorical(
                [np.nan] * len(test_indices),
                categories=train_hashtags.cat.categories,
            )
        )
    cache_directory.mkdir(parents=True, exist_ok=True)
    train_clip_path = cache_directory / f"{cache_key}_train_pca64.npy"
    test_clip_path = cache_directory / f"{cache_key}_test_pca64.npy"
    if not train_clip_path.exists():
        np.save(train_clip_path, transform_clip(pca, raw_clip, clip_present, train_indices))
    if not test_clip_path.exists():
        np.save(test_clip_path, transform_clip(pca, raw_clip, clip_present, test_indices))
    return PreparedSplit(
        fold=fold,
        train_indices=train_indices,
        test_indices=test_indices,
        train_hashtags=train_hashtags,
        test_hashtags=test_hashtags,
        train_clip_path=train_clip_path,
        test_clip_path=test_clip_path,
        pca_fit_samples=len(pca_rows),
        explained_variance=float(pca.explained_variance_ratio_.sum()),
    )


def prepare_mixed_splits(data, raw_clip, clip_present, cache_directory: Path) -> list[PreparedSplit]:
    indices = np.arange(len(data))
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    prepared = []
    for fold, (train, test) in enumerate(splitter.split(indices, joint_strata(data)), start=1):
        print(f"Preparing Mixed fold {fold}/3 PCA...", flush=True)
        prepared.append(
            prepare_split(
                fold, train, test, raw_clip, clip_present, data["hashtag"],
                cache_directory, f"mixed_fold{fold}",
            )
        )
    return prepared


def prepare_ood_splits(data, raw_clip, clip_present, cache_directory: Path) -> dict[str, list[PreparedSplit]]:
    tags = data["hashtag"].astype("object").fillna("missing").astype(str)
    result: dict[str, list[PreparedSplit]] = {}
    for tag_number, held_tag in enumerate(sorted(tags.unique()), start=1):
        held_indices = np.flatnonzero(tags.eq(held_tag).to_numpy())
        pool_indices = np.flatnonzero(~tags.eq(held_tag).to_numpy())
        pool_frame = data.iloc[pool_indices]
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        result[held_tag] = []
        for fold, (train_positions, _unused) in enumerate(
            splitter.split(np.arange(len(pool_indices)), joint_strata(pool_frame)), start=1
        ):
            train_indices = pool_indices[train_positions]
            print(
                f"Preparing OOD PCA: {held_tag} ({tag_number}/10), fold {fold}/3...",
                flush=True,
            )
            result[held_tag].append(
                prepare_split(
                    fold, train_indices, held_indices, raw_clip, clip_present,
                    data["hashtag"], cache_directory, f"ood_{held_tag}_fold{fold}",
                    held_out_hashtag=held_tag,
                )
            )
    return result


def build_matrix(
    base_values: np.ndarray,
    base_columns: list[str],
    pca_columns: list[str],
    split: PreparedSplit,
    remaining_features: list[str],
    train: bool,
) -> pd.DataFrame:
    indices = split.train_indices if train else split.test_indices
    clip_values = np.load(split.train_clip_path if train else split.test_clip_path, mmap_mode="r")
    base_lookup = {name: number for number, name in enumerate(base_columns)}
    pca_lookup = {name: number for number, name in enumerate(pca_columns)}
    remaining_base = [name for name in remaining_features if name in base_lookup]
    remaining_pca = [name for name in remaining_features if name in pca_lookup]
    parts = []
    if remaining_base:
        parts.append(base_values[indices][:, [base_lookup[name] for name in remaining_base]])
    if remaining_pca:
        parts.append(np.asarray(clip_values[:, [pca_lookup[name] for name in remaining_pca]]))
    values = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    matrix = pd.DataFrame(values, columns=remaining_base + remaining_pca)
    matrix.insert(0, "hashtag", split.train_hashtags if train else split.test_hashtags)
    return matrix


def fit_split(
    split: PreparedSplit,
    data: pd.DataFrame,
    base_values: np.ndarray,
    base_columns: list[str],
    pca_columns: list[str],
    remaining_features: list[str],
    n_jobs: int,
    return_gain: bool,
):
    x_train = build_matrix(
        base_values, base_columns, pca_columns, split, remaining_features, train=True
    )
    x_test = build_matrix(
        base_values, base_columns, pca_columns, split, remaining_features, train=False
    )
    y_train = data["label"].to_numpy(dtype=int)[split.train_indices]
    y_test = data["label"].to_numpy(dtype=int)[split.test_indices]
    model = XGBClassifier(**PARAMETERS, n_jobs=n_jobs, random_state=SEED + split.fold)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    f1 = float(f1_score(y_test, prediction))
    gain = None
    if return_gain:
        # Jonas uses XGBClassifier.feature_importances_: average split gain
        # normalised to sum to one within each fitted model.
        names = list(model.feature_names_in_)
        importances = model.feature_importances_
        normalised_gain = dict(zip(names, importances, strict=True))
        gain = {name: float(normalised_gain.get(name, 0.0)) for name in remaining_features}
        # Diagnostic only: hashtag remains a control input, never a ranked feature.
        gain["__hashtag_control_gain__"] = float(normalised_gain.get("hashtag", 0.0))
    del x_train, x_test, model
    return f1, gain


def initial_gain_frame(fold_gains, remaining_features, feature_groups):
    mean_gain = pd.DataFrame(fold_gains).reindex(columns=remaining_features).fillna(0).mean(axis=0)
    ranking = mean_gain.sort_values(ascending=False, kind="mergesort").rename("mean_gain").reset_index()
    ranking = ranking.rename(columns={"index": "feature_name"})
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.insert(2, "feature_group", ranking["feature_name"].map(feature_groups))
    return ranking


def plot_curves(summary: pd.DataFrame, output_dir: Path, max_steps: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is unavailable; CSV results were saved but the plot was skipped.", flush=True)
        return

    ordered = summary.sort_values("step")
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(
        ordered["step"], ordered["mixed_f1_mean"], marker="o", markersize=3.5,
        linewidth=1.7, color="#356A9A", label="Mixed F1",
    )
    ax.plot(
        ordered["step"], ordered["ood_macro_f1"], marker="o", markersize=3.5,
        linewidth=1.7, color="#D17832", label="OOD Avg F1",
    )
    # Add visual breathing room around the first and last data points.  The
    # endpoints are intentionally not tick-labelled: analytical steps remain
    # 0, 7, ..., 70.
    ax.set_xlim(-5, max_steps + 5)
    ax.set_xticks(np.arange(0, max_steps + 1, 7))
    ax.set_xlabel("Cumulative Number of Features Removed")
    ax.set_ylabel("F1 Score")
    ax.set_title("Effect of Sequential UMI Feature Removal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "umi_sequential_removal_mixed_ood.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "umi_sequential_removal_mixed_ood.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--shallow37", type=Path, required=True)
    parser.add_argument("--extra2", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--initial-gain-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.steps > 340:
        raise ValueError("--steps must be between 0 and 340.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        data,
        base_values,
        raw_clip,
        clip_present,
        base_columns,
        pca_columns,
        feature_groups,
    ) = load_data(args)
    all_features = base_columns + pca_columns
    if len(all_features) != 341:
        raise ValueError(f"Expected 341 rankable UMI features, found {len(all_features)}.")

    cache_directory = args.output_dir / "pca64_cache"
    mixed_splits = prepare_mixed_splits(data, raw_clip, clip_present, cache_directory / "mixed")

    if args.initial_gain_only:
        fold_gains = []
        fold_rows = []
        for split in mixed_splits:
            f1, gain = fit_split(
                split, data, base_values, base_columns, pca_columns,
                all_features, args.n_jobs, return_gain=True,
            )
            fold_rows.append({
                "fold": split.fold,
                "f1": f1,
                "hashtag_control_gain": gain["__hashtag_control_gain__"],
            })
            fold_gains.append(gain)
        ranking = initial_gain_frame(fold_gains, all_features, feature_groups)
        atomic_csv(ranking, args.output_dir / "umi_initial_gain_full.csv")
        atomic_csv(ranking.head(20), args.output_dir / "umi_initial_gain_top20.csv")
        atomic_csv(pd.DataFrame(fold_rows), args.output_dir / "umi_initial_gain_mixed_folds.csv")
        print(ranking.head(20).to_string(index=False), flush=True)
        return

    print("Preparing the 30 fixed OOD train/test splits and PCA transforms...", flush=True)
    ood_splits = prepare_ood_splits(data, raw_clip, clip_present, cache_directory / "ood")

    summary_path = args.output_dir / "sequential_removal_summary.csv"
    mixed_path = args.output_dir / "mixed_fold_results.csv"
    ood_path = args.output_dir / "ood_fold_results.csv"
    gain_path = args.output_dir / "gain_by_step.csv"
    run_configuration_path = args.output_dir / "run_configuration.json"

    if run_configuration_path.exists():
        existing_configuration = json.loads(run_configuration_path.read_text(encoding="utf-8"))
        if existing_configuration != RUN_CONFIGURATION:
            raise ValueError(
                "This output directory contains results from an incompatible run. "
                "Use a new --output-dir so gain definitions are not mixed."
            )
    elif summary_path.exists():
        raise ValueError(
            "This output directory contains legacy partial results without a run configuration. "
            "Use a new --output-dir so gain definitions are not mixed."
        )
    else:
        run_configuration_path.write_text(
            json.dumps(RUN_CONFIGURATION, indent=2), encoding="utf-8"
        )

    summary_rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    mixed_rows = pd.read_csv(mixed_path).to_dict("records") if mixed_path.exists() else []
    ood_rows = pd.read_csv(ood_path).to_dict("records") if ood_path.exists() else []
    gain_rows = pd.read_csv(gain_path).to_dict("records") if gain_path.exists() else []

    removed = [
        str(row["removed_feature"])
        for row in sorted(summary_rows, key=lambda item: int(item["step"]))
        if int(row["step"]) > 0
    ]
    start_step = 0
    if summary_rows:
        last = max(summary_rows, key=lambda item: int(item["step"]))
        start_step = int(last["step"]) + 1
        if start_step <= args.steps:
            next_feature = str(last["next_removed_feature"])
            if next_feature and next_feature != "nan" and next_feature not in removed:
                removed.append(next_feature)
        print(f"Resuming from step {start_step}; already removed {len(removed)} features.", flush=True)

    tags = data["hashtag"].astype("object").fillna("missing").astype(str)
    for step in range(start_step, args.steps + 1):
        remaining = [feature for feature in all_features if feature not in set(removed)]
        print(f"\n=== Step {step}/{args.steps}: {len(remaining)} features remain ===", flush=True)

        fold_gains = []
        step_mixed_rows = []
        for split in mixed_splits:
            f1, gain = fit_split(
                split, data, base_values, base_columns, pca_columns,
                remaining, args.n_jobs, return_gain=True,
            )
            step_mixed_rows.append({"step": step, "fold": split.fold, "f1": f1})
            fold_gains.append(gain)
            print(f"Mixed fold {split.fold}: F1={f1:.6f}", flush=True)

        ranking = initial_gain_frame(fold_gains, remaining, feature_groups)
        next_removed = str(ranking.iloc[0]["feature_name"]) if step < args.steps else ""
        for row in ranking.to_dict("records"):
            gain_rows.append({"step": step, **row, "selected_next": row["feature_name"] == next_removed})

        step_ood_rows = []
        hashtag_means = []
        for held_tag, splits in ood_splits.items():
            tag_scores = []
            for split in splits:
                f1, _ = fit_split(
                    split, data, base_values, base_columns, pca_columns,
                    remaining, args.n_jobs, return_gain=False,
                )
                tag_scores.append(f1)
                step_ood_rows.append(
                    {"step": step, "held_out_hashtag": held_tag, "fold": split.fold, "f1": f1}
                )
            tag_mean = float(np.mean(tag_scores))
            hashtag_means.append(tag_mean)
            print(f"OOD {held_tag}: mean F1={tag_mean:.6f}", flush=True)

        mixed_values = np.array([row["f1"] for row in step_mixed_rows], dtype=float)
        hashtag_values = np.array(hashtag_means, dtype=float)
        summary_rows.append(
            {
                "step": step,
                "removed_feature": "None" if step == 0 else removed[-1],
                "remaining_feature_count": len(remaining),
                "mixed_f1_mean": float(mixed_values.mean()),
                "mixed_f1_sd": float(mixed_values.std(ddof=1)),
                "ood_macro_f1": float(hashtag_values.mean()),
                "ood_hashtag_sd": float(hashtag_values.std(ddof=1)),
                "next_removed_feature": next_removed,
                "next_removed_mean_gain": float(ranking.iloc[0]["mean_gain"]) if next_removed else np.nan,
            }
        )
        mixed_rows.extend(step_mixed_rows)
        ood_rows.extend(step_ood_rows)

        atomic_csv(pd.DataFrame(summary_rows), summary_path)
        atomic_csv(pd.DataFrame(mixed_rows), mixed_path)
        atomic_csv(pd.DataFrame(ood_rows), ood_path)
        atomic_csv(pd.DataFrame(gain_rows), gain_path)
        if step == 0:
            atomic_csv(ranking, args.output_dir / "umi_initial_gain_full.csv")
            atomic_csv(ranking.head(20), args.output_dir / "umi_initial_gain_top20.csv")
        plot_curves(pd.DataFrame(summary_rows), args.output_dir, args.steps)

        print(json.dumps(summary_rows[-1]), flush=True)
        if step < args.steps:
            removed.append(next_removed)

    metadata = {
        "feature_set": "UMI: hashtag control + U172 + M66 + 39 non-semantic image + CLIP PCA64",
        "initial_feature_count": len(all_features),
        "always_retained_control": "hashtag (categorical; excluded from ranking and removal)",
        "steps": args.steps,
        "selection_rule": "highest mean gain across the three fixed Mixed-ID folds",
        "mixed_protocol": "fixed three-fold CV jointly stratified by hashtag and label",
        "ood_protocol": "ten fixed held-out hashtags; three joint hashtag-label stratified training folds per hashtag",
        "ood_hashtag_encoding": "held-out test hashtag encoded as missing, following Jonas-style unseen-category handling",
        "ood_aggregation": "mean across three runs per hashtag, followed by an unweighted mean across hashtags",
        "x_axis_groups": "0--70 with major ticks every 7 removals",
        "parameters": {**PARAMETERS, "n_jobs": args.n_jobs},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
