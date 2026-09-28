"""Select fixed XGBoost parameters using MI validation F1 across three splits.

All 75 candidates are evaluated on the same three independently stratified
63/7/30 development splits.  Only MI is used for selection because it is the
thesis's primary multimodal model.  The 30% portion of each split is not used
in this development-stage selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from selection_common import BASE, EXTRA2, add_hashtag, load_data, metric_row, split_indices


SPLIT_SEEDS = (42, 101, 202)
CANDIDATES = {
    f"lr{learning_rate:g}_depth{depth}_weight{weight}": dict(
        learning_rate=learning_rate,
        max_depth=depth,
        scale_pos_weight=weight,
    )
    for learning_rate in (0.1, 0.2, 0.3)
    for depth in (3, 4, 5, 6, 7)
    for weight in (1, 2, 3, 4, 5)
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--shallow37", type=Path, required=True)
    parser.add_argument("--extra2", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, m_cols, shallow_cols, clip_cols = load_data(args.hybrid, args.shallow37, args.extra2, args.clip)
    y = data["label"].to_numpy(dtype=int)
    tags = data["hashtag"].fillna("missing").astype(str).astype("category")
    message = data[m_cols].fillna(0).astype("float32")
    nonsemantic = data[shallow_cols + EXTRA2].fillna(0).astype("float32")
    raw_clip = data[clip_cols].fillna(0).astype("float32").to_numpy()
    clip_present = data[clip_cols].notna().all(axis=1).to_numpy()

    rows: list[dict[str, float | int | str]] = []
    for seed in SPLIT_SEEDS:
        train, validation = split_indices(data, seed)
        pca_rows = train[clip_present[train]]
        pca = PCA(n_components=16, svd_solver="randomized", random_state=seed)
        pca.fit(raw_clip[pca_rows])
        pcs = np.zeros((len(data), 16), dtype="float32")
        present_rows = np.flatnonzero(clip_present)
        pcs[present_rows] = pca.transform(raw_clip[present_rows]).astype("float32")
        clip_pcs = pd.DataFrame(
            pcs, columns=[f"I-clip-pca16-{index:02d}" for index in range(16)]
        )
        mi = add_hashtag([message, nonsemantic, clip_pcs], tags)
        print(
            f"seed={seed}, train={len(train)}, validation={len(validation)}, "
            f"PCA-fit rows={len(pca_rows)}",
            flush=True,
        )

        for candidate, parameters in CANDIDATES.items():
            model = XGBClassifier(
                **BASE, **parameters, random_state=seed,
            )
            model.fit(mi.iloc[train], y[train])
            metrics = metric_row(y[validation], model.predict(mi.iloc[validation]))
            row = {
                "split_seed": seed,
                "candidate": candidate,
                **parameters,
                **metrics,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "pca16_fit_rows": len(pca_rows),
                "pca16_explained_variance": float(pca.explained_variance_ratio_.sum()),
            }
            rows.append(row)
            print(
                f"seed={seed} {candidate} MI_F1={metrics['f1']:.6f}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(
            ["candidate", "learning_rate", "max_depth", "scale_pos_weight"],
            as_index=False,
        )
        .agg(
            mi_f1_mean=("f1", "mean"),
            mi_f1_sd=("f1", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
        )
    )
    split_winners = results.loc[results.groupby("split_seed")["f1"].idxmax()]
    winner_counts = split_winners["candidate"].value_counts()
    summary["split_wins"] = summary["candidate"].map(winner_counts).fillna(0).astype(int)
    summary = summary.sort_values(
        ["mi_f1_mean", "mi_f1_sd"], ascending=[False, True]
    ).reset_index(drop=True)
    selected = summary.iloc[0]

    results.to_csv(args.output_dir / "candidate_run_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "selection_rule": "highest mean MI validation F1 across three development splits",
        "split_seeds": list(SPLIT_SEEDS),
        "split_protocol": "jointly stratified by hashtag and label, 63% train, 7% validation, 30% unused",
        "pca": "16 components fitted only on CLIP-present training records within each split",
        "fixed_parameters": BASE,
        "candidate_count": len(CANDIDATES),
        "selected_candidate": str(selected["candidate"]),
        "selected_parameters": {
            **BASE,
            "learning_rate": float(selected["learning_rate"]),
            "max_depth": int(selected["max_depth"]),
            "scale_pos_weight": float(selected["scale_pos_weight"]),
        },
        "selected_scores": {
            "MI_F1_mean": float(selected["mi_f1_mean"]),
            "MI_F1_sd": float(selected["mi_f1_sd"]),
            "split_wins": int(selected["split_wins"]),
        },
    }
    (args.output_dir / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nSelected training parameters")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
