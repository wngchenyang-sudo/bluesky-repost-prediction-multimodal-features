"""Verify and aggregate full-image CLIP features to one row per Bluesky post.

No source file is changed.  Matching requires both local_path and sha256, then
post-level CLIP is the arithmetic mean over that post's successful image pairs.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np


KEY_COLUMNS = ("local_path", "sha256")
PAIR_COLUMNS = ("local_path", "sha256", "image_url", "image_cid", "post_uri", "image_number")


def open_csv(path: Path):
    return path.open("r", encoding="utf-8-sig", newline="")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--reference-map", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["mean", "first_downloaded"], default="mean")
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Load one vector per physical image; fail rather than silently accept a
    # duplicated key or a non-512-dimensional feature row.
    vectors: dict[tuple[str, str], np.ndarray] = {}
    with open_csv(args.features) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Feature CSV has no header")
        clip_cols = [c for c in reader.fieldnames if c.startswith("I-clip-")]
        if len(clip_cols) != 512:
            raise ValueError(f"Expected 512 I-clip-* columns, found {len(clip_cols)}")
        for line, row in enumerate(reader, start=2):
            key = tuple(row[c] for c in KEY_COLUMNS)
            if not all(key) or key in vectors:
                raise ValueError(f"Invalid or duplicate feature key at line {line}: {key}")
            vector = np.asarray([float(row[c]) for c in clip_cols], dtype=np.float64)
            if not np.isfinite(vector).all():
                raise ValueError(f"Non-finite CLIP value at feature line {line}")
            vectors[key] = vector

    # Manifest is the immutable source-of-truth for every post/image relation.
    manifest_pairs: set[tuple[str, ...]] = set()
    with open_csv(args.manifest) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "downloaded":
                pair = tuple(row[c] for c in PAIR_COLUMNS)
                if pair in manifest_pairs:
                    raise ValueError("Duplicate successful manifest pair: " + repr(pair))
                manifest_pairs.add(pair)

    sums: OrderedDict[str, np.ndarray] = OrderedDict()
    counts: Counter[str] = Counter()
    first_by_post: OrderedDict[str, tuple[int, str, str, np.ndarray]] = OrderedDict()
    map_pairs: set[tuple[str, ...]] = set()
    missing_feature_pairs: list[tuple[str, ...]] = []
    with open_csv(args.reference_map) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or []) != PAIR_COLUMNS:
            raise ValueError("Reference-map columns differ from expected order: " + repr(reader.fieldnames))
        for line, row in enumerate(reader, start=2):
            pair = tuple(row[c] for c in PAIR_COLUMNS)
            if pair in map_pairs:
                raise ValueError(f"Duplicate reference-map pair at line {line}")
            map_pairs.add(pair)
            if pair not in manifest_pairs:
                raise ValueError(f"Reference-map row not present in successful manifest: line {line}")
            key = (row["local_path"], row["sha256"])
            vector = vectors.get(key)
            if vector is None:
                missing_feature_pairs.append(pair)
                continue
            post_uri = row["post_uri"]
            if post_uri not in sums:
                sums[post_uri] = np.zeros(512, dtype=np.float64)
            sums[post_uri] += vector
            counts[post_uri] += 1
            image_number = int(row["image_number"])
            candidate = (image_number, row["local_path"], row["image_cid"], vector)
            existing = first_by_post.get(post_uri)
            if existing is None or candidate[0] < existing[0]:
                first_by_post[post_uri] = candidate

    if len(map_pairs) != len(manifest_pairs):
        raise ValueError(f"Reference-map/manifest successful-pair count differs: {len(map_pairs)} vs {len(manifest_pairs)}")
    if missing_feature_pairs:
        raise ValueError(f"{len(missing_feature_pairs)} successful pairs lack CLIP vectors")

    if args.mode == "mean":
        post_output = out / "full_clip_features_by_post.csv"
        summary_output = out / "full_clip_aggregation_summary.json"
        fields = ["post_uri", "clip_image_pair_count"] + clip_cols
    else:
        post_output = out / "full_clip_features_by_post_first_downloaded.csv"
        summary_output = out / "full_clip_first_downloaded_summary.json"
        fields = ["post_uri", "downloaded_clip_image_pair_count", "selected_image_number", "selected_local_path", "selected_image_cid"] + clip_cols
    with post_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        if args.mode == "mean":
            for post_uri, vector_sum in sums.items():
                writer.writerow([post_uri, counts[post_uri], *map(float, vector_sum / counts[post_uri])])
        else:
            for post_uri, first in first_by_post.items():
                image_number, local_path, image_cid, vector = first
                writer.writerow([post_uri, counts[post_uri], image_number, local_path, image_cid, *map(float, vector)])

    summary = {
        "matching_key": ["local_path", "sha256"],
        "feature_rows_unique_images": len(vectors),
        "feature_dimension": len(clip_cols),
        "successful_manifest_pairs": len(manifest_pairs),
        "reference_map_pairs": len(map_pairs),
        "reference_map_exactly_matches_successful_manifest": map_pairs == manifest_pairs,
        "pairs_missing_clip_vector": len(missing_feature_pairs),
        "posts_with_clip_features": len(sums),
        "aggregation": (
            "arithmetic mean of L2-normalized per-image CLIP vectors; no post-level re-normalization"
            if args.mode == "mean" else
            "lowest image_number among successful downloaded image pairs; selected per-image CLIP vector unchanged"
        ),
        "mode": args.mode,
        "output": str(post_output),
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
