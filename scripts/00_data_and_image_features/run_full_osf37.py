"""Calculate the original MATLAB OSF37 shallow image features for full images.

This intentionally produces image- and post-level feature tables only.  It
does not touch the 24k interaction data or any existing small-sample outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
WORKER = CODE / "full_osf37_worker.m"
REFERENCE = Path(r"E:\Blue\image_downloads\small sample\shallow_features\osf_trial\reference")
MATLAB = Path(r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe")
SUCCESS = {"downloaded", "skipped_existing"}
NAMES = [
    "use_of_light", "saturation_mean", "saturation_std", "brightness_std",
    "valence", "dominance", "arousal", "hue_circular_variance",
    "low_dof_hue", "low_dof_saturation", "low_dof_brightness",
    "rule_of_thirds_saturation", "rule_of_thirds_brightness",
]
CHANNELS = ["hue", "saturation", "brightness"]
NAMES += [f"{c}_wavelet_level_{i}" for c in CHANNELS for i in (1, 2, 3)]
NAMES += [f"{c}_wavelet_sum" for c in CHANNELS]
NAMES += [f"{c}_{p}_glcm" for c in CHANNELS for p in ("contrast", "correlation", "energy", "homogeneity")]
COLS = ["I-nonsem-" + name for name in NAMES]
FUNCTIONS = ["use_of_light", "indicators", "pda", "huestats", "trt", "low_depth_of_field_indicators", "wavelet_features", "greylev"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def output_files(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    copied = out / WORKER.name
    if copied.exists() and sha(copied) != sha(WORKER):
        raise ValueError(f"Existing worker differs: {copied}; choose a new output directory")
    if not copied.exists():
        shutil.copy2(WORKER, copied)
    (out / "per_image").mkdir(exist_ok=True)


def prepare(root: Path, out: Path, limit: int | None) -> None:
    output_files(out)
    manifest_path = root / "image_manifest.csv"
    for path in [manifest_path, WORKER, MATLAB, *[REFERENCE / (name + ".m") for name in FUNCTIONS]]:
        if not path.exists():
            raise FileNotFoundError(path)
    config = {
        "download_root": str(root), "manifest": str(manifest_path), "reference": str(REFERENCE),
        "matlab": str(MATLAB), "features": NAMES, "feature_columns": COLS,
        "manifest_sha256": sha(manifest_path), "worker_sha256": sha(WORKER),
        "source_sha256": {name: sha(REFERENCE / (name + ".m")) for name in FUNCTIONS},
        "policy": "Native MATLAB imread; RGB uint8; grayscale replicated; sym db4; no normalization",
        "limit_unique_images": limit,
    }
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Input/code changed; choose a new output directory rather than mixing checkpoints")
        return
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    success = manifest[manifest.status.isin(SUCCESS)].copy()
    for col in ["sha256", "image_cid", "image_url"]:
        if (success.groupby("local_path")[col].nunique() > 1).any():
            raise ValueError(f"Ambiguous local_path → {col}")
    jobs = []
    unique = success.drop_duplicates("local_path").sort_values("local_path")
    if limit is not None:
        unique = unique.head(limit)
    for index, row in enumerate(unique.itertuples(index=False)):
        image = (root / row.local_path).resolve()
        if not image.is_relative_to((root / "images").resolve()):
            raise ValueError(f"Path outside images directory: {image}")
        if not image.exists() or sha(image) != row.sha256:
            raise ValueError(f"Local image/checksum mismatch: {image}")
        jobs.append({"key": f"{index:05d}", "path": str(image), "local_path": row.local_path,
                     "sha256": row.sha256, "image_cid": row.image_cid})
    write_json(out / "jobs.json", jobs)
    write_json(config_path, config)
    print(json.dumps({"prepared_unique_images": len(jobs), "output": str(out)}, ensure_ascii=False), flush=True)


def aggregate(root: Path, out: Path, limit: int | None) -> None:
    prepare(root, out, limit)
    jobs = json.loads((out / "jobs.json").read_text(encoding="utf-8"))
    records = []
    for job in jobs:
        checkpoint = out / "per_image" / (job["key"] + ".json")
        if not checkpoint.exists():
            raise ValueError(f"Not complete: missing {checkpoint.name}")
        row = json.loads(checkpoint.read_text(encoding="utf-8"))
        good = row.get("status") == "ok"
        values = row.get("values", [])
        if row.get("key") != job["key"] or row.get("sha256") != job["sha256"]:
            raise ValueError(f"Checkpoint identity mismatch: {checkpoint.name}")
        if good and (len(values) != 37 or not np.isfinite(values).all()):
            raise ValueError(f"Invalid feature values: {checkpoint.name}")
        records.append({"local_path": job["local_path"], "image_cid": job["image_cid"], "sha256": job["sha256"],
                        "feature_status": row.get("status"), "error": row.get("error", ""),
                        "elapsed_seconds": row.get("elapsed_seconds"),
                        **dict(zip(COLS, values if good else [np.nan] * 37))})
    per_image = pd.DataFrame(records)
    manifest = pd.read_csv(root / "image_manifest.csv", keep_default_na=False)
    successful_pairs = manifest[manifest.status.isin(SUCCESS)].copy()
    # A pilot intentionally contains only a subset of physical images.
    successful_pairs = successful_pairs[successful_pairs.local_path.isin(set(per_image.local_path))].copy()
    paired = successful_pairs.merge(per_image, on=["local_path", "image_cid", "sha256"], how="left", validate="many_to_one")
    if paired.feature_status.isna().any():
        raise ValueError("A successful manifest pair has no computed image record")
    usable = paired[paired.feature_status.eq("ok")]
    post = usable.groupby("post_uri", sort=False)[COLS].mean().reset_index()
    post["downloaded_image_count"] = post.post_uri.map(successful_pairs.groupby("post_uri").size()).astype(int)
    post["usable_feature_image_count"] = post.post_uri.map(usable.groupby("post_uri").size()).astype(int)
    per_image.to_csv(out / "image_features_by_file.csv", index=False, encoding="utf-8-sig")
    post.to_csv(out / "image_features_by_post.csv", index=False, encoding="utf-8-sig")
    per_image[per_image.feature_status.ne("ok")].to_csv(out / "image_feature_failures.csv", index=False, encoding="utf-8-sig")
    summary = {"status": "complete" if per_image.feature_status.eq("ok").all() else "complete_with_failures",
               "unique_images": len(per_image), "successful_images": int(per_image.feature_status.eq("ok").sum()),
               "failed_images": int(per_image.feature_status.ne("ok").sum()), "features": 37,
               "posts_with_features": int(post.post_uri.nunique()), "aggregation": "mean over successful post-image pairs"}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run(root: Path, out: Path, workers: int, limit: int | None) -> None:
    prepare(root, out, limit)
    jobs = json.loads((out / "jobs.json").read_text(encoding="utf-8"))
    started, initial = time.monotonic(), len(list((out / "per_image").glob("*.json")))
    processes = []
    try:
        for worker in range(workers):
            processes.append(subprocess.Popen([str(MATLAB), "-singleCompThread", "-batch", f"full_osf37_worker({worker},{workers})", "-logfile", str(out / f"worker_{worker}.log")], cwd=out, creationflags=subprocess.CREATE_NO_WINDOW))
        while any(p.poll() is None for p in processes):
            complete = len(list((out / "per_image").glob("*.json")))
            elapsed = time.monotonic() - started
            report = {"completed": complete, "total": len(jobs), "newly_completed": complete - initial,
                      "elapsed_minutes": round(elapsed / 60, 2), "active_workers": sum(p.poll() is None for p in processes)}
            if complete > initial:
                report["estimated_remaining_minutes"] = round((len(jobs) - complete) * elapsed / (complete - initial) / 60, 1)
            write_json(out / "progress.json", report); print(json.dumps(report), flush=True); time.sleep(30)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("MATLAB worker failed; inspect worker logs. Re-run to resume.")
        aggregate(root, out, limit)
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "run", "aggregate"])
    ap.add_argument("--download-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Pilot only: first N sorted unique local paths")
    args = ap.parse_args()
    if not 1 <= args.workers <= 4:
        ap.error("--workers must be 1..4")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    root, out = args.download_root.resolve(), args.output_dir.resolve()
    {"prepare": lambda: prepare(root, out, args.limit), "aggregate": lambda: aggregate(root, out, args.limit), "run": lambda: run(root, out, args.workers, args.limit)}[args.action]()
