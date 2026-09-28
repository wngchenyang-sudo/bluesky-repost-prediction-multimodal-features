"""Calculate the 21 additional shallow image features for full Bluesky images.

Creates independent image-level and post-level outputs.  It does not depend on
or overwrite the completed OSF37 output, so the two feature blocks remain
separately auditable before being joined into a 58-feature table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
WORKER = CODE / "full_extra21_worker.m"
TAMURA = CODE / "tamura3sigs_no_stats.m"
MATLAB = Path(r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe")
SUCCESS = {"downloaded", "skipped_existing"}
NAMES = [
    "colorfulness", "color_black", "color_blue", "color_brown", "color_gray",
    "color_green", "color_orange", "color_pink", "color_purple", "color_red",
    "color_white", "color_yellow", "edge_fraction", "segmentation_region_count",
    "region_size_mean", "image_pixel_count", "aspect_ratio", "gray_local_entropy_mean",
    "tamura_coarseness", "tamura_contrast", "tamura_directionality",
]
COLS = ["I-nonsem-extra-" + name for name in NAMES]


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


def install_matlab_files(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for source in [WORKER, TAMURA]:
        destination = out / source.name
        if destination.exists() and sha(destination) != sha(source):
            raise ValueError(f"Existing MATLAB dependency differs: {destination}; choose a new output directory")
        if not destination.exists():
            shutil.copy2(source, destination)
    (out / "per_image").mkdir(exist_ok=True)


def prepare(root: Path, out: Path, limit: int | None) -> None:
    install_matlab_files(out)
    manifest_path = root / "image_manifest.csv"
    for path in [manifest_path, WORKER, TAMURA, MATLAB]:
        if not path.exists():
            raise FileNotFoundError(path)
    config = {
        "download_root": str(root), "manifest": str(manifest_path), "matlab": str(MATLAB),
        "features": NAMES, "feature_columns": COLS, "limit_unique_images": limit,
        "manifest_sha256": sha(manifest_path), "worker_sha256": sha(WORKER), "tamura_sha256": sha(TAMURA),
        "policy": {
            "input": "native MATLAB imread; uint8 RGB; grayscale replicated; no resize or EXIF rotation",
            "colorfulness": "Hasler-Susstrunk RGB opponent-colour formula",
            "named_colours": "mutually exclusive HSV threshold proportions",
            "edge_fraction": "edge(gray,Canny)",
            "segmentation": "MATLAB SLIC superpixels requested=250, compactness=10; documented substitute for unavailable EDISON Mean Shift",
            "geometry": "native image dimensions", "entropy": "mean(entropyfilt(gray))",
            "tamura": "Tamura3Sigs-compatible three measures; local kurtosis implementation because Statistics Toolbox is absent",
            "post_aggregation": "arithmetic mean over successful post-image pairs",
        },
    }
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Input/code changed; choose a new output directory rather than mixing checkpoints")
        return
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    success = manifest[manifest.status.isin(SUCCESS)].copy()
    for column in ["sha256", "image_cid", "image_url"]:
        if (success.groupby("local_path")[column].nunique() > 1).any():
            raise ValueError(f"Ambiguous local_path → {column}")
    unique = success.drop_duplicates("local_path").sort_values("local_path")
    if limit is not None:
        unique = unique.head(limit)
    jobs = []
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
        good, values = row.get("status") == "ok", row.get("values", [])
        if row.get("key") != job["key"] or row.get("sha256") != job["sha256"]:
            raise ValueError(f"Checkpoint identity mismatch: {checkpoint.name}")
        if good and (len(values) != len(NAMES) or not np.isfinite(values).all()):
            raise ValueError(f"Invalid feature values: {checkpoint.name}")
        records.append({"local_path": job["local_path"], "image_cid": job["image_cid"], "sha256": job["sha256"],
                        "feature_status": row.get("status"), "error": row.get("error", ""),
                        "elapsed_seconds": row.get("elapsed_seconds"),
                        **dict(zip(COLS, values if good else [np.nan] * len(NAMES)))})
    per_image = pd.DataFrame(records)
    manifest = pd.read_csv(root / "image_manifest.csv", keep_default_na=False)
    pairs = manifest[manifest.status.isin(SUCCESS) & manifest.local_path.isin(set(per_image.local_path))].copy()
    paired = pairs.merge(per_image, on=["local_path", "image_cid", "sha256"], how="left", validate="many_to_one")
    if paired.feature_status.isna().any():
        raise ValueError("A successful manifest pair has no computed image record")
    usable = paired[paired.feature_status.eq("ok")]
    post = usable.groupby("post_uri", sort=False)[COLS].mean().reset_index()
    post["downloaded_image_count"] = post.post_uri.map(pairs.groupby("post_uri").size()).astype(int)
    post["usable_feature_image_count"] = post.post_uri.map(usable.groupby("post_uri").size()).astype(int)
    per_image.to_csv(out / "extra21_features_by_file.csv", index=False, encoding="utf-8-sig")
    post.to_csv(out / "extra21_features_by_post.csv", index=False, encoding="utf-8-sig")
    per_image[per_image.feature_status.ne("ok")].to_csv(out / "extra21_feature_failures.csv", index=False, encoding="utf-8-sig")
    summary = {"status": "complete" if per_image.feature_status.eq("ok").all() else "complete_with_failures",
               "unique_images": len(per_image), "successful_images": int(per_image.feature_status.eq("ok").sum()),
               "failed_images": int(per_image.feature_status.ne("ok").sum()), "features": len(NAMES),
               "posts_with_features": int(post.post_uri.nunique()), "aggregation": "mean over successful post-image pairs"}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run(root: Path, out: Path, workers: int, limit: int | None) -> None:
    prepare(root, out, limit)
    jobs = json.loads((out / "jobs.json").read_text(encoding="utf-8"))
    started, initial = time.monotonic(), len(list((out / "per_image").glob("*.json")))
    processes, logs = [], []
    try:
        for worker in range(workers):
            log = (out / f"worker_{worker}.log").open("a", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen([str(MATLAB), "-nojvm", "-batch", f"full_extra21_worker({worker},{workers})"], cwd=out, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW))
        while any(process.poll() is None for process in processes):
            complete, elapsed = len(list((out / "per_image").glob("*.json"))), time.monotonic() - started
            report = {"completed": complete, "total": len(jobs), "newly_completed": complete - initial,
                      "elapsed_minutes": round(elapsed / 60, 2), "active_workers": sum(p.poll() is None for p in processes)}
            if complete > initial:
                report["estimated_remaining_minutes"] = round((len(jobs) - complete) * elapsed / (complete - initial) / 60, 1)
            write_json(out / "progress.json", report); print(json.dumps(report), flush=True); time.sleep(30)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("MATLAB worker failed; inspect worker logs. Re-run to resume.")
        aggregate(root, out, limit)
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "run", "aggregate"])
    ap.add_argument("--download-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not 1 <= args.workers <= 4:
        ap.error("--workers must be 1..4")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    root, out = args.download_root.resolve(), args.output_dir.resolve()
    {"prepare": lambda: prepare(root, out, args.limit), "aggregate": lambda: aggregate(root, out, args.limit), "run": lambda: run(root, out, args.workers, args.limit)}[args.action]()
