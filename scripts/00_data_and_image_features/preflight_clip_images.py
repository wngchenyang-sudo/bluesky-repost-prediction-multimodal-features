"""Read-only CLIP preflight for a full Bluesky image download.

Checks every unique downloaded physical image against the manifest before any
file transfer or CLIP inference.  It writes a unique-image input manifest and
an explicit failures file; it never edits source images or source manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def value(row: dict[str, str], name: str) -> str:
    return (row.get(name) or "").strip()


def file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    root = args.download_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (root / "image_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    downloaded = [r for r in rows if value(r, "status") == "downloaded"]
    by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in downloaded:
        by_path[value(row, "local_path")].append(row)

    images_root = (root / "images").resolve()
    output: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for number, (relative, group) in enumerate(sorted(by_path.items()), start=1):
        reference = group[0]
        path = (root / relative).resolve()
        error = ""
        observed_sha, observed_bytes = "", 0
        observed_width, observed_height, observed_format = "", "", ""
        try:
            if not relative or not path.is_relative_to(images_root):
                raise ValueError("local_path is empty or outside images directory")
            if not path.is_file():
                raise FileNotFoundError(path)
            observed_sha, observed_bytes = file_sha256(path)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                observed_width, observed_height, observed_format = image.width, image.height, image.format
            # The same physical file must have internally consistent metadata.
            expected_sha = {value(r, "sha256") for r in group}
            expected_bytes = {value(r, "bytes") for r in group}
            expected_dims = {(value(r, "width"), value(r, "height"), value(r, "format")) for r in group}
            if expected_sha != {observed_sha}:
                raise ValueError("SHA256 mismatch: manifest=" + repr(expected_sha))
            if expected_bytes != {str(observed_bytes)}:
                raise ValueError("byte-count mismatch: manifest=" + repr(expected_bytes))
            if expected_dims != {(str(observed_width), str(observed_height), observed_format)}:
                raise ValueError("dimension/format mismatch: manifest=" + repr(expected_dims))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        item = {
            "clip_index": number - 1,
            "local_path": relative,
            "absolute_path": str(path),
            "image_cid": value(reference, "image_cid"),
            "sha256": observed_sha,
            "bytes": observed_bytes,
            "width": observed_width,
            "height": observed_height,
            "format": observed_format,
            "linked_manifest_rows": len(group),
            "example_post_uri": value(reference, "post_uri"),
            "example_hashtag": value(reference, "hashtag"),
            "preflight_ok": not error,
            "error": error,
        }
        output.append(item)
        if error:
            failures.append(item)
        if number % args.progress_every == 0 or number == len(by_path):
            print(json.dumps({"checked_unique_images": number, "total_unique_images": len(by_path),
                              "failures": len(failures)}, ensure_ascii=False), flush=True)

    valid = [r for r in output if r["preflight_ok"]]
    fields = list(output[0]) if output else []
    write_csv(out / "clip_unique_images_preflight.csv", output, fields)
    write_csv(out / "clip_preflight_failures.csv", failures, fields)
    summary = {
        "manifest_rows": len(rows),
        "downloaded_manifest_rows": len(downloaded),
        "failed_manifest_rows": len(rows) - len(downloaded),
        "unique_physical_images": len(by_path),
        "preflight_passed_images": len(valid),
        "preflight_failed_images": len(failures),
        "total_validated_bytes": sum(int(r["bytes"]) for r in valid),
        "formats": dict(Counter(str(r["format"]) for r in valid)),
        "unique_input_csv": str(out / "clip_unique_images_preflight.csv"),
        "failures_csv": str(out / "clip_preflight_failures.csv"),
        "note": "Use only rows where preflight_ok=True for CLIP. The input has one row per physical image, not one row per post-image pair.",
    }
    (out / "clip_preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
