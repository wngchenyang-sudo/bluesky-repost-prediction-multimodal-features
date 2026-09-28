"""Export the two additional non-semantic features used in the final models.

``run_full_extra21.py`` produces a broader table of 21 candidate visual
features. The formal models use only image pixel count and aspect ratio. This
script creates the compact formal input table from that intermediate output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FORMAL_COLUMNS = [
    "post_uri",
    "I-nonsem-extra-image_pixel_count",
    "I-nonsem-extra-aspect_ratio",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Input extra21_features_by_post.csv from run_full_extra21.py.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for extra2_features_by_post.csv.")
    args = parser.parse_args()

    header = pd.read_csv(args.input, nrows=0).columns
    missing = [column for column in FORMAL_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"Input table does not contain required columns: {missing}")

    output = pd.read_csv(args.input, usecols=FORMAL_COLUMNS)
    if output["post_uri"].duplicated().any():
        raise ValueError("Input table contains duplicate post_uri values.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(output):,} rows and 2 formal features to {args.output}")


if __name__ == "__main__":
    main()
