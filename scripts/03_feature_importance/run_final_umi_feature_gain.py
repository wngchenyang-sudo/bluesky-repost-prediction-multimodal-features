"""Generate only the dissertation Table 7.1 UMI Top-20 gain ranking.

This is a convenience entry point for the initial-gain stage of
``run_umi_sequential_removal_mixed_ood.py``.  It fits the three fixed Mixed
folds, normalises XGBoost gain within each fold, and averages the gain across
folds.  It does not run sequential removal or OOD evaluation.
"""
from __future__ import annotations

import sys

from run_umi_sequential_removal_mixed_ood import main


if __name__ == "__main__":
    # The shared implementation treats zero removal steps as initial-gain-only
    # mode.  Keep this detail out of the user-facing command.
    if "--steps" in sys.argv:
        raise SystemExit("This script always generates only the initial Top 20; do not pass --steps.")
    sys.argv.extend(["--steps", "0"])
    main()
