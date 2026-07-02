#!/usr/bin/env python3
"""Backfill val_codebook_util for SIM_VQ eval_metrics.csv (was 0, should be 100 * active / codebook_size)."""

import csv
import json
from pathlib import Path


def main():
    results_dir = Path("results/sim_vq")
    if not results_dir.exists():
        print("results/sim_vq not found")
        return

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        csv_path = run_dir / "eval_metrics.csv"
        if not config_path.exists() or not csv_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)
        codebook_size = config.get("codebook_size")
        if not codebook_size:
            continue

        rows = []
        updated = False
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                try:
                    active = int(float(row.get("val_active_codes", 0)))
                    current_util = float(row.get("val_codebook_util", 0))
                    # Only update if current util is 0 or very small (< 1%)
                    if current_util < 1.0:
                        util = 100.0 * active / codebook_size if codebook_size > 0 else 0.0
                        row["val_codebook_util"] = f"{util:.4f}"
                        updated = True
                except (ValueError, TypeError):
                    pass
                rows.append(row)

        if updated:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            print(f"Updated {run_dir.name}: codebook_size={codebook_size}, util backfilled from 0")


if __name__ == "__main__":
    main()
