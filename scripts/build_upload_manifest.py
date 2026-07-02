"""Build a flat staging directory + manifest + index for rclone uploads.

For every run under `results/`, finds `best_model.pt` and `latest_model.pt`,
hardlinks them into `uploads/staging/` under names of the form
`<run_name>__<best|latest>_e<epoch>.pt`, and writes:

- `uploads/gdrive_manifest.txt` — one staged-file path per line (for rclone --files-from)
- `uploads/gdrive_index.csv`    — lookup table (model, codebook_size, kind, epoch, val_rec_loss, source_path)

Epoch is read from `eval_metrics.csv`:
- best:   row with min `val_rec_loss`
- latest: last row (max epoch)

Falls back to the .pt's mtime-derived index only if eval_metrics.csv is missing.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

CB_RE = re.compile(r"cb(\d+)([kKmM]?)")
# In this repo the "k"-suffixed run names map to powers of two, not multiples of 1024.
# E.g. cb65k means 65536 (=2^16), not 66560.
CB_K_ABBREV = {4: 4096, 8: 8192, 16: 16384, 32: 32768, 65: 65536}


def parse_eval_metrics(path: Path):
    """Return (best_epoch, best_val_rec_loss, latest_epoch, latest_val_rec_loss)."""
    if not path.is_file():
        return None, None, None, None
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("epoch")]
    if not rows:
        return None, None, None, None

    def fnum(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    best_row = min(
        (r for r in rows if fnum(r.get("val_rec_loss")) is not None),
        key=lambda r: fnum(r["val_rec_loss"]),
        default=None,
    )
    latest_row = max(rows, key=lambda r: int(r["epoch"]))
    best_epoch = int(best_row["epoch"]) if best_row else None
    best_loss = fnum(best_row["val_rec_loss"]) if best_row else None
    latest_epoch = int(latest_row["epoch"])
    latest_loss = fnum(latest_row.get("val_rec_loss"))
    return best_epoch, best_loss, latest_epoch, latest_loss


def parse_config(path: Path):
    if not path.is_file():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def codebook_size_from(config: dict, run_name: str):
    if "codebook_size" in config and config["codebook_size"]:
        return int(config["codebook_size"])
    m = CB_RE.search(run_name)
    if m:
        n = int(m.group(1))
        suffix = m.group(2).lower()
        if suffix == "k":
            return CB_K_ABBREV.get(n, n * 1024)
        return n
    return None


def hardlink_or_copy(src: Path, dst: Path) -> str:
    """Hardlink src→dst (replacing dst). Falls back to symlink if cross-device."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        dst.symlink_to(src.resolve())
        return "symlink"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results", type=Path)
    ap.add_argument("--staging-dir", default="uploads/staging", type=Path)
    ap.add_argument("--manifest", default="uploads/gdrive_manifest.txt", type=Path)
    ap.add_argument("--index", default="uploads/gdrive_index.csv", type=Path)
    ap.add_argument(
        "--kinds",
        nargs="+",
        default=["best", "latest"],
        choices=["best", "latest"],
        help="Which checkpoint kinds to include.",
    )
    ap.add_argument("--clean-staging", action="store_true",
                    help="Remove existing staging dir before building.")
    args = ap.parse_args()

    if not args.results_root.is_dir():
        sys.exit(f"results-root not found: {args.results_root}")

    if args.clean_staging and args.staging_dir.exists():
        for p in args.staging_dir.iterdir():
            p.unlink()
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    manifest_paths = []

    targets = [f"{kind}_model.pt" for kind in args.kinds]
    for ckpt in sorted(args.results_root.rglob("checkpoints/*_model.pt")):
        if ckpt.name not in targets:
            continue
        kind = ckpt.stem.replace("_model", "")  # "best" or "latest"
        run_dir = ckpt.parent.parent
        run_name = run_dir.name
        config = parse_config(run_dir / "config.json")
        model = config.get("model") or run_dir.parent.name
        cb_size = codebook_size_from(config, run_name)

        best_e, best_loss, latest_e, latest_loss = parse_eval_metrics(run_dir / "eval_metrics.csv")
        if kind == "best":
            epoch, val_loss = best_e, best_loss
        else:
            epoch, val_loss = latest_e, latest_loss

        epoch_tag = f"e{epoch:03d}" if epoch is not None else "eUNK"
        staged_name = f"{run_name}__{kind}_{epoch_tag}.pt"
        staged_path = args.staging_dir / staged_name
        link_kind = hardlink_or_copy(ckpt, staged_path)
        manifest_paths.append(str(staged_path))

        rows.append({
            "staged_file": staged_name,
            "model": model,
            "codebook_size": cb_size if cb_size is not None else "",
            "kind": kind,
            "epoch": epoch if epoch is not None else "",
            "val_rec_loss": f"{val_loss:.6f}" if val_loss is not None else "",
            "run_name": run_name,
            "source_path": str(ckpt),
            "link_kind": link_kind,
        })

    rows.sort(key=lambda r: (r["model"], r["codebook_size"] or 0, r["run_name"], r["kind"]))

    with args.index.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["staged_file", "model", "codebook_size", "kind", "epoch",
                        "val_rec_loss", "run_name", "source_path", "link_kind"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with args.manifest.open("w") as f:
        for p in sorted(manifest_paths):
            f.write(p + "\n")
        f.write(str(args.index) + "\n")

    print(f"Staged {len(rows)} checkpoints → {args.staging_dir}")
    print(f"Manifest: {args.manifest} ({len(manifest_paths) + 1} entries incl. index)")
    print(f"Index:    {args.index}")
    missing_epoch = sum(1 for r in rows if r["epoch"] == "")
    if missing_epoch:
        print(f"  warning: {missing_epoch} entries missing epoch (eval_metrics.csv absent or empty)")


if __name__ == "__main__":
    main()
