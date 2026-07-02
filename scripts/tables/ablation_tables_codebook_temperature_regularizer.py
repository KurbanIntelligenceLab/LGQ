#!/usr/bin/env python3
"""
Build tables of codebook-size, temperature, and regularizer ablations with last epoch.
Scans results/{fsq,lfq,lmb,vq,sim_vq} for eval_metrics.csv + config.json.
"""

import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
ABLATIONS = RESULTS / "ablations"
MODEL_DIRS = ["fsq", "lfq", "lmb", "vq", "sim_vq"]


def get_codebook_size(config: dict, model: str) -> int | None:
    if model == "fsq":
        levels = config.get("levels")
        if levels:
            n = 1
            for L in levels:
                n *= int(L)
            return n
    if model == "lmb":
        return config.get("num_bins")
    return config.get("codebook_size")


def get_last_epoch(csv_path: Path) -> int | None:
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    epochs = []
    for r in rows:
        try:
            e = r.get("epoch")
            if e is not None and str(e).strip():
                epochs.append(int(float(e)))
        except (ValueError, TypeError):
            pass
    return max(epochs) if epochs else None


def main() -> None:
    # Collect all runs with eval_metrics.csv
    runs: list[dict] = []
    for model_dir in MODEL_DIRS:
        base = RESULTS / model_dir
        if not base.is_dir():
            continue
        model = model_dir.upper().replace("_", "-")  # sim_vq -> SIM_VQ
        for exp_dir in base.iterdir():
            if not exp_dir.is_dir():
                continue
            csv_path = exp_dir / "eval_metrics.csv"
            config_path = exp_dir / "config.json"
            if not csv_path.exists() or not config_path.exists():
                continue
            last_epoch = get_last_epoch(csv_path)
            if last_epoch is None:
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except Exception:
                config = {}
            cb = get_codebook_size(config, model_dir)
            run_info = {
                "model": model,
                "run": exp_dir.name,
                "last_epoch": last_epoch,
                "codebook_size": cb,
                "tau_start": config.get("tau_start"),
                "tau_end": config.get("tau_end"),
                "lambda_peak": config.get("lambda_peak"),
                "lambda_bins": config.get("lambda_bins"),
            }
            runs.append(run_info)

    # Codebook size ablations: every run with its codebook size and last epoch
    cb_runs = [r for r in runs if r["codebook_size"] is not None]
    cb_table = []
    for r in sorted(cb_runs, key=lambda x: (x["model"], x["codebook_size"], -x["last_epoch"])):
        cb_table.append({
            "model": r["model"],
            "codebook_size": r["codebook_size"],
            "run": r["run"],
            "last_epoch": r["last_epoch"],
        })

    # Temperature (LMB only): tau_start → tau_end
    tau_runs = [r for r in runs if r["model"] == "LMB" and (r["tau_start"] is not None or r["tau_end"] is not None)]
    tau_table = []
    for r in tau_runs:
        ts = r["tau_start"] if r["tau_start"] is not None else "?"
        te = r["tau_end"] if r["tau_end"] is not None else "?"
        tau_table.append({
            "run": r["run"],
            "tau_schedule": f"{ts} → {te}",
            "last_epoch": r["last_epoch"],
        })
    tau_table.sort(key=lambda x: -x["last_epoch"])

    # Regularizer (LMB only): lambda_peak, lambda_bins
    reg_runs = [r for r in runs if r["model"] == "LMB" and (r["lambda_peak"] is not None or r["lambda_bins"] is not None)]
    reg_table = []
    for r in reg_runs:
        reg_table.append({
            "run": r["run"],
            "lambda_peak": r["lambda_peak"] if r["lambda_peak"] is not None else "—",
            "lambda_bins": r["lambda_bins"] if r["lambda_bins"] is not None else "—",
            "last_epoch": r["last_epoch"],
        })
    reg_table.sort(key=lambda x: -x["last_epoch"])

    # Format output
    ABLATIONS.mkdir(parents=True, exist_ok=True)
    out_path = ABLATIONS / "ablation_codebook_temperature_regularizer_last_epoch.txt"
    lines = [
        "=" * 90,
        "ABLATIONS — CODEBOOK SIZE, TEMPERATURE, REGULARIZER (last epoch per run)",
        "=" * 90,
        "",
        "——— CODEBOOK SIZE ———",
        f"{'Model':<12} {'Codebook Size':<16} {'Run (name)':<50} {'Last Epoch':<12}",
        "-" * 90,
    ]
    for row in cb_table:
        run_short = row["run"][:48] + ".." if len(row["run"]) > 50 else row["run"]
        lines.append(f"{row['model']:<12} {row['codebook_size']:<16} {run_short:<50} {row['last_epoch']:<12}")
    lines.extend([
        "",
        "——— TEMPERATURE (LMB) ———",
        f"{'Run (name)':<55} {'τ schedule':<20} {'Last Epoch':<12}",
        "-" * 90,
    ])
    for row in tau_table:
        run_short = row["run"][:53] + ".." if len(row["run"]) > 55 else row["run"]
        lines.append(f"{run_short:<55} {row['tau_schedule']:<20} {row['last_epoch']:<12}")
    lines.extend([
        "",
        "——— REGULARIZER (LMB) ———",
        f"{'Run (name)':<55} {'λ_peak':<12} {'λ_bins':<12} {'Last Epoch':<12}",
        "-" * 90,
    ])
    for row in reg_table:
        run_short = row["run"][:53] + ".." if len(row["run"]) > 55 else row["run"]
        lines.append(f"{run_short:<55} {str(row['lambda_peak']):<12} {str(row['lambda_bins']):<12} {row['last_epoch']:<12}")
    lines.extend(["", "=" * 90, f"Source: eval_metrics.csv under results/{{fsq,lfq,lmb,vq,sim_vq}}/", "=" * 90])

    text = "\n".join(lines)
    out_path.write_text(text + "\n")
    print(text)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
