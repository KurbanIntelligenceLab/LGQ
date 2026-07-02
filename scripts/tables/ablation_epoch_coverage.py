#!/usr/bin/env python3
"""
Parse results/model_comparison_all_epochs.txt and report the last epoch
each ablation has data for (based on rFID row).
"""

import re
from pathlib import Path

MODELS = ["FSQ", "LFQ", "LMB", "LMB-Fixed", "LMB-Fair", "SIM_VQ", "VQ", "ROT_VQ"]
RESULTS_FILE = Path(__file__).resolve().parents[1] / "results" / "model_comparison_all_epochs.txt"


def main() -> None:
    text = RESULTS_FILE.read_text()
    lines = text.split("\n")
    last_epoch: dict[str, int] = {m: 0 for m in MODELS}
    for i, line in enumerate(lines):
        epoch_m = re.match(r"EPOCH (\d+) - MODEL COMPARISON", line)
        if not epoch_m:
            continue
        epoch_num = int(epoch_m.group(1))
        # rFID line is a few lines below (after separator and Metric header)
        for j in range(i + 1, min(i + 10, len(lines))):
            if lines[j].strip().startswith("rFID"):
                rfid_line = lines[j]
                break
        else:
            continue
        tokens = rfid_line.split()
        if tokens[0] != "rFID" or len(tokens) < 9:
            continue
        values = tokens[1:9]
        for k, model in enumerate(MODELS):
            if k >= len(values):
                break
            s = values[k]
            if s != "N/A":
                try:
                    float(s)
                    last_epoch[model] = max(last_epoch[model], epoch_num)
                except ValueError:
                    pass
    # Print table
    out_lines = [
        "Ablation epoch coverage (last epoch with data in model_comparison_all_epochs.txt)",
        "=" * 60,
        f"{'Ablation':<15} {'Last Epoch':<12} {'Has data'}",
        "-" * 60,
    ]
    for model in MODELS:
        ep = last_epoch[model]
        status = "Yes" if ep > 0 else "No (N/A throughout)"
        out_lines.append(f"{model:<15} {ep:<12} {status}")
    out_lines.append("=" * 60)
    max_ep = max(last_epoch.values())
    out_lines.append(f"\nMax epoch in file: {max_ep}")
    out_lines.append(f"Epochs in file: 1 .. {max_ep}")
    output = "\n".join(out_lines)
    print(output)
    # Write to results
    out_file = RESULTS_FILE.parent / "ablation_epoch_coverage.txt"
    out_file.write_text(output + "\n")
    print(f"\nTable written to {out_file}")


if __name__ == "__main__":
    main()
