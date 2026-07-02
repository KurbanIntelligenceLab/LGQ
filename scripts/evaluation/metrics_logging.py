"""Training logging utilities."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class CsvMetricLogger:
    """CSV logger for training metrics."""
    path: Path
    fieldnames: Optional[list[str]] = None

    def log(self, row: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.path.exists() and self.path.stat().st_size > 0

        if self.fieldnames is None:
            if file_exists:
                # Resume: lock fieldnames to the existing header so column count
                # stays stable across restarts. Without this, a resumed process
                # whose first row lacks a key (e.g. val_rfid when --no-rfid is
                # toggled) would write rows with fewer fields than the header,
                # silently shifting all subsequent column data.
                with self.path.open("r", newline="") as f:
                    header = next(csv.reader(f), None)
                if header:
                    self.fieldnames = header
            if self.fieldnames is None:
                # Fresh file: stabilize column order — common fields first, rest sorted.
                common = ["epoch", "global_step", "batch_idx"]
                remaining = sorted([k for k in row.keys() if k not in common])
                self.fieldnames = [k for k in common if k in row] + remaining

        # If the row introduces a key we haven't seen, rewrite the file with the
        # extended schema rather than dropping data on the floor.
        new_keys = [k for k in row.keys() if k not in self.fieldnames]
        if new_keys and file_exists:
            with self.path.open("r", newline="") as f:
                existing_rows = list(csv.DictReader(f))
            self.fieldnames = list(self.fieldnames) + sorted(new_keys)
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow(r)
            file_exists = True
        elif new_keys:
            self.fieldnames = list(self.fieldnames) + sorted(new_keys)

        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})
            f.flush()


def batch_perplexity_from_indices(indices: torch.Tensor) -> Optional[float]:
    """
    Compute approximate batch perplexity from discrete code indices.
    Returns None if indices are not usable.
    """
    if not isinstance(indices, torch.Tensor):
        return None
    if indices.numel() == 0:
        return None

    x = indices.detach().reshape(-1)
    if not torch.is_floating_point(x):
        x = x.to(torch.long)
    else:
        x = x.to(torch.long)

    # Counts over unique indices in this batch
    uniques, counts = torch.unique(x, return_counts=True)
    if counts.numel() == 0:
        return None
    p = counts.float() / counts.sum().float()
    entropy = -(p * (p + 1e-12).log()).sum()
    return float(torch.exp(entropy).item())
