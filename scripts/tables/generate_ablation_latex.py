#!/usr/bin/env python3
"""
Generate LaTeX tables for LMB ablation studies.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

def load_all_epochs_metrics(experiment_dir: Path) -> Optional[Dict[int, Dict[str, float]]]:
    """Load metrics from ALL epochs in eval_metrics.csv.
    For duplicate epochs, uses the LAST occurrence (most recent evaluation).
    Returns a dict mapping epoch -> metrics.
    """
    eval_csv = experiment_dir / "eval_metrics.csv"
    if not eval_csv.exists():
        return None
    
    # Store all epochs, keeping only the last occurrence of each
    epochs_metrics = {}
    
    with open(eval_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('epoch'):
                continue
            try:
                epoch = int(row['epoch'])
                metrics = {}
                for key, value in row.items():
                    if key not in ['epoch', 'global_step']:
                        try:
                            if value and value.strip() and value != 'None':
                                metrics[key] = float(value)
                            else:
                                metrics[key] = None
                        except (ValueError, TypeError):
                            metrics[key] = None
                metrics['epoch'] = epoch
                # Always update (keeps last occurrence for each epoch)
                epochs_metrics[epoch] = metrics
            except (ValueError, KeyError):
                continue
    
    if not epochs_metrics:
        return None
    
    return epochs_metrics


def load_latest_metrics(experiment_dir: Path) -> Optional[Dict[str, float]]:
    """Load metrics from the latest epoch in eval_metrics.csv."""
    all_epochs = load_all_epochs_metrics(experiment_dir)
    if not all_epochs:
        return None
    latest_epoch = max(all_epochs.keys())
    return all_epochs[latest_epoch]

def format_value(value: Optional[float], decimals: int = 2) -> str:
    """Format a metric value for LaTeX."""
    if value is None:
        return "---"
    return f"{value:.{decimals}f}"

def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    return text.replace('_', '\\_').replace('%', '\\%')

def generate_latex_table(
    title: str,
    experiments: List[Tuple[str, Dict[str, float]]],
    metrics: List[Tuple[str, str, int, bool]],
    caption: str = "",
    label: str = ""
) -> str:
    """Generate a LaTeX table from experiment results.
    Since all experiments now use the same epoch, we don't need an epoch column.
    """
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    lines.append("\\toprule")
    
    # Header
    header = "Experiment"
    for _, display_name, _, _ in metrics:
        header += f" & {display_name}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # Data rows
    for exp_name, metrics_dict in experiments:
        row = escape_latex(exp_name)
        for csv_key, _, decimals, _ in metrics:
            value = metrics_dict.get(csv_key)
            row += f" & {format_value(value, decimals)}"
        row += " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{escape_latex(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("")
    
    return "\n".join(lines)

def main():
    results_dir = Path("results/lmb")
    
    # Define metrics to include
    metrics = [
        ("val_rfid", "rFID", 2, False),
        ("val_psnr", "PSNR", 2, True),
        ("val_ssim", "SSIM", 4, True),
        ("val_lpips", "LPIPS", 4, False),
        ("val_rec_loss", "Rec Loss", 4, False),
        ("val_codebook_util", "Codebook Util (\\%)", 2, True),
        ("val_perplexity", "Perplexity", 1, True),
        ("val_active_codes", "Active Codes", 0, True),
    ]
    
    # Group experiments by category
    categories = {
        "Flattening": {
            "lmb_ablation_flattened": "Flattened",
            "lmb_ablation_perchannel": "Per-Channel",
        },
        "Temperature": {
            "lmb_ablation_tau_fast": "Fast (1.0 → 0.05)",
            "lmb_ablation_tau_slow": "Slow (1.0 → 0.2)",
            "lmb_ablation_tau_fixed": "Fixed (1.0 → 1.0)",
        },
        "Regularization": {
            "lmb_ablation_reg_none": "None (0.0, 0.0)",
            "lmb_ablation_reg_weak": "Weak (0.002, 0.002)",
            "lmb_ablation_reg_strong": "Strong (0.01, 0.01)",
            "lmb_ablation_reg_peak_only": "Peak Only",
            "lmb_ablation_reg_bins_only": "Bins Only",
        },
        "Codebook Size": {
            "lmb_ablation_cb4k": "4K",
            "lmb_ablation_cb8k": "8K",
            "lmb_ablation_cb16k": "16K",
            "lmb_ablation_cb32k": "32K",
            "lmb_ablation_cb65k": "65K",
        },
        "Initialization": {
            "lmb_ablation_init_uniform": "Uniform",
            "lmb_ablation_init_gaussian": "Gaussian",
        },
    }
    
    # Also include regularization grid if available
    reg_grid_experiments = {}
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir() and "reg_grid" in exp_dir.name:
            # Parse grid parameters from name
            name = exp_dir.name
            if "peak002" in name:
                peak = "0.002"
            elif "peak005" in name:
                peak = "0.005"
            elif "peak010" in name:
                peak = "0.010"
            else:
                peak = "?"
            
            if "bins002" in name:
                bins = "0.002"
            elif "bins005" in name:
                bins = "0.005"
            elif "bins010" in name:
                bins = "0.010"
            else:
                bins = "?"
            
            reg_grid_experiments[exp_dir.name] = f"({peak}, {bins})"
    
    if reg_grid_experiments:
        categories["Regularization Grid"] = reg_grid_experiments
    
    # Generate tables for each category
    all_tables = []
    
    for category_name, experiment_map in categories.items():
        # First pass: load all epochs for each experiment and find common epoch
        all_experiments_epochs = {}
        
        for exp_dir_name, display_name in experiment_map.items():
            exp_dir = results_dir / exp_dir_name
            if not exp_dir.exists():
                continue
            
            epochs_dict = load_all_epochs_metrics(exp_dir)
            if epochs_dict:
                all_experiments_epochs[display_name] = (exp_dir_name, epochs_dict)
        
        if not all_experiments_epochs:
            continue
        
        # Find the maximum common epoch (highest epoch that all experiments have)
        available_epochs_per_exp = [set(data[1].keys()) for data in all_experiments_epochs.values()]
        common_epochs = set.intersection(*available_epochs_per_exp)
        
        if not common_epochs:
            # No common epoch - fall back to epoch 1 if all have it, otherwise skip
            print(f"Warning: No common epoch for {category_name}, using minimum available")
            # Use the minimum of each experiment's max epoch
            min_max_epoch = min(max(data[1].keys()) for data in all_experiments_epochs.values())
            common_epoch = min_max_epoch
        else:
            common_epoch = max(common_epochs)
        
        # Second pass: get metrics at the common epoch
        experiments_data = []
        for display_name, (exp_dir_name, epochs_dict) in all_experiments_epochs.items():
            if common_epoch in epochs_dict:
                metrics_dict = epochs_dict[common_epoch]
                experiments_data.append((display_name, metrics_dict))
        
        if experiments_data:
            # Sort by rFID (lower is better)
            experiments_data.sort(key=lambda x: x[1].get('val_rfid', float('inf')))
            
            caption = f"LMB ablation study: {category_name}. All metrics from epoch {common_epoch}."
            label = f"tab:lmb_ablation_{category_name.lower().replace(' ', '_')}"
            
            table = generate_latex_table(
                title=f"LMB Ablation: {category_name}",
                experiments=experiments_data,
                metrics=metrics,
                caption=caption,
                label=label
            )
            all_tables.append((category_name, table))
            print(f"  {category_name}: using epoch {common_epoch} for {len(experiments_data)} experiments")
    
    # Write all tables to file
    output_file = Path("results/lmb_ablation_tables.tex")
    with open(output_file, 'w') as f:
        f.write("% LMB Ablation Study Tables\n")
        f.write("% Generated automatically from experiment results\n\n")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{multirow}\n")
        f.write("\\begin{document}\n\n")
        
        for category_name, table in all_tables:
            f.write(f"% ========== {category_name} ==========\n")
            f.write(table)
            f.write("\n")
        
        f.write("\\end{document}\n")
    
    print(f"Generated LaTeX tables: {output_file}")
    print(f"\nTotal categories: {len(all_tables)}")
    for category_name, _ in all_tables:
        print(f"  - {category_name}")

if __name__ == "__main__":
    main()
