#!/usr/bin/env python3
"""
Generate LaTeX tables comparing codebook sizes across all models.
"""

import csv
from pathlib import Path
from typing import Dict, Optional

def load_metrics_at_epoch(experiment_dir: Path, target_epoch: int = 1) -> Optional[Dict[str, float]]:
    """Load metrics from a specific epoch in eval_metrics.csv."""
    eval_csv = experiment_dir / "eval_metrics.csv"
    if not eval_csv.exists():
        return None
    
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
                epochs_metrics[epoch] = metrics
            except (ValueError, KeyError):
                continue
    
    if not epochs_metrics:
        return None
    
    if target_epoch in epochs_metrics:
        return epochs_metrics[target_epoch]
    return None


def format_value(value, decimals=2):
    if value is None:
        return "---"
    return f"{value:.{decimals}f}"


def main():
    results_dir = Path("results")
    output_dir = Path("results")
    
    # Models and their configurations
    models_config = {
        "LMB": ("lmb", "lmb_ablation_cb"),
        "FSQ": ("fsq", "fsq_cb"),
        "LFQ": ("lfq", "lfq_cb"),
        "VQ": ("vq", "vq_cb"),
        "SIM_VQ": ("sim_vq", "sim_vq_cb"),
    }
    
    codebook_sizes = ["4k", "8k", "16k", "32k", "65k"]
    size_labels = {"4k": "4K", "8k": "8K", "16k": "16K", "32k": "32K", "65k": "65K"}
    
    target_epoch = 1
    
    # Load all data
    all_data = {}  # model -> size -> metrics
    for model_name, (model_dir, prefix) in models_config.items():
        all_data[model_name] = {}
        for size in codebook_sizes:
            exp_dir = results_dir / model_dir / f"{prefix}{size}"
            metrics = load_metrics_at_epoch(exp_dir, target_epoch)
            if metrics:
                all_data[model_name][size] = metrics
    
    # Generate one table per model
    tables = []
    
    for model_name, (model_dir, prefix) in models_config.items():
        if not all_data[model_name]:
            continue
        
        lines = []
        lines.append(f"% {model_name} Codebook Size Comparison")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lcccccccc}")
        lines.append("\\toprule")
        lines.append("Codebook & rFID & PSNR & SSIM & LPIPS & Rec Loss & Codebook Util (\\%) & Perplexity & Active Codes \\\\")
        lines.append("\\midrule")
        
        # Sort by rFID
        sorted_sizes = sorted(
            [s for s in codebook_sizes if s in all_data[model_name]],
            key=lambda s: all_data[model_name][s].get('val_rfid', float('inf'))
        )
        
        for size in sorted_sizes:
            m = all_data[model_name][size]
            row = f"{size_labels[size]} & "
            row += f"{format_value(m.get('val_rfid'), 2)} & "
            row += f"{format_value(m.get('val_psnr'), 2)} & "
            row += f"{format_value(m.get('val_ssim'), 4)} & "
            row += f"{format_value(m.get('val_lpips'), 4)} & "
            row += f"{format_value(m.get('val_rec_loss'), 4)} & "
            row += f"{format_value(m.get('val_codebook_util'), 2)} & "
            row += f"{format_value(m.get('val_perplexity'), 1)} & "
            row += f"{format_value(m.get('val_active_codes'), 0)} \\\\"
            lines.append(row)
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{{model_name} codebook size comparison. All metrics from epoch {target_epoch}.}}")
        lines.append(f"\\label{{tab:{model_name.lower()}_codebook_size}}")
        lines.append("\\end{table}")
        lines.append("")
        
        tables.append("\n".join(lines))
    
    # Generate combined table (all models, one row per model-size combination)
    combined_lines = []
    combined_lines.append("% Combined Codebook Size Comparison - All Models")
    combined_lines.append("\\begin{table}[htbp]")
    combined_lines.append("\\centering")
    combined_lines.append("\\small")
    combined_lines.append("\\begin{tabular}{llcccccccc}")
    combined_lines.append("\\toprule")
    combined_lines.append("Model & CB Size & rFID & PSNR & SSIM & LPIPS & Rec Loss & CB Util (\\%) & Perplexity & Active \\\\")
    combined_lines.append("\\midrule")
    
    # Collect all entries and sort by rFID
    all_entries = []
    for model_name in models_config.keys():
        for size in codebook_sizes:
            if size in all_data[model_name]:
                m = all_data[model_name][size]
                rfid = m.get('val_rfid', float('inf'))
                all_entries.append((model_name, size, m, rfid))
    
    all_entries.sort(key=lambda x: x[3])
    
    for model_name, size, m, _ in all_entries:
        row = f"{model_name} & {size_labels[size]} & "
        row += f"{format_value(m.get('val_rfid'), 2)} & "
        row += f"{format_value(m.get('val_psnr'), 2)} & "
        row += f"{format_value(m.get('val_ssim'), 4)} & "
        row += f"{format_value(m.get('val_lpips'), 4)} & "
        row += f"{format_value(m.get('val_rec_loss'), 4)} & "
        row += f"{format_value(m.get('val_codebook_util'), 2)} & "
        row += f"{format_value(m.get('val_perplexity'), 1)} & "
        row += f"{format_value(m.get('val_active_codes'), 0)} \\\\"
        combined_lines.append(row)
    
    combined_lines.append("\\bottomrule")
    combined_lines.append("\\end{tabular}")
    combined_lines.append(f"\\caption{{Codebook size comparison across all models. All metrics from epoch {target_epoch}. Sorted by rFID.}}")
    combined_lines.append("\\label{tab:all_models_codebook_size}")
    combined_lines.append("\\end{table}")
    
    tables.append("\n".join(combined_lines))
    
    # Write to file
    output_file = output_dir / "codebook_size_comparison_all_models.tex"
    with open(output_file, 'w') as f:
        f.write("% Codebook Size Comparison Tables - All Models\n")
        f.write("% Generated automatically from experiment results\n")
        f.write(f"% All metrics from epoch {target_epoch}\n\n")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\begin{document}\n\n")
        
        for table in tables:
            f.write(table)
            f.write("\n\n")
        
        f.write("\\end{document}\n")
    
    print(f"Generated: {output_file}")
    
    # Also print to console
    print("\n" + "="*80)
    for table in tables:
        print(table)
        print()


if __name__ == "__main__":
    main()
