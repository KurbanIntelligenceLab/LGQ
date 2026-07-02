#!/usr/bin/env python3
"""
Resume stopped experiments on empty GPUs.
This script finds experiments that have stopped (not reached max epochs) and
resumes them on available GPUs.
"""

import json
import csv
from pathlib import Path
import subprocess
import sys

def get_gpu_usage():
    """Check GPU usage using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpu_id = int(parts[0])
                memory_used = int(parts[1])
                utilization = int(parts[2])
                gpus.append({
                    'id': gpu_id,
                    'memory_used_mb': memory_used,
                    'utilization': utilization
                })
        return gpus
    except:
        return []

def find_stopped_experiments(results_dir: Path, image_size: int = 128):
    """Find experiments that have stopped before reaching max epochs."""
    stopped = []
    
    for model_dir in ['fsq', 'vq', 'sim_vq', 'lmb']:
        base_path = results_dir / model_dir
        if not base_path.exists():
            continue
        
        for exp_dir in sorted(base_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            config_path = exp_dir / 'config.json'
            eval_path = exp_dir / 'eval_metrics.csv'
            
            if not config_path.exists():
                continue
            
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                if config.get('image_size') != image_size:
                    continue
                
                max_epochs = config.get('epochs', 100)
                current_epochs = 0
                
                if eval_path.exists():
                    with open(eval_path, 'r') as f:
                        reader = csv.DictReader(f)
                        current_epochs = len(list(reader))
                
                if current_epochs < max_epochs:
                    stopped.append({
                        'model': model_dir,
                        'exp_dir': exp_dir,
                        'config': config,
                        'current_epochs': current_epochs,
                        'max_epochs': max_epochs,
                        'gpu': config.get('gpu', 0)
                    })
            except Exception as e:
                print(f"Error checking {exp_dir}: {e}")
                continue
    
    return stopped

def main():
    results_dir = Path('results')
    
    print("=" * 80)
    print("Finding stopped experiments and available GPUs")
    print("=" * 80)
    
    # Get GPU status
    gpus = get_gpu_usage()
    if not gpus:
        print("Warning: Could not check GPU status. Proceeding anyway...")
        available_gpus = [0, 1, 2, 3]  # Assume 4 GPUs
    else:
        print(f"\nGPU Status:")
        available_gpus = []
        for gpu in gpus:
            status = "AVAILABLE" if gpu['utilization'] < 10 and gpu['memory_used_mb'] < 1000 else "BUSY"
            print(f"  GPU {gpu['id']}: {status} (Util: {gpu['utilization']}%, Mem: {gpu['memory_used_mb']}MB)")
            if gpu['utilization'] < 10 and gpu['memory_used_mb'] < 1000:
                available_gpus.append(gpu['id'])
    
    if not available_gpus:
        print("\n❌ No available GPUs found!")
        return
    
    print(f"\n✅ Available GPUs: {available_gpus}")
    
    # Find stopped experiments
    stopped = find_stopped_experiments(results_dir, image_size=128)
    
    if not stopped:
        print("\n✅ No stopped experiments found!")
        return
    
    print(f"\nFound {len(stopped)} stopped experiments:")
    print("-" * 80)
    for exp in stopped:
        print(f"  {exp['model'].upper():8} {exp['exp_dir'].name[:50]:50} "
              f"Epochs: {exp['current_epochs']:3}/{exp['max_epochs']:3} "
              f"(GPU {exp['gpu']})")
    
    # Assign stopped experiments to available GPUs
    print(f"\n{'=' * 80}")
    print("Resume Commands (run these to resume training):")
    print("=" * 80)
    
    gpu_idx = 0
    for exp in stopped:
        if gpu_idx >= len(available_gpus):
            print(f"\n⚠️  No more available GPUs. {len(stopped) - gpu_idx} experiments cannot be resumed.")
            break
        
        new_gpu = available_gpus[gpu_idx]
        exp_dir = exp['exp_dir']
        model = exp['model']
        
        # Determine the training script
        script_map = {
            'fsq': 'shell/train_fsq.sh',
            'vq': 'shell/train_vq.sh',
            'sim_vq': 'shell/train_sim_vq.sh',
            'lmb': 'shell/train_lmb.sh'
        }
        
        script = script_map.get(model)
        if not script:
            continue
        
        # Get config values
        config = exp['config']
        
        print(f"\n# Resume {model.upper()} on GPU {new_gpu}:")
        print(f"bash {script} \\")
        print(f"    --gpu {new_gpu} \\")
        print(f"    --data-root {config.get('data_root', 'data/imagenet')} \\")
        print(f"    --batch-size {config.get('batch_size', 64)} \\")
        print(f"    --epochs {config.get('epochs', 100)} \\")
        print(f"    --lr {config.get('lr', 3e-4)} \\")
        print(f"    --dim {config.get('dim', 128)} \\")
        print(f"    --embedding-dim {config.get('embedding_dim', 128)} \\")
        
        # Model-specific args
        if model == 'fsq' and 'levels' in config:
            levels = ' '.join(map(str, config['levels']))
            print(f"    --levels {levels} \\")
        elif model == 'vq':
            print(f"    --codebook-size {config.get('codebook_size', 16384)} \\")
        elif model == 'sim_vq':
            print(f"    --codebook-size {config.get('codebook_size', 16384)} \\")
        elif model == 'lmb':
            print(f"    --num-bins {config.get('num_bins', 16384)} \\")
            print(f"    --tau-start {config.get('tau_start', 1.0)} \\")
            print(f"    --tau-end {config.get('tau_end', 0.1)} \\")
            if config.get('flatten_channels'):
                print(f"    --flatten-channels \\")
        
        print(f"    --run-name {exp_dir.name}")
        print(f"# Training will auto-resume from checkpoint in {exp_dir}")
        
        gpu_idx += 1

if __name__ == "__main__":
    main()
