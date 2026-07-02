#!/usr/bin/env python3
"""
Benchmark wall-clock time and peak GPU memory across quantization models.

Compares training-step cost (forward + backward + optimizer step) so practitioners
can see the compute/memory tradeoff. LGQ-style methods (e.g. LMB softmax over K
bins per token) can be more expensive than hard VQ with EMA or FSQ; this script
quantifies that.

Usage:
    python scripts/benchmark_compute_memory.py
        # 16K comparison: FSQ, LFQ, VQ, LMB 16K flat (main), SimVQ, LMB-Fixed 16K
    python scripts/benchmark_compute_memory.py --preset all   # all configs
    python scripts/benchmark_compute_memory.py --batch-size 128 --steps 100 --warmup 10
    python scripts/benchmark_compute_memory.py --device cpu  # CPU-only (no GPU memory)
    python scripts/benchmark_compute_memory.py --breakdown  # forward/backward/optimizer split

Output: results/compute_memory_benchmark.json and results/compute_memory_benchmark.txt

Metrics reported:
  - runtime: total time for timed steps (seconds)
  - memory / peak memory: peak GPU memory (MB/GB)
  - FLOPs: forward-pass FLOPs per step (optional, requires fvcore)
  - tokens/sec: latent tokens per second (imgs/sec × tokens per image)
  - per-step time: ms per training step
  - K = 16K: default preset compares FSQ, LFQ, VQ, LMB, SimVQ, LMB-Fixed at 16K
"""
# Add project root so configs/ and quantization/ are importable when run as script
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_external = _project_root / "external"
if _external.exists() and str(_external) not in sys.path:
    sys.path.insert(0, str(_external))

import argparse
import json
import time

import torch

# Optional FLOP counting (forward pass only)
try:
    from fvcore.nn import FlopCountAnalysis
    _HAS_FVCORE = True
except ImportError:
    _HAS_FVCORE = False

from configs import MODEL_CONFIGS
from configs.base import LossOutput
from quantization.model import (
    ModelConfig,
    QuantizerConfig,
    UnifiedAutoEncoder,
    count_model_parameters,
)


# Model configs (same backbone: base_ch=64, embedding_dim=128)
# Format: (display_name, quantizer_type, QuantizerConfig, config_class for compute_loss)
BENCHMARK_CONFIGS_ALL = [
    ("FSQ [8,5,5,5]", "fsq", QuantizerConfig(levels=[8, 5, 5, 5]), MODEL_CONFIGS["fsq"]),
    ("FSQ 16K", "fsq", QuantizerConfig(levels=[16, 16, 8, 8]), MODEL_CONFIGS["fsq"]),
    ("VQ 8K", "vq", QuantizerConfig(codebook_size=8192), MODEL_CONFIGS["vq"]),
    ("VQ 16K", "vq", QuantizerConfig(codebook_size=16384), MODEL_CONFIGS["vq"]),
    ("Rot-VQ 8K", "rot_vq", QuantizerConfig(codebook_size=8192, rotation_trick=True), MODEL_CONFIGS["rot_vq"]),
    ("LFQ 16K", "lfq", QuantizerConfig(codebook_size=16384), MODEL_CONFIGS["lfq"]),
    ("LFQ 65K", "lfq", QuantizerConfig(codebook_size=65536), MODEL_CONFIGS["lfq"]),
    ("SimVQ 16K", "sim_vq", QuantizerConfig(codebook_size=16384, use_mlp=False), MODEL_CONFIGS["sim_vq"]),
    ("SimVQ 16K+MLP", "sim_vq", QuantizerConfig(codebook_size=16384, use_mlp=True), MODEL_CONFIGS["sim_vq"]),
    ("LMB 128", "lmb", QuantizerConfig(num_bins=128), MODEL_CONFIGS["lmb"]),
    ("LMB 16K flat", "lmb", QuantizerConfig(num_bins=16384, flatten_channels=True), MODEL_CONFIGS["lmb"]),
    ("LMB 16K fair", "lmb", QuantizerConfig(perchannel_fair=True, lmb_levels=[16, 16, 8, 8]), MODEL_CONFIGS["lmb"]),
]

# 16K matched-capacity comparison: FSQ, LFQ, VQ, LMB (flattened), SimVQ, LMB-Fixed (fair)
# LMB = main experiment: 16K bins, flatten_channels (single codebook over 128-d vector)
# LMB-Fixed = per-channel fair [16,16,8,8] = 16K (FSQ-style structure, learnable bins)
BENCHMARK_CONFIGS_16K = [
    ("FSQ 16K", "fsq", QuantizerConfig(levels=[16, 16, 8, 8]), MODEL_CONFIGS["fsq"]),
    ("LFQ 16K", "lfq", QuantizerConfig(codebook_size=16384), MODEL_CONFIGS["lfq"]),
    ("VQ 16K", "vq", QuantizerConfig(codebook_size=16384), MODEL_CONFIGS["vq"]),
    ("LMB 16K flat", "lmb", QuantizerConfig(num_bins=16384, flatten_channels=True), MODEL_CONFIGS["lmb"]),  # main experiment
    ("SimVQ 16K", "sim_vq", QuantizerConfig(codebook_size=16384, use_mlp=False), MODEL_CONFIGS["sim_vq"]),
    ("LMB-Fixed 16K", "lmb", QuantizerConfig(perchannel_fair=True, lmb_levels=[16, 16, 8, 8]), MODEL_CONFIGS["lmb"]),
]


def get_num_image_tokens(model: torch.nn.Module, x: torch.Tensor) -> int:
    """Return latent tokens per image (H*W of encoder output)."""
    model.eval()
    with torch.no_grad():
        z = model.enc(x)
    # z: (B, C, H, W) -> tokens per image = H * W
    return int(z.shape[2] * z.shape[3])


def count_forward_flops(model: torch.nn.Module, config_class, x: torch.Tensor, device: torch.device) -> int | None:
    """Return forward-pass FLOPs per step (one compute_loss call), or None if fvcore unavailable."""
    if not _HAS_FVCORE:
        return None
    model.eval()
    try:
        # FlopCountAnalysis expects (model, inputs); compute_loss may take model, x
        flops = FlopCountAnalysis(model, x)
        total = flops.total()
        return int(total)
    except Exception:
        return None


def run_benchmark(
    model: torch.nn.Module,
    config_class,
    x: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    timed_steps: int,
    breakdown: bool = False,
) -> dict:
    """Run warmup then timed steps; return timing, peak memory, and optional FLOPs."""
    model.train()
    results = {}

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss_output: LossOutput = config_class.compute_loss(model, x)
        loss_output.total_loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed run
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step_times_ms = []
    if breakdown:
        fwd_times_ms = []
        bwd_times_ms = []
        opt_times_ms = []

    for _ in range(timed_steps):
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True) if breakdown else None
            t2 = torch.cuda.Event(enable_timing=True) if breakdown else None
            t3 = torch.cuda.Event(enable_timing=True)
            t0.record()
        else:
            t0_sec = time.perf_counter()

        loss_output: LossOutput = config_class.compute_loss(model, x)
        if device.type == "cuda" and breakdown:
            t1.record()
            torch.cuda.synchronize()
            fwd_times_ms.append(t0.elapsed_time(t1))

        loss_output.total_loss.backward()
        if device.type == "cuda" and breakdown:
            t2.record()
            torch.cuda.synchronize()
            bwd_times_ms.append(t1.elapsed_time(t2))

        optimizer.step()
        if device.type == "cuda":
            t3.record()
            torch.cuda.synchronize()
            step_times_ms.append(t0.elapsed_time(t3))
            if breakdown:
                opt_times_ms.append(t2.elapsed_time(t3))
        else:
            step_times_ms.append((time.perf_counter() - t0_sec) * 1000)

    results["step_time_ms_mean"] = sum(step_times_ms) / len(step_times_ms) if step_times_ms else 0.0
    results["step_time_ms_std"] = (
        (sum((t - results["step_time_ms_mean"]) ** 2 for t in step_times_ms) / len(step_times_ms)) ** 0.5
        if len(step_times_ms) > 1 else 0.0
    )
    results["step_time_ms_min"] = min(step_times_ms) if step_times_ms else 0.0
    results["step_time_ms_max"] = max(step_times_ms) if step_times_ms else 0.0
    # Total runtime (seconds) for the timed steps
    results["runtime_s"] = sum(step_times_ms) / 1000.0 if step_times_ms else 0.0

    if breakdown and device.type == "cuda" and fwd_times_ms and bwd_times_ms and opt_times_ms:
        results["forward_ms_mean"] = sum(fwd_times_ms) / len(fwd_times_ms)
        results["backward_ms_mean"] = sum(bwd_times_ms) / len(bwd_times_ms)
        results["optimizer_ms_mean"] = sum(opt_times_ms) / len(opt_times_ms)

    if device.type == "cuda":
        results["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        results["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        results["peak_memory_gb"] = None
        results["peak_memory_mb"] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark compute and memory across quantization models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, default=128, help="Spatial size (H=W)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps")
    parser.add_argument("--steps", type=int, default=50, help="Timed steps")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cuda:0, cpu); default auto")
    parser.add_argument("--breakdown", action="store_true", help="Report forward/backward/optimizer time")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for JSON/table output")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU (no GPU memory)")
    parser.add_argument("--preset", type=str, default="16k", choices=("16k", "all"),
                        help="Preset: '16k' = FSQ/LFQ/VQ/LMB/SimVQ/LMB-Fixed at 16K (default); 'all' = full set")
    args = parser.parse_args()

    configs = BENCHMARK_CONFIGS_16K if args.preset == "16k" else BENCHMARK_CONFIGS_ALL

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU (no GPU memory reported).")
    else:
        device = torch.device(args.device or "cuda")

    model_cfg = ModelConfig(base_ch=64, embedding_dim=128)
    batch_size = args.batch_size
    image_size = args.image_size
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    all_results = []
    print(f"\nBenchmark: batch_size={batch_size}, image_size={image_size}, warmup={args.warmup}, steps={args.steps}")
    print(f"Device: {device}\n")

    for name, qtype, qconfig, config_class in configs:
        try:
            model = UnifiedAutoEncoder(qtype, model_cfg, qconfig).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            params = count_model_parameters(model)
            total_params = params["total"]

            num_tokens = get_num_image_tokens(model, x)
            flops_forward = count_forward_flops(model, config_class, x, device)

            stats = run_benchmark(
                model, config_class, x, device, optimizer,
                warmup_steps=args.warmup, timed_steps=args.steps, breakdown=args.breakdown,
            )
            throughput = (batch_size * 1000.0 / stats["step_time_ms_mean"]) if stats["step_time_ms_mean"] > 0 else 0
            tokens_per_sec = throughput * num_tokens

            row = {
                "model": name,
                "quantizer_type": qtype,
                "total_params": total_params,
                "runtime_s": round(stats["runtime_s"], 2),
                "step_time_ms": round(stats["step_time_ms_mean"], 2),
                "step_time_ms_std": round(stats["step_time_ms_std"], 2),
                "throughput_imgs_per_sec": round(throughput, 1),
                "tokens_per_image": num_tokens,
                "tokens_per_sec": round(tokens_per_sec, 0),
                "peak_memory_mb": round(stats["peak_memory_mb"], 1) if stats["peak_memory_mb"] is not None else None,
                "peak_memory_gb": round(stats["peak_memory_gb"], 3) if stats["peak_memory_gb"] is not None else None,
                "flops_forward": flops_forward,
            }
            if args.breakdown and "forward_ms_mean" in stats:
                row["forward_ms"] = round(stats["forward_ms_mean"], 2)
                row["backward_ms"] = round(stats["backward_ms_mean"], 2)
                row["optimizer_ms"] = round(stats["optimizer_ms_mean"], 2)
            all_results.append(row)

            mem_str = f"{row['peak_memory_mb']:.1f} MB" if row["peak_memory_mb"] is not None else "N/A"
            flops_str = f"{row['flops_forward'] / 1e9:.2f}G" if row.get("flops_forward") else "N/A"
            print(
                f"  {name:<18}  {row['step_time_ms']:>7.2f} ms/step  {throughput:>6.1f} img/s  "
                f"{tokens_per_sec:>8.0f} tok/s  {mem_str:>10}  {flops_str:>8}  ({total_params:,} params)"
            )
        except Exception as e:
            print(f"  {name:<18}  ERROR: {e}")
            all_results.append({"model": name, "quantizer_type": qtype, "error": str(e)})

    # Table
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "compute_memory_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "config": {
                    "preset": args.preset,
                    "batch_size": batch_size,
                    "image_size": image_size,
                    "warmup_steps": args.warmup,
                    "timed_steps": args.steps,
                    "device": str(device),
                },
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {json_path}")

    # Text table: runtime, memory, FLOPs, tokens/sec, per-step time (K=16K preset)
    txt_path = out_dir / "compute_memory_benchmark.txt"
    width = 115
    with open(txt_path, "w") as f:
        f.write("Compute & memory benchmark (one training step: forward + backward + optimizer)\n")
        f.write(f"preset={args.preset} (K=16K), batch_size={batch_size}, image_size={image_size}, device={device}\n")
        f.write("Metrics: runtime, peak memory, FLOPs (forward), tokens/sec, per-step time\n")
        f.write("=" * width + "\n")
        f.write(
            f"{'Model':<20} {'runtime_s':>10} {'ms/step':>10} {'img/s':>8} {'tok/s':>10} "
            f"{'Peak Mem':>12} {'FLOPs':>12} {'Params':>14}\n"
        )
        f.write("-" * width + "\n")
        for r in all_results:
            if "error" in r:
                f.write(f"{r['model']:<20}  error: {r['error']}\n")
                continue
            mem = f"{r['peak_memory_mb']:.1f} MB" if r.get("peak_memory_mb") is not None else "N/A"
            flops = f"{r['flops_forward'] / 1e9:.2f}G" if r.get("flops_forward") else "N/A"
            f.write(
                f"{r['model']:<20} {r['runtime_s']:>10.2f} {r['step_time_ms']:>10.2f} "
                f"{r['throughput_imgs_per_sec']:>8.1f} {r['tokens_per_sec']:>10.0f} {mem:>12} {flops:>12} {r['total_params']:>14,}\n"
            )
        f.write("-" * width + "\n")
    print(f"Table saved to {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
