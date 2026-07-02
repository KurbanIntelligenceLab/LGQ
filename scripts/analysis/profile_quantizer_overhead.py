#!/usr/bin/env python3
"""
Profile quantizer-only forward time (all-to-all soft assignment over K codes).

Quantifies overhead of LMB's cdist + softmax over K per token vs hard VQ/SimVQ.
Useful for scalability to larger K, higher resolution, or multi-scale tokenizers.

Usage:
    python scripts/profile_quantizer_overhead.py
        # LMB at K=4K,8K,16K,32K; single image size
    python scripts/profile_quantizer_overhead.py --sweep-resolution
        # Same + sweep image_size 64, 128, 256
    python scripts/profile_quantizer_overhead.py --compare-vq
        # Add VQ and SimVQ at 16K for comparison
"""

import argparse
import sys
import time
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from configs import MODEL_CONFIGS
from quantization.model import ModelConfig, QuantizerConfig, UnifiedAutoEncoder


def get_num_image_tokens(model: torch.nn.Module, x: torch.Tensor) -> int:
    """Latent tokens per image (H*W of encoder output)."""
    model.eval()
    with torch.no_grad():
        z = model.enc(x)
    return int(z.shape[2] * z.shape[3])


def profile_quantizer_forward(
    model: torch.nn.Module,
    config_class,
    x: torch.Tensor,
    device: torch.device,
    warmup: int = 5,
    steps: int = 100,
) -> dict:
    """
    Time full forward and quantizer-only forward via hook on model.quantize.
    Returns dict with full_forward_ms, quantizer_ms, ratio, step_time_ms.
    """
    model.eval()
    quantizer_times_ms = []

    # Time only the quantizer forward (cdist + softmax over K for LMB)
    quantizer = model.quantize
    original_forward = quantizer.forward

    def timed_forward(*args, **kwargs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = original_forward(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        quantizer_times_ms.append((t1 - t0) * 1000)
        return out

    quantizer.forward = timed_forward

    try:
        # Warmup
        for _ in range(warmup):
            config_class.compute_loss(model, x)

        quantizer_times_ms.clear()
        full_times_ms = []
        for _ in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            config_class.compute_loss(model, x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            full_times_ms.append((t1 - t0) * 1000)

        full_mean = sum(full_times_ms) / len(full_times_ms)
        quant_mean = sum(quantizer_times_ms) / len(quantizer_times_ms) if quantizer_times_ms else 0.0
        ratio = (quant_mean / full_mean * 100) if full_mean > 0 else 0.0
        return {
            "full_forward_ms": full_mean,
            "quantizer_ms": quant_mean,
            "quantizer_pct": ratio,
            "steps": steps,
        }
    finally:
        quantizer.forward = original_forward


def main():
    ap = argparse.ArgumentParser(description="Profile quantizer-only forward time (LMB soft assignment over K)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--sweep-resolution", action="store_true", help="Sweep image_size 64, 128, 256")
    ap.add_argument("--compare-vq", action="store_true", help="Add VQ and SimVQ at 16K")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default="results", help="Output dir for profile_quantizer_overhead.txt")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = ModelConfig(base_ch=64, embedding_dim=128)

    # LMB at several K (flattened, main setting)
    configs = [
        ("LMB 4K flat", "lmb", QuantizerConfig(num_bins=4096, flatten_channels=True), MODEL_CONFIGS["lmb"]),
        ("LMB 8K flat", "lmb", QuantizerConfig(num_bins=8192, flatten_channels=True), MODEL_CONFIGS["lmb"]),
        ("LMB 16K flat", "lmb", QuantizerConfig(num_bins=16384, flatten_channels=True), MODEL_CONFIGS["lmb"]),
        ("LMB 32K flat", "lmb", QuantizerConfig(num_bins=32768, flatten_channels=True), MODEL_CONFIGS["lmb"]),
    ]
    if args.compare_vq:
        configs.extend([
            ("VQ 16K", "vq", QuantizerConfig(codebook_size=16384), MODEL_CONFIGS["vq"]),
            ("SimVQ 16K", "sim_vq", QuantizerConfig(codebook_size=16384, use_mlp=False), MODEL_CONFIGS["sim_vq"]),
        ])

    image_sizes = [64, 128, 256] if args.sweep_resolution else [args.image_size]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Quantizer overhead profile (quantizer-only forward time via hook)",
        f"device={device}, batch_size={args.batch_size}, warmup={args.warmup}, steps={args.steps}",
        "Metrics: full_forward_ms, quantizer_ms, quantizer_pct, tokens/image",
        "",
    ]

    for image_size in image_sizes:
        x = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
        lines.append(f"=== image_size={image_size} ===")
        print(f"\nimage_size={image_size}")

        for name, qtype, qconfig, config_class in configs:
            try:
                model = UnifiedAutoEncoder(qtype, model_cfg, qconfig).to(device)
                num_tokens = get_num_image_tokens(model, x)
                stats = profile_quantizer_forward(
                    model, config_class, x, device,
                    warmup=args.warmup, steps=args.steps,
                )
                K = getattr(qconfig, "num_bins", None) or getattr(qconfig, "codebook_size", 0)
                line = (
                    f"  {name:<18}  full={stats['full_forward_ms']:.2f} ms  quantizer={stats['quantizer_ms']:.2f} ms  "
                    f"({stats['quantizer_pct']:.1f}%)  tokens/img={num_tokens}  K={K}"
                )
                lines.append(line)
                print(line)
            except Exception as e:
                lines.append(f"  {name:<18}  ERROR: {e}")
                print(f"  {name:<18}  ERROR: {e}")
        lines.append("")

    txt_path = out_dir / "profile_quantizer_overhead.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
