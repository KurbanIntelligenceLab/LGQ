#!/usr/bin/env python3
"""
Count parameters for all unified models to verify fair comparison.

All models use the same Encoder/Decoder backbone, differing only
in their quantization layer.

Usage:
    python scripts/count_params.py
"""

from pathlib import Path

from quantization.model import (
    ModelConfig,
    QuantizerConfig,
    UnifiedAutoEncoder,
    count_model_parameters,
)


def main():
    # Match experiment config: 128 dim, 16K codebook where applicable
    model_config = ModelConfig(base_ch=64, embedding_dim=128)

    configs = [
        ("FSQ [8,5,5,5]", "fsq", QuantizerConfig(levels=[8, 5, 5, 5]), False),
        ("FSQ 16K", "fsq", QuantizerConfig(levels=[16, 16, 8, 8]), True),
        ("VQ 8K", "vq", QuantizerConfig(codebook_size=8192), False),
        ("VQ 16K", "vq", QuantizerConfig(codebook_size=16384), True),
        ("LFQ 16K", "lfq", QuantizerConfig(codebook_size=16384), True),
        ("SimVQ 16K", "sim_vq", QuantizerConfig(codebook_size=16384, use_mlp=False), True),
        ("SimVQ 16K+MLP", "sim_vq", QuantizerConfig(codebook_size=16384, use_mlp=True), False),
        ("LMB 128", "lmb", QuantizerConfig(num_bins=128), True),
        ("LMB 16K flat", "lmb", QuantizerConfig(num_bins=16384, flatten_channels=True), False),
        (
            "LMB 16K fair",
            "lmb",
            QuantizerConfig(perchannel_fair=True, lmb_levels=[16, 16, 8, 8]),
            False,
        ),
    ]

    results = {}
    for name, qtype, qconfig, is_main in configs:
        try:
            model = UnifiedAutoEncoder(qtype, model_config, qconfig)
            p = count_model_parameters(model)
            p["quantization"] = p["quantizer"] + p["projections"]
            results[name] = p
        except Exception as e:
            results[name] = {"error": str(e)}

    # Table
    width = 98
    print("=" * width)
    print("Quantization parameter counts (embedding_dim=128, same Encoder/Decoder)")
    print("=" * width)
    print()
    header = (
        f"{'Model':<14} {'Main':<6} {'Encoder':>12} {'Decoder':>12} {'Quantizer':>12} {'Projections':>12} "
        f"{'Quantization':>14} {'Total':>12}"
    )
    print(header)
    print("-" * len(header))

    for (name, _qtype, _qconfig, is_main) in configs:
        p = results.get(name, {})
        if "error" in p:
            print(f"{name:<14} {'*' if is_main else '':<6}  error: {p['error']}")
            continue
        q = p["quantizer"]
        proj = p["projections"]
        qtotal = p["quantization"]
        main_str = "yes" if is_main else ""
        print(
            f"{name:<14} {main_str:<6} {p['encoder']:>12,} {p['decoder']:>12,} {q:>12,} {proj:>12,} "
            f"{qtotal:>14,} {p['total']:>12,}"
        )

    print("-" * len(header))
    print()
    print("Main = used in primary model comparison (train_*.sh defaults / compare_all_models).")
    print("  SimVQ: main = SimVQ 16K (no MLP). SimVQ 16K+MLP is optional.")
    print("  LMB:   main = LMB 128 (per-channel, train_lmb.sh default).")
    print("         LMB 16K flat / 16K fair are ablations (comparison may use 16K flat).")
    print("Quantization = Quantizer + Projections (pre_quant, post_quant, pre_vq_norm).")
    print("Encoder and Decoder are identical across models.")
    print()

    # Write table to files
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "quantization_param_counts.txt"
    with open(txt_path, "w") as f:
        f.write("Quantization parameter counts (embedding_dim=128)\n")
        f.write("=" * width + "\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for (name, _qtype, _qconfig, is_main) in configs:
            p = results.get(name, {})
            if "error" in p:
                f.write(f"{name:<14} {'*' if is_main else '':<6}  error: {p['error']}\n")
                continue
            main_str = "yes" if is_main else ""
            q, proj = p["quantizer"], p["projections"]
            qtotal = p["quantization"]
            f.write(
                f"{name:<14} {main_str:<6} {p['encoder']:>12,} {p['decoder']:>12,} {q:>12,} {proj:>12,} "
                f"{qtotal:>14,} {p['total']:>12,}\n"
            )
        f.write("-" * len(header) + "\n")
        f.write("\nMain = primary comparison. SimVQ main = 16K (no MLP). LMB main = 128 (per-ch).\n")
        f.write("Quantization = Quantizer + Projections.\n")
    print(f"Table saved to {txt_path}")

    csv_path = out_dir / "quantization_param_counts.csv"
    with open(csv_path, "w") as f:
        f.write("model,main,encoder,decoder,quantizer,projections,quantization,total\n")
        for (name, _qtype, _qconfig, is_main) in configs:
            p = results.get(name, {})
            if "error" in p:
                continue
            qtotal = p["quantization"]
            f.write(
                f"{name},{str(is_main).lower()},{p['encoder']},{p['decoder']},{p['quantizer']},"
                f"{p['projections']},{qtotal},{p['total']}\n"
            )
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
