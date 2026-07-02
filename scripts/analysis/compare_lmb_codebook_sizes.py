#!/usr/bin/env python3
"""Quick comparison of LMB flattened at 4K, 8K, 16K - 128D metrics only (no UMAP)."""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class FlatImageDataset(Dataset):
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    def __init__(self, root, transform=None, max_images=None):
        root = Path(root)
        paths = []
        for ext in self.EXTENSIONS:
            paths.extend(root.glob(f'*{ext}'))
            paths.extend(root.glob(f'*{ext.upper()}'))
            paths.extend(root.glob(f'**/*{ext}'))
            paths.extend(root.glob(f'**/*{ext.upper()}'))
        self.paths = sorted(set(paths))
        if max_images:
            self.paths = self.paths[:max_images]
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.transform(img) if self.transform else img, 0


def transform(img):
    img = img.resize((128, 128), Image.LANCZOS)
    x = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)
    m = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (x - m) / s


def load_model(ckpt_path, device):
    from configs import MODEL_CONFIGS
    from quantization.model import ModelConfig
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    mc = ModelConfig(base_ch=cfg.get("dim", 128) // 2, embedding_dim=cfg.get("embedding_dim", 128))
    model = MODEL_CONFIGS[cfg["model"]].create_model(argparse.Namespace(**cfg), mc)
    sd = ckpt["model_state_dict"].copy()
    if "quantize.centers" in sd and hasattr(model.quantize, "centers") and sd["quantize.centers"].shape != model.quantize.centers.shape:
        if sd["quantize.centers"].shape[0] == model.quantize.centers.shape[1]:
            sd["quantize.centers"] = sd["quantize.centers"].t()
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, cfg, ckpt.get("epoch", "?")


@torch.no_grad()
def collect_z_and_active(model, loader, device, n_img=512, subsample=6000):
    all_z, active = [], set()
    seen = 0
    for x, _ in tqdm(loader, total=min(len(loader), n_img // loader.batch_size + 1), desc="Collecting"):
        if seen >= n_img:
            break
        x = x.to(device)
        seen += x.size(0)
        z = model.enc(x)
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, C)
        z_norm = model.pre_vq_norm(z_flat)
        all_z.append(z_norm.cpu())
        out = model(x)
        idx = out[1]
        active.update(idx.reshape(-1).cpu().numpy().tolist())
    z = torch.cat(all_z, dim=0)
    if z.size(0) > subsample:
        z = z[torch.randperm(z.size(0))[:subsample]]
    return z, list(active)


def metrics_128d(z, e):
    from sklearn.neighbors import NearestNeighbors
    z_np = z.numpy() if isinstance(z, torch.Tensor) else z
    e_np = e.numpy() if isinstance(e, torch.Tensor) else e
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(e_np)
    d, _ = nn.kneighbors(z_np)
    latent_to_code = d.flatten()
    nn2 = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(z_np)
    d2, _ = nn2.kneighbors(e_np)
    code_to_latent = d2.flatten()
    z_std = float(z_np.std())
    orphan = (code_to_latent > 2 * latent_to_code.mean()).mean() * 100
    return {
        "latent_to_code_mean": float(latent_to_code.mean()),
        "latent_to_code_median": float(np.median(latent_to_code)),
        "code_to_latent_mean": float(code_to_latent.mean()),
        "relative_quant_error": float(latent_to_code.mean() / z_std),
        "orphan_codes_pct": float(orphan),
        "z_std": z_std,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/imagenet")
    ap.add_argument("--num-images", type=int, default=512)
    ap.add_argument("--subsample", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only", type=str, default=None, help="Only run 4k, 8k, or 16k")
    args = ap.parse_args()

    device = "cpu"
    data_root = Path(args.data_root).expanduser()
    test = data_root / "test"
    if not test.exists():
        test = data_root / "val"
    ds = FlatImageDataset(str(test), transform=transform, max_images=args.num_images * 2)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    all_configs = [
        ("LMB 4K", "results/lmb/lmb_ablation_cb4k/checkpoints/best_model.pt"),
        ("LMB 8K", "results/lmb/lmb_ablation_cb8k/checkpoints/best_model.pt"),
        ("LMB 16K", "results/lmb/lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd/checkpoints/best_model.pt"),
    ]
    only = (args.only or "").lower()
    configs = [(n, p) for n, p in all_configs if not only or only in n.lower()]

    print("LMB flattened codebook size comparison (128D metrics)")
    print("=" * 60)

    for name, ckpt_path in configs:
        ckpt_path = project_root / ckpt_path
        if not ckpt_path.exists():
            print(f"{name}: checkpoint not found, skip")
            continue
        model, cfg, epoch = load_model(str(ckpt_path), device)
        K = model.quantize.centers.shape[0]
        print(f"\n{name} (epoch {epoch}, codebook {K})")
        z, active_ids = collect_z_and_active(model, loader, device, args.num_images, args.subsample)
        E = model.quantize.centers.detach().cpu()[active_ids]
        m = metrics_128d(z, E)
        print(f"  Active codes: {len(active_ids)} ({100*len(active_ids)/K:.1f}%)")
        print(f"  latent_to_code_mean: {m['latent_to_code_mean']:.4f}")
        print(f"  relative_quant_error: {m['relative_quant_error']:.4f}")
        print(f"  orphan_codes_pct: {m['orphan_codes_pct']:.1f}%")
        del model

    print("\nDone.")


if __name__ == "__main__":
    main()
