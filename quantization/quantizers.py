"""
LGQ (Learnable Geometric Quantization): multi-channel bin discretizer with
learnable bin centers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBinDiscretizer(nn.Module):
    """
    LGQ (Learnable Geometric Quantization): multi-channel bin discretizer with learnable bin centers.

    Two modes:
    1. Per-channel mode (flatten_channels=False): Each of C channels has K bins independently
    2. Flattened mode (flatten_channels=True): Channels are flattened into single vector,
       using one codebook of K bins for the entire C-dimensional vector
    
    Input: [B, C, T] where C = embedding_dim, T = spatial tokens
    Output idx: 
      - Per-channel: [B, T, C] with values in [0..K-1] per channel
      - Flattened: [B, T] with values in [0..K-1] (single index per token)
    
    Args:
        embedding_dim: Number of latent channels (C)
        num_bins: Number of bins (K)
        tau: Softmax temperature for soft assignments
        lambda_peak: Weight for peakedness regularization (encourages confident assignments)
        lambda_bins: Weight for bin diversity regularization (discourages concentration)
        lambda_floor: Weight for perplexity floor (prevents collapse)
        p_min: Minimum soft perplexity target
        learn_widths: Enable learnable bin widths (adaptive temperature per bin)
        width_reg: L1 regularization on bin widths
        flatten_channels: If True, flatten channels into single vector before quantization
    """
    def __init__(self, embedding_dim: int, num_bins: int, tau: float = 1.0,
                 lambda_peak: float = 0.0, lambda_bins: float = 0.0,
                 lambda_floor: float = 0.0, p_min: float = 0.0,
                 learn_widths: bool = False, width_reg: float = 1e-4,
                 beta_rate: float = 0.0, rate_target: float = -1.0,
                 flatten_channels: bool = True, init_method: str = "random",
                 distance_type: str = "l2_sq"):
        super().__init__()
        self.C = int(embedding_dim)
        self.K = int(num_bins)
        self.tau = float(tau)
        self.lambda_peak = float(lambda_peak)
        self.lambda_bins = float(lambda_bins)
        self.lambda_floor = float(lambda_floor)
        self.p_min = float(p_min)
        self.learn_widths = bool(learn_widths)
        self.width_reg = float(width_reg)
        self.beta_rate = float(beta_rate)
        self.rate_target = float(rate_target)
        self.flatten_channels = bool(flatten_channels)
        self.init_method = str(init_method)  # "random", "uniform", "gaussian", "kmeans"
        self.distance_type = str(distance_type)  # "l2_sq" or "cosine"

        if self.flatten_channels:
            # Flattened mode: Single codebook of K bins for C-dimensional vectors
            # Centers: [K, C] - K codebook entries, each is a C-dimensional vector
            self.centers = nn.Parameter(torch.zeros(self.K, self.C))
            if self.learn_widths:
                # Alpha: [K] - learnable widths per codebook entry
                self.alpha = nn.Parameter(torch.zeros(self.K))
            self.register_buffer("_dbg_usage", torch.zeros(self.K))
        else:
            # Per-channel mode: One set of K bins per channel
            # Centers: [C, K] - one set of K bins per channel
            self.centers = nn.Parameter(torch.zeros(self.C, self.K))
            if self.learn_widths:
                # Alpha: [C, K] - learnable widths per channel per bin
                self.alpha = nn.Parameter(torch.zeros(self.C, self.K))
            self.register_buffer("_dbg_usage", torch.zeros(self.C, self.K))

        self.register_buffer("_inited", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("_dbg_avg_max_p", torch.tensor(0.0))
        self.register_buffer("_dbg_soft_perplexity", torch.tensor(0.0))

    def set_tau(self, tau: float):
        """Update temperature for annealing schedule."""
        self.tau = float(tau)

    @torch.no_grad()
    def _maybe_init_from_batch(self, x_flat: torch.Tensor):
        """Initialize centers from first batch.
        
        Args:
            x_flat: [B*T, C] for flattened mode, or [B, T, C] for per-channel mode
        """
        if self._inited.item() == 1:
            return
        
        if self.flatten_channels:
            # Flattened mode: x_flat is [B*T, C]
            B_T, C = x_flat.shape
            assert C == self.C, f"Input channels {C} != embedding_dim {self.C}"
            
            if self.init_method == "random":
                # Initialize codebook entries using random sampling from input
                indices = torch.randperm(B_T, device=x_flat.device)[:min(self.K, B_T)]
                sampled = x_flat[indices]  # [min(K, B*T), C]
                
                if sampled.shape[0] < self.K:
                    # If we have fewer samples than K, repeat and add noise
                    n_repeat = (self.K // sampled.shape[0]) + 1
                    sampled = sampled.repeat(n_repeat, 1)[:self.K]
                    noise = torch.randn_like(sampled) * 0.01
                    sampled = sampled + noise
                
                self.centers.data.copy_(sampled[:self.K])
            elif self.init_method == "uniform":
                # Initialize uniformly in the range of input data
                x_min = x_flat.min(dim=0)[0]  # [C]
                x_max = x_flat.max(dim=0)[0]  # [C]
                x_range = x_max - x_min
                x_range = torch.clamp(x_range, min=1e-6)
                
                # Generate K uniform samples in [x_min, x_max] for each dimension
                for k in range(self.K):
                    alpha = k / max(self.K - 1, 1)  # 0 to 1
                    self.centers.data[k] = x_min + alpha * x_range
            elif self.init_method == "gaussian":
                # Initialize from Gaussian distribution matching input statistics
                x_mean = x_flat.mean(dim=0)  # [C]
                x_std = x_flat.std(dim=0)  # [C]
                x_std = torch.clamp(x_std, min=1e-6)
                
                # Sample K vectors from N(x_mean, x_std^2)
                self.centers.data.normal_(0, 1)
                self.centers.data = self.centers.data * x_std.unsqueeze(0) + x_mean.unsqueeze(0)
            elif self.init_method == "kmeans":
                # K-means initialization: run a few iterations of k-means on first batch
                # This gives better spread of centers than random sampling
                B_T, C = x_flat.shape
                num_iters = 10
                
                # Initialize centers by random sampling
                indices = torch.randperm(B_T, device=x_flat.device)[:min(self.K, B_T)]
                centers = x_flat[indices].clone()  # [min(K, B_T), C]
                
                if centers.shape[0] < self.K:
                    # If we have fewer samples than K, repeat and add noise
                    n_repeat = (self.K // centers.shape[0]) + 1
                    centers = centers.repeat(n_repeat, 1)[:self.K]
                    noise = torch.randn_like(centers) * 0.01
                    centers = centers + noise
                
                # Run k-means iterations
                for _ in range(num_iters):
                    # Compute distances: [B_T, K]
                    dists = torch.cdist(x_flat, centers, p=2)  # [B_T, K]
                    # Assign to nearest center
                    assignments = dists.argmin(dim=1)  # [B_T]
                    
                    # Update centers: mean of assigned points
                    new_centers = torch.zeros_like(centers)  # [K, C]
                    counts = torch.zeros(self.K, device=x_flat.device, dtype=torch.long)
                    
                    for k in range(self.K):
                        mask = (assignments == k)
                        if mask.any():
                            new_centers[k] = x_flat[mask].mean(dim=0)
                            counts[k] = mask.sum()
                        else:
                            # If no points assigned, keep old center
                            new_centers[k] = centers[k]
                    
                    centers = new_centers
                
                self.centers.data.copy_(centers)
            else:
                raise ValueError(f"Unknown init_method: {self.init_method}. Use 'random', 'uniform', 'gaussian', or 'kmeans'")
        else:
            # Per-channel mode: x_flat is [B, T, C]
            B, T, C = x_flat.shape
            assert C == self.C, f"Input channels {C} != embedding_dim {self.C}"
            
            # Initialize per channel
            for c in range(C):
                x_c = x_flat[:, :, c]  # [B, T]
                xmin = x_c.min()
                xmax = x_c.max()
                if float(xmin) == float(xmax):
                    xmin = xmin - 0.5
                    xmax = xmax + 0.5
                bins = torch.linspace(xmin, xmax, self.K, device=x_flat.device, dtype=x_flat.dtype)
                self.centers[c, :].copy_(bins)
        
        self._inited.fill_(1)

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass with soft-to-hard quantization.
        
        Input: [B, C, T]
        Output: (loss, q_bct, perplexity, emb_like, idx, onehot)
        where idx: 
          - Flattened mode: [B, T] with values in [0..K-1]
          - Per-channel mode: [B, T, C] with values in [0..K-1] per channel
        """
        # Convert to [B, T, C]
        x_btc = inputs.permute(0, 2, 1).contiguous()  # [B, T, C]
        B, T, C = x_btc.shape
        assert C == self.C, f"Input channels {C} != embedding_dim {self.C}"
        
        if self.flatten_channels:
            # FLATTENED MODE: Flatten channels into single vector
            # x_flat: [B*T, C] - each row is a C-dimensional vector
            x_flat = x_btc.view(-1, C)  # [B*T, C]
            self._maybe_init_from_batch(x_flat)
            
            # Compute logits from distances/similarities
            tau = 1.0 if (self.tau is None or self.tau <= 0) else self.tau

            if self.distance_type == "cosine":
                # Cosine similarity — bounded in [-1, 1], prevents softmax saturation
                x_norm = F.normalize(x_flat, dim=-1)
                c_norm = F.normalize(self.centers, dim=-1)
                sim = torch.matmul(x_norm, c_norm.T)  # [B*T, K]
                if self.learn_widths:
                    s = F.softplus(self.alpha) + 1e-6  # [K]
                    logits = sim / (tau * s.unsqueeze(0))
                    width_loss = self.width_reg * s.mean()
                else:
                    logits = sim / tau
                    width_loss = 0.0
            elif self.distance_type == "dot":
                # Raw dot-product (no normalization) — ablation baseline
                sim = torch.matmul(x_flat, self.centers.T)  # [B*T, K]
                if self.learn_widths:
                    s = F.softplus(self.alpha) + 1e-6
                    logits = sim / (tau * s.unsqueeze(0))
                    width_loss = self.width_reg * s.mean()
                else:
                    logits = sim / tau
                    width_loss = 0.0
            else:
                # Squared L2 distance — matches paper Eq. 2 (free-energy)
                d_sq = torch.cdist(x_flat, self.centers, p=2).pow(2)  # [B*T, K]
                if self.learn_widths:
                    s = F.softplus(self.alpha) + 1e-6  # [K]
                    logits = -d_sq / (tau * s.unsqueeze(0))
                    width_loss = self.width_reg * s.mean()
                else:
                    logits = -d_sq / tau
                    width_loss = 0.0
            
            # Softmax over K dimension: [B*T, K]
            p = F.softmax(logits, dim=1)
            
            # Debug metrics
            avg_max_p = p.max(dim=1).values.mean()
            u = p.mean(dim=0)  # [K] - average usage per codebook entry
            eps = 1e-12
            soft_Pu = torch.exp(-(u.clamp_min(eps) * u.clamp_min(eps).log()).sum())
            
            self._dbg_avg_max_p = avg_max_p.detach()
            self._dbg_soft_perplexity = soft_Pu.detach()
            self._dbg_usage = u.detach()

            # Hard quantization: [B*T] with values in [0..K-1]
            hard_idx_flat = p.argmax(dim=1)  # [B*T]

            # Onehot encoding: [B*T, K]
            onehot = torch.zeros_like(p)
            onehot.scatter_(1, hard_idx_flat.unsqueeze(1), 1.0)

            # Hard usage histogram (true argmax distribution over codes).
            # Used for reported perplexity and dead-code restart EMA so they
            # reflect actual code utilization, not the soft softmax spread.
            u_hard = onehot.mean(dim=0)  # [K]

            # Quantized values: [B*T, C] - select codebook entry for each vector
            q_soft = torch.matmul(p, self.centers)  # [B*T, K] @ [K, C] = [B*T, C]
            q_hard = self.centers[hard_idx_flat]  # [B*T, C] - select hard assignment
            q = q_soft + (q_hard - q_soft).detach()  # Straight-through estimator
            
            # Reshape back: [B, T, C]
            q_btc = q.view(B, T, C)
            q_bct = q_btc.permute(0, 2, 1).contiguous()  # [B, C, T]
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(q_btc, x_btc)
            
            # Regularization losses
            p_squared_sum = (p ** 2).sum(dim=1)  # [B*T]
            L_peak = F.relu(1.0 - p_squared_sum).mean()  # Peakedness
            L_bins = (u ** 2).sum() * self.K  # Bin diversity (scaled by K, same as LGQ)
            L_floor = F.relu(self.p_min - soft_Pu) ** 2  # Perplexity floor
            
            loss = recon_loss + self.lambda_peak * L_peak + self.lambda_bins * L_bins + self.lambda_floor * L_floor

            # Perplexity from HARD assignments (true codebook utilization).
            # Bounded above by the number of unique argmax indices in the batch,
            # so it cannot mask collapse the way the soft-distribution version did.
            probs = u_hard.clamp_min(eps)
            perplexity = torch.exp(-(probs * probs.log()).sum())
            
            # Embedding-like tensor: [K*C, 1] for compatibility
            emb_like = self.centers.view(-1, 1)  # [K*C, 1]
            
            # Reshape idx to [B, T] (single index per token)
            idx = hard_idx_flat.view(B, T)
            
            # Onehot for compatibility: [B, T, K]
            onehot_reshaped = onehot.view(B, T, self.K)
            
            return loss, q_bct, perplexity, emb_like, idx, onehot_reshaped
        
        else:
            # PER-CHANNEL MODE: Original implementation
            self._maybe_init_from_batch(x_btc)
            
            # Process each channel independently
            x_flat = x_btc.view(-1, C)  # [B*T, C]
            
            # Compute distances: [B*T, C, K] — for scalar channels,
            # squared L2 = (x - c)^2; absolute difference for legacy compat.
            centers_expanded = self.centers.unsqueeze(0)  # [1, C, K]
            x_expanded = x_flat.unsqueeze(2)  # [B*T, C, 1]
            d_sq = (x_expanded - centers_expanded).pow(2)  # [B*T, C, K]

            # Compute logits and probabilities per channel
            tau = 1.0 if (self.tau is None or self.tau <= 0) else self.tau
            if self.learn_widths:
                s = F.softplus(self.alpha) + 1e-6  # [C, K]
                logits = -d_sq / (tau * s.unsqueeze(0))  # [B*T, C, K]
                width_loss = self.width_reg * s.mean()
            else:
                logits = -d_sq / tau
                width_loss = 0.0
            
            # Softmax over K dimension: [B*T, C, K]
            p = F.softmax(logits, dim=2)
            
            # Debug metrics
            avg_max_p = p.max(dim=2).values.mean()
            u = p.mean(dim=0)  # [C, K] - average usage per channel per bin
            eps = 1e-12
            soft_Pu_per_channel = (-(u.clamp_min(eps) * u.clamp_min(eps).log()).sum(dim=1)).exp()
            soft_Pu = soft_Pu_per_channel.mean()
            
            self._dbg_avg_max_p = avg_max_p.detach()
            self._dbg_soft_perplexity = soft_Pu.detach()
            self._dbg_usage = u.detach()
            
            # Hard quantization: [B*T, C] with values in [0..K-1]
            hard_idx_flat = p.argmax(dim=2)  # [B*T, C]
            
            # Onehot encoding: [B*T, C, K]
            onehot = torch.zeros_like(p)
            hard_idx_expanded = hard_idx_flat.unsqueeze(2)  # [B*T, C, 1]
            onehot.scatter_(2, hard_idx_expanded, 1.0)
            
            # Quantized values: [B*T, C]
            q_soft = torch.sum(p * self.centers.unsqueeze(0), dim=2)  # [B*T, C]
            q_hard = torch.sum(onehot * self.centers.unsqueeze(0), dim=2)  # [B*T, C]
            q = q_soft + (q_hard - q_soft).detach()  # Straight-through estimator
            
            # Reshape back: [B, T, C]
            q_btc = q.view(B, T, C)
            q_bct = q_btc.permute(0, 2, 1).contiguous()  # [B, C, T]
            
            # Reconstruction loss (MSE for quantization loss component)
            recon_loss = F.mse_loss(q_btc, x_btc)
            
            # Regularization losses
            p_squared_sum = (p ** 2).sum(dim=2)  # [B*T, C]
            L_peak = F.relu(1.0 - p_squared_sum).mean()  # Peakedness (Eq. 5)
            L_bins = (u ** 2).sum() * self.K  # Bin diversity (scaled by K)
            L_floor = F.relu(self.p_min - soft_Pu) ** 2  # Perplexity floor

            loss = recon_loss + self.lambda_peak * L_peak + self.lambda_bins * L_bins + self.lambda_floor * L_floor

            # Perplexity: average over channels
            hard_onehot_avg = onehot.mean(dim=0)  # [C, K]
            perplexity_per_channel = []
            for c in range(C):
                probs = hard_onehot_avg[c].clamp_min(eps)
                perplexity_per_channel.append(torch.exp(-(probs * probs.log()).sum()))
            perplexity = torch.stack(perplexity_per_channel).mean()
            
            # Embedding-like tensor: [C*K, 1] for compatibility
            emb_like = self.centers.view(-1, 1)  # [C*K, 1]
            
            # Reshape idx to [B, T, C]
            idx = hard_idx_flat.view(B, T, C)
            
            return loss, q_bct, perplexity, emb_like, idx, onehot.view(B, T, C, self.K)


class MultiBinDiscretizerFair(nn.Module):
    """
    LGQ per-channel quantizer with FSQ-style structure (fair comparison).
    Maps encoder to len(levels) channels, then scalar-quantizes each channel
    with levels[c] bins. Same setup as FSQ: e.g. 4 channels [16,16,8,8] -> 16K codes.
    """

    def __init__(
        self,
        levels: list[int],
        tau: float = 1.0,
        lambda_peak: float = 0.005,
        lambda_bins: float = 0.005,
        lambda_floor: float = 0.0,
        p_min: float = 0.0,
        init_method: str = "random",
    ):
        super().__init__()
        self.levels = levels
        self.C = len(levels)
        self.K_max = max(levels)
        self.tau = float(tau)
        self.lambda_peak = float(lambda_peak)
        self.lambda_bins = float(lambda_bins)
        self.lambda_floor = float(lambda_floor)
        self.p_min = float(p_min)
        self.init_method = str(init_method)
        self.codebook_size = int(__import__("math").prod(levels))

        self.centers = nn.Parameter(torch.zeros(self.C, self.K_max))
        self.register_buffer("_dbg_usage", torch.zeros(self.C, self.K_max))
        self.register_buffer("_inited", torch.tensor(0, dtype=torch.uint8))

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    @torch.no_grad()
    def _maybe_init_from_batch(self, x_btc: torch.Tensor) -> None:
        if self._inited.item() == 1:
            return
        B, T, C = x_btc.shape
        assert C == self.C
        for c in range(C):
            L = self.levels[c]
            x_c = x_btc[..., c].reshape(-1)
            xmin = x_c.min().item()
            xmax = x_c.max().item()
            if xmin == xmax:
                xmin -= 0.5
                xmax += 0.5
            bins = torch.linspace(xmin, xmax, L, device=x_btc.device, dtype=x_btc.dtype)
            self.centers.data[c, :L].copy_(bins)
        self._inited.fill_(1)

    def forward(self, inputs: torch.Tensor):
        x_btc = inputs.permute(0, 2, 1).contiguous()
        B, T, C = x_btc.shape
        assert C == self.C
        self._maybe_init_from_batch(x_btc)

        device = x_btc.device
        eps = 1e-12
        tau = max(1e-6, self.tau)
        losses = []
        q_list = []
        idx_list = []
        p_list = []
        u_list = []
        u_hard_list = []

        for c in range(C):
            L = self.levels[c]
            x_c = x_btc[..., c]
            cen = self.centers[c, :L]
            d_sq = (x_c.unsqueeze(-1) - cen.unsqueeze(0).unsqueeze(0)).pow(2)
            logits = -d_sq / tau
            p = F.softmax(logits, dim=-1)
            hard_idx = p.argmax(dim=-1)
            onehot = torch.zeros_like(p)
            onehot.scatter_(-1, hard_idx.unsqueeze(-1), 1.0)
            q_soft = (p * cen.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            q_hard = (onehot * cen.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            q_c = q_soft + (q_hard - q_soft).detach()
            q_list.append(q_c)
            idx_list.append(hard_idx)
            p_list.append(p)
            u = p.mean(dim=(0, 1))
            u_list.append(u)
            # Hard usage for honest perplexity reporting (true argmax histogram).
            u_hard_list.append(onehot.mean(dim=(0, 1)))
            recon_c = F.mse_loss(q_c, x_c)
            L_peak = F.relu(1.0 - (p ** 2).sum(dim=-1)).mean()
            L_bins = (u ** 2).sum() * len(cen)  # Bin diversity (scaled by K)
            loss_c = recon_c + self.lambda_peak * L_peak + self.lambda_bins * L_bins
            losses.append(loss_c)

        loss = torch.stack(losses).mean()
        soft_Pu = 1.0
        for c in range(C):
            L = self.levels[c]
            u = u_list[c].clamp_min(eps)
            soft_Pu = soft_Pu * torch.exp(-(u * u.log()).sum())
        L_floor = F.relu(self.p_min - soft_Pu) ** 2
        loss = loss + self.lambda_floor * L_floor

        q_btc = torch.stack(q_list, dim=-1)
        q_bct = q_btc.permute(0, 2, 1).contiguous()
        idx_btc = torch.stack(idx_list, dim=-1)
        # Perplexity from HARD assignments per channel (true codebook utilization).
        perplexity_per_ch = []
        for c in range(C):
            u_h = u_hard_list[c].clamp_min(eps)
            perplexity_per_ch.append(torch.exp(-(u_h * u_h.log()).sum()))
        perplexity = torch.stack(perplexity_per_ch).mean()

        onehot_full = torch.zeros(B, T, C, self.K_max, device=device, dtype=q_btc.dtype)
        for c in range(C):
            L = self.levels[c]
            onehot_full[..., c, :L] = p_list[c]
        emb_like = self.centers.view(-1, 1)
        usage = torch.zeros(self.C, self.K_max, device=device)
        for c in range(C):
            usage[c, : self.levels[c]] = u_list[c].detach()
        self._dbg_usage = usage

        return loss, q_bct, perplexity, emb_like, idx_btc, onehot_full


def prune_codebook(quantizer, keep_mask):
    """
    Permanently remove codebook entries from an LGQ quantizer.

    Args:
        quantizer: MultiBinDiscretizer or MultiBinDiscretizerFair instance.
        keep_mask:
          - MultiBinDiscretizer (flattened or per-channel): BoolTensor [K],
            True = keep.  For per-channel mode the same columns are removed
            across all C channels.
          - MultiBinDiscretizerFair: list of C BoolTensors, each [levels[c]].

    Returns:
        old_to_new: dict mapping old index -> new contiguous index for kept
        entries (flattened / per-channel), or list of such dicts (fair).
    """
    if isinstance(quantizer, MultiBinDiscretizerFair):
        return _prune_fair(quantizer, keep_mask)

    # --- MultiBinDiscretizer (flattened or per-channel) ---
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]
    new_K = len(keep_indices)
    device = quantizer.centers.device

    if quantizer.flatten_channels:
        # centers: [K, C] -> [K', C]
        new_centers = quantizer.centers.data[keep_indices].clone()
    else:
        # centers: [C, K] -> [C, K']
        new_centers = quantizer.centers.data[:, keep_indices].clone()

    quantizer.centers = nn.Parameter(new_centers)

    # Resize _dbg_usage buffer
    if quantizer.flatten_channels:
        quantizer._dbg_usage = torch.zeros(new_K, device=device)
    else:
        quantizer._dbg_usage = torch.zeros(quantizer.C, new_K, device=device)

    # Resize alpha if learnable widths
    if quantizer.learn_widths:
        if quantizer.flatten_channels:
            new_alpha = quantizer.alpha.data[keep_indices].clone()
        else:
            new_alpha = quantizer.alpha.data[:, keep_indices].clone()
        quantizer.alpha = nn.Parameter(new_alpha)

    quantizer.K = new_K

    # Build old -> new index mapping
    old_to_new = {}
    for new_idx, old_idx in enumerate(keep_indices.tolist()):
        old_to_new[old_idx] = new_idx
    return old_to_new


def _prune_fair(quantizer, keep_masks):
    """Prune MultiBinDiscretizerFair with per-channel masks."""
    assert len(keep_masks) == quantizer.C
    device = quantizer.centers.device

    new_levels = []
    old_to_new_list = []

    for c in range(quantizer.C):
        mask_c = keep_masks[c]
        keep_idx = mask_c.nonzero(as_tuple=True)[0]
        new_levels.append(len(keep_idx))
        mapping = {}
        for new_i, old_i in enumerate(keep_idx.tolist()):
            mapping[old_i] = new_i
        old_to_new_list.append(mapping)

    new_K_max = max(new_levels)
    new_centers = torch.zeros(quantizer.C, new_K_max,
                              device=device, dtype=quantizer.centers.dtype)
    for c in range(quantizer.C):
        mask_c = keep_masks[c]
        keep_idx = mask_c.nonzero(as_tuple=True)[0]
        kept = quantizer.centers.data[c, :quantizer.levels[c]][keep_idx]
        new_centers[c, :new_levels[c]] = kept

    quantizer.centers = nn.Parameter(new_centers)
    quantizer._dbg_usage = torch.zeros(quantizer.C, new_K_max, device=device)
    quantizer.levels = new_levels
    quantizer.K_max = new_K_max
    quantizer.codebook_size = int(math.prod(new_levels))

    return old_to_new_list
