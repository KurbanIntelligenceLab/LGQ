"""PatchGAN discriminator with hinge loss and adaptive weight (VQGAN-style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization."""

    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for i in range(1, n_layers + 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            stride = 2 if i < n_layers else 1
            layers += [
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride, 1)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)
            ),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def hinge_loss_d(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss."""
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def hinge_loss_g(logits_fake: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss."""
    return -logits_fake.mean()


def calculate_adaptive_weight(
    rec_loss: torch.Tensor,
    g_loss: torch.Tensor,
    last_layer_weight: torch.Tensor,
    max_weight: float = 1e4,
) -> torch.Tensor:
    """
    VQGAN adaptive weight: balances reconstruction and adversarial gradients
    at the last decoder layer so neither dominates.
    """
    rec_grads = torch.autograd.grad(
        rec_loss, last_layer_weight, retain_graph=True
    )[0]
    g_grads = torch.autograd.grad(
        g_loss, last_layer_weight, retain_graph=True
    )[0]
    weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-6)
    weight = torch.clamp(weight, 0.0, max_weight).detach()
    return weight
