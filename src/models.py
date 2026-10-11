"""DCGAN (Radford et al., 2015), siguiendo el tutorial oficial de PyTorch, para 64x64.

Generador:      z (nz x 1 x 1) -> 512x4x4 -> 256x8x8 -> 128x16x16 -> 64x32x32 -> 3x64x64 (tanh)
Discriminador:  espejo con Conv stride 2 -> 1 logit (SIN sigmoid: la pérdida trabaja sobre logits,
                lo que es numéricamente más estable y además lo necesita la pérdida hinge).
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def weights_init(m: nn.Module) -> None:
    """Inicialización del paper DCGAN: conv ~ N(0, 0.02); BatchNorm: gamma ~ N(1, 0.02), beta = 0."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()

        def up(cin, cout, stride=2, padding=1):
            # ConvTranspose 4x4 stride 2 duplica la resolución; bias=False porque BatchNorm lo anula.
            return [nn.ConvTranspose2d(cin, cout, 4, stride, padding, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(True)]

        self.net = nn.Sequential(
            *up(nz, ngf * 8, stride=1, padding=0),   # 1x1 -> 4x4
            *up(ngf * 8, ngf * 4),                   # -> 8x8
            *up(ngf * 4, ngf * 2),                   # -> 16x16
            *up(ngf * 2, ngf),                       # -> 32x32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # -> 64x64
            nn.Tanh(),                               # salida en [-1, 1], igual que los datos
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64, nc: int = 3, spectral_norm_d: bool = False):
        super().__init__()

        def down(cin, cout, bn=True):
            layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *down(nc, ndf, bn=False),     # 64 -> 32 (sin BN en la primera capa, como en el paper)
            *down(ndf, ndf * 2),          # -> 16
            *down(ndf * 2, ndf * 4),      # -> 8
            *down(ndf * 4, ndf * 8),      # -> 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # 4x4 -> 1x1: un logit por imagen
        )
        self.apply(weights_init)
        if spectral_norm_d:
            # Experimento de estabilización: cada Conv se envuelve con normalización espectral, que divide
            # el peso por su mayor valor singular -> cada capa es 1-Lipschitz y D no puede volverse
            # arbitrariamente "afilado". Se aplica DESPUÉS de inicializar para que la init N(0, 0.02)
            # quede en `weight_orig`. Todo lo demás (incluida BatchNorm) se mantiene igual a base.
            for i, layer in enumerate(self.net):
                if isinstance(layer, nn.Conv2d):
                    self.net[i] = spectral_norm(layer)

    def forward(self, x):
        return self.net(x).view(-1)


def build_models(cfg: dict):
    G = Generator(cfg["nz"], cfg["ngf"])
    G.apply(weights_init)
    D = Discriminator(cfg["ndf"], spectral_norm_d=cfg.get("spectral_norm_d", False))
    return G, D
