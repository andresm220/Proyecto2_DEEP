"""Funciones de pérdida. Todas reciben LOGITS del discriminador (sin sigmoid).

Cada pérdida es un par (d_loss, g_loss):
    d_loss(real_logits, fake_logits) -> escalar que D minimiza
    g_loss(fake_logits)              -> escalar que G minimiza
"""
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ BCE no saturante (base)
def bce_d_loss(real_logits, fake_logits):
    # D quiere: reales -> 1, falsas -> 0. BCEWithLogits aplica la sigmoide internamente
    # de forma estable (log-sum-exp), en vez de hacer sigmoid + BCE por separado.
    return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))


def bce_nonsat_g_loss(fake_logits):
    # No saturante: G maximiza log D(G(z)) en vez de minimizar log(1 - D(G(z))).
    # Al inicio D rechaza fácilmente las falsas; con log(1-D) el gradiente sería ~0
    # justo cuando G más lo necesita. Con -log D el gradiente es grande en ese régimen.
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


# ------------------------------------------------------------------ Hinge (experimento de pérdida)
def hinge_d_loss(real_logits, fake_logits):
    # D: mean(relu(1 - D(x))) + mean(relu(1 + D(G(z)))).
    # Las muestras ya clasificadas con margen (D(x) > 1 o D(G(z)) < -1) aportan pérdida 0:
    # D deja de "empujar" sobre ellas y se concentra en las dudosas.
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits):
    # G: -mean(D(G(z))). Lineal en el logit: el gradiente no se satura aunque D esté muy seguro.
    return -fake_logits.mean()


LOSSES = {
    "bce_nonsat": (bce_d_loss, bce_nonsat_g_loss),
    "hinge": (hinge_d_loss, hinge_g_loss),
}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"Pérdida desconocida '{name}'. Opciones: {list(LOSSES)}")
    return LOSSES[name]
