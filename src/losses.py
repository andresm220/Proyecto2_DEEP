"""Funciones de pérdida. Todas reciben LOGITS del discriminador (sin sigmoid).

Cada pérdida es un par (d_loss, g_loss):
    d_loss(real_logits, fake_logits) -> escalar que D minimiza
    g_loss(fake_logits)              -> escalar que G minimiza
"""
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ BCE no saturante (base)
def bce_d_loss(real_logits, fake_logits, label_noise: float = 0.0):
    # D quiere: reales -> 1, falsas -> 0. BCEWithLogits aplica la sigmoide internamente
    # de forma estable (log-sum-exp), en vez de hacer sigmoid + BCE por separado.
    real_t, fake_t = torch.ones_like(real_logits), torch.zeros_like(fake_logits)
    if label_noise > 0:
        # Etiquetas ruidosas (Salimans et al. 2016; receta de la autora de TinyHero): reales ~ U[1-a, 1],
        # falsas ~ U[0, a]. D nunca puede llevar la pérdida a 0 volviéndose infinitamente seguro, así que
        # no "satura" y sigue dando a G gradientes que distinguen entre muestras.
        real_t = real_t - label_noise * torch.rand_like(real_t)
        fake_t = fake_t + label_noise * torch.rand_like(fake_t)
    return (F.binary_cross_entropy_with_logits(real_logits, real_t)
            + F.binary_cross_entropy_with_logits(fake_logits, fake_t))


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


def get_loss(name: str, label_noise: float = 0.0):
    """label_noise solo aplica a BCE (las etiquetas 0/1 son propias de esa pérdida; hinge usa márgenes)."""
    if name not in LOSSES:
        raise ValueError(f"Pérdida desconocida '{name}'. Opciones: {list(LOSSES)}")
    d_loss, g_loss = LOSSES[name]
    if name == "bce_nonsat" and label_noise > 0:
        return (lambda r, f: bce_d_loss(r, f, label_noise)), g_loss
    return d_loss, g_loss
