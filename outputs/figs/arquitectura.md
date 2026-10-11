# Arquitectura DCGAN (configuración base)

## Generador
| Capa | Salida (C×H×W) | Parámetros |
|---|---|---|
| ConvTranspose2d | 512×4×4 | 819,200 |
| BatchNorm2d | 512×4×4 | 1,024 |
| ReLU | 512×4×4 | 0 |
| ConvTranspose2d | 256×8×8 | 2,097,152 |
| BatchNorm2d | 256×8×8 | 512 |
| ReLU | 256×8×8 | 0 |
| ConvTranspose2d | 128×16×16 | 524,288 |
| BatchNorm2d | 128×16×16 | 256 |
| ReLU | 128×16×16 | 0 |
| ConvTranspose2d | 64×32×32 | 131,072 |
| BatchNorm2d | 64×32×32 | 128 |
| ReLU | 64×32×32 | 0 |
| ConvTranspose2d | 3×64×64 | 3,072 |
| Tanh | 3×64×64 | 0 |
| **Total** | | **3,576,704** |

## Discriminador
| Capa | Salida (C×H×W) | Parámetros |
|---|---|---|
| Conv2d | 64×32×32 | 3,072 |
| LeakyReLU | 64×32×32 | 0 |
| Conv2d | 128×16×16 | 131,072 |
| BatchNorm2d | 128×16×16 | 256 |
| LeakyReLU | 128×16×16 | 0 |
| Conv2d | 256×8×8 | 524,288 |
| BatchNorm2d | 256×8×8 | 512 |
| LeakyReLU | 256×8×8 | 0 |
| Conv2d | 512×4×4 | 2,097,152 |
| BatchNorm2d | 512×4×4 | 1,024 |
| LeakyReLU | 512×4×4 | 0 |
| Conv2d | 1×1×1 | 8,192 |
| **Total** | | **2,765,568** |

## Configuración base (control)

| Clave | Valor |
|---|---|
| `name` | `base` |
| `seed` | `42` |
| `image_size` | `64` |
| `nz` | `100` |
| `ngf` | `64` |
| `ndf` | `64` |
| `batch_size` | `128` |
| `epochs` | `100` |
| `lr_g` | `0.0002` |
| `lr_d` | `0.0002` |
| `betas` | `[0.5, 0.999]` |
| `loss` | `bce_nonsat` |
| `spectral_norm_d` | `False` |
| `augment_hflip` | `True` |
| `fixed_noise_seed` | `1234` |
| `n_fixed` | `16` |
| `snapshot_every` | `5` |
| `ckpt_every` | `5` |
