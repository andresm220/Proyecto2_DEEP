# Proyecto 2 — Diseño generativo de personajes con GANs

**Los Errantes de Kaminal**: un reino de fantasía medieval en pixel art estilo RPG de 16 bits, visto desde arriba
como en los juegos clásicos de aventura. Sus habitantes son aventureros errantes: guerreros con armadura, magos con
túnica, exploradores con capucha y gente común del reino. Una DCGAN entrenada por nosotros sobre sprites LPC de 64×64
genera los personajes. **Todas las imágenes de la galería salen de ese generador.**

Curso: Deep Learning y Sistemas Inteligentes 2026, UVG.

## Instalación

```bash
pip install -r requirements.txt
```

Versiones exactas con las que se desarrolló y probó el código:

| Paquete | Versión |
|---|---|
| Python | 3.12 |
| torch | 2.10.0 |
| torchvision | 0.25.0 |
| numpy | 2.4.6 |
| matplotlib | 3.10.1 |
| scikit-learn | 1.8.0 |
| pyyaml | 6.0.2 |
| pillow | 11.1.0 |
| pytest | 8.3.5 |

En Google Colab, torch y torchvision ya vienen instalados. El flujo completo en Colab está en
`notebooks/02_entrenamiento_y_experimentos.ipynb`.

## Estructura

```
configs/      base.yaml (control), exp_hinge.yaml, exp_sn.yaml, final.yaml
src/
  data.py       dataset TinyHero: descarga, RGBA→RGB sobre blanco, [-1, 1], flip horizontal
  models.py     DCGAN (Generator, Discriminator con flag spectral_norm_d)
  losses.py     bce_nonsat y hinge
  train.py      loop de entrenamiento genérico por config, checkpoints, --resume
  compare.py    rejillas y curvas de varios runs lado a lado
  utils.py      semillas, configs, rejillas, gráficas
  evaluate.py   diversidad, novedad y vecinos más cercanos (Persona B)
  gallery.py    selección y regeneración de la galería (Persona B)
notebooks/    01_datos, 02_entrenamiento_y_experimentos, 03_galeria_y_vecinos
tests/        pytest (CPU, unos 3 min)
outputs/      figs/, data_stats.json, fixed_noise.pt, runs/<name>/
galeria/      10 PNG + seeds.json + z_vectors.pt
```

## Datos

```bash
python -m src.data            # descarga data.zip, genera outputs/data_stats.json y outputs/figs/muestras_*.png
```

TinyHero tiene 3 648 sprites de 64×64: 912 personajes × 4 vistas (espalda, izquierda, frente, derecha).
Los PNG (RGBA) se componen sobre **fondo negro**, se normalizan a [−1, 1] y se usa flip horizontal p=0.5.
Los detalles de la preparación están en `notebooks/01_datos.ipynb`.

## Problemas encontrados

1. **Con fondo blanco, G colapsaba en todas las configuraciones.** La primera base (DCGAN del tutorial, fondo
   blanco, 100 épocas) aprendió personajes reconocibles entre las épocas 20 y 60, pero **los 16 z de la rejilla
   fija daban el mismo personaje** (mode collapse), y después de la época 60 divergió a ruido: la pérdida de D
   cayó a 0.004 con D(x) ≈ 0.998 y D(G(z)) ≈ 0.002 (D gana y G se queda sin gradiente útil).
2. **No era un problema de `G.eval()`:** en modo `train` las 16 muestras también eran idénticas. Medido capa por
   capa, la variación entre muestras se apagaba dentro de G (de 0.99 a 0.02) con los γ de BatchNorm intactos
   (≈ 1): G aprendió a ignorar z.
3. **Pruebas cortas de 30 épocas** (`configs/probes/`, todas con fondo blanco salvo donde se indica): SN, hinge,
   instance noise, dropout en D, etiquetas ruidosas y sus combinaciones **siguieron colapsando**. La receta de la
   autora del dataset (fondo negro + etiquetas ruidosas + dropout + batch 64) y **solo fondo negro** dieron
   personajes distintos para cada z.
4. **Decisión:** el control pasa a ser la DCGAN del tutorial **+ fondo negro** (un solo cambio, en los datos;
   negro es además el RGB que los PNG guardan bajo los píxeles transparentes). Base, hinge y SN se entrenaron
   de nuevo desde cero con ese control. Los runs con fondo blanco quedan en Drive en `runs_fondo_blanco/`.
   No sabemos con certeza *por qué* el fondo blanco provoca el colapso; es un hallazgo empírico.

## Entrenamiento

Cada run lo define un YAML. Los experimentos heredan de `base.yaml` (clave `inherit`) y cambian **una sola** clave:

```bash
python -m src.train --config configs/base.yaml        # control
python -m src.train --config configs/exp_hinge.yaml   # experimento de pérdida: loss: hinge
python -m src.train --config configs/exp_sn.yaml      # experimento de estabilización: spectral_norm_d: true
python -m src.train --config configs/final.yaml       # modelo final (genera la galería)

# opciones: --out-root <carpeta> (en Colab, una carpeta de Drive) · --resume (continuar desde last.pt)
```

Cada run deja en `outputs/runs/<name>/`: `checkpoints/epoch_XXX.pt` y `last.pt`, `samples/epoch_XXX.png`
(rejilla de ruido fijo), `losses.csv` (`epoch, iter, loss_D, loss_G, D_x, D_G_z`), `fixed_grid.png`,
`losses.png`, `progress.png`, `config.yaml` y `train_log.json`.

`--resume` continúa desde `last.pt` (que se guarda cada `ckpt_every` épocas) con los estados del optimizador y de
todos los generadores de números aleatorios. En CPU reproduce **exactamente** una corrida sin interrupciones
(lo verifica `tests/test_train.py`); en GPU queda muy cerca pero no idéntica, porque ConvTranspose en cuDNN no es
determinista bit a bit. Si la config cambió respecto al checkpoint (salvo `epochs`), se niega a continuar.
Correr de nuevo un run que ya tiene `last.pt` sin `--resume` falla a propósito, para no borrar un run terminado;
para empezar de cero hay que pasar `--overwrite`.

Comparar runs:

```bash
python -m src.compare --runs base exp_hinge --out outputs/figs/cmp_base_vs_hinge
```

Tests: `python -m pytest tests/ -p no:asyncio`

## Galería

```bash
python -m src.gallery --regenerate --weights <ruta a los pesos finales> --seeds galeria/seeds.json
```

- Se generan **N = 200** muestras. La muestra *i* usa `torch.Generator().manual_seed(i)` → `z_i`, así que cada
  personaje queda identificado por un entero.
- **Tasa de selección:** 10 de 200. El criterio de selección (k-means sobre features más revisión visual) está en
  `galeria/seeds.json` y en la presentación.
- **La galería se genera en CPU y de a una muestra (batch 1)**, tanto al seleccionar como al regenerar. En GPU,
  cuDNN puede elegir algoritmos distintos según el tamaño del batch, y una diferencia de 1e-6 basta para cambiar
  algún píxel al redondear a 0–255. En CPU y con batch 1, la regeneración es exacta píxel a píxel.
- **Siempre `G.eval()`.** En modo `eval`, BatchNorm usa las estadísticas acumuladas durante el entrenamiento, así que
  una misma `z` produce siempre la misma imagen, sin importar con qué otras `z` vaya en el batch. En modo `train`,
  BatchNorm normaliza con las estadísticas del batch actual y la imagen cambia. Por eso hay que usar el mismo modo
  al seleccionar y al regenerar; si se mezclan, la galería no se reproduce igual.

**Pesos finales:** *(enlace de Google Drive, se agrega al terminar el entrenamiento final)*

## Semillas

| Uso | Semilla |
|---|---|
| Global (inicialización, barajado, ruido de entrenamiento) | 42 |
| Ruido fijo de la rejilla (`outputs/fixed_noise.pt`, 16 × 100) | 1234 |
| Galería | 0–199 (una por muestra) |

## Atribución y licencia del dataset

- **Dataset:** TinyHero, de Agnieszka Mikołajczyk — https://github.com/AgaMiko/pixel_character_generator
- **Generado con:** Universal LPC Spritesheet Character Generator, de makrohn — https://github.com/makrohn/Universal-LPC-spritesheet
- **Arte:** contribuidores del Liberated Pixel Cup (LPC).
- **Licencia:** doble, **CC-BY-SA 3.0** y **GNU GPL 3.0**. Los derivados de este proyecto (pesos y galería)
  se publican bajo las mismas licencias.

## Declaración de uso de IA

Se usó Claude para planificar y estructurar el proyecto, y Claude Code para escribir y depurar el código. El equipo
revisó, entendió y verificó el código y los resultados. Ninguna imagen de la galería proviene de un generador
externo: todas salen del generador DCGAN entrenado por nosotros.
