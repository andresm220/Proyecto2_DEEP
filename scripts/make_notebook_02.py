"""Genera notebooks/02_entrenamiento_y_experimentos.ipynb (se ejecuta en Google Colab con GPU)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

HINGE_HYPOTHESIS = (
    "Creemos que la pérdida hinge mejorará la estabilidad y la diversidad porque no satura cuando D está muy "
    "seguro y su margen limita cuánto se 'castiga' a muestras ya bien clasificadas. Lo consideraremos útil si la "
    "rejilla de ruido fijo muestra menos filas repetidas y la métrica de diversidad sube respecto a base sin bajar "
    "la calidad visual."
)

cells = [
    md("""# 02 — Entrenamiento y experimentos (Colab)

**Antes de empezar:** *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*.

Todo lo que produce el entrenamiento se guarda en Google Drive (`OUT`), porque Colab borra el disco al desconectarse.
Si Colab se cae a mitad de un run, basta con volver a correr las celdas (o "Ejecutar todo"): `entrenar()` detecta
el `last.pt` del run y reanuda desde ahí con `--resume`."""),
    code("""from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/proyecto2-gan/outputs'   # carpeta compartida con Persona B
!mkdir -p "$OUT\""""),
    code("""import os

def entrenar(config, name):
    \"\"\"Entrena un run; si ya existe su last.pt (Colab se cayó, o se usó 'Ejecutar todo'), reanuda en vez de
    fallar. train.py se niega a sobrescribir un run existente sin --resume / --overwrite.\"\"\"
    resume = '--resume' if os.path.exists(f'{OUT}/runs/{name}/checkpoints/last.pt') else ''
    !python -m src.train --config {config} --out-root "{OUT}" {resume}"""),
    code("""REPO_URL = 'https://github.com/andresm220/Proyecto2_DEEP.git'
!rm -rf /content/proyecto2-gan && git clone -q $REPO_URL /content/proyecto2-gan
%cd /content/proyecto2-gan
!git log --oneline -1
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    md("""## 1. Medir el tiempo por época

Con 28 iteraciones por época, una época en T4 debería tardar pocos segundos. Medimos 2 épocas en un run aparte
(`timing`) para fijar `epochs` en `configs/base.yaml`: el objetivo es que **cada run tarde ≲ 30–40 min**.
Como hinge y SN heredan de base, cambiar `epochs` en base lo cambia para todos los experimentos a la vez."""),
    code("""!python -m src.train --config configs/base.yaml --name timing --epochs 2 --out-root "$OUT" --overwrite
import json
t = json.load(open(f'{OUT}/runs/timing/train_log.json'))
print(f"{t['sec_per_epoch_mean']:.1f} s/época -> en 30 min caben ~{int(1800 / t['sec_per_epoch_mean'])} épocas")"""),
    md("""**Decisión de épocas:** en la T4 una época tardó **5.8 s** (con el guardado en Drive incluido), así que en
30 min caben ~308 épocas. Elegimos **250 épocas ≈ 24 min por run** (con margen), `snapshot_every: 10` y
`ckpt_every: 25`. El modelo final usa 500 épocas (≈ 48 min), ver la sección 5.

> Cada checkpoint pesa ~75 MB (G + D + los dos Adam) y `last.pt` se guarda cada `ckpt_every` épocas.
> Poner `ckpt_every ≈ epochs / 10` (≈ 10 checkpoints, ~750 MB por run) para no llenar Drive. Revisar también la
> papelera de Drive después de esta celda: si los archivos reemplazados se acumulan ahí, vaciarla.
> `ckpt_every` y `snapshot_every` **no** afectan el entrenamiento (solo guardan cosas), así que cambiarlos no rompe
> la regla de "una sola diferencia". El tiempo medido ya incluye guardar en Drive."""),
    md("""## 2. Entrenar base (el control)"""),
    code("""entrenar('configs/base.yaml', 'base')"""),
    code("""from IPython.display import Image, display
for f in ['fixed_grid.png', 'progress.png', 'losses.png']:
    display(Image(f'{OUT}/runs/base/{f}'))"""),
    md("""### Problema encontrado: con fondo blanco G colapsaba

La primera base se entrenó con los sprites compuestos sobre **fondo blanco**. Entre las épocas 20 y 60 aprendió
personajes reconocibles, pero **los 16 z de la rejilla fija daban el mismo personaje** (mode collapse), y después
de la época 60 divergió a ruido: pérdida de D 0.004, D(x) ≈ 0.998, D(G(z)) ≈ 0.002.

- No era el modo `eval` de BatchNorm: en modo `train` las 16 muestras también eran idénticas. Medido capa por capa,
  la variación entre muestras se apagaba dentro de G (0.99 → 0.02) con los γ de BN ≈ 1: G aprendió a ignorar z.
- Pruebas de 30 épocas (`configs/probes/`): SN, hinge, instance noise, dropout en D y etiquetas ruidosas
  **siguieron colapsando** con fondo blanco. **Solo el fondo negro** dio personajes distintos para cada z.
- **Decisión:** el control pasa a ser la DCGAN del tutorial + fondo negro (`background: black` en `base.yaml`,
  un solo cambio, en los datos). Base, hinge y SN se entrenaron de nuevo desde cero. Los runs con fondo blanco
  quedan en Drive en `runs_fondo_blanco/` (figuras en `outputs/runs_fondo_blanco/base/` del repo).
  No sabemos con certeza *por qué* el fondo blanco provoca el colapso: es un hallazgo empírico.

**Base con fondo negro (250 épocas):** 16 personajes distintos en la rejilla fija, diversidad 0.102 (el dataset
real tiene 0.189); última época D(x) 0.97 y D(G(z)) 0.035: D domina, pero sin colapsar."""),
    md("""**Sincronización S1:** en cuanto exista `runs/base/checkpoints/epoch_005.pt`, avisar a Persona B. B necesita
la carpeta `proyecto2-gan/outputs` compartida (incluye `fixed_noise.pt`, que deben usar TODOS los runs)."""),
    md(f"""## 3. Experimento de pérdida: hinge

**Hipótesis (escrita antes de entrenar):** {HINGE_HYPOTHESIS}

**Cambio único respecto a base:** `loss: hinge` (ver `configs/exp_hinge.yaml`). La celda siguiente deja impresa
la fecha y hora en que se registró la hipótesis, antes de lanzar el entrenamiento."""),
    code("""import datetime
print('Hipótesis registrada:', datetime.datetime.now().isoformat(timespec='minutes'))
!python -c "from src.utils import load_config, diff_configs; print(diff_configs(load_config('configs/exp_hinge.yaml'), load_config('configs/base.yaml')))\""""),
    code("""entrenar('configs/exp_hinge.yaml', 'exp_hinge')"""),
    code("""for f in ['fixed_grid.png', 'progress.png', 'losses.png']:
    display(Image(f'{OUT}/runs/exp_hinge/{f}'))"""),
    md("## 4. Comparación base vs hinge"),
    code("""!python -m src.compare --out-root "$OUT" --runs base exp_hinge --out "$OUT/figs/cmp_base_vs_hinge"
display(Image(f'{OUT}/figs/cmp_base_vs_hinge_grids.png'))
display(Image(f'{OUT}/figs/cmp_base_vs_hinge_curves.png'))"""),
    md("""**Análisis base vs hinge** (hipótesis registrada el 2026-10-11 a las 04:59 UTC, antes de entrenar):

| Run | Diversidad ↑ | Novedad | D(x) / D(G(z)) última época |
|---|---|---|---|
| base | 0.102 | 0.171 | 0.97 / 0.035 |
| exp_hinge | **0.104** | 0.141 | 0.92 / 0.074 |
| *dataset real* | *0.189* | | |

- *Rejilla:* ningún run repite el mismo personaje en las 16 celdas; hinge da siluetas más nítidas y contornos más
  limpios. Los dos tienden a pocos modos (hinge, sobre todo hacia personajes pelirrojos).
- *Curvas:* ninguna diverge. Con hinge D domina menos: D(x) 0.92 contra 0.97 y D(G(z)) 0.074 contra 0.035 (las
  pérdidas BCE y hinge están en escalas distintas, por eso comparamos D(x) y D(G(z))).
- *Métricas:* la diversidad sube apenas un 2 %; la novedad baja un poco (las muestras quedan más cerca del train).
- **Veredicto:** hipótesis cumplida a medias. Hinge es útil (más nitidez, D menos dominante, diversidad igual o algo
  mayor), pero la mejora en diversidad es modesta."""),
    md("""## 4b. Experimento de estabilización: normalización espectral en D (Persona B)

**Hipótesis (escrita antes de entrenar):** Creemos que la normalización espectral en D mejorará la estabilidad
porque limita la constante de Lipschitz de D, evitando que se vuelva demasiado fuerte y deje a G sin gradiente útil.
Lo consideraremos útil si la pérdida de D no colapsa a ~0, las curvas oscilan menos y la diversidad no baja.

**Cambio único respecto a base:** `spectral_norm_d: true` (ver `configs/exp_sn.yaml`): cada Conv de D se envuelve
con `torch.nn.utils.spectral_norm`. Todo lo demás (pérdida BCE no saturante, BatchNorm, datos, semilla, épocas,
ruido fijo) queda igual a base."""),
    code("""import datetime
print('Hipótesis registrada:', datetime.datetime.now().isoformat(timespec='minutes'))
!python -c "from src.utils import load_config, diff_configs; print(diff_configs(load_config('configs/exp_sn.yaml'), load_config('configs/base.yaml')))\""""),
    code("""entrenar('configs/exp_sn.yaml', 'exp_sn')"""),
    code("""for f in ['fixed_grid.png', 'progress.png', 'losses.png']:
    display(Image(f'{OUT}/runs/exp_sn/{f}'))
!python -m src.compare --out-root "$OUT" --runs base exp_sn --out "$OUT/figs/cmp_base_vs_sn"
!python -m src.compare --out-root "$OUT" --runs base exp_hinge exp_sn --out "$OUT/figs/cmp_todos"
display(Image(f'{OUT}/figs/cmp_base_vs_sn_grids.png'))
display(Image(f'{OUT}/figs/cmp_base_vs_sn_curves.png'))"""),
    md("""**Análisis SN** (hipótesis registrada el 2026-10-11 a las 04:59 UTC, antes de entrenar):

| Run | Diversidad ↑ | Novedad | Pérdida de D | D(x) / D(G(z)) última época |
|---|---|---|---|---|
| base | 0.102 | 0.171 | 0.096 | 0.97 / 0.035 |
| exp_sn | 0.090 (**−12 %**) | 0.136 | 0.045 | 0.98 / 0.022 |

- La pérdida de D quedó **más cerca de 0** que en base y D(G(z)) más baja: con SN, D es todavía más dominante.
- La diversidad baja un 12 %.
- **Veredicto:** hipótesis refutada. Explicación probable: mantuvimos BatchNorm en D (para cambiar una sola cosa) y
  BatchNorm contrarresta la restricción de Lipschitz que impone SN. Es un resultado negativo honesto."""),
    md("""## 5. Modelo final (después de S2)

**Decisión S2 (con `comparacion.csv`):** el modelo final es **hinge solo, 500 épocas** (`configs/final.yaml`).
Hinge fue el único cambio que mejoró algo (nitidez, D menos dominante, diversidad +2 %); SN empeoró la diversidad,
así que no lo combinamos. Duplicamos las épocas porque las curvas de hinge seguían estables a las 250."""),
    code("""entrenar('configs/final.yaml', 'final')"""),
    code("""for f in ['fixed_grid.png', 'progress.png', 'losses.png']:
    display(Image(f'{OUT}/runs/final/{f}'))"""),
    md("""**Lectura del modelo final:**

- *Rejilla de ruido fijo* (`progress.png`): época 0 = ruido; desde la época 100 hay 16 personajes reconocibles y
  distintos; de la 100 a la 500 se definen siluetas y paletas. No colapsa a un personaje ni diverge.
- *Curvas* (`losses.png`): estables después de ~1 000 iteraciones, pero D gana terreno poco a poco: D(x) 0.80 → 0.99,
  D(G(z)) 0.20 → 0.02, pérdida de G 3.5 → 6. Es una deriva lenta, no el colapso brusco del run con fondo blanco.
- *Métricas* (notebook 03): diversidad 0.104 y novedad 0.131 (mínimo 0.083), casi igual que hinge a 250 épocas:
  las épocas extra no subieron la diversidad.
- **Limitación:** colapso de modo parcial. Dentro de cada cluster de la galería las muestras son casi idénticas, y la
  diversidad (0.10) queda lejos de la del dataset real (0.19)."""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nbf.write(nb, "notebooks/02_entrenamiento_y_experimentos.ipynb")
print("Escrito notebooks/02_entrenamiento_y_experimentos.ipynb")
