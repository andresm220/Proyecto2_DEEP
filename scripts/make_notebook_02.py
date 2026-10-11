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
Si Colab se cae a mitad de un run, se vuelve a correr la misma celda agregando `--resume`: el entrenamiento sigue
desde `last.pt` y da exactamente las mismas pérdidas que si no se hubiera interrumpido (lo verifica `tests/test_train.py`)."""),
    code("""from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/proyecto2-gan/outputs'   # carpeta compartida con Persona B
!mkdir -p "$OUT\""""),
    code("""REPO_URL = 'https://github.com/andresm220/Proyecto2_DEEP.git'
!rm -rf /content/proyecto2-gan && git clone -q $REPO_URL /content/proyecto2-gan
%cd /content/proyecto2-gan
!git log --oneline -1
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    md("""## 1. Medir el tiempo por época

Con 28 iteraciones por época, una época en T4 debería tardar pocos segundos. Medimos 2 épocas en un run aparte
(`timing`) para fijar `epochs` en `configs/base.yaml`: el objetivo es que **cada run tarde ≲ 30–40 min**.
Como hinge y SN heredan de base, cambiar `epochs` en base lo cambia para todos los experimentos a la vez."""),
    code("""!python -m src.train --config configs/base.yaml --name timing --epochs 2 --out-root "$OUT"
import json
t = json.load(open(f'{OUT}/runs/timing/train_log.json'))
print(f"{t['sec_per_epoch_mean']:.1f} s/época -> en 30 min caben ~{int(1800 / t['sec_per_epoch_mean'])} épocas")"""),
    md("""**Decisión de épocas:** escribir aquí el valor elegido y por qué (por ejemplo "300 épocas ≈ 25 min").
Después: editar `epochs` en `configs/base.yaml`, hacer commit y push, y volver a correr la celda del `git clone`.

> Cada checkpoint pesa ~75 MB (G + D + los dos Adam). Si `epochs` sube mucho, subir también `ckpt_every`
> (por ejemplo `epochs / 10`) para no llenar Drive. `ckpt_every` y `snapshot_every` **no** afectan el
> entrenamiento (solo guardan cosas), así que cambiarlos no rompe la regla de "una sola diferencia"."""),
    md("""## 2. Entrenar base (el control)"""),
    code("""!python -m src.train --config configs/base.yaml --out-root "$OUT"
# Si se cayó: !python -m src.train --config configs/base.yaml --out-root "$OUT" --resume"""),
    code("""from IPython.display import Image, display
for f in ['fixed_grid.png', 'progress.png', 'losses.png']:
    display(Image(f'{OUT}/runs/base/{f}'))"""),
    md("""**Sincronización S1:** en cuanto exista `runs/base/checkpoints/epoch_005.pt`, avisar a Persona B. B necesita
la carpeta `proyecto2-gan/outputs` compartida (incluye `fixed_noise.pt`, que deben usar TODOS los runs)."""),
    md(f"""## 3. Experimento de pérdida: hinge

**Hipótesis (escrita antes de entrenar):** {HINGE_HYPOTHESIS}

**Cambio único respecto a base:** `loss: hinge` (ver `configs/exp_hinge.yaml`). La celda siguiente deja impresa
la fecha y hora en que se registró la hipótesis, antes de lanzar el entrenamiento."""),
    code("""import datetime
print('Hipótesis registrada:', datetime.datetime.now().isoformat(timespec='minutes'))
!python -c "from src.utils import load_config, diff_configs; print(diff_configs(load_config('configs/exp_hinge.yaml'), load_config('configs/base.yaml')))\""""),
    code("""!python -m src.train --config configs/exp_hinge.yaml --out-root "$OUT"
# Si se cayó: agregar --resume"""),
    code("""for f in ['fixed_grid.png', 'progress.png', 'losses.png']:
    display(Image(f'{OUT}/runs/exp_hinge/{f}'))"""),
    md("## 4. Comparación base vs hinge"),
    code("""!python -m src.compare --out-root "$OUT" --runs base exp_hinge --out "$OUT/figs/cmp_base_vs_hinge"
display(Image(f'{OUT}/figs/cmp_base_vs_hinge_grids.png'))
display(Image(f'{OUT}/figs/cmp_base_vs_hinge_curves.png'))"""),
    md("""**Análisis (completar con lo observado):**

- *Rejilla:* ¿hay filas o celdas que se repiten (señal de mode collapse)? ¿Cuál run da siluetas más legibles?
- *Curvas sanas:* pérdidas de D y G "enfrentadas" y estables. **Alarma:** pérdida de D → ~0 con D(x) ≈ 1 y
  D(G(z)) ≈ 0 de forma sostenida (D gana y G se queda sin gradiente útil).
- *Métricas (de Persona B, `metrics.json`):* diversidad (distancia coseno media entre pares, más alta = más
  diversa) y novedad (distancia al vecino más cercano del train).
- *Veredicto vs. el criterio de la hipótesis:* ¿se cumplió "menos filas repetidas y más diversidad sin bajar la calidad"?"""),
    md("""## 5. Modelo final (después de S2)

Decidir con B la configuración final a partir de `comparacion.csv`, escribir aquí la decisión y su porqué,
crear `configs/final.yaml` (hacer commit) y entrenar:"""),
    code("""!python -m src.train --config configs/final.yaml --out-root "$OUT"
for f in ['fixed_grid.png', 'progress.png', 'losses.png']:
    display(Image(f'{OUT}/runs/final/{f}'))"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nbf.write(nb, "notebooks/02_entrenamiento_y_experimentos.ipynb")
print("Escrito notebooks/02_entrenamiento_y_experimentos.ipynb")
