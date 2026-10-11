"""Genera notebooks/03_galeria_y_vecinos.ipynb (se ejecuta en Google Colab)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# 03 — Métricas, galería y vecinos más cercanos (Colab)

1. **Métricas** de cada run (diversidad y novedad sobre 500 muestras con seeds 0–499) → `comparacion.csv`.
2. **Galería**: 200 candidatos del modelo final (seeds 0–199) → k-means (k=10) → revisión visual → 10 personajes.
3. **Regeneración exacta** de la galería desde los pesos + seeds (verificación píxel a píxel).
4. **Vecinos más cercanos** de los 10 personajes en el set de entrenamiento, para juzgar si son nuevos.

La galería se genera en **CPU**, con `G.eval()` y una muestra a la vez, así que se puede regenerar exacta en
cualquier máquina. No hace falta GPU para este notebook (la GPU solo acelera las métricas)."""),
    code("""from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/proyecto2-gan/outputs'
GAL = '/content/drive/MyDrive/proyecto2-gan/galeria'
!rm -rf /content/proyecto2-gan && git clone -q https://github.com/andresm220/Proyecto2_DEEP.git /content/proyecto2-gan
%cd /content/proyecto2-gan
!git log --oneline -1
from IPython.display import Image, display
import pandas as pd, json"""),
    md("""## 1. Métricas por run

- **Diversidad**: distancia coseno media entre pares de las 500 muestras (features ResNet-18). Más alta = más
  diversa. `train_diversity` es el mismo número para 500 imágenes reales: es la referencia de "cuánta variedad hay".
- **Novedad**: distancia coseno media de cada muestra a su vecino más cercano del train. Muy baja = el generador
  podría estar copiando. Ojo: el train tiene cada personaje en 4 vistas, así que un vecino muy cercano puede ser
  otra vista del mismo personaje."""),
    code("""for run in ['base', 'exp_hinge', 'exp_sn']:
    !python -m src.evaluate --run "$OUT/runs/{run}" --out-root "$OUT" > /dev/null && echo "{run}: ok"
!python -m src.evaluate --out-root "$OUT" --table base exp_hinge exp_sn
pd.read_csv(f'{OUT}/comparacion.csv')"""),
    code("""for run in ['base', 'exp_hinge', 'exp_sn']:
    print(run); display(Image(f'{OUT}/runs/{run}/vecinos_muestra.png', width=700))"""),
    md("""## 2. Galería: 200 candidatos del modelo final

`FINAL` es `galeria/G_final.pt`: solo los pesos de G del modelo final. Si se cambia, hay que volver a generar los
candidatos (el código lo verifica con el hash SHA-256 de los pesos)."""),
    code("""# Exportar solo G (~14 MB) desde el checkpoint completo (~75 MB): este archivo va en el zip de entrega y es el
# que se usa para generar, seleccionar y regenerar la galería (seeds.json guarda su SHA-256).
!python -m src.gallery --export-g --weights "{OUT}/runs/final/checkpoints/last.pt" --out "$GAL"
FINAL = f'{GAL}/G_final.pt'
!python -m src.gallery --candidates --weights "$FINAL" --n 200 --out "$GAL\""""),
    code("""import glob
for f in sorted(glob.glob(f'{GAL}/candidatos/cluster_*.png')):
    display(Image(f, width=900))"""),
    md("""### Revisión visual (el criterio que se declara en la diapositiva 9)

Cada hoja muestra un cluster ordenado por cercanía al centroide. `picks.json` trae por defecto la primera
muestra de cada cluster. Para cada cluster, elegir la muestra **más legible según la vara del universo**:
(a) figura humanoide completa, (b) atuendo coherente con una clase del reino, (c) distinta de las otras 9.
Si hace falta, cambiar la seed en la celda siguiente (debe pertenecer a ese cluster) y anotar el porqué."""),
    code("""picks = json.load(open(f'{GAL}/picks.json'))
# Ejemplo: picks['3'] = 57   # cluster 3: la 57 tiene la silueta completa; la del centroide sale cortada
json.dump(picks, open(f'{GAL}/picks.json', 'w'), indent=2)
picks"""),
    code("""!python -m src.gallery --select --weights "$FINAL" --out "$GAL"
display(Image(f'{GAL}/galeria.png'))
json.load(open(f'{GAL}/seeds.json'))['tasa_seleccion']"""),
    md("## 3. Regeneración exacta (regla del proyecto: el código debe regenerar la galería exacta)"),
    code("""!python -m src.gallery --regenerate --weights "$FINAL" --seeds "$GAL/seeds.json\""""),
    md("""## 4. Vecinos más cercanos de los 10 personajes

Dos criterios: similitud coseno sobre features ResNet-18 (parecido "perceptual") y distancia en píxeles (RMSE,
detecta copias casi exactas). Cada vecino indica qué personaje del dataset es y en qué vista."""),
    code("""!python -m src.evaluate --gallery "$GAL/seeds.json" --weights "$FINAL" --out-root "$OUT" > /dev/null
display(Image(f'{GAL}/vecinos.png'))
display(Image(f'{GAL}/vecinos_pixeles.png'))"""),
    md("""**Interpretación (completar con honestidad):**

- ¿Algún personaje es casi idéntico a su vecino (similitud ≈ 1 o RMSE ≈ 0)? Si sí, **reemplazarlo** por otra
  muestra de su cluster (`picks.json` → `--select` → `--regenerate`) y **declararlo** en el PDF.
- ¿El vecino más cercano es el mismo personaje en otra vista? Eso indica que la GAN reprodujo un personaje
  existente, no que inventó uno.
- ¿Qué personaje costó más defender como nuevo, y qué dice eso de lo que aprendió la GAN (reflexión final)?"""),
    md("""## 5. Nombres y clases (bonificación)

Escribir nombre y clase en `seeds.json` y volver a correr `--select`: conserva los nombres (busca por seed) y
actualiza `galeria.png`. Después, `--regenerate` sigue pasando porque las imágenes no cambian."""),
    code("""meta = json.load(open(f'{GAL}/seeds.json'))
# meta['characters'][0].update(nombre='Ixbalam, Guardiana del Paso', clase='guerrera')
json.dump(meta, open(f'{GAL}/seeds.json', 'w'), indent=2, ensure_ascii=False)
!python -m src.gallery --select --weights "$FINAL" --out "$GAL" && python -m src.gallery --regenerate --weights "$FINAL" --seeds "$GAL/seeds.json"
display(Image(f'{GAL}/galeria.png'))"""),
    md("""Al final, copiar la carpeta `galeria/` de Drive a la carpeta `galeria/` del repo (10 PNG + versiones 256,
`seeds.json`, `z_vectors.pt` y `G_final.pt`) para la entrega. Con eso, `python -m src.gallery --regenerate
--weights galeria/G_final.pt --seeds galeria/seeds.json` funciona sin Drive."""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": []}
nbf.write(nb, "notebooks/03_galeria_y_vecinos.ipynb")
print("Escrito notebooks/03_galeria_y_vecinos.ipynb")
