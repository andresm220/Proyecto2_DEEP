# Guía para armar `presentacion.pdf` (máx. 12 diapositivas)

Todo lo que hace falta para cada diapositiva: qué figura usar, qué números citar y qué conclusión sacar.
Rutas: `Drive:` = `MyDrive/proyecto2-gan/` · `repo:` = este repositorio.
Pixel art siempre ampliado con NEAREST (las figuras ya vienen así).

---

## 1. Universo — *Los Errantes de Kaminal*
- **Descripción:** reino de fantasía medieval en pixel art RPG de 16 bits, visto desde arriba. Sus habitantes son
  aventureros errantes: guerreros con armadura, magos con túnica, exploradores con capucha y gente común. Siluetas
  legibles en 64×64 con contornos oscuros, sobre la noche del reino (fondo negro).
- **Variedad esperada:** tono de piel, tipo de cuerpo, cabello, ropa/armadura, accesorios (cascos, capas, armas).
- **Vara de juicio:** un personaje cuenta si (a) se lee como figura humanoide completa, (b) tiene un atuendo
  coherente con una clase del reino, (c) se distingue claramente de los otros 9.

## 2. Datos
- **Figura:** `repo: outputs/figs/muestras_dataset.png` (64 reales) y `outputs/figs/muestras_por_vista.png`.
- **Origen:** TinyHero (A. Mikołajczyk), generado con el Universal LPC Spritesheet (makrohn), arte del Liberated
  Pixel Cup. Descarga: github.com/AgaMiko/pixel_character_generator.
- **Licencia:** CC-BY-SA 3.0 y GPL 3.0 → atribución + compartir igual.
- **Tamaño:** 3 648 imágenes 64×64 = **912 personajes × 4 vistas** (el mismo `k.png` es el mismo personaje).
- **Preparación:** RGBA → **fondo negro** → RGB; [−1, 1] (por el `tanh`); flip horizontal p=0.5; batch 128 con
  `drop_last` (28 iteraciones/época).

## 3. Arquitectura y control
- **Tabla:** `repo: outputs/figs/arquitectura.md` (G 3.58 M parámetros, D 2.77 M).
- DCGAN (Radford 2015) del tutorial de PyTorch: G con ConvTranspose + BatchNorm + ReLU → `tanh`; D espejo con
  LeakyReLU(0.2), sin sigmoid (`BCEWithLogitsLoss`). Init N(0, 0.02).
- **Por qué DCGAN:** la más simple que funciona en 64×64, explicable línea por línea, buen control.
- **Control (`configs/base.yaml`):** semilla 42, Adam 2e-4 (β 0.5/0.999), BCE no saturante, 250 épocas
  (5.8 s/época en T4 ≈ 24 min), ruido fijo 1234, fondo negro.

## 4. Experimento de pérdida: hinge
- **Hipótesis (registrada 22:59 del 10/10, antes de entrenar):** "Creemos que la pérdida hinge mejorará la
  estabilidad y la diversidad porque no satura cuando D está muy seguro… Lo consideraremos útil si la rejilla de
  ruido fijo muestra menos filas repetidas y la diversidad sube respecto a base sin bajar la calidad visual."
- **Cambio único:** `loss: hinge`.
- **Figuras:** `Drive: outputs/figs/cmp_todos_grids.png`, `cmp_todos_curves.png`.
- **Resultado (250 épocas):** diversidad 0.104 vs 0.102 de base (+2 %); D menos dominante: D(x) 0.92 vs 0.97 y
  D(G(z)) 0.074 vs 0.035; siluetas más nítidas. Limitación: colapso parcial hacia personajes pelirrojos.
- **Veredicto:** útil, pero modesto (la diversidad casi no sube).

## 5. Experimento de estabilización: normalización espectral en D
- **Hipótesis (registrada 22:59 del 10/10):** "Creemos que la normalización espectral en D mejorará la estabilidad
  porque limita la constante de Lipschitz de D… Lo consideraremos útil si la pérdida de D no colapsa a ~0, las
  curvas oscilan menos y la diversidad no baja."
- **Cambio único:** `spectral_norm_d: true` (todo lo demás igual, BatchNorm incluido).
- **Resultado:** diversidad 0.090 (**−12 %**), D **más** dominante (D(x) 0.98, D(G(z)) 0.02).
- **Veredicto:** hipótesis refutada. Explicación probable: BatchNorm en D (que mantuvimos para cambiar una sola
  cosa) contrarresta el efecto de SN. Resultado negativo honesto.

## 6. Evidencia 1: rejilla de ruido fijo a lo largo del entrenamiento (modelo final)
- **Figura:** `repo: outputs/runs/final/progress.png` (épocas 0, 100, …, 500) y `outputs/runs/final/fixed_grid.png`.
- **Qué mostrar:** época 0 = ruido gris; desde la época 100 ya hay 16 personajes reconocibles y distintos para los
  16 z fijos; de la 100 a la 500 se definen siluetas y paletas. Comparar con
  `outputs/runs_fondo_blanco/base/progress.png`, donde los 16 z daban el mismo personaje.
- **Conclusión:** el modelo final no colapsa a un solo personaje ni diverge en 500 épocas.

## 7. Evidencia 2: curvas de pérdida
- **Figura:** `Drive: outputs/runs/final/losses.png` (y `cmp_todos_curves.png`).
- Cómo leerlas: sanas = pérdidas de D y G enfrentadas y estables; **alarma** = pérdida de D → ~0 con D(x) ≈ 1 y
  D(G(z)) ≈ 0 sostenido (D gana, G sin gradiente). Ojo: BCE y hinge están en escalas distintas.
- **Lectura del modelo final:** después de ~1 000 iteraciones las curvas se estabilizan (no hay divergencia), pero D
  gana terreno poco a poco: D(x) 0.80 → 0.96, D(G(z)) 0.20 → 0.05, pérdida de G 3.5 → 6, pérdida de D (hinge) → ~0.
  Es una deriva lenta, no el colapso brusco del run con fondo blanco (D(G(z)) 0.002 en la época 76).
- **Conclusión:** entrenamiento estable pero con D cada vez más dominante; coincide con el colapso de modo parcial
  que se ve en la galería.

## 8. Problemas y fallos
- **Colapso con fondo blanco (el hallazgo principal):** la primera base colapsó: los 16 z daban el mismo personaje
  y después de la época 60 divergió a ruido (pérdida de D 0.004, D(G(z)) 0.002). Figuras en
  `Drive: outputs/runs_fondo_blanco/base/progress.png`.
- No era el modo eval de BatchNorm (en modo train también era idéntico); medido capa por capa, G aprendía a ignorar
  z (variación entre muestras 0.99 → 0.02 con γ de BN ≈ 1).
- Pruebas de 30 épocas (`repo: configs/probes/`): SN, hinge, instance noise, dropout en D, etiquetas ruidosas →
  todas colapsaban con fondo blanco. **Solo fondo negro** lo resolvió. Se re-entrenaron base, hinge y SN.
- Otros: Colab sin `--resume` en GPU (bug arreglado), reanudación segura, galería generada en CPU para
  regeneración exacta.
- Detalle en `repo: README.md`, sección "Problemas encontrados".

## 9. Galería
- **Figura:** `repo: galeria/galeria.png` (10 personajes con nombre y clase).
- **Tasa de selección:** 10 de 200 (seeds 0–199), k-means (k=10) sobre features ResNet-18 + revisión visual
  (la más legible de cada cluster según la vara). Está en `galeria/seeds.json`.
- **Personajes:**

  | # | Nombre | Clase | seed |
  |---|---|---|---|
  | 00 | Tohil, Guardabosques del Cerro | explorador | 64 |
  | 01 | Ixkik, la Espada Carmesí | guerrera | 114 |
  | 02 | Balam el Buhonero | mercader errante | 132 |
  | 03 | Kukul, Arquero del Paso | explorador | 190 |
  | 04 | Itzam, Sabio de Ceniza | mago | 97 |
  | 05 | Gorrak el Desterrado | guerrero orco | 188 |
  | 06 | Chak, Aprendiz de Herrero | gente del reino | 83 |
  | 07 | Ixchel, Hechicera Índigo | maga | 53 |
  | 08 | Hunab, Mercenario Rojo | guerrero | 112 |
  | 09 | Xquic, Vidente de la Niebla | maga | 138 |
- **Honestidad sobre la selección:** dentro de cada cluster las muestras son casi idénticas (ver
  `galeria/candidatos/cluster_XX.png`), así que la revisión visual confirmó en todos los clusters la muestra más
  cercana al centroide. Pares que se parecen: 02/06, 01/08, 07/09.
- **Reproducibilidad:** `python -m src.gallery --regenerate --weights galeria/G_final.pt --seeds galeria/seeds.json`
  → "OK", verificado en dos máquinas (Colab y una laptop Windows).

## 10. Evidencia 3: vecinos más cercanos
- **Figuras:** `repo: galeria/vecinos.png` (coseno ResNet-18) y `galeria/vecinos_pixeles.png` (RMSE).
- **Resultado:** similitud coseno 0.84–0.90 con el vecino más cercano; ningún personaje es copia de uno real
  (el máximo, 0.90, es personaje_00 con el personaje 251 del dataset en dos vistas).
- **Interpretación:** los vecinos comparten el color de pelo o la paleta con el personaje generado, pero no el cuerpo
  ni la ropa. La GAN parece combinar partes (pelo de uno, cuerpo/atuendo de otro), igual que el generador LPC con el
  que se hizo el dataset. Ojo: el dataset tiene cada personaje en 4 vistas, por eso a veces aparecen dos vistas del
  mismo personaje entre los top-3. Varios vecinos son orcos verdes: ResNet-18 se fija mucho en el color dominante.

## 11. Conclusiones, limitaciones y reflexión final
- **Conclusiones:** (1) con este dataset, el fondo fue más decisivo que la pérdida o la estabilización: con fondo
  blanco todo colapsaba; (2) hinge ayudó poco en diversidad y algo en nitidez; SN (con BatchNorm en D) empeoró;
  (3) la galería es reproducible píxel a píxel en otra máquina.
- **Limitaciones:** un solo run por configuración (no medimos variación entre semillas); diversidad 0.10 frente a
  0.19 del dataset real y colapso de modo parcial; las métricas usan ResNet-18 de ImageNet, que no es un juez ideal
  de pixel art; imágenes con ruido de alta frecuencia.
- **Reflexión final (propuesta):** el personaje que más cuesta defender como nuevo es **personaje_00 (Tohil)**: su
  vecino más cercano (personaje 251, sim 0.90) aparece en dos vistas, y la silueta es la del sprite LPC estándar. Lo
  que aprendió la GAN no es a inventar personajes desde cero sino a **recombinar las piezas** del generador LPC
  (cuerpos, peinados, paletas) en combinaciones que no están en el dataset, y además con pocos modos.

## 12. Matriz de evidencias
| Evidencia | Diapositiva | Conclusión | Limitación |
|---|---|---|---|
| Rejillas de ruido fijo base/hinge/SN | 4–5 | hinge más nítido; SN no ayuda | una sola rejilla de 16 z |
| `comparacion.csv` (diversidad, novedad) | 4–5 | hinge +2 %, SN −12 % | ResNet-18 de ImageNet no es un juez perfecto de pixel art; un run por config |
| Evolución de la rejilla (final) | 6 | sin colapso a un personaje ni divergencia en 500 épocas | 16 z no muestran el colapso parcial |
| Curvas de pérdida | 7 | estable, pero D gana terreno poco a poco | escalas distintas BCE/hinge |
| Pruebas de fondo blanco | 8 | el fondo blanco provoca el colapso | hallazgo empírico, sin explicación causal |
| Vecinos más cercanos | 10 | ninguna copia (sim ≤ 0.90); recombina partes | el train tiene 4 vistas por personaje; ResNet-18 prioriza el color |
| Regeneración exacta | 9 | la galería se reproduce píxel a píxel | en CPU con batch 1 |
