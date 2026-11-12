# Ejercicio 1a — Autoencoder básico sobre font.h

## Objetivo

Aprender los 32 patrones de 7×5 de `data/font.h` con un autoencoder cuyo espacio latente
tiene 2 dimensiones, con un error máximo de 1 píxel incorrecto por letra (consigna, ej. 1a).
Además: describir y estudiar las arquitecturas y técnicas de optimización probadas (barrido
secuencial), graficar el espacio latente y mostrar cómo la red genera letras nuevas.

## Decisiones del equipo (no se cambian)

- Pérdida para entrenar: entropía cruzada binaria (BCE) con salida sigmoidea.
- Métrica para comparar TODO (incluido PCA): píxeles mal. Se redondea la salida (≥ 0,5 → 1) y
  se cuentan los píxeles distintos a la letra original. "Llega" = la peor de las 32 letras
  tiene ≤ 1 píxel mal (lo pide el enunciado).
- Línea base: PCA a 2 componentes, implementado a mano con NumPy (sin scikit-learn). Si un
  autoencoder da peor que PCA, hay algo mal.
- Espacio latente de 2 dimensiones.
- 5 semillas (0 a 4) por configuración; todo resultado se reporta como media ± desvío.
- Máximo 10.000 épocas por corrida; se corta apenas llega.

Referencias que ya medimos con este motor (para verificar):

- PCA 2D sobre font.h: MSE 0,1491; 7,19 píxeles mal por letra en promedio; peor letra 15;
  0 de 32 letras con ≤ 1.
- AE lineal 35→2→35 (todo identidad, MSE, Adam lr 0,01, 5.000 épocas): da exactamente lo mismo
  que PCA.
- Config base: 5 de 5 semillas llegan, en 360–540 épocas.
- GD y momentum con lr 0,003 no llegan nunca; GD necesita lr ≈ 1 y momentum ≈ 0,3.

## Barrido secuencial (diseño de Leo)

Se arranca de la config base y se recorren los pasos en orden. En cada paso se prueban los
valores indicados cambiando SOLO ese parámetro; se elige el mejor y queda fijo para los pasos
siguientes.

Config base: ocultas 30-20-10 (encoder 35→30→20→10→2, decoder espejado 2→10→20→30→35), tanh en
ocultas, latente lineal, salida sigmoidea, Xavier, Adam lr 0,003, lote completo (32), sin
decaimiento, BCE, sin L2.

0. Líneas de base (no eligen nada): PCA 2D · AE lineal (como arriba) · sin ocultas (35→2→35,
   salida sigmoidea, BCE). Si el AE lineal no da el MSE de PCA con ±2 %, hay un bug.
1. Profundidad: 0, 1, 2, 3 y 4 capas ocultas por lado. Anchos de 30 a 10, repartidos parejo y
   redondeados → 0: [] · 1: [30] · 2: [30, 10] · 3: [30, 20, 10] · 4: [30, 23, 17, 10].
2. Ancho: ×0,5 · ×1 · ×2 de cada capa oculta del mejor (redondeado).
3. Activación de ocultas: tanh · sigmoide · ReLU · leaky ReLU (α = 0,01).
4. Activación del latente: lineal · tanh.
5. Inicialización: U(±0,1) · U(±1) · N(0, 1) · Xavier · He.
6 + 7. Optimizador y lr juntos (para no castigar a GD con el lr de Adam): GD · momentum 0,9 ·
   Adam, cada uno con lr ∈ {0,0003, 0,001, 0,003, 0,01, 0,03, 0,1, 0,3, 1}. Se elige el mejor
   par (optimizador, lr).
8. Decaimiento del lr: ninguno · exponencial (lr × 0,999 por época) · escalonado (lr × 0,5 cada
   250 épocas).
9. Tamaño de lote: 32 · 8 · 4 · 1.
10. Pérdida: BCE · MSE · MSE con lr × 5.
11. Regularización L2: λ = 0 · 1e-5 · 1e-4 · 1e-3.

Final: con la config final, volver a correr el paso 3. Si gana otra activación, NO se vuelve a
barrer: se anota como hallazgo (muestra que los pasos no son independientes).

Cómo se elige el mejor de cada paso:

1. Más % de semillas que llegan.
2. Si empatan: menos épocas promedio hasta llegar (contando solo las semillas que llegaron).
3. Si ninguna llega: el menor promedio de píxeles mal de la peor letra al final.
4. Si siguen empatados: se queda el valor que ya estaba.

Por cada corrida se guarda: paso, valor probado, config completa, semilla, si llegó, época en
que llegó, peor letra y promedio de píxeles mal al final, píxeles mal de cada una de las 32
letras, segundos, cantidad de actualizaciones de pesos (épocas × lotes por época) y estado
(ok/diverged). Se elige por épocas, pero los segundos y las actualizaciones quedan guardados
para discutir el paso 9.

## Piezas

En `core/` (genérico, con tests):

- `activations.py`: `LeakyReLU(alpha=0.01)`, registrada como `"leaky_relu"`.
- `initializers.py`: `Normal(std=1.0)`, W ~ N(0, std²), registrado como `"normal"`.
- `callbacks.py`: `StopAtTarget` (corta cuando una métrica llega a un objetivo y guarda la
  época), `ExponentialDecay` y `StepDecay` (modifican `optimizer.lr`, como `AdaptiveEta`).
- `metrics.py`: `pixel_errors` (por muestra), `max_pixel_errors`, `mean_pixel_errors`.
- `pca.py`: `PCA` con `fit` / `transform` / `inverse_transform`.

En `ej1/`:

- `datos.py`: `cargar_font()` → X (32, 35) en {0, 1} y las 32 etiquetas.
- `config.json`: base, líneas de base, grillas de cada paso, semillas, épocas máximas.
- `barrido.py`: corre el barrido y solo escribe en `resultados/ej1/` (un CSV por paso,
  `elecciones.json`, y en `final/` los pesos `.npz` y la historia por época de las 5 semillas
  de la config final). Si el CSV de un paso ya existe, lo saltea.
- `graficos.py`: solo lee de `resultados/ej1/` y guarda en `resultados/ej1/figuras/`.

Comandos:

```
python -m ej1.barrido      # barrido completo -> resultados/ej1/
python -m ej1.graficos     # figuras -> resultados/ej1/figuras/
```

## Fases

| # | Fase | Estado |
|---|------|--------|
| 1 | Spec, referencia en CLAUDE.md §6, docs/structure.md | hecha |
| 2 | core: `LeakyReLU` | hecha |
| 3 | core: inicializador `Normal` | hecha |
| 4 | core: `StopAtTarget`, `ExponentialDecay`, `StepDecay` | hecha |
| 5 | core: métricas de píxeles mal | hecha |
| 6 | core: `PCA` | hecha |
| 7 | ej1: cargador de font.h | hecha |
| 8 | ej1: config + script del barrido + smoke test | hecha |
| 9 | ej1: script de gráficos (+ matplotlib en requirements) | hecha |
| 10 | Corrida completa, figuras y resultados | hecha |

## Supuestos

- **Sin partición ni normalización.** El enunciado pide aprender los 32 patrones, así que todas
  las decisiones se toman con el error sobre esos 32. Los datos ya son 0/1. Las reglas 2
  (validación separada) y 7 (normalización ajustada en train) de CLAUDE.md no aplican a este
  ejercicio.
- **Épocas hasta llegar:** la primera época al final de la cual la peor letra tiene ≤ 1 píxel
  mal, medida sobre los 32 patrones con los pesos ya actualizados (como mide `fit`).
- **Redondeo de anchos (paso 2):** al entero más cercano con las mitades hacia arriba
  (11,5 → 12), mínimo 1. El decoder es siempre el espejo del encoder.
- **Desvío:** desvío estándar muestral (ddof = 1) sobre las semillas. PCA es determinístico:
  se guarda una sola fila, sin semilla, y su desvío se reporta como 0.
- **Líneas de base AE:** también cortan si llegan (no se espera que lleguen). El AE lineal usa
  5.000 épocas como máximo; "sin ocultas" usa el máximo general (10.000).
- **Decaimiento exponencial:** al final de cada época, lr ← lr × 0,999 (la época e entrena con
  lr₀ · 0,999^(e−1)). **Escalonado:** al final de las épocas 250, 500, …, lr ← lr × 0,5.
- **Momentum 0,9:** es el α de la fórmula de la cátedra, Δθ ← −lr·g + α·Δθ.
- **Sigmoide:** la de `core`, que es la logística de la cátedra con β = 1,
  f(z) = 1 / (1 + e^(−2z)) = ½ (1 + tanh z). Vale para la salida y para el valor "sigmoide" del
  paso 3.
- **"MSE con lr × 5"** multiplica el lr vigente en ese paso (el elegido en 6 + 7).
- **Lotes:** 32 es el lote completo y no se baraja (así lo hace `fit`); con 8, 4 y 1 el orden
  se baraja cada época con el generador de la semilla.
- **Semilla:** `np.random.default_rng(semilla)` genera los pesos de encoder y decoder y el orden
  de los lotes.
- **Divergencia:** si la pérdida deja de ser finita, `fit` corta, la corrida queda con estado
  `diverged` y cuenta como que no llegó. Los píxeles mal se miden igual sobre la salida final
  (un NaN no supera 0,5, así que cuenta como 0).
- **Letra "mal" (figura 7):** la que termina la corrida con más de 1 píxel mal. Se cuentan todas
  las corridas de autoencoder de todos los pasos (líneas de base AE y verificación final
  incluidas); PCA no.
- **Modelo para las figuras 4 y 5:** la config final con la menor semilla que llegó.
- **Grilla de letras (figura 5a):** 12 × 12 puntos sobre el rango del código de las 32 letras,
  con 8 % de margen. Una letra generada es "nueva" si está a ≥ 2 píxeles de todas las del set.
- **Interpolaciones (figura 5b):** cada letra con su vecina más cercana en el latente; de esos
  pares, los 3 cuyas letras más difieren en píxeles y cuyo segmento pasa por al menos una letra
  nueva. 9 puntos por segmento, extremos incluidos.
- **Distancia entre letras:** cantidad de píxeles distintos (Hamming) entre letras binarizadas.

## Resultados

Barrido completo: 315 corridas de autoencoder más PCA, unos 43 minutos en secuencial (42 de
entrenamiento; solo el lote de 1 del paso 9 se lleva 19). Todo sale
de `resultados/ej1/` (CSV por paso, `elecciones.json`, `final/`).

### Verificación del motor

- PCA 2D: MSE 0,1491; 7,19 píxeles mal por letra; peor letra 15; 0 de 32 letras con ≤ 1.
  Coincide con la referencia.
- AE lineal: el MSE de PCA en las 5 semillas (desvío relativo máximo 2,3·10⁻⁶; tolerancia 2 %) y
  los mismos píxeles mal (7,19 de promedio y 15 en la peor letra).
- Config base (paso 1, valor 30-20-10): llegan 5 de 5, en 361–535 épocas (430 ± 72). Coincide
  con la referencia (360–540).

### Elecciones por paso

Épocas hasta llegar: media ± desvío sobre las semillas que llegaron.

| Paso | Antes | Gana | Criterio | Llegan | Épocas | Resto de los valores (llegan · épocas) |
|---|---|---|---|---|---|---|
| 1 profundidad | 30-20-10 | **30-20-10** | épocas | 5/5 | 430 ± 72 | sin ocultas 0/5 · 30: 5/5, 616 ± 38 · 30-10: 5/5, 600 ± 92 · 30-23-17-10: 5/5, 467 ± 62 |
| 2 ancho | ×1 | **×2** (60-40-20) | épocas | 5/5 | 258 ± 51 | ×0,5: 5/5, 1402 ± 276 · ×1: 5/5, 430 ± 72 |
| 3 activación ocultas | tanh | **tanh** | épocas | 5/5 | 258 ± 51 | sigmoide 5/5, 820 ± 290 · ReLU 5/5, 293 ± 63 · leaky ReLU 5/5, 365 ± 179 |
| 4 activación latente | lineal | **lineal** | épocas | 5/5 | 258 ± 51 | tanh 5/5, 328 ± 64 |
| 5 inicialización | Xavier | **He** | épocas | 5/5 | 257,8 ± 24 | Xavier 5/5, 258,0 ± 51 · U(±0,1) 5/5, 592 ± 373 · U(±1) 5/5, 488 ± 111 · N(0,1) 5/5, 814 ± 132 |
| 6 + 7 optimizador y lr | Adam 0,003 | **Adam 0,01** | épocas | 5/5 | 141 ± 19 | Adam 0,003: 258 ± 24 · Adam 0,03: 261 ± 107 · mejor momentum: 0,1, 5/5, 572 ± 102 · mejor GD: 0,3, 5/5, 1788 ± 174 |
| 8 decaimiento lr | ninguno | **ninguno** | empate | 5/5 | 141 ± 19 | escalonado: idéntico (141 ± 19) · exponencial 5/5, 145 ± 21 |
| 9 tamaño de lote | 32 | **32** | épocas | 5/5 | 141 ± 19 | 8: 5/5, 185 ± 43 · 4: 1/5 · 1: 0/5 |
| 10 pérdida | BCE | **BCE** | llegan | 5/5 | 141 ± 19 | MSE 2/5 · MSE con lr × 5: 0/5 |
| 11 L2 | 0 | **0** | épocas | 5/5 | 141 ± 19 | 1e-5: 142 ± 20 · 1e-4: 153 ± 13 · 1e-3: 238 ± 47 (todas 5/5) |
| Final: paso 3 repetido | tanh | **tanh** | épocas | 5/5 | 141 ± 19 | leaky ReLU 143 ± 35 · ReLU 177 ± 34 · sigmoide 507 ± 170 (todas 5/5) |

En la repetición final vuelve a ganar tanh: no hay hallazgo de dependencia entre pasos para la
activación de las ocultas (leaky ReLU queda a 2 épocas).

### Config final

Encoder 35 → 60 → 40 → 20 → 2 y decoder espejado 2 → 20 → 40 → 60 → 35; tanh en las ocultas,
latente lineal, salida sigmoidea, inicialización He, Adam con lr 0,01, lote completo (32), sin
decaimiento, BCE, sin L2, máximo 10.000 épocas (`resultados/ej1/final/config.json`).

| Modelo | Llegan | Épocas hasta llegar | Píxeles mal por letra | Peor letra al final | Actualizaciones |
|---|---|---|---|---|---|
| PCA 2D | 0/1 | — | 7,19 ± 0 | 15 ± 0 | — |
| AE lineal | 0/5 | — | 7,19 ± 0,00 | 15,0 ± 0,0 | 5000 |
| AE sin ocultas | 0/5 | — | 6,47 ± 0,29 | 15,2 ± 6,1 | 10.000 |
| **Config final** | **5/5** | **141 ± 19** (121–163) | **0,100 ± 0,026** | **1,0 ± 0,0** | 141 |

La config final aprende los 32 patrones con a lo sumo 1 píxel mal en las 5 semillas, en 0,2 a
0,5 segundos por corrida.

### Observaciones

- **He contra Xavier (paso 5):** He gana por 0,2 épocas de media (257,8 contra 258,0). La regla
  lo elige porque no es un empate exacto, pero en la práctica es un empate; He tiene menos
  dispersión (24 contra 51).
- **Escalonado (paso 8)** da exactamente las mismas corridas que "ninguno": todas las semillas
  llegan antes de la época 250, así que el lr nunca cambia. Gana "ninguno" por la regla 4.
- **Tamaño de lote (paso 9):** el lote de 8 también llega 5/5, pero en más épocas (185 ± 43),
  con 5,2 veces más actualizaciones (738 contra 141) y más tiempo (1,2 s contra 0,45 s). Con el
  lr elegido para lote completo, los lotes de 4 y 1 no llegan (lote 1: peor letra 24,6 ± 4,2 tras
  10.000 épocas, 320.000 actualizaciones y unos 227 s por corrida). Los segundos son ruidosos a
  escala de décimas (la misma corrida mide entre 0,15 y 0,7 s según la carga de la máquina): solo
  sirven cuando difieren en órdenes de magnitud, como acá.
- **Pérdida (paso 10):** con MSE llegan 2 de 5, y multiplicar el lr por 5 empeora (0 de 5, sin
  divergir: la peor letra queda en 26,4 ± 4,7).
- **AE sin ocultas contra PCA:** mejora a PCA en MSE y en píxeles mal promedio con las 5
  semillas, y en la peor letra con 4 de 5. La semilla 2 sacrifica la letra DEL (26 píxeles mal):
  BCE optimiza el promedio, no la peor letra. Su peor letra media (15,2) queda apenas por encima
  de la de PCA (15) solo por esa semilla.
- **Letras más difíciles** (figura 7, 315 corridas de autoencoder): v y w quedan mal en 90,
  e en 89 y k en 88; las más fáciles son | (61), l (68) y ` (70).
- **Letras nuevas:** en la grilla de 12 × 12 del latente, 65 de 144 letras decodificadas están
  a ≥ 2 píxeles de todas las del set. La que se presenta como nueva sale de interpolar de «d» a
  «z» en t = 0,25 (z = (0,23; −5,87)): está a 4 píxeles de la letra más parecida («z»). Entre
  «j» e «y» el segmento pasa por la zona de la «c» y genera un glifo a 4 píxeles de la «c».

### Figuras

En `resultados/ej1/figuras/`, todas generadas por `python -m ej1.graficos` desde lo guardado:

1. `01_paso_01_profundidad.png` … `01_paso_11_l2.png` y `01_paso_final_activacion_ocultas.png`:
   una por paso, con % de semillas que llegan y épocas hasta llegar (media ± desvío).
2. `02_optimizador_y_lr.png`: épocas hasta llegar contra lr (eje log), una curva por optimizador;
   × = no llega con ninguna semilla, círculo vacío = llegan algunas.
3. `03_lineas_de_base_vs_final.png`: píxeles mal promedio y peor letra, líneas de base contra
   config final.
4. `04_espacio_latente.png`: las 32 letras en el latente de la config final (semilla 0) y en PCA 2D.
5. `05a_grilla_latente.png` (letras decodificadas en una grilla del latente, con los 32 códigos
   encima) y `05b_interpolacion.png` (3 interpolaciones); `05_letras_generadas.csv` tiene la
   distancia de cada letra generada a la más parecida del set.
6. `06_curvas_final.png`: pérdida y peor letra por época de la config final, 5 semillas.
7. `07_letras_dificiles.png`: en cuántas corridas del barrido quedó mal cada letra.
