# Ejercicio 2 — Corrección: resoluciones, datos y colapso del posterior

Este archivo reemplaza las partes de `ej2-vae.md` que quedaron obsoletas después de la
primera corrida: §1.2–§1.4 (resoluciones y muestras), §2 (normalización del KL), §5–§7
(barrido, finales y métricas), §8 (figuras) y §12.6 (Qué se commitea). El desarrollo
teórico de la corrección del KL está en `ej2-kl-fix.md`.

## 0. Diagnóstico

La primera corrida terminó con los cuatro modelos elegidos, pero los tres VAE de imagen
colapsaron: 0 unidades activas en test y reconstrucción igual a la imagen media de train.
La causa no es el dataset ni el motor: es la normalización del término KL. El motor promedia
la reconstrucción sobre muestras **y features** (`n·d`) pero sumaba el KL sobre el latente y
lo promediaba solo sobre muestras. Así el gradiente del KL queda multiplicado por `d`
(256–480) respecto del de reconstrucción y el encoder aprende `μ = 0`, `σ = 1`.

La corrección es la normalización estándar de β-VAE cuando la reconstrucción es un promedio
por píxel:

```text
J = L_recon_promedio + β · KL_promedio / d
  = (1/d) · [ Σ_pixels L_recon + β · KL_promedio ]
```

El factor `1/d` es global; lo que cambia es el peso relativo, que pasa a ser `β` y deja de
depender de la resolución. `ej2-kl-fix.md` deriva esto con gradientes y discute warm-up,
free bits y unidades activas.

## 1. Fase 1 — Especificación

Archivos:

- `docs/context/ej2-vae-fix.md` (este archivo).
- `docs/context/ej2-kl-fix.md` (teoría de la corrección del KL).
- `docs/context/ej2-vae.md`: agregar al final `## 14. Correcciones`, con un resumen de esta
  corrección y links a los dos archivos nuevos. No se reescribe el historial: las secciones
  viejas quedan como registro y §14 dice qué las reemplaza.
- `CLAUDE.md` §6: agregar los dos links.
- `docs/structure.md`: actualizar resoluciones, nombres de `.npz`, `latente_4/` en lugar de
  `latente_2/`, y la ruta `data/procesadas/`.
- `commits.txt`: mantener `data/procesadas` en la línea de `git add` de la fase 3 y agregar
  los procesados nuevos en la fase 11. Agregar los cuatro bloques de la corrección
  (`Fase 10` a `Fase 13`, una por cada fase de este plan).

Criterio de aceptación: los tres documentos existen, los links resuelven, `ruff` sigue
verde y no se tocó código todavía.

## 2. Fase 2 — Conversiones nuevas

### 2.1 Rutas

- `utils/preparar_ej2.py`: `PROCESADAS = RAIZ / "data" / "procesadas"`.
- `ej2/datos.py`: `RUTA_DATOS = RAIZ / "data" / "procesadas"`.
- La carpeta movida `data/banderas/procesadas/` queda fuera de uso; el usuario puede
  borrarla. No se lee ni se escribe ahí.
- `data/procesadas/` se crea si falta y se commitea: los `.npz` y PNG procesados van al repo
  para que el entrenamiento sea reproducible. Los raw (`data/headshots`,
  `data/player_images`, `data/chineseMNIST.csv`) no se commitean.

### 2.2 Resoluciones (reemplaza §1.2–§1.4)

| dataset | resolución PIL (ancho×alto) | array (alto×ancho) | vector `d` | `.npz` | carpeta PNG |
|---|---|---|---:|---|---|
| F1 | 32×32 | 32×32 | 1024 | `f1_32x32.npz` | `f1_32x32/` |
| NBA | 30×36 | 36×30 | 1080 | `nba_30x36.npz` | `nba_30x36/` |
| Chino | 32×32 | 32×32 | 1024 | `chino_32x32.npz` | no se guardan PNG |
| Banderas | — | — | 12 | `flags.npz` | — |

Los nombres nuevos evitan pisar los archivos viejos de 20×20/20×24/16×16 que el usuario
movió. Los manifiestos de F1 no cambian (158 imágenes, 29 códigos).

### 2.3 Chino: 32×32, 20 por clase, negro sobre blanco

- Leer 20 filas por clase hasta 15 clases = **300 imágenes**.
- Downsample 64×64 → 32×32 con promedio de bloques 2×2.
- Normalizar cada imagen por su máximo: `x = x / max(x)` (si `max = 0`, dejar negro).
- Invertir: `x = 1 − x`. Resultado: fondo blanco (1), trazo más intenso negro (0).
  La normalización por máximo es necesaria porque el promedio 2×2 diluye el trazo y, sin
  estirarlo, el “negro” quedaría gris.
- Split estratificado por clase: 16 train / 2 val / 2 test por clase = **240/30/30**.
  `semilla_split = 2026`, mismo generador por clase en orden de clase.
- `meta` incluye `por_clase: 20`, `resolucion: [32, 32]`, `invertido: true`.
- Hojas de contacto: 15 archivos `resultados/ej2/chino_<clase>.png` con 8 muestras por
  clase (antes 16), para que se lean los trazos.

### 2.4 Banderas

- `(67, 12)` y split 54/7/6 no cambian.
- Regenerar `flags.npz` desde el CSV con el mismo parser.
- Para las figuras, correr `python3 utils/generar_banderas.py --ancho 240 --salida
  data/banderas` y usar esos PNG como fila “original” del resumen. `data/banderas/` ya está
  ignorado.
- Agregar a `utils/generar_banderas.py` una función `generar_vector(nombre, vector, ancho,
  salida)` que arme la fila del CSV a partir del vector de 12 features y llame a `generar`.
  Así el resumen de banderas usa el mismo renderer para originales, reconstrucciones y
  muestras: los tres se escriben como PNG auxiliares en
  `resultados/ej2/banderas/figuras/auxiliares/` y la figura los lee con Pillow.

### 2.5 Verificación

- Formas: `(158, 1024)`, `(347, 1080)`, `(300, 1024)`, `(67, 12)`.
- Rangos `[0, 1]`, sin NaN/Inf.
- Splits: 126/16/16, 277/35/35, 240/30/30, 54/7/6.
- En chino: el fondo debe ser 1.0 y el píxel de mayor tinta 0.0.
- Tests nuevos en `tests/test_ej2_dedup.py` (o un `test_ej2_preparar.py`):
  - `_leer_chino` con un CSV sintético de 15 clases × 1 muestra: invierte, normaliza y
    devuelve 15 filas.
  - el split de 20/clase da 16/2/2 por clase.

### 2.6 Comandos y tiempos

```bash
python3 utils/preparar_ej2.py --dataset f1
python3 utils/preparar_ej2.py --dataset nba
python3 utils/preparar_ej2.py --dataset chino
python3 utils/preparar_ej2.py --dataset banderas
```

Tiempos esperados: F1 y NBA < 1 min cada uno; chino < 10 s (solo 300 filas del CSV);
banderas < 1 s. Los `.npz` y PNG procesados se agregan al repo.

## 3. Fase 3 — Implementación

### 3.1 Corrección del KL en el núcleo

`core/vae.py`:

- `_scores`: `result["loss"] += self.kl_weight * _kl_sum(...) / (y.shape[0] * y.shape[1])`.
- `backward`: `scale = self.kl_weight / y_true.size`.
- Docstring: explicar que la reconstrucción promedia sobre `n·d`, así que el KL se normaliza
  por `d` para que `β` no dependa de la resolución.
- `kl_divergence`: sigue devolviendo `KL` promedio por muestra **sin** dividir por `d` (es
  una métrica, no el término de la pérdida). Documentarlo.

`tests/test_vae.py`:

- Reemplazar `test_kl_no_se_divide_por_la_cantidad_de_features` por
  `test_kl_se_normaliza_por_la_cantidad_de_features`, parametrizado `d ∈ {6, 12}`:
  `evaluate(kl_weight=1) − evaluate(kl_weight=0) == KL / d`.
- Ajustar `test_loss_es_reconstruccion_mas_lambda_por_kl` a `recon + 0.5 * KL / d`.

### 3.2 Warm-up y free bits

En `core/vae.py`:

- `free_bits: float = 0.0` en `__init__`. Con `KL_nk` la KL de la muestra `n` en la
  dimensión `k`, el término pasa a `Σ_k max(mean_n KL_nk, free_bits)`. El backward enmascara
  la dimensión `k` cuando `mean_n KL_nk <= free_bits` (su gradiente es 0). `free_bits=0`
  reproduce el comportamiento actual y los tests viejos. La métrica `kl_divergence` sigue
  devolviendo la KL real, sin el piso.

En `ej2/vae_exp.py`:

- Callback `CalentamientoKL(beta, epocas)`:
  - `on_train_begin`: `network.kl_weight = 0.0`.
  - `on_epoch_end`: `network.kl_weight = beta * min(1.0, epoch / epocas)`.
- Al terminar el fit, fijar `modelo.kl_weight = beta` antes de evaluar y guardar.
- Early stopping sobre `val_recon` (MSE determinística) con `patience` 200, no sobre
  `val_loss`, para no cortar durante el warm-up ni en una meseta de colapso.
- Métrica `recon` en `fit` para que `val_recon` esté en los logs.

### 3.3 Configuración

`ej2/config.json`:

- `base`: `latente = 4`, `capacidad = [256, 128]`, `beta = 1.0`, `lr = 0.001`,
  `perdida = "mse"`, `max_epocas = 1000`, `paciencia = 200`.
- Pasos:
  - `1_latente`: `[4, 8, 16]` (se eliminan 2 y 32).
  - `2_capacidad`: `[[128, 64], [256, 128]]`.
  - `3_beta`: `[0.01, 0.1, 1.0, 5.0]`.
  - `4_lr`: `[0.0003, 0.001, 0.003]`.
  - `5_perdida`: `["mse"]` (se saltea; MSE + identidad para gris continuo).
- Por dataset:

| dataset | d | forma (ancho, alto) | batch | semillas barrido | épocas barrido |
|---|---:|---|---:|---:|---:|
| nba | 1080 | [30, 36] | 128 | 5 | 400 |
| f1 | 1024 | [32, 32] | 16 | 5 | 400 |
| chino | 1024 | [32, 32] | 64 | 5 | 300 |
| banderas | 12 | [1, 12] | 32 | 5 | 400 |

- `semillas_final = 5`; `semillas_latente = 5` (chino también, porque 300 muestras es
  barato).
- `calentamiento_frac = 0.2` (20 % de las épocas, mínimo 50).
- `free_bits = 0.0` en el nivel 0.

### 3.4 Filtro de colapso y escalera de reintentos

En `_elegir` de `ej2/vae_exp.py`, para **todos** los pasos (no solo β):

1. Descartar corridas inválidas o con `unidades_activas < k/2` o `kl_medio <= 0.01`.
2. Elegir por ELBO con el criterio de simplicidad actual.
3. Si ninguna corrida del paso cumple, elegir la de más unidades activas y registrar el
   paso como `colapsado`.

Después de elegir el ganador del barrido, si el modelo final sigue colapsado, subir de
nivel y repetir el barrido/final con los overrides:

| nivel | β | warm-up | free bits | capacidad | épocas / paciencia |
|---|---|---|---:|---|---|
| 0 | 0.01, 0.1, 1, 5 | 20 % | 0.0 | 128-64, 256-128 | 1000 / 200 |
| 1 | 0.01, 0.1, 1 | 40 % | 0.05 | 128-64, 256-128 | 1200 / 250 |
| 2 | 0.001, 0.01, 0.1 | 50 % | 0.10 | 256-128, 512-256 | 1500 / 300 |
| 3 | 0.0001, 0.001, 0.01 | 50 % | 0.20 | 256-128, 512-256 | 1500 / 300 |

- Se guarda `nivel_anticolapso` en `elecciones.json` y en `config.json` de cada carpeta.
- Criterio para aceptar un nivel: en validación, `activas >= k/2`, `KL > 0.01` y
  `recon < baseline de la imagen media de train`. Si no, siguiente nivel.
- El nivel 3 es el último; si aun así colapsa, se reporta como hallazgo y se usa el de más
  unidades activas (no se falsifica el resultado).

### 3.5 Carga y artefactos

- `ej2/datos.py`: `ARCHIVOS_IMAGEN = {"f1": "f1_32x32.npz", "nba": "nba_30x36.npz",
  "chino": "chino_32x32.npz"}`; `d` y `forma` salen de `config.json`.
- `ej2/vae_exp.py`: carpetas `latente_4/` y `latente_16/` (se elimina `latente_2/`),
  `ae_latente_4/`, `ae_latente_16/`. Si el ganador es 4 o 16, se reutiliza `final/`.
- Los nombres de pesos/historias/muestras no cambian.
- `config.json` de cada carpeta incluye `nivel_anticolapso`, `free_bits`,
  `calentamiento_epocas`.

### 3.6 Figuras

Nuevo `00_resumen.png` por dataset, **primero** en el orden y en la presentación:

- 8 columnas × 3 filas: original / reconstrucción determinística / muestra de `N(0, I)`.
- NBA: `lebron`, `Luka-Doncic`, `Nikola-Jokic`, `Stephen-Curry` + 4 aleatorios.
- F1: `ALO`, `VER` + 6 aleatorios. `COL` y `ANT` no existen en el dataset; la caption lo
  aclara y se completan con aleatorios.
- Chino: 8 clases distintas al azar, una muestra de test por clase.
- Banderas: 8 países al azar de test. Las tres filas se generan con
  `utils/generar_banderas.py` (función `generar_vector`): los originales desde el CSV, las
  reconstrucciones y las muestras desde los vectores decodificados. Los PNG auxiliares van a
  `resultados/ej2/banderas/figuras/auxiliares/` y la figura los lee con Pillow.
- El resto: `01_dataset`, `02_reconstruccion` (train/test), `03_latente_4`,
  `04_latente_16_pca`, `05_generacion`, `06_curvas`, `07_latente_4_vs_16`; en banderas se
  agrega `08_banderas_decodificadas`.
- `resultados/ej2/README.md`: índice con una línea por figura y el comando para regenerar.
- `python3 -m ej2.graficos` imprime la ruta absoluta de la carpeta al terminar.

### 3.7 Tests

- `test_vae.py`: normalización `/d`, free bits, warm-up.
- `test_ej2_vae.py`: constructores con `beta` y `free_bits`; smoke con k=4/8/16.
- `test_ej2_datos.py`: nombres nuevos, formas, `forma` (ancho, alto).
- `test_ej2_preparar.py` (nuevo): chino invertido y 20/clase.
- `test_ej2_graficos.py`: selección de 8 muestras, resumen, banderas auxiliares.
- `check-cicd.py` verde antes de la fase 4.

## 4. Fase 4 — Reentrenamiento y cierre

1. Barridos: `nba`, `f1`, `chino`, `banderas` con `--trabajadores 5`.
2. Verificar `elecciones.json`: una decisión por paso y `nivel_anticolapso`.
3. Si el ganador colapsa, subir de nivel y repetir (automático o con `--nivel`).
4. Finales: `final`, `latente_4`, `latente_16`, `ae_latente_4`, `ae_latente_16`.
5. Verificar que los pesos recargados reproducen `test.csv` (semilla 0, 20 carpetas).
6. Regenerar las 33 figuras con `python3 -m ej2.graficos`.
7. CI completo, tests de `ej1`/`ej1b`, `git status` sin cambios en `ej1/`, `ej1b/`,
   `resultados/ej1/`, `resultados/ej1b/`, `data/font.h`.
8. Actualizar `ej2-vae-fix.md` con resultados medidos y agregar la fase 13 a `commits.txt`.

Tiempos estimados con 5 workers:

| dataset | barrido | finales |
|---|---:|---:|
| NBA 30×36 | ~4 min | ~1 min |
| F1 32×32 | ~4 min | ~1 min |
| Chino 32×32, 300 | ~4 min | ~1 min |
| Banderas | < 1 min | < 1 min |

Total: ~20 min de cómputo, más reintentos si hacen falta.

Para referencia, 64×64 con 20 muestras por clase (300) sería ~2× el costo de 16×16 con 2400
(16× los píxeles, 1/8 las muestras): ~13 min de barrido y ~5 min de finales. No se usa en
esta corrección.

## 5. Criterios de aceptación de la corrección

- Los cuatro datasets tienen las formas nuevas y los splits indicados.
- Chino se ve negro sobre blanco y las hojas de contacto son legibles.
- La pérdida usa `KL/d`; los tests lo verifican.
- El ganador de cada dataset tiene `activas >= k/2` y `KL > 0.01`, o se documenta el nivel
  de reintento usado.
- La reconstrucción del VAE es claramente mejor que la imagen media de train.
- `00_resumen.png` existe para los cuatro datasets y muestra original/reconstrucción/muestras
  con los nombres pedidos.
- `python3 check-cicd.py` pasa; `ej1`/`ej1b` no cambian.
- `data/procesadas/` se agrega al repo con los cuatro datasets y sus PNG.

## 6. Riesgos

- **Chino con 20/clase**: 300 muestras y 2 de validación por clase. La validación es ruidosa;
  el early stopping puede cortar antes. Si la reconstrucción queda pobre, subir a 40/clase
  (600) es la palanca, no tocar el test.
- **F1 sin Colapinto/Antonelli**: los códigos no están en las 158 imágenes. Se usan ALO/VER y
  aleatorios; la caption lo dice.
- **Banderas**: el resumen depende de los PNG de `data/banderas`; si faltan, se corre
  `utils/generar_banderas.py` antes de las figuras.
- **Warm-up**: si el early stopping corta durante el warm-up, el modelo queda casi AE. Por eso
  se monitorea `val_recon` y la paciencia es 200.
- **Reintentos**: cada nivel cuesta un barrido; el nivel 0 debería alcanzar con la
  normalización corregida. Los niveles 1–3 son el seguro.
