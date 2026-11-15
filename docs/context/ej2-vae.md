# Ejercicio 2 — VAE sobre cuatro datasets

## Objetivo

Entrenar y estudiar autoencoders variacionales (VAE) sobre cuatro conjuntos de datos de
naturaleza distinta, cumpliendo la consigna del ejercicio 2:

1. Implementar el VAE (ya existe en `core/vae.py`).
2. Elegir una arquitectura y una configuración de entrenamiento mediante un barrido
   secuencial, con validación, sobre cada dataset.
3. Mostrar el espacio latente, reconstrucciones y generación de muestras nuevas.
4. Comparar una dimensión latente chica (2) contra una grande (16) en la presentación.

Los cuatro datasets son:

| Clave | Dataset | Tipo de entrada | Uso |
|---|---|---|---|
| `nba` | Retratos de jugadores NBA | imagen 20×24 gris | comparación cuantitativa latente 2 vs 16 |
| `f1` | Fotos de pilotos de F1 | imagen 20×20 gris | ejemplo cualitativo con pocos datos |
| `chino` | Caracteres chinos (subconjunto) | imagen 16×16 gris | dataset grande y regular |
| `banderas` | Banderas tricolores | vector numérico de 12 features | VAE tabular, sin imágenes de entrada |

## Decisiones del equipo (no se cambian)

- Motor propio: NumPy, sin scikit-learn/PyTorch/TensorFlow/Keras para modelos, splits ni
  métricas. scikit-learn solo puede usarse como oráculo dentro de `tests/`.
- `core/vae.py` pasa a la normalización estándar de β-VAE:
  `J = L_reconstrucción_promedio + β · KL_promedio_por_muestra`, **sin dividir el KL por la
  cantidad de features**. Esto solo afecta al VAE; `ej1` y `ej1b` no usan esta clase.
- Se reporta media ± desvío (ddof = 1) sobre ≥ 3 semillas. Los modelos finales usan 5.
- Toda decisión de hiperparámetros sale de validación. Test se lee solo para la evaluación
  final.
- Banderas: la entrada del VAE es **siempre** el vector de 12 columnas del CSV. Nunca se
  construye una imagen para entrenar. Renderizar banderas está permitido únicamente para
  figuras y lo hace `ej2/banderas.py`, que no importa el cargador de datos ni el script de
  entrenamiento.
- Semillas: 0–4. `semilla_split` fija y guardada.
- No se commitea nada. Al terminar se muestra `git status` y `git diff --stat`, se propone
  mensaje de commit y se espera permiso.
- Código, comentarios y nombres de tests en español.
- `data/f1hq` está en `.gitignore` y se ignora: no es fuente de este ejercicio. Tampoco se
  borra.
- Los `.npz` procesados y las imágenes de baja resolución se agregan al repo (no están en
  `.gitignore`), para que el entrenamiento sea reproducible. Los raw (`data/headshots`,
  `data/player_images`, `data/chineseMNIST.csv`) no se commitean.

## Datos: fuentes y conteos exactos

### F1 — `data/headshots/`

- 347 archivos PNG en total.
- 172 en el nivel superior `data/headshots/<temporada>/*.png`, temporadas 2017–2026.
- 175 dentro de `data/headshots/<temporada>/by_ref/*.png`.
- Los `by_ref` son **duplicados exactos** (mismo SHA-256) de un archivo del nivel superior.
  Verificado sobre los 175.
- Los 172 del nivel superior contienen **158 hashes únicos** y **29 códigos de piloto**.
  Un mismo piloto puede tener fotos distintas en distintas temporadas; se conservan.
- Tamaños: 141 de 840×840, 16 de 1000×1000, 13 de 839×839, 1 de 849×849, 1 de 800×800.
- Modos: 156 `P` con transparencia y 16 `RGBA`. Todos tienen fondo transparente.
- Dataset final: **158 imágenes únicas**, 20×20 gris.

### NBA — `data/player_images/`

- 347 archivos, todos con nombre de jugador único (no hay duplicados de nombre).
- 345 son JPEG de 75×100 (aspecto 3:4 exacto).
- 1 es `Ryan-...` de 75×99 (diferencia despreciable).
- 1 es `lebron.png`, 600×436 RGBA, aspecto 1.38, único apaisado y con transparencia.
- Dataset final: **347 imágenes**, sin descartar ninguna, 20×24 gris.

### Chino — `data/chineseMNIST.csv`

- 15.001 líneas: encabezado + 15.000 filas.
- Columnas `pixel_0 … pixel_4095, label, character` (4098 columnas).
- 15 clases × 1000 filas: `零 一 二 三 四 五 六 七 八 九 十 百 千 万 亿`.
- Cada fila es una imagen de 64×64 en escala de grises, fondo 0 y tinta clara. No hay que
  invertir.
- Dataset final: **160 por clase = 2400 imágenes**, 16×16 gris continuo.

### Banderas — `data/flags_tricolor.csv`

- 67 banderas, una por país.
- 46 horizontales y 21 verticales (`rotada` 0/1).
- `proporcion`: 57 valen 1.0; el resto 0.5, 0.667, 1.333, 2.0 o 3.0.
- `aspecto`: entre 1.1538 y 2.0 (mayoría 1.5 y 2.0).
- Cada fila tiene exactamente 3 colores RGB.
- El CSV está mal escapado: el encabezado declara 13 columnas con `;`, pero las filas tienen
  7–8 campos porque los triples RGB contienen comas. Se parsea con la lógica tolerante de
  `utils/generar_banderas.py` (ver §2.5).
- Dataset final: **matriz `(67, 12)`**. Ninguna imagen entra al VAE.

## 1. Preparación de datos — `utils/preparar_ej2.py`

Script único, fuera de `ej2/` y de ruff (como el resto de `utils/`). Usa Pillow para leer y
escribir imágenes; se agrega `pillow==11.3.0` a `requirements.txt`.

Interfaz:

```bash
python3 utils/preparar_ej2.py --dedup-f1     # destructivo, una sola vez
python3 utils/preparar_ej2.py --dataset f1
python3 utils/preparar_ej2.py --dataset nba
python3 utils/preparar_ej2.py --dataset chino
python3 utils/preparar_ej2.py --dataset banderas
python3 utils/preparar_ej2.py --todo         # todo lo anterior, en orden
```

Reglas generales:

- `--forzar` rehace las salidas existentes; sin él, cada salida existente se saltea.
- Todas las rutas son relativas a la raíz del repo (`Path(__file__).resolve().parents[1]`).
- El script imprime un resumen por dataset: cantidad de muestras, forma de `X`, rango,
  conteo por split y rutas escritas.
- Si una fuente falta o un conteo no coincide con el esperado, aborta con un mensaje claro
  y no escribe nada parcial.

### 1.1 Deduplicación de F1 — destructiva, una sola vez

`--dedup-f1` ejecuta exactamente estos pasos:

1. Listar todos los `data/headshots/<temporada>/by_ref/*.png`.
2. Calcular SHA-256 de cada uno y de todos los `data/headshots/<temporada>/*.png` del nivel
   superior.
3. Para cada `by_ref`, buscar un top-level con el mismo hash **en la misma temporada**. Si
   alguno no tiene duplicado exacto, abortar y listar los archivos problemáticos. La
   excepción comprobada en los datos reales: `2023/by_ref/LOGSAR1.png` es único y
   `2024/by_ref/LOGSAR1.png` duplica a `2023/SAR.png` (otra temporada). Con
   `--permitir-huerfanos` se copian antes de borrar a
   `data/procesadas/f1_byref_sin_par/` y se registran en
   `resultados/ej2/dedup_byref_sin_par.csv`; el dataset principal sigue teniendo 158
   imágenes. Sin ese flag, el script aborta sin borrar nada.
4. Escribir `resultados/ej2/dedup_headshots.csv` con columnas
   `ruta_borrada,sha256,ruta_conservada` (175 filas).
5. Borrar cada directorio `data/headshots/<temporada>/by_ref/` completo.
6. Verificar que `find data/headshots -type d -name by_ref` no devuelve nada y que quedan
   172 archivos de nivel superior.
7. Con los `by_ref` todavía presentes (paso previo al borrado), construir el mapa
   `código → piloto`:
   - `codigo` = stem del archivo top-level (`HAM`).
   - `piloto` = stem del `by_ref` asociado (`LEWHAM01`).
   - Si un top-level no tuviera `by_ref`, se usa `codigo` como `piloto`.
8. Guardar ese mapa en `data/procesadas/f1_manifest_completo.csv` antes de borrar.

Después del borrado, deduplicar los 172 top-level por hash:

- Agrupar por SHA-256.
- De cada grupo conservar, en este orden de prioridad: temporada más reciente, mayor
  resolución, ruta lexicográfica menor.
- **No borrar** los archivos descartados: solo se excluyen del dataset. Los archivos siguen
  en disco y en `.gitignore`.
- Escribir `resultados/ej2/dedup_hash.csv` con
  `ruta_conservada,ruta_descartada,sha256,temporada_conservada,temporada_descartada`.
- Resultado esperado: 158 filas conservadas, 14 descartadas.

Escribir los manifiestos:

- `data/procesadas/f1_manifest.csv`: 158 filas, columnas
  `archivo,codigo,piloto,temporada,sha256,ancho,alto`.
- `data/procesadas/f1_manifest_piloto.csv`: 29 filas, una por `codigo`, con la imagen de
  temporada más reciente. Es solo una variante opcional; **el dataset principal es el de 158**.

### 1.2 F1 — imágenes 20×20

Para cada fila de `f1_manifest.csv`:

1. Abrir con Pillow. Si el modo es `P` con transparencia o `RGBA`, convertir a `RGBA`.
2. Componer sobre fondo blanco:
   `salida = alpha·color + (1−alpha)·255`.
3. Calcular el bounding box del sujeto con `alpha > 8`.
4. Agregar un margen del 2 % del lado mayor del bbox.
5. Recortar el bbox y llevarlo a un cuadrado blanco del lado mayor (padding, sin estirar).
6. Convertir a `L` (gris) y redimensionar a 20×20 con LANCZOS.
7. Normalizar a `[0,1]` con `float32`.
8. Guardar el PNG de baja resolución en
   `data/procesadas/f1_20x20/<codigo>_<temporada>_<hash8>.png`.
9. Agregar el vector aplanado a `X`.

Salidas:

- `data/procesadas/f1_20x20.npz`
- `data/procesadas/f1_20x20/` (158 PNG)
- `resultados/ej2/f1_contacto.png` (grilla con las 158 imágenes y su código)

Split: permutación aleatoria con `np.random.default_rng(semilla_split)` sobre los índices
ordenados por ruta, 80/10/10 → **126/16/16**. Se guarda la etiqueta por muestra. No es
disjunto por piloto; para un VAE generativo está bien. La variante disjunta por piloto queda
disponible con `f1_manifest_piloto.csv`, pero no se usa por defecto.

### 1.3 NBA — imágenes 20×24

Para cada archivo de `data/player_images/`:

1. Abrir. Si es `RGBA`, componer sobre blanco.
2. Si el aspecto es 3:4 (todos menos LeBron): redimensionar directo a 20×24. El único
   75×99 se redimensiona también; la distorsión es < 1,5 % y no justifica otro tratamiento.
3. `lebron.png`:
   - Componer sobre blanco.
   - Calcular el bbox del sujeto con `alpha > 8` (resultado medido: x 41–559, y 61–435).
   - Recortar a 3:4 centrado en el bbox: se conserva toda la altura y se toma el ancho
     `0.75·alto`, centrado en el centro horizontal del bbox. Si el recorte se sale de la
     imagen, se completa con blanco.
   - Redimensionar a 20×24.
4. Convertir a `L` y redimensionar a 20×24 con LANCZOS.
5. Normalizar a `[0,1]` con `float32`.
6. Guardar `data/procesadas/nba_20x24/<nombre>.png`.
7. Agregar el vector aplanado a `X`.

Salidas:

- `data/procesadas/nba_20x24.npz`
- `data/procesadas/nba_20x24/` (347 PNG)
- `resultados/ej2/nba_contacto.png` (grilla con las 347 imágenes)

Split: 80/10/10 → **277/35/35**. Cada archivo es un jugador distinto, así que el split es
naturalmente disjunto por identidad.

Si el recorte de LeBron se ve mal en la hoja de contacto, se descarta **solo esa imagen** y
se documenta en el resumen; no se descarta ninguna de las 75×100.

### 1.4 Chino — imágenes 16×16

Leer el CSV en streaming con `csv.reader`; no cargar los 120 MB como matriz densa.

1. Saltear el encabezado.
2. Acumular filas hasta tener **160 por cada una de las 15 clases** (2400 en total).
3. Para cada fila: `np.asarray(fila[:4096], dtype=np.float32).reshape(64, 64)`.
4. Downsample por promedio de bloques 4×4:
   `chico = pix.reshape(16, 4, 16, 4).mean(axis=(1, 3))`.
5. Normalizar `/255` a `[0,1]`.
6. Guardar el vector aplanado en `X` y la clase/character en `y`.

No se binariza la entrada: se conserva gris continuo. La binarización con umbral 0,5 se usa
solo para la métrica de píxeles mal.

Salidas:

- `data/procesadas/chino_16x16.npz`
- 15 hojas de contacto en `resultados/ej2/chino_<clase>.png`, 16 muestras por clase.
- Por defecto **no** se escriben 2400 PNG individuales (evita miles de archivos en git). La
  opción `--guardar-png` los escribe en `data/procesadas/chino_16x16/` si se piden.

Split: estratificado por clase, 80/10/10 por clase → **128/16/16 por clase = 1920/240/240**.
Se permuta cada clase con el mismo generador, en orden de clase.

### 1.5 Banderas — vector de 12 features

Parseo tolerante (idéntico al de `utils/generar_banderas.py`):

```python
nombre, *resto = fila
numeros = list(map(int, ",".join(resto[:-3]).split(",")))
colores = [numeros[i : i + 3] for i in range(0, len(numeros), 3)]
rotada, proporcion, aspecto = int(resto[-3]), float(resto[-2]), float(resto[-1])
```

Vector, en este orden exacto:

```text
[r1/255, g1/255, b1/255,
 r2/255, g2/255, b2/255,
 r3/255, g3/255, b3/255,
 rotada,          # 0/1
 proporcion,      # crudo, 0.5–3.0
 aspecto]         # crudo, 1.15–2.0
```

Salida `data/procesadas/flags.npz`:

- `X` `(67, 12) float32` **crudo** (colores ya en `[0,1]`, `proporcion` y `aspecto` sin
  escalar).
- `nombres` `(67,)` con el país.
- `split` `(67,)` con `train/val/test`.
- `columnas` `(12,)` con los nombres de las columnas.

Split: permutación aleatoria 80/10/10 → **54/7/6**. El escalado de `proporcion` y `aspecto`
se ajusta **solo con las filas de train**, en `ej2/datos.py`, no en la preparación. El
escalador (mínimos y rangos) se guarda en `resultados/ej2/banderas/escalador.json` cuando
corre el entrenamiento.

No se generan imágenes de banderas como dato. `data/banderas/` (PNG ya existentes) no se usa.

### 1.6 Formato de los `.npz`

Todos usan `np.savez_compressed` y guardan `meta` como string JSON.

`f1_20x20.npz`:

| clave | forma | tipo | contenido |
|---|---|---|---|
| `X` | (158, 400) | float32 | píxeles en `[0,1]` |
| `y` | (158,) | `<U3` | código (`HAM`) |
| `piloto` | (158,) | `<U16` | identificador de piloto (`LEWHAM01`) |
| `temporada` | (158,) | int16 | año |
| `archivo` | (158,) | `<U128` | ruta fuente |
| `split` | (158,) | `<U5` | `train/val/test` |
| `meta` | () | `<U...` | JSON con resolución, fuente, `semilla_split`, conteos |

`nba_20x24.npz`: `X (347,480)`, `y (347,)` nombre, `archivo (347,)`, `split (347,)`, `meta`.

`chino_16x16.npz`: `X (2400,256)`, `y (2400,)` carácter, `clase (2400,)` entero 0–14,
`split (2400,)`, `meta`.

`flags.npz`: `X (67,12)`, `nombres (67,)`, `split (67,)`, `columnas (12,)`, `meta`.

Verificación en la preparación: formas, `dtype`, rango `[0,1]` para imágenes, suma de conteos
por split, y que ningún split esté vacío.

## 2. Cambio en `core/vae.py` — β estándar

Hoy la clase normaliza el KL igual que la reconstrucción (divide por `n·d`). Pasa a:

```text
J = L_recon_promedio + β · mean_n KL(q(z|x_n) || N(0,I))
```

donde `KL(q||N(0,I)) = ½ Σ_k (exp(s_k) + μ_k² − 1 − s_k)` se suma sobre el latent y se
promedia solo sobre muestras.

Cambios concretos:

- En `_scores`: `result["loss"] += self.kl_weight * _kl_sum(...) / y.shape[0]`.
- En `backward`: `scale = self.kl_weight / y_true.shape[0]`.
- Docstring: reemplazar la fórmula que divide por `d` por la de β-VAE estándar y aclarar que
  el KL se promedia por muestra, no por píxel/feature.
- `docs/structure.md`: la fórmula ya dice `L + kl_weight·KL`; se ajusta la nota para que
  coincida con el código.
- No se toca `core/autoencoder.py`, `core/network.py` ni `core/losses.py`.
- `ej1` y `ej1b` no usan `VariationalAutoencoder`, así que sus resultados no cambian.

Tests en `tests/test_vae.py`:

- `test_loss_es_reconstruccion_mas_lambda_por_kl`: esperar
  `reconstruccion + 0.5 * vae.kl_divergence(X)` (sin `/ X.shape[1]`).
- Nuevo `test_kl_no_se_divide_por_la_cantidad_de_features`, parametrizado sobre
  `d ∈ {6, 12}`: con ε fijo, la diferencia entre `evaluate` con `kl_weight=1` y con
  `kl_weight=0` es exactamente `kl_divergence(X)`.
- Los tests de diferencias finitas ya existentes deben seguir pasando sin cambios.
- El resto de los tests del VAE no se modifica.

## 3. Estructura de código

```text
utils/
  preparar_ej2.py             dedup + datasets procesados (Pillow; fuera de ruff)
ej2/
  __init__.py
  datos.py                    carga de npz, splits, escalador de banderas
  banderas.py                 parseo del CSV + render_flag (solo figuras)
  config.json                 grilla del barrido + configuración por dataset
  modelos.py                  constructores de VAE de imagen, VAE tabular y AE determinístico
  vae_exp.py                  barrido compacto y entrenamiento final (paralelo)
  graficos.py                 figuras; solo lee resultados/ej2/
tests/
  test_vae.py                 actualizado (normalización del KL)
  test_ej2_dedup.py           dedup sobre tmp_path
  test_ej2_datos.py           formas, rangos, splits, banderas 12 columnas
  test_ej2_vae.py             constructores, smoke training, reproducibilidad
  test_ej2_banderas.py        parser y renderer
data/procesadas/
  f1_20x20.npz, nba_20x24.npz, chino_16x16.npz, flags.npz
  f1_20x20/, nba_20x24/       PNG de baja resolución
  f1_manifest.csv, f1_manifest_piloto.csv, f1_manifest_completo.csv
resultados/ej2/
  dedup_headshots.csv, dedup_hash.csv
  <dataset>/
    barrido/paso_<n>_<nombre>.csv
    elecciones.json
    final/                     config, pesos, historias, test.csv, muestras
    latente_2/
    latente_16/
    ae_latente_2/, ae_latente_16/
    figuras/
```

Reglas de dependencia:

- `ej2/datos.py` no importa `ej2/banderas.py`.
- `ej2/vae_exp.py` no importa `ej2/banderas.py`.
- `ej2/modelos.py` no importa datos ni banderas.
- `ej2/graficos.py` es el único que importa `ej2/banderas.render_flag`.
- `ej2/graficos.py` no entrena: solo lee `resultados/ej2/` y escribe figuras.

## 4. Modelos

### 4.1 VAE de imagen (NBA, F1, chino)

```text
encoder: d → c1 → c2 → 2k      tanh en ocultas, identidad en la salida
decoder: k → c2 → c1 → d      tanh en ocultas
salida:   sigmoid si la pérdida es BCE; identidad si es MSE
init:     Xavier
```

`d`: NBA 480, F1 400, chino 256. `k` es la dimensión latente del barrido. La capacidad
`c1–c2` es la del barrido: `64–32`, `128–64` o `256–128`.

`build_vae_imagen(d, latente, capacidad, salida, rng) -> VariationalAutoencoder`.
`build_ae_imagen(d, latente, capacidad, salida, rng) -> Autoencoder` (encoder termina en `k`).

### 4.2 VAE tabular (banderas)

```text
encoder: 12 → c1 → c2 → 2k    tanh, identidad
decoder: k → c2 → c1 → 12     tanh, identidad
pérdida: MSE sobre features escaladas
init: Xavier
```

`build_vae_tabular(latente, capacidad, rng)` y `build_ae_tabular(...)`.

### 4.3 Autoencoder determinístico de comparación

Misma arquitectura y misma `k`, sin muestreo ni KL. Sirve para comparar la generación
`N(0,I)` por el decoder del VAE contra el decoder de un AE. Se entrena con la misma pérdida,
lr y épocas que el modelo final correspondiente.

## 5. Barrido compacto

### 5.1 Configuración base

```json
{
  "semillas": [0, 1, 2, 3, 4],
  "semilla_split": 2026,
  "base": {
    "latente": 16,
    "capacidad": [128, 64],
    "beta": 1.0,
    "lr": 0.001,
    "perdida": "binary_crossentropy"
  },
  "pasos": [
    {"id": "1_latente",   "valores": [2, 8, 16, 32]},
    {"id": "2_capacidad", "valores": [[64, 32], [128, 64], [256, 128]]},
    {"id": "3_beta",      "valores": [0.1, 1.0, 5.0]},
    {"id": "4_lr",        "valores": [0.0003, 0.001, 0.003]},
    {"id": "5_perdida",   "valores": ["binary_crossentropy", "mse"]}
  ]
}
```

Cada paso cambia **solo** su parámetro respecto del ganador acumulado; los pasos se recorren
en orden y el ganador queda fijo para los siguientes.

Configuración por dataset en `ej2/config.json`:

| dataset | tipo | d | batch | semillas barrido | épocas barrido | pérdidas del paso 5 |
|---|---|---:|---:|---:|---:|---|
| `nba` | imagen | 480 | 128 | 5 | 500 | BCE, MSE |
| `f1` | imagen | 400 | 8 | 5 | 500 | BCE, MSE |
| `chino` | imagen | 256 | 128 | 3 | 200 | BCE |
| `banderas` | tabular | 12 | 32 | 5 | 500 | MSE |

El paso 5 se saltea cuando la lista de pérdidas tiene un solo valor, así que chino y banderas
tienen 13 configuraciones (4+3+3+3) y NBA/F1 tienen 15 (4+3+3+3+2).

Base por dataset:

- `latente = 16`, `capacidad = [128, 64]`, `beta = 1.0`, `lr = 0.001`.
- `perdida`: BCE para imágenes, MSE para banderas.
- Optimizador Adam, sin decaimiento ni L2.
- Máximo 1000 épocas, `EarlyStopping` sobre `val_loss` con paciencia 100 y
  `restore_best=True`. F1 usa el mismo criterio: con 158 imágenes, validación tiene 16
  muestras, suficiente para cortar.
- El barrido usa las épocas de la tabla (500 o 200) con el mismo early stopping.

### 5.2 Reglas de selección

Definiciones:

- `val_elbo` = pérdida de validación del VAE (`L + β·KL`), promediada sobre semillas.
- `val_recon` = error de reconstrucción determinística `decode(encode(X_val))`:
  - imágenes: MSE en `[0,1]` (comparable entre BCE y MSE);
  - banderas: MSE en unidades originales (revirtiendo el escalado).
- `unidades_activas` por dimensión latente `j`:
  `mean_n ½(exp(s_j) + μ_j² − 1 − s_j)`; activa si `> 0.01`.
- `kl_medio` = `kl_divergence(X_val)`.

Para los pasos 1, 2 y 4 (latente, capacidad, lr):

1. Descartar corridas con NaN/Inf o que no bajaron de la pérdida inicial.
2. Agrupar por valor; quedarse con el menor `val_elbo` medio.
3. Si otro valor está a menos de 1 desvío estándar del mejor, preferir el más simple:
   menor latente, menor capacidad (en el orden dado) o menor lr.
4. Empate exacto: mantener el valor actual.

Para el paso 3 (β): el ELBO no es comparable entre β. Entre los modelos con
`unidades_activas ≥ 0.5·k` y `kl_medio > 0.01`, elegir el menor `val_recon`. Si ninguno
cumple, elegir el de mayor `unidades_activas` y reportarlo como hallazgo. Empate: β actual.

Para el paso 5 (pérdida): elegir el menor `val_recon` (MSE determinístico). Si están a menos
de 1 desvío, preferir BCE para imágenes por ser la pérdida natural de píxeles en `[0,1]`;
para banderas no hay paso 5.

Todas las decisiones se registran en `elecciones.json`: paso, valor ganador, métricas medias
y desvíos, motivo.

### 5.3 Ejecución paralela y reanudación

`ej2/vae_exp.py`:

```bash
python3 -m ej2.vae_exp --dataset nba --fase barrido --trabajadores 5
python3 -m ej2.vae_exp --dataset nba --fase final   --trabajadores 5
```

- Cada corrida es `(valor, semilla)`. Se ejecutan con `multiprocessing` contexto `spawn`.
- Antes de importar NumPy en cada worker se fijan
  `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=4` (o `--hilos`), para no sobre-suscribir.
- `--trabajadores 5` por defecto; 1 para smoke tests.
- Un CSV por paso: `barrido/paso_<n>_<nombre>.csv`, una fila por corrida con
  `paso,valor,semilla,val_elbo,val_recon,kl_medio,unidades_activas,epocas,segundos,valido`.
- Si el CSV del paso existe, se lee y se usa para elegir; no se reentrena. Si el hash de la
  configuración acumulada no coincide con el guardado en el CSV, se aborta pidiendo borrar
  ese CSV (mismo criterio que `ej1/barrido.py`).
- `--forzar` borra y rehace el paso indicado.
- Al terminar cada paso se reescribe `elecciones.json`.

## 6. Entrenamiento final

Con la configuración elegida por dataset se entrena:

1. `final/`: el modelo ganador, 5 semillas × 1000 épocas (chino: 5 semillas × 1000).
2. `latente_2/`: la misma config forzando `k = 2`, 5 semillas × 1000 (chino: 3 semillas).
3. `latente_16/`: la misma config forzando `k = 16`, 5 semillas × 1000 (chino: 3 semillas).
4. `ae_latente_2/` y `ae_latente_16/`: AE determinístico, misma capacidad, lr, pérdida y
   épocas, 5 semillas (chino: 3).

Si el ganador ya es 2 o 16, se reutiliza esa corrida para no duplicar.

Cada carpeta guarda:

```text
config.json                 config efectiva + semilla_split + versión del spec
pesos_semilla_<s>.npz       params del encoder y decoder
historia_semilla_<s>.csv    epoch, train_loss, val_loss, kl, unidades_activas
test.csv                    una fila por semilla con las métricas de test
muestras.npz                z, muestras generadas, reconstrucciones determinísticas
```

El test se evalúa una sola vez por semilla, con los pesos restaurados del mejor `val_loss`.

## 7. Métricas

Todas se reportan como media ± desvío sobre semillas.

### Imágenes

- `recon_bce` y `recon_mse` determinísticos: `decode(encode(X_test))`.
- `pixeles_mal`: fracción de píxeles con `|x − x'| > 0.5` (para chino, binarizar ambos a 0,5).
- `kl_medio`: `kl_divergence(X_test)`.
- `unidades_activas`: sobre `X_test`.
- `nn_dist`: distancia euclídea media de cada muestra generada `N(0,I)` a su imagen de train
  más cercana. Sirve para detectar muestras degeneradas o copias.
- Figuras: grillas de muestras y de reconstrucciones.

### Banderas

- `recon_mse` y `recon_mae` en el espacio escalado.
- `recon_mse_original` y `recon_mae_original`: se revierte el escalado de `proporcion` y
  `aspecto`, se recortan al rango de train, se redondean los colores a 0–255 y `rotada` a
  0/1; se calcula el error contra el vector original.
- `kl_medio`, `unidades_activas`.
- Render de banderas decodificadas **solo** para figuras.

### Comparación latente 2 vs 16

Para cada dataset se compara `latente_2` contra `latente_16` con las mismas métricas. La
presentación usa:

- NBA como evidencia cuantitativa principal (347 imágenes).
- F1 como ejemplo cualitativo de dataset chico.
- Chino como ejemplo de muchas muestras y clases.
- Banderas como ejemplo de VAE tabular y de generación de combinaciones plausibles.

## 8. Figuras — `ej2/graficos.py`

`python3 -m ej2.graficos [--dataset nba]` lee solo `resultados/ej2/` y escribe en
`resultados/ej2/<dataset>/figuras/`:

1. `01_dataset.png`: grilla de muestras del dataset (para banderas, renderizadas).
2. `02_reconstruccion.png`: original vs reconstrucción determinística (train y test).
3. `03_latente_2.png`: dispersión 2D de `μ` coloreada por clase/piloto/jugador y grilla
   decodificada sobre una malla; para banderas, banderas renderizadas.
4. `04_latente_16_pca.png`: PCA (implementado en `core/pca.py`) de `μ` a 2D, dispersión y
   grilla/interpolación decodificada.
5. `05_generacion.png`: muestras de `N(0,I)` y distribución de `nn_dist`.
6. `06_curvas.png`: train/val loss, KL y unidades activas por época, media ± banda.
7. `07_latente_2_vs_16.png`: reconstrucciones y muestras lado a lado.
8. `08_banderas_decodificadas.png`: solo banderas; vectores decodificados renderizados.

Cada figura se genera a partir de artefactos guardados. Ninguna entrena. Si falta un
artefacto, el script falla con un mensaje que indica qué fase correr.

## 9. Tests y criterios de aceptación

Tests nuevos (sin depender de los raw ni del CSV de 120 MB; usan `tmp_path` y fixtures
sintéticas):

- `tests/test_ej2_dedup.py`:
  - aborta sin borrar si un `by_ref` no tiene duplicado;
  - borra los `by_ref` cuando todos son duplicados;
  - deduplica por hash con la prioridad temporada/resolución/ruta;
  - los manifiestos tienen los conteos esperados en el fixture.
- `tests/test_ej2_datos.py`:
  - formas y rangos de los cuatro `.npz` sintéticos;
  - splits disjuntos y no vacíos;
  - `flags` tiene `X.shape == (67, 12)` y no contiene claves de imagen;
  - el escalador de banderas se ajusta solo con train (cambiar val/test no cambia mínimos ni
    rangos);
  - `import ej2.datos` no importa `ej2.banderas`.
- `tests/test_ej2_vae.py`:
  - constructores de imagen y tabulares con las formas correctas;
  - smoke training de 2 épocas con datos sintéticos;
  - `sample(n)` y `encode(X)` con las formas correctas;
  - reproducibilidad por semilla;
  - `k = 2` y `k = 16` construyen y entrenan.
- `tests/test_ej2_banderas.py`:
  - parseo de una fila con comas dentro de los colores;
  - vector de 12 columnas y orden correcto;
  - `render_flag` devuelve una imagen RGB válida, determinística, con las proporciones
    esperadas para una bandera horizontal y una vertical.
- `tests/test_vae.py`: actualizado según §2.

Aceptación:

- `find data/headshots -type d -name by_ref` no devuelve nada.
- Quedan 172 PNG top-level; el dataset F1 tiene 158 imágenes y 29 códigos.
- NBA: 347 imágenes. Chino: 2400. Banderas: `(67, 12)`.
- Formas/rangos de los `.npz` correctos; splits disjuntos.
- El barrido completa sin NaN; `elecciones.json` tiene una decisión por paso.
- Los modelos finales se pueden recargar desde `pesos_semilla_<s>.npz` y reproducen
  `test.csv`.
- `python3 check-cicd.py` pasa; los tests de `ej1`/`ej1b` siguen pasando.
- `python3 -m ej2.graficos` regenera todas las figuras sin entrenar.
- `git status` no muestra cambios en `ej1/`, `resultados/ej1/`, `resultados/ej1b/` ni
  `data/font.h`.

## 10. Orden de ejecución

1. **Línea base**: `git status`, `python3 check-cicd.py`, `python3 -m pytest -q`. Confirmar
   que todo pasa antes de tocar nada.
2. **Spec**: este archivo; actualizar `docs/structure.md` (nuevo `ej2/`, `utils/`,
   `data/procesadas/`, `resultados/ej2/`) y `CLAUDE.md` §6 (agregar el link a este spec).
3. **Núcleo**: cambio de β en `core/vae.py` + tests de `tests/test_vae.py`. CI verde.
4. **Preparación**: implementar `utils/preparar_ej2.py`.
   - Correr `--dedup-f1`; revisar `dedup_headshots.csv` y `dedup_hash.csv`; verificar los
     conteos.
   - Correr `--todo`; revisar manifiestos y hojas de contacto.
5. **Datos y modelos**: `ej2/datos.py`, `ej2/banderas.py`, `ej2/modelos.py`,
   `ej2/config.json`; tests de datos, banderas y modelos. CI verde.
6. **Entrenamiento**: `ej2/vae_exp.py`; smoke con `--trabajadores 1` y 1 semilla × 2 épocas
   por dataset; verificar archivos y formas.
7. **Barrido**: correr los cuatro barridos con `--trabajadores 5`; revisar CSVs y
   `elecciones.json`; documentar divergencias.
8. **Finales**: entrenar `final`, `latente_2`, `latente_16`, `ae_latente_2`,
   `ae_latente_16`; guardar test y muestras.
9. **Figuras**: implementar `ej2/graficos.py`; generar todas; revisar visualmente las de
   banderas y las reconstrucciones.
10. **Cierre**: CI completo, reejecutar tests de `ej1`/`ej1b`, actualizar el estado en este
    spec con los resultados, mostrar `git status` y `git diff --stat`, proponer mensaje de
    commit y esperar permiso.

## 11. Tiempos estimados

Medidos en esta máquina (24 hilos) con 5 workers × 4 hilos BLAS. Los tiempos de imagen se
escalan por cantidad de píxeles y de muestras.

| Dataset | Resolución / d | 5 semillas × 1000 épocas | Barrido | Final + latente 2/16 + AE |
|---|---|---:|---:|---:|
| NBA | 20×24, d=480, n=347 | ~42 s | ~5,3 min | ~4 min |
| F1 | 20×20, d=400, n=158 | ~39 s | ~5 min | ~3 min |
| Chino | 16×16, d=256, n=2400 | ~2,3 min | ~3,6 min | ~6 min |
| Banderas | 12 features, n=67 | < 1 s | < 1 min | < 1 min |

Total aproximado: **~30 min** de cómputo, más implementación y revisión. El único dataset
cerca del presupuesto de 10 minutos es chino; por eso usa 3 semillas en el barrido y en
latente 2/16, y 5 semillas solo en `final`.

## 12. Supuestos confirmados

1. F1: se borran todos los `by_ref` y el dataset principal son las **158 imágenes únicas**
   (no las 29 por piloto). El manifiesto de 29 queda como variante opcional.
2. NBA: se conservan las **347**; solo LeBron lleva recorte 3:4 especial.
3. Chino: **160 por clase = 2400** imágenes, 16×16 gris continuo, sin binarizar la entrada.
4. Banderas: **12 features numéricas**; nunca imágenes como entrada; el render es solo para
   figuras.
5. `data/f1hq` se ignora (ya está en `.gitignore`), no se borra.
6. No se commitea nada; el usuario agrega los `.npz`/PNG procesados y commitea.
7. F1 pasa a usar early stopping (validación de 16 imágenes). El plan original preveía
   épocas fijas porque asumía un dataset de 16–20 imágenes; con 158 confirmadas, la
   validación alcanza para cortar por `val_loss`.
8. Se agrega `pillow==11.3.0` a `requirements.txt` para la preparación de imágenes.

## 13. Estado y resultados

Ejecutado completo el 2026-11-15. Todos los pasos de §10 quedaron corridos y verificados; los
cuatro barridos, los cinco modelos por dataset y las 29 figuras existen en `resultados/ej2/`.

### 13.1 Configuración elegida por validación

| dataset | latente | capacidad | β | lr | pérdida | pasos |
|---|---:|---|---:|---:|---|---:|
| `nba` | 2 | 64–32 | 1.0 | 0.003 | MSE | 5 |
| `f1` | 2 | 64–32 | 0.1 | 0.0003 | BCE | 5 |
| `chino` | 2 | 64–32 | 1.0 | 0.0003 | BCE | 4 (una sola pérdida) |
| `banderas` | 2 | 64–32 | 0.1 | 0.003 | MSE | 4 (una sola pérdida) |

Los cuatro datasets eligieron el latente 2 y la capacidad más chica porque, a menos de un
desvío del mejor ELBO, el criterio de simplicidad prefiere el modelo más chico. Por eso
`latente_2/` reutiliza la corrida de `final/` y solo se entrena aparte `latente_16/`.

### 13.2 Métricas de test (media ± desvío sobre semillas)

MSE de reconstrucción determinística; para banderas está en el espacio escalado.

| dataset | final / latente 2 | latente 16 | AE latente 2 | AE latente 16 |
|---|---:|---:|---:|---:|
| `nba` | 0.0498 ± 0.0002 | 0.0505 ± 0.0011 | 0.0305 ± 0.0032 | 0.0135 ± 0.0004 |
| `f1` | 0.0472 ± 0.0001 | 0.0480 ± 0.0003 | 0.0250 ± 0.0004 | 0.0196 ± 0.0006 |
| `chino` | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 | 0.0008 ± 0.0000 | 0.0005 ± 0.0000 |
| `banderas` | 0.1056 ± 0.0020 | 0.1123 ± 0.0024 | 0.0665 ± 0.0108 | 0.0004 ± 0.0001 |

Unidades activas medias: 0 en los tres datasets de imagen y 0.2 en banderas. `nn_dist` de las
muestras de `N(0, I)` es chico en las imágenes (0.013–0.016 en NBA/F1) y grande en los AE
(0.11–0.23), señal de que el VAE genera la cara media.

### 13.3 Hallazgo: colapso del posterior en las imágenes

Con la normalización de §2 (reconstrucción promediada sobre muestras y píxeles + β · KL
promedio por muestra) el término KL domina por un factor igual a la cantidad de features
(400–480 en NBA/F1, 256 en chino). Resultado medido:

- Los tres datasets de imagen terminaron con **0 unidades activas** en test.
- La reconstrucción del VAE iguala a la imagen media de train: NBA 0.0498 vs 0.0496, F1 0.0472
  vs 0.0470, chino 0.0010 vs 0.0010.
- El paso 3 (β) disparó la regla de excepción en NBA, F1 y chino: ningún β de
  `{0.1, 1, 5}` logró `activas ≥ k/2` y `KL > 0.01`; se eligió el de más unidades activas
  (0) y, entre esos, la menor reconstrucción.
- Banderas (d = 12) no colapsa del todo: 2/2 unidades activas en el barrido y KL 2.12 / 1.11 /
  0.66 para β 0.1 / 1 / 5; en el final de 1000 épocas queda 1 semilla con 1 unidad activa.
- Los AE determinísticos sí usan el latente (0.0135 en NBA con k = 16), así que la diferencia
  es del objetivo β-VAE, no del motor ni de los datos.

Durante el barrido se detectó y corrigió un error del runner: `beta` no se pasaba a
`VariationalAutoencoder` (`kl_weight` quedaba en 1.0), por lo que el paso 3 era un no-op. Ahora
`construir_vae_imagen`/`construir_vae_tabular` reciben `beta`, un test verifica
`kl_weight == beta`, y los cuatro barridos se repitieron con `--forzar`.

No se cambia la normalización: es una decisión del equipo (§ Decisiones). Si se quisiera
evitar el colapso habría que sumar la reconstrucción sobre píxeles o escalar β por `d`, y
repetir barrido y finales.

### 13.4 Figuras

`python3 -m ej2.graficos` regenera 29 archivos (7 por dataset de imagen, 8 en banderas) en
`resultados/ej2/<dataset>/figuras/`:

| archivo | contenido |
|---|---|
| `01_dataset.png` | muestras del dataset (banderas renderizadas) |
| `02_reconstruccion.png` | original vs reconstrucción en train y test |
| `03_latente_2.png` | μ del test y grilla decodificada |
| `04_latente_16_pca.png` | PCA de μ a 2D y grilla decodificada |
| `05_generacion.png` | muestras de `N(0, I)` y distancias a train |
| `06_curvas.png` | pérdidas, KL y unidades activas por época |
| `07_latente_2_vs_16.png` | reconstrucciones y muestras lado a lado |
| `08_banderas_decodificadas.png` | banderas de test, reconstruidas y generadas |

### 13.5 Reproducción y tiempos medidos

```bash
python3 utils/preparar_ej2.py --dedup-f1 --permitir-huerfanos
python3 utils/preparar_ej2.py --todo
python3 -m ej2.vae_exp --dataset nba --fase barrido --trabajadores 5
python3 -m ej2.vae_exp --dataset f1 --fase barrido --trabajadores 5
python3 -m ej2.vae_exp --dataset chino --fase barrido --trabajadores 5
python3 -m ej2.vae_exp --dataset banderas --fase barrido --trabajadores 5
python3 -m ej2.vae_exp --dataset nba --fase final --trabajadores 5
python3 -m ej2.vae_exp --dataset f1 --fase final --trabajadores 5
python3 -m ej2.vae_exp --dataset chino --fase final --trabajadores 5
python3 -m ej2.vae_exp --dataset banderas --fase final --trabajadores 5
python3 -m ej2.graficos
python3 check-cicd.py
```

Barridos: NBA 1m43s, F1 1m27s, chino 6m29s, banderas 3s. Finales: NBA 21s, F1 27s, chino
2m16s, banderas 3s. Figuras: 5s. CI completo: 26s. El barrido es reanudable: al repetir NBA
releyó 20/15/15/15/10 corridas y solo reescribió `elecciones.json`.

### 13.6 Criterios de aceptación

- `find data/headshots -type d -name by_ref` no devuelve nada; quedan 172 PNG top-level y el
  dataset F1 tiene 158 imágenes y 29 códigos.
- NBA 347, chino 2400 (`160 × 15`), banderas `(67, 12)`; conteos por split 277/35/35,
  126/16/16, 1920/240/240 y 54/7/6.
- Los cuatro barridos terminaron sin NaN/Inf y `elecciones.json` tiene una decisión por paso.
- Los 20 modelos finales se recargan desde `pesos_semilla_<s>.npz` y reproducen el
  `recon_mse` de `test.csv` (verificado para la semilla 0 en las 20 carpetas).
- `python3 check-cicd.py` pasa: 53 archivos formateados y toda la suite en verde.
- `ej1/`, `ej1b/`, `resultados/ej1/`, `resultados/ej1b/` y `data/font.h` no tienen cambios.
- `python3 -m ej2.graficos` regenera las 29 figuras sin entrenar.

## 14. Correcciones (2026-11-15)

La primera corrida dejó los tres VAE de imagen colapsados: 0 unidades activas y reconstrucción
igual a la imagen media de train. La causa es la normalización del KL de §2. Este apartado
reemplaza §1.2–§1.4, §2, §5–§7, §8 y §12.6; el historial de arriba queda como registro de lo
que se hizo y este es el estado vigente.

Documentos nuevos:

- [ej2-vae-fix.md](ej2-vae-fix.md): plan de corrección en cuatro fases (especificación,
  conversiones, implementación, reentrenamiento), hiperparámetros y criterios de aceptación.
- [ej2-kl-fix.md](ej2-kl-fix.md): desarrollo teórico de por qué colapsa el VAE, la
  normalización correcta, warm-up, free bits y unidades activas.

### 14.1 Diagnóstico

El motor promedia la reconstrucción sobre muestras **y** features (`n·d`), pero el VAE sumaba
el KL sobre el latente y lo promediaba solo sobre muestras. El gradiente del KL queda
multiplicado por `d` (256–480) respecto del de reconstrucción, así que el encoder aprende
`μ = 0`, `σ = 1` y el decoder emite la imagen media. Evidencia: NBA `KL=9e-05`,
`recon=0.0498` vs baseline 0.0496; F1 `KL=0.0014`, `0.0472` vs `0.0470`; chino `KL≈0`,
`0.0010` vs `0.0010`. Los AE determinísticos sí usan el latente (NBA AE16 `0.0135`), así que
el problema es el objetivo β-VAE, no los datos ni el motor.

### 14.2 Normalización corregida

```text
J = L_recon_promedio + β · KL_promedio / d
  = (1/d) · [ Σ_pixels L_recon + β · KL_promedio ]
```

Es la forma estándar de β-VAE con reconstrucción sumada sobre píxeles. El `1/d` es un factor
global: no cambia el minimizador y hace que `β` no dependa de la resolución. `β = 1` recupera
el ELBO sin ponderar. La métrica `kl_divergence` sigue devolviendo la KL promedio real, sin
dividir por `d`.

### 14.3 Datos y resoluciones

| dataset | antes | ahora | vector `d` | `.npz` |
|---|---|---|---:|---|
| F1 | 20×20 | **32×32** | 1024 | `f1_32x32.npz` |
| NBA | 20×24 | **30×36** (ancho×alto) | 1080 | `nba_30x36.npz` |
| Chino | 16×16, 2400 | **32×32, 20 por clase (300)** | 1024 | `chino_32x32.npz` |
| Banderas | 12 | sin cambio | 12 | `flags.npz` |

- Chino se downsamplea 64×64 → 32×32 con promedio de bloques 2×2, se normaliza por el máximo
  de cada imagen y se invierte (`1 − x`): **fondo blanco y trazo negro**. Split 16/2/2 por
  clase = 240/30/30.
- Los procesados vuelven a `data/procesadas/` y se commitean, igual que antes: los `.npz` y
  PNG procesados van al repo para que el entrenamiento sea reproducible.
- `data/banderas/procesadas/` (la carpeta movida) queda fuera de uso.

### 14.4 Hiperparámetros y antídotos contra el colapso

Barrido: latente `{4, 8, 16}`, capacidad `{128-64, 256-128}`, β `{0.01, 0.1, 1, 5}`,
lr `{3e-4, 1e-3, 3e-3}`, MSE + identidad, 1000 épocas, paciencia 200 sobre `val_recon`.
Warm-up del KL lineal en el 20 % de las épocas y filtro `activas ≥ k/2` y `KL > 0.01` en
**todos** los pasos del barrido, no solo en β. Si aun así colapsa, se sube por una escalera de
cuatro niveles que agrega free bits (`0.05`, `0.10`, `0.20`), warm-ups más largos y más
capacidad; el nivel usado se guarda en `elecciones.json`. El detalle está en
[ej2-vae-fix.md](ej2-vae-fix.md) §3.4.

### 14.5 Figuras

Se agrega `00_resumen.png` como primera figura y la que se usa en la presentación: 8 columnas
por tres filas (original / reconstrucción / muestra de `N(0, I)`). NBA usa LeBron, Dončić,
Jokić y Curry más 4 aleatorios; F1 usa ALO y VER más 6 aleatorios (COL y ANT no están en las
158 imágenes); chino usa 8 clases distintas; banderas usa el mismo renderer de
`utils/generar_banderas.py` para las tres filas. El resto del índice se mantiene, con
`latente_4` en lugar de `latente_2`. `resultados/ej2/README.md` indexa las figuras.
