# Ejercicio 1b — Denoising autoencoder sobre font.h

## Objetivo

Sobre el mismo dataset del 1a (los 32 patrones de 7×5 de `data/font.h`), entrenar un
autoencoder que reciba una letra con ruido y devuelva la letra limpia (consigna, ej. 1b):

1. Plantear una arquitectura de red adecuada para la tarea y explicar por qué se eligió.
2. Distorsionar las entradas con distintos niveles de ruido y estudiar qué tan bien elimina
   el ruido el autoencoder.

## Decisiones del equipo (no se cambian)

- Ruido: solo bit-flip para entrenar, con p ∈ {0, 0,05, 0,1, 0,2, 0,3} (ver "Ruido"). El
  gaussiano se evalúa sin reentrenar (ver "Evaluación con ruido gaussiano").
- Arquitectura: se estudia la dimensión del latente, con valores {2, 4, 8, 16} (experimento A).
- Corte: `EarlyStopping` sobre `val_loss`, con paciencia de 500, `restore_best` y un máximo de
  10.000 épocas.
- Validación y test son conjuntos ruidosos separados, con semillas 1000 y 2000 y K = 50 copias
  por letra y por nivel.
- Además de los p fijos, se prueba un nivel de ruido variable (`mixto`, experimento B).

## Qué se reutiliza del 1a

- Pérdida BCE con salida sigmoidea, y la métrica de **píxeles mal**: se binariza la salida
  (≥ 0,5 → 1) y se compara con la letra **limpia**. Una letra cuenta como **recuperada** si
  le quedan ≤ 1 píxel mal, el mismo umbral que pide el 1a.
- Punto de partida: la config final del 1a (`resultados/ej1/final/config.json`): ocultas
  60-40-20 con el decoder espejado, tanh, latente lineal, He, Adam con lr 0,01, lote
  completo, BCE y sin L2.
- Semillas 0 a 4. Todo resultado se reporta como media ± desvío (ddof = 1).
- El código para armar el modelo desde la config: `ej1.barrido.armar_autoencoder`, `aplicar`
  y `cargar_modelo`, y `cargar_font`.

## Entrenamiento

El objetivo es siempre la letra limpia: `ae.fit(X, X, augment=BitFlip(p_train), ...)`. Hoy
`fit` aplica `augment` solo a la entrada de los lotes de entrenamiento y deja el objetivo
sin tocar (`core/network.py`), y lo hace con un generador hijo del de la red. Por eso el ruido
se sortea de nuevo en cada época: la red nunca ve dos veces la misma versión ruidosa de una
letra. No hace falta tocar `Network` ni `Autoencoder`.

## Ruido

**Bit-flip (sal y pimienta).** Cada píxel se invierte (0 ↔ 1) con probabilidad p,
de manera independiente. Elegimos este ruido porque los datos son binarios: el ruido
gaussiano deja valores que no son ni 0 ni 1, y si se lo recorta a [0, 1] y se binariza,
termina siendo un bit-flip con una probabilidad que depende de σ. Con bit-flip el nivel de
ruido se lee directo en píxeles: p · 35 píxeles mal esperados por letra.

| p | 0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| Píxeles invertidos esperados | 0 | 1,75 | 3,5 | 7 | 10,5 |

Con p = 0,5 la entrada es ruido puro y no tiene información de la letra, así que no se usa.

El ruido gaussiano no se usa para entrenar, pero **sí se evalúa**: no se binariza la entrada,
así que el DAE recibe valores continuos y el problema no es el mismo que un bit-flip. El DAE
se entrenó solo con bit-flip (p ∈ {0; 0,05; 0,1; 0,2; 0,3}) y `ej1/gaussiano.py` mide cuánto
generaliza a σ ∈ {0; 0,1; 0,2; 0,3; 0,4; 0,5} sin reentrenar. Los resultados están en
"Evaluación con ruido gaussiano".

**Nivel variable (`mixto`):** a cada letra de cada lote se le sortea su propio
p ~ U(0; 0,3), y el sorteo se repite en cada época. Tiene sentido porque, con bit-flip, p < 0,5
y las 32 letras igual de probables, el denoiser óptimo (el vecino más cercano en Hamming) es el
mismo para cualquier p. Entonces un único modelo puede ser bueno en todos los niveles.
Entrenar con un p fijo ajusta la red a ese nivel, mientras que la mezcla le muestra todos los
niveles en cada lote. Se elige con la misma regla que los p fijos.

## Conjuntos de evaluación

Acá las 32 letras limpias son las de entrenamiento, y lo que se generaliza son las
realizaciones del ruido. Por eso la regla 2 de CLAUDE.md sí aplica: se separan dos conjuntos
de letras ruidosas, generados con semillas fijas distintas de las de entrenamiento.

- **Validación:** K = 50 copias ruidosas de cada letra para cada p de evaluación (1.600 por
  nivel). Se generan con `np.random.default_rng(1000)`. Con este conjunto se toman todas las
  decisiones: arquitectura, p de entrenamiento y épocas.
- **Test:** K = 50 copias por letra y por nivel, generadas con `np.random.default_rng(2000)`.
  Se lee solo en la evaluación final.
- Niveles de evaluación: p_test ∈ {0, 0,05, 0,1, 0,2, 0,3}.

Las semillas 1000 y 2000 son fijas y quedan anotadas en la config. Los conjuntos no dependen
de la semilla del modelo, así que todos los modelos se evalúan sobre las mismas letras
ruidosas.

## Cuándo se corta el entrenamiento

Con ruido no hay un "llega" exacto como en el 1a. Se usa `EarlyStopping` sobre la
pérdida de validación (`val_loss`), con `restore_best=True` y paciencia de 500 épocas. La
validación que se le pasa a `fit` es la del nivel p = p_train. Para p_train = 0 se usan las
32 letras limpias, y para `mixto`, la unión de la validación de todos los p_test > 0. El máximo es de 10.000 épocas, como en el 1a.

## Referencias de comparación

Cada modelo se compara contra tres referencias, todas evaluadas en los mismos conjuntos:

1. **No hacer nada:** devolver la entrada ruidosa. Tiene ≈ 35p píxeles mal. Un DAE que no le
   gana a esto no limpia.
2. **AE del 1a:** la config final del 1a, entrenada sin ruido. Se cargan los pesos ya
   guardados en `resultados/ej1/final/`, sin reentrenar. Muestra cuánto aporta entrenar con
   ruido.
3. **Vecino más cercano (techo):** devolver la letra del set que está a menor distancia de
   Hamming de la entrada. Con bit-flip independiente, p < 0,5 y las 32 letras igual de
   probables, este es el mejor denoiser posible (MAP). Ningún modelo puede superarlo en promedio,
   y sirve para ver cuánto se acerca el DAE.

## Experimentos

Los experimentos van en orden, como en el 1a: se elige el mejor valor de cada uno y queda fijo
para el siguiente.

**A. Arquitectura, que es la parte 1b-1: dimensión del latente.** Se prueban latente ∈ {2, 4,
8, 16}, con p_train = 0,1 y el resto igual a la config final del 1a. La hipótesis a contrastar
es esta: un latente chico obliga a proyectar la entrada sobre la "variedad" de las 32 letras,
y eso descarta el ruido. Un latente grande reconstruye mejor la entrada, pero puede dejar
pasar parte del ruido. Esta comparación es la que justifica la arquitectura elegida.

**B. Nivel de ruido de entrenamiento.** Con el latente elegido en A, se prueba p_train ∈ {0,
0,05, 0,1, 0,2, 0,3, `mixto`}. Cada modelo se evalúa en todos los p_test, lo que da una matriz
p_train × p_test.

Cómo se elige en A y en B, siempre con el conjunto de **validación**:

1. Menor promedio de píxeles mal, promediado sobre los p_test ∈ {0,05, 0,1, 0,2, 0,3}
   (p_test = 0 no se cuenta).
2. Si empatan: más % de letras recuperadas sobre esos mismos niveles.
3. Si siguen empatados: se queda el valor actual de la config (en A, el latente 2 del 1a; en
   B, p = 0,1); si no está entre los empatados, el primero de la grilla.

Son 20 corridas en A y 30 en B, más la evaluación de las referencias.

**Final.** Con la config elegida se reportan los resultados en el conjunto de **test**: para
cada p_test, los píxeles mal y el % de letras recuperadas, con las 5 semillas, junto a las
tres referencias.

Por cada corrida se guarda: experimento, valor, semilla, config completa, épocas, época de
la mejor validación, segundos y estado. Además, para cada p_test del conjunto de validación:
el promedio de píxeles mal, la peor letra y el % de letras recuperadas.

## Piezas

En `core/`, genérico y con tests:

- `augmentation.py`: `BitFlip(p)`, registrada como `"bit_flip"`. Invierte cada elemento con
  probabilidad p y supone datos en {0, 1}. `p` puede ser un float (nivel fijo) o un par
  `(p_min, p_max)`: en ese caso cada fila sortea su propio p ~ U(p_min, p_max). Tests:
  - p = 0 deja la entrada igual y p = 1 la invierte entera.
  - La tasa empírica de píxeles invertidos se acerca a p.
  - Con un par, cada fila usa un p distinto y la tasa media se acerca a (p_min + p_max) / 2.
  - No modifica X in place.
  - Con el mismo generador da siempre el mismo resultado.

En `ej1/`:

- `config_dae.json`: base (la config final del 1a, más `copias_epoca` y `monitor`), tipo de
  ruido, niveles, K, semillas de validación y test, grillas de A, B y C, y paciencia.
- `dae.py`: corre A, B, C y la evaluación final. Solo escribe en `resultados/ej1b/`: un CSV
  por experimento, `elecciones.json`, y en `final/` los pesos, la historia y `test.csv`. Igual
  que el barrido del 1a, si el CSV de un experimento ya existe, lo saltea.
- `gaussiano.py`: evalúa el DAE final con ruido gaussiano, sin reentrenar; escribe
  `resultados/ej1b/gaussiano/test.csv`.
- `graficos_dae.py`: solo lee de `resultados/ej1b/` y guarda en `resultados/ej1b/figuras/`.

## Figuras

1. Píxeles mal (media ± desvío) contra p_test: una curva por latente del experimento A, y las
   tres referencias en línea punteada.
2. Mapa de calor p_train (incluido `mixto`) × p_test con los píxeles mal promedio del experimento B (validación).
3. Resultado final en test: píxeles mal y % de letras recuperadas contra p_test, el DAE
   elegido frente a las tres referencias.
4. Ejemplos visuales: para algunas letras y cada p_test, tres filas con la letra limpia, la
   letra con ruido y la reconstrucción del DAE.
5. Espacio latente del DAE con latente 2 del experimento A, aunque no gane: las letras
   ruidosas coloreadas por letra de origen y unidas a ella con un segmento, con los códigos de
   las letras limpias encima. Muestra si el encoder agrupa las versiones ruidosas cerca de su
   letra.
6. Ruido gaussiano: píxeles mal y % de letras recuperadas contra σ, el DAE final contra las
   tres referencias.

## Fases

| # | Fase | Estado |
|---|------|--------|
| 1 | Spec aprobado, referencia en CLAUDE.md §6, docs/structure.md | hecha |
| 2 | core: `BitFlip` + tests | hecha |
| 3 | ej1: conjuntos ruidosos de validación y test, y referencias + tests | hecha |
| 4 | ej1: config + `dae.py` + smoke test | hecha |
| 5 | ej1: `graficos_dae.py` | hecha |
| 6 | Corrida completa, figuras y resultados en este spec | hecha |

## Supuestos

- El "mismo dataset" de la consigna son las 32 letras de font.h. No se agregan letras.
- El DAE no tiene que aprender a reconstruir con 2 dimensiones de latente: la consigna del 1b
  no lo exige. Por eso la dimensión del latente es la variable de arquitectura que se estudia.
- Con lote completo, cada época ve 32 letras ruidosas, una por letra. Si eso hace que el
  entrenamiento sea muy lento o muy ruidoso, se puede repetir X k veces por época. Eso se
  decide después de la fase 4 y queda anotado acá.
- **Conjuntos ruidosos:** un solo generador, `default_rng(semilla)`, recorre los niveles en el
  orden de la config. Las k copias de cada letra van intercaladas (`np.tile`), y un vector
  `idx` dice de qué letra limpia sale cada fila. Todo esto vive en `ej1/dae.py`
  (`conjuntos_ruidosos`, `medir_limpieza`, `vecino_mas_cercano`, `cargar_ae_1a`).
- **Vecino más cercano:** en un empate de distancia gana la letra de menor índice. El vecino más
  cercano no es infalible ni con poco ruido: «l» y «|» están a solo 2 píxeles, así que un único
  píxel invertido puede dejar la entrada a la misma distancia de las dos.
- **Config del 1b:** `config_dae.json` copia la config final del 1a, salvo `decaimiento`
  (era null y no se usa), y le agrega `paciencia` y `ruido`. `ruido` es un float (p fijo) o un
  par `[p_min, p_max]` (mixto). Un test verifica que la base coincide con
  `resultados/ej1/final/config.json`.
- **Salidas de `dae.py`:** `experimento_A_latente.csv` y `experimento_B_ruido.csv` (una fila
  por valor × semilla, con columnas `val_<medida>_<p>` para cada nivel);
  `referencias_validacion.csv` (nada, vecino y AE del 1a en validación, para la figura 1);
  `elecciones.json`; y `final/` con `config.json`, pesos e historia por semilla, y
  `test.csv` (modelo, semilla, nivel y medidas, para el DAE y las tres referencias). La
  config final se reentrena con las 5 semillas. Es determinístico: los pesos finales
  reproducen la corrida del experimento que fijó la config, y el smoke test lo verifica.
- **Figuras:** se guardan en `resultados/ej1b/figuras/`: `01_latente.png`,
  `02_ruido_entrenamiento.png`, `03_final_test.png`, `04_ejemplos.png`,
  `05_latente_dae.png` y `06_gaussiano.png` (esta última solo si existe
  `gaussiano/test.csv`). Las figuras 4 y 5 usan la menor semilla y las letras de test, que se
  regeneran con su semilla fija. La figura 4 usa el DAE final y muestra la primera copia
  ruidosa de «a e g k m s l |» con cada p > 0. La figura 5 usa el latente 2 del experimento A,
  con p = 0,1. Para eso, A tiene `"guardar_pesos": ["2"]` en la config, y `dae.py` guarda esos
  pesos en `pesos/A_2_semilla_<s>.npz`, porque el script de gráficos no entrena.
- **Sin currículum:** no se prueba un p que suba o baje con las épocas. Habría que barrer
  también ese schedule, y el p sorteado por muestra (`mixto`) ya pone todos los niveles en
  cada lote.
- Un píxel ruidoso que coincide por azar con la letra limpia no se distingue de uno sin
  ruido: la métrica siempre compara contra la letra limpia.

## Resultados

Corrida completa: 50 corridas de DAE (20 en A y 30 en B), más el reentrenamiento final con 5
semillas. Fueron unos 52 minutos de entrenamiento en secuencial, sin ninguna corrida divergida.
Todo sale de `resultados/ej1b/`. Píxeles mal: media ± desvío sobre semillas. "Promedio" es el
promedio sobre los p de validación > 0, que es el criterio de elección.

### Experimento A: dimensión del latente (p_train = 0,1, validación)

| Latente | Promedio | Recuperadas | p = 0 | 0,05 | 0,1 | 0,2 | 0,3 | Épocas |
|---|---|---|---|---|---|---|---|---|
| 2 | 3,78 ± 0,24 | 59,6 ± 6,7 % | 0,40 | 0,56 | 1,06 | 4,29 | 9,22 | 2516 ± 797 |
| 4 | 3,24 ± 0,08 | 67,6 ± 1,0 % | 0,01 | 0,13 | 0,62 | 3,64 | 8,56 | 1807 ± 120 |
| **8** | **3,06 ± 0,13** | **68,7 ± 0,8 %** | 0,03 | 0,13 | 0,53 | 3,33 | 8,26 | 1556 ± 241 |
| 16 | 3,07 ± 0,18 | 68,1 ± 1,8 % | 0,03 | 0,13 | 0,55 | 3,33 | 8,25 | 1476 ± 226 |

Gana el latente 8. La hipótesis del spec (un latente chico filtra mejor el ruido) **no se
cumple**: el latente 2 es el peor en todos los niveles. Con 2 dimensiones la red ni siquiera
reconstruye bien las letras limpias (0,40 píxeles mal con p = 0), porque el corte mira la
validación ruidosa y no las limpias. A partir del latente 4 el resultado se satura: 8 y 16
empatan dentro del desvío (3,06 contra 3,07), y 8 gana por la regla.

### Experimento B: ruido de entrenamiento (latente 8, validación)

| p_train | Promedio | Recuperadas | p = 0 | 0,05 | 0,1 | 0,2 | 0,3 | Épocas |
|---|---|---|---|---|---|---|---|---|
| 0 | 6,48 ± 0,42 | 34,7 ± 0,9 % | 0,00 | 1,78 | 4,06 | 8,26 | 11,82 | 10.000 (no corta) |
| 0,05 | 3,55 ± 0,13 | 63,9 ± 0,8 % | 0,00 | 0,16 | 0,86 | 4,20 | 8,98 | 1355 ± 351 |
| 0,1 | 3,06 ± 0,13 | 68,7 ± 0,8 % | 0,03 | 0,13 | 0,53 | 3,33 | 8,26 | 1556 ± 241 |
| **0,2** | **2,72 ± 0,10** | **66,0 ± 3,4 %** | 0,28 | 0,38 | 0,71 | 2,66 | 7,12 | 2349 ± 730 |
| 0,3 | 4,15 ± 0,06 | 28,7 ± 2,1 % | 2,29 | 2,45 | 2,74 | 4,25 | 7,15 | 2254 ± 248 |
| mixto | 2,88 ± 0,08 | 65,4 ± 1,7 % | 0,07 | 0,21 | 0,65 | 3,14 | 7,52 | 2088 ± 273 |

Gana p_train = 0,2.

- **Cada p_train rinde mejor cerca de su propio nivel** (la diagonal de la figura 2). Entrenar
  con poco ruido deja a la red indefensa con mucho ruido, y entrenar con mucho ruido la vuelve
  menos precisa con poco: con p_train = 0,3 hay 2,29 píxeles mal sobre letras limpias.
- **p_train = 0 (sin ruido) no limpia nada:** con p = 0,1 deja 4,06 píxeles mal, más que la
  entrada (3,5). Además, nunca corta: su validación son las letras limpias, que sigue
  mejorando hasta las 10.000 épocas.
- **Mixto:** queda segundo (2,88 contra 2,72) y es el más parejo entre niveles. Le gana a 0,2
  con p ≤ 0,1 (0,07 / 0,21 / 0,65 contra 0,28 / 0,38 / 0,71) y pierde con p ≥ 0,2. Que gane 0,2
  depende del criterio: al promediar píxeles mal, los niveles altos (que tienen muchos más
  errores) pesan más en el promedio y favorecen al p_train alto. Si importaran más los niveles
  bajos, mixto sería la mejor opción.

### Config final

Encoder 35 → 60 → 40 → 20 → **8** y decoder espejado; tanh en las ocultas, latente lineal,
salida sigmoidea, He, Adam con lr 0,01, lote completo, BCE y sin L2. Se entrena con bit-flip de
**p = 0,2** y se corta con paciencia 500. Corta entre las épocas 1656 y 3467
(`resultados/ej1b/final/config.json`).

### Resultado final en test

Píxeles mal por letra · % de letras recuperadas (≤ 1 píxel mal). DAE y AE del 1a: media ±
desvío sobre 5 semillas. Las otras dos referencias son determinísticas.

| Modelo | p = 0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| No hacer nada | 0,00 · 100 % | 1,74 · 48 % | 3,50 · 11 % | 7,14 · 0 % | 10,58 · 0 % |
| AE del 1a | 0,10 ± 0,03 · 100 % | 4,62 ± 0,48 · 46 % | 7,50 ± 0,41 · 24 % | 11,14 ± 0,51 · 9 % | 13,28 ± 0,25 · 5 % |
| **DAE** | 0,28 ± 0,16 · 92 % | **0,36 ± 0,11 · 91 %** | **0,72 ± 0,11 · 85 %** | **2,99 ± 0,11 · 59 %** | **7,40 ± 0,08 · 28 %** |
| Vecino más cercano (techo) | 0,00 · 100 % | 0,01 · 100 % | 0,12 · 98 % | 1,43 · 85 % | 5,88 · 51 % |

El test confirma lo que mostraba la validación: no hay sobreajuste a los conjuntos ruidosos de
validación.

### Observaciones

- **El AE del 1a no sirve para limpiar:** con cualquier ruido da peor que devolver la entrada.
  Su latente de 2 dimensiones lleva una letra con 2 o 3 píxeles cambiados a otra letra entera.
- **El DAE limpia:** con p = 0,1, de 3,5 píxeles mal en la entrada pasa a 0,72, y recupera el
  85 % de las letras contra el 11 % de la entrada. Con p = 0,3 sigue mejorando la entrada (7,40
  contra 10,58), pero recupera pocas letras.
- **Queda lejos del techo** con ruido alto (con p = 0,2, 2,99 píxeles mal contra 1,43 del vecino
  más cercano): la red no aprende a "elegir una letra" tan bien como una búsqueda exhaustiva
  sobre las 32.
- **Costo con letras limpias:** el DAE final recupera solo el 92 ± 5 % de las letras sin ruido,
  contra el 100 % del AE del 1a. Las que falla varían con la semilla: «`», «r», «|», «o» y
  otras, con 2 a 4 píxeles mal. Entrenar para limpiar ruido le resta precisión para copiar.
- **«l» y «|»** (a 2 píxeles): en la figura 4, el DAE de la semilla 0 devuelve «l» cuando le
  llega una «|» con p ≤ 0,2. En el latente 2 (figura 5), sus códigos casi coinciden.
- **Latente 2 (figura 5):** las letras ruidosas quedan, en su mayoría, cerca del código de su
  letra limpia. Las que saltan lejos caen cerca de otra letra, y esos saltos son los errores.

## Corrección (2026-11-16): elegir por letras recuperadas

Reemplaza la regla de "Cómo se elige en A y en B". Los resultados de arriba (p_train = 0,2)
quedan archivados en `resultados/ej1b/final_p02/` (final, figuras y `elecciones.json`).

**Por qué.** El promedio de píxeles mal sobre los p de validación lo dominan los niveles
altos: con p = 0,3 hay ~8 píxeles mal y con p = 0,05 menos de 1. Eso favorecía al p_train
alto (0,2), que deja de copiar bien las letras limpias (92 % recuperadas sin ruido). El % de
letras recuperadas pesa igual cada nivel y usa el umbral de la consigna (≤ 1 píxel mal).

**Regla nueva** (en A y en B, sobre validación): 1. más % de letras recuperadas promediado
sobre p > 0; 2. si empatan, menos píxeles mal; 3. si siguen empatados, el valor actual o el
primero de la grilla. Con los CSV ya guardados (no se reentrenó A ni B):

- A: sigue ganando el latente 8 (68,7 % contra 68,1 % del 16).
- B: gana **p_train = 0,1** (68,7 %) en lugar de 0,2 (66,0 %); mixto queda 65,4 %.

**Resultado en test** (píxeles mal por letra · % recuperadas; media sobre 5 semillas):

| Modelo | p = 0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| DAE p_train = 0,2 (antes) | 0,28 · 92 % | 0,36 · 91 % | 0,72 · 85 % | 2,99 · 59 % | 7,40 · 28 % |
| **DAE p_train = 0,1** | **0,03 · 99 %** | **0,13 · 97 %** | **0,57 · 90 %** | 3,55 · 59 % | 8,58 · 25 % |
| Vecino más cercano (techo) | 0,00 · 100 % | 0,01 · 100 % | 0,12 · 98 % | 1,43 · 85 % | 5,88 · 51 % |
| No hacer nada | 0,00 · 100 % | 1,74 · 48 % | 3,50 · 11 % | 7,14 · 0 % | 10,58 · 0 % |

- Con p ≤ 0,1 el DAE nuevo casi alcanza al techo y ya no rompe las letras limpias (99 %
  recuperadas; antes 92 %). En la figura 4, la «|» limpia ahora sale bien.
- Con p = 0,3 recupera menos (25 % contra 28 %) y deja más píxeles mal: es el costo de
  entrenar con menos ruido. Con p = 0,2 las recuperadas no cambian (59 %).
- La brecha con el techo en ruido alto sigue abierta; es lo que estudia el experimento C.

## Experimento C (2026-11-16): arquitectura y copias por época

Se probaron las ocultas {60-40-20, 128-64, 128-128-64} × `copias_epoca` {1, 8, 16}, con
latente 8, p_train = 0,1, lote 32 y 5 semillas (45 corridas). Con `copias_epoca = k` cada
época entrena sobre `np.tile(X, k)`; el bit-flip se sortea de nuevo en cada lote, así que la
red ve k versiones ruidosas distintas de cada letra por época. Se elige con la misma regla:
más letras recuperadas y, si empatan, menos píxeles mal. Todo sale de
`resultados/ej1b/experimento_C_arquitectura_y_copias.csv`.

Letras recuperadas (%) · píxeles mal por letra, media ± desvío sobre 5 semillas:

| Ocultas | k = 1 | k = 8 | k = 16 |
|---|---|---|---|
| 60-40-20 | 68,0 ± 0,8 · 3,037 ± 0,122 | **69,6 ± 0,9 · 3,008 ± 0,088** | 68,3 ± 0,7 · 3,093 ± 0,195 |
| 128-64 | 67,0 ± 1,1 · 2,919 ± 0,098 | 68,2 ± 0,9 · 2,903 ± 0,065 | 68,3 ± 1,0 · 2,965 ± 0,128 |
| 128-128-64 | 65,8 ± 0,5 · 3,213 ± 0,045 | 65,3 ± 0,7 · 3,252 ± 0,062 | 65,2 ± 1,3 · 3,275 ± 0,134 |

Gana **60-40-20 con k = 8** (69,6 % contra 68,0 % del k = 1), así que la config final cambia
solo en `copias_epoca`: el resto (latente 8, p_train = 0,1, lote 32) queda igual. Las
arquitecturas más anchas y la más profunda son peores: no compensan con más capacidad lo que
el 1a ya había elegido para memorizar. Más copias ayudan poco (1,6 puntos con k = 8) y k = 16
vuelve a bajar; la brecha con el vecino más cercano en ruido alto sigue abierta.

El final anterior (k = 1) quedó archivado en `resultados/ej1b/final_c0/`. Test del final
nuevo (píxeles mal · % recuperadas, media sobre 5 semillas):

| Modelo | p = 0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| DAE k = 1 (antes) | 0,03 · 99 % | 0,13 · 97 % | 0,57 · 90 % | 3,55 · 59 % | 8,58 · 25 % |
| **DAE k = 8 (ahora)** | **0,00 · 100 %** | **0,08 · 98 %** | **0,50 · 92 %** | **3,50 · 60 %** | **8,51 · 26 %** |
| Vecino más cercano | 0,00 · 100 % | 0,01 · 100 % | 0,12 · 98 % | 1,43 · 85 % | 5,88 · 51 % |
| No hacer nada | 0,00 · 100 % | 1,74 · 48 % | 3,50 · 11 % | 7,14 · 0 % | 10,58 · 0 % |

La mejora es chica pero consistente: con k = 8 el DAE ya no rompe ninguna letra limpia
(0,00 píxeles mal) y gana 1–2 puntos de recuperadas en casi todos los niveles. No alcanza para
cerrar la brecha con el vecino más cercano con p = 0,2–0,3.

## Evaluación con ruido gaussiano (2026-11-16)

El DAE se entrenó solo con bit-flip, pero la consigna pide distorsionar las entradas en
distintos niveles. Se evalúa sin reentrenar con `conjuntos_gaussianos`: `X[idx] + N(0, σ²)`
recortado a [0, 1] y **sin binarizar**. σ ∈ {0; 0,1; 0,2; 0,3; 0,4; 0,5}, K = 50 copias por
letra, semilla 3000 (distinta de validación 1000 y test 2000). La entrada es continua, así
que es otro problema: la métrica binariza la salida en 0,5 y muchos píxeles perturbados no
llegan a cambiar de lado. Las referencias son devolver la entrada binarizada en 0,5 ("nada")
y el vecino más cercano sobre esa entrada. Sale de `resultados/ej1b/gaussiano/test.csv`;
figura `06_gaussiano.png`.

Píxeles mal por letra · % recuperadas (media ± desvío sobre 5 semillas; las referencias son
determinísticas):

| Modelo | σ = 0 | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 |
|---|---|---|---|---|---|---|
| No hacer nada | 0,00 · 100 % | 0,00 · 100 % | 0,23 · 98 % | 1,73 · 48 % | 3,67 · 10 % | 5,59 · 2 % |
| AE del 1a | 0,10 ± 0,03 · 100 % | 0,97 ± 0,14 · 80 % | 3,27 ± 0,27 · 51 % | 5,79 ± 0,42 · 32 % | 8,09 ± 0,50 · 19 % | 9,74 ± 0,54 · 14 % |
| **DAE** | 0,00 · 100 % | 0,00 · 100 % | 0,01 · 100 % | 0,05 ± 0,01 · 99 % | 0,28 ± 0,02 · 94 % | 1,12 ± 0,06 · 84 % |
| Vecino más cercano | 0,00 · 100 % | 0,00 · 100 % | 0,00 · 100 % | 0,03 · 99 % | 0,16 · 97 % | 0,58 · 92 % |

El DAE generaliza bien aunque nunca vio ruido gaussiano: con σ = 0,5 pasa de 5,59 píxeles mal
en la entrada a 1,12, y recupera 84 % de las letras contra 2 % de la entrada. Queda cerca del
vecino más cercano (0,58 · 92 %) y muy por encima del AE del 1a. La conclusión es que el
bit-flip no es el único ruido que el DAE aprende a limpiar; el gaussiano, al ser continuo,
es más fácil de corregir con el umbral de 0,5, no más difícil.

## Notas para la presentación (2026-11-16)

- **El latente satura desde 4, no "gana 8".** Con 4, 8 y 16 el promedio de píxeles mal queda
  en 3,24, 3,06 y 3,07; 8 y 16 empatan dentro del desvío y la regla elige 8. La conclusión es
  que a partir de 4 dimensiones el resultado se satura, no que 8 sea mejor que 16.
- **p_train = 0** entrena las 10.000 épocas sin cortar (su validación son las letras limpias,
  que siguen mejorando). Es, en la práctica, un AE con latente 8: copia bien las letras
  limpias (0,00 píxeles mal con p = 0) pero no limpia ruido.
