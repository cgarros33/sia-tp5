# Ejercicio 1b — Denoising autoencoder sobre font.h

## Objetivo

Sobre el mismo dataset del 1a (los 32 patrones de 7×5 de `data/font.h`), entrenar un
autoencoder que reciba una letra con ruido y devuelva la letra limpia (consigna, ej. 1b):

1. Plantear una arquitectura de red adecuada para la tarea y explicar por qué se eligió.
2. Distorsionar las entradas con distintos niveles de ruido y estudiar qué tan bien elimina
   el ruido el autoencoder.

## Decisiones del equipo (no se cambian)

- Ruido: solo bit-flip, con p ∈ {0, 0,05, 0,1, 0,2, 0,3} (ver "Ruido").
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

El ruido gaussiano no se estudia: sobre datos 0/1, recortado a [0, 1] y binarizado, se reduce
a un bit-flip, y agregarlo duplicaría las corridas sin aportar un fenómeno nuevo.

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

- `config_dae.json`: base (la config final del 1a), tipo de ruido, niveles, K, semillas de
  validación y test, grillas de A y B, y paciencia.
- `dae.py`: corre A, B y la evaluación final. Solo escribe en `resultados/ej1b/`: un CSV por
  experimento, `elecciones.json`, y en `final/` los pesos, la historia y `test.csv`. Igual que
  el barrido del 1a, si el CSV de un experimento ya existe, lo saltea.
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

## Fases

| # | Fase | Estado |
|---|------|--------|
| 1 | Spec aprobado, referencia en CLAUDE.md §6, docs/structure.md | hecha |
| 2 | core: `BitFlip` + tests | hecha |
| 3 | ej1: conjuntos ruidosos de validación y test, y referencias + tests | hecha |
| 4 | ej1: config + `dae.py` + smoke test | hecha |
| 5 | ej1: `graficos_dae.py` | hecha |
| 6 | Corrida completa, figuras y resultados en este spec | pendiente |

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
  config final se reentrena con las 5 semillas. Es determinístico: da los mismos pesos que la
  corrida de B, y el smoke test lo verifica.
- **Figuras:** se guardan en `resultados/ej1b/figuras/`: `01_latente.png`,
  `02_ruido_entrenamiento.png`, `03_final_test.png`, `04_ejemplos.png` y
  `05_latente_dae.png`. Las figuras 4 y 5 usan la menor semilla y las letras de test, que se
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

Pendiente.
