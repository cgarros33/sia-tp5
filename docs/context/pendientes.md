# Pendientes del TP5 — plan para ejecutar

Lo que quedó sin hacer después de la revisión del 2026-11-16 (rama `fix/ej2-generacion`).
Cada tarea es independiente, salvo donde se aclara. Están en orden de prioridad.

## Antes de empezar

1. `git fetch && git checkout fix/ej2-generacion` (o la rama a la que se haya mergeado).
2. `python3 check-cicd.py` tiene que dar "Todo en verde".
3. Leer `CLAUDE.md`. Lo que más importa acá:
   - **No borrar resultados.** Si una corrida reemplaza otra, primero se mueve la carpeta
     vieja (ej.: `resultados/ej1b/final/` → `resultados/ej1b/final_<algo>/`). Así se hizo con
     `resultados/ej2_beta001/` y `resultados/ej1b/final_p02/`.
   - **Tests primero**, después código; `check-cicd.py` verde antes de commitear.
   - **Commits de una sola línea**, sin menciones de autoría ni de IA.
   - Comentarios al mínimo, en español.
   - Al terminar cada tarea, agregar una sección "Corrección/Experimento (fecha)" al spec del
     ejercicio con los números medidos, no reescribir el historial.
4. Si una tarea obliga a elegir algo que este plan no fija, preguntarle al equipo antes.

Estado de partida, para no rehacerlo: la generación del VAE ya funciona (β por tolerancia y
corte después del warm-up), el 1b ya elige por letras recuperadas (gana p_train = 0,1), el 1a
ya tiene `final_cero/` (0 píxeles mal) y la verificación con 20 semillas. Detalle en
`ej2-vae-fix.md` §8, `ej1b-denoising.md` y `ej1-autoencoder.md` (secciones del 2026-11-16).

---

## 1. Decidir qué pasa con el dataset chino (ej. 2) — decisión del equipo

> **Resuelto (2026-11-16):** se intentó la opción B con 100 imágenes por clase,
> binarización y BCE, pero la reconstrucción quedó en 0,87 × la imagen media (se exigía
> < 0,8). Chino queda fuera de la entrega. Detalle en `ej2-vae-fix.md` §10.

**Problema.** No aprende: la reconstrucción de validación (0,0085) es igual a devolver la
imagen media (0,00856), y las reconstrucciones son la misma mancha gris para las 15 clases.
Se pasó de 2400 a 300 imágenes en 32×32. Además sus resultados son de la regla de β vieja
(β = 0,01): nunca se volvió a correr.

**Opciones.**

- **A (recomendada): sacarlo de la entrega.** La consigna pide un dataset; NBA, F1 y
  banderas ya cubren el ejercicio. Pasos: no mostrarlo en la presentación y anotarlo en
  `ej2-vae-fix.md` como hallazgo ("un MLP con latente 4 no aprende 300 caracteres de 32×32").
  No hace falta borrar `resultados/ej2/chino/`.
- **B: arreglarlo.** En `utils/preparar_ej2.py` volver a ≥ 100 imágenes por clase, binarizar
  la entrada y entrenar con salida sigmoide + BCE (`"perdidas": ["binary_crossentropy"]` en
  `ej2/config.json`). Endurecer `_evaluar_nivel` en `ej2/vae_exp.py` para exigir
  `recon < 0,8 × baseline` en vez de `recon < baseline` (con test). Archivar
  `resultados/ej2/chino/` antes de correr. Costo: ~1 h de cómputo.

**Hecho cuando:** la decisión está anotada en el spec y, si es B, la reconstrucción de
validación es menor a 0,8 × la imagen media y `00_resumen.png` muestra caracteres legibles.

---

## 2. B3 — Evaluar el DAE del 1b con ruido gaussiano

**Problema.** `ej1b-denoising.md` (sección "Ruido") dice que no se estudia ruido gaussiano
porque "recortado a [0, 1] y binarizado se reduce a bit-flip". Es falso: nadie binariza la
entrada; el DAE recibe valores continuos, que es otro problema. La consigna pide "distorsionar
las entradas en diferentes niveles", así que conviene medirlo.

**Qué hacer** (solo evaluación: no se reentrena nada).

1. Test en `tests/test_ej1_dae.py` para una función nueva `conjuntos_gaussianos(X, sigmas, k,
   semilla)` en `ej1/dae.py`, análoga a `conjuntos_ruidosos`: `X[idx] + N(0, σ²)`, recortado a
   [0, 1], sin binarizar. Tests: σ = 0 deja X igual; mismas formas e `idx` que
   `conjuntos_ruidosos`; misma semilla, mismo resultado.
2. Script `ej1/gaussiano.py` (o una función en `dae.py`, a elegir con el equipo) que carga el
   DAE final (`resultados/ej1b/final/pesos_semilla_<s>.npz`, con `cargar_modelo`) y el AE del
   1a (`cargar_ae_1a`), y los evalúa con `evaluar(...)` sobre σ ∈ {0; 0,1; 0,2; 0,3; 0,4; 0,5},
   K = 50 copias, semilla 3000 (distinta de validación 1000 y test 2000). Referencias: "no
   hacer nada" (devolver la entrada binarizada a 0,5) y el vecino más cercano sobre la entrada
   binarizada. Escribe `resultados/ej1b/gaussiano/test.csv`.
3. Figura en `ej1/graficos_dae.py`: `06_gaussiano.png`, píxeles mal y % recuperadas contra σ,
   con el mismo estilo que `fig_final` (se puede reusar `curvas_por_modelo`).
4. Reescribir el párrafo del spec: el gaussiano es otro tipo de ruido (entrada continua); se
   entrenó solo con bit-flip y acá se mide cuánto generaliza.

**Hecho cuando:** existe la figura, la tabla está en el spec y el párrafo incorrecto quedó
corregido. Cómputo: segundos.

---

## 3. Experimento C del 1b — acercarse al vecino más cercano con ruido alto

**Problema.** Con p = 0,2 el DAE recupera 59 % de las letras contra 85 % del vecino más
cercano (techo). Nunca se estudió por qué. Hipótesis: (1) cada época ve una sola versión
ruidosa de cada letra (32 ejemplos por época); (2) la arquitectura se heredó del 1a, que se
ajustó para memorizar, no para limpiar; (3) el corte mira `val_loss` (BCE) y no la métrica con
la que se elige.

**Cómo justificarlo en la presentación** (la pregunta de "por qué barren otra cosa que en el
1a"): se hereda del 1a todo lo que no depende de la tarea (activación, inicialización,
optimizador, lr, pérdida, L2) y se vuelve a estudiar solo lo que la consigna pide (1b-1:
"arquitectura conveniente para esta tarea") y lo que el ruido introduce (cuántas
realizaciones ve por época; eso no existe en el 1a). 60-40-20 queda en la grilla como control.

**Qué hacer.**

1. Config: en `ej1/config_dae.json` agregar a la base `"copias_epoca": 1` y
   `"monitor": "val_loss"` (los valores de hoy) y un experimento C:
   ocultas ∈ {[60, 40, 20], [128, 64], [128, 128, 64]} × copias_epoca ∈ {1, 8, 16}, con
   latente 8, p_train 0,1, lote 32. Son 9 valores × 5 semillas = 45 corridas.
2. `ej1/dae.py`, `entrenar_dae`: con `copias_epoca = k`, entrenar sobre `np.tile(X, (k, 1))`
   (el ruido se sortea distinto en cada copia porque `BitFlip` se aplica por lote). Agregar a
   `fit` una métrica `recuperadas` (fracción con ≤ 1 píxel mal) y cortar con
   `EarlyStopping(monitor="val_recuperadas", mode="max")` cuando `monitor` lo pida.
   Tests: con k = 3 cada época hace 3× las actualizaciones; el corte por recuperadas restaura
   la mejor época; con `copias_epoca = 1` y `monitor = "val_loss"` da lo mismo que hoy (las
   corridas de A y B no cambian).
3. **Costo:** `dae.py` corre en secuencia y con k = 16 cada época cuesta ~16 veces más.
   Estimado 1–2 h. Si molesta, agregar `--trabajadores` con `multiprocessing` como en
   `ej2/vae_exp.py` (`_correr_tareas`), o achicar la grilla a k ∈ {1, 16} × ocultas
   {60-40-20, 128-64} (20 corridas).
4. Elegir con la misma regla (más recuperadas, después menos píxeles mal). Si gana algo
   distinto de lo actual: mover `resultados/ej1b/final/` a `resultados/ej1b/final_c0/`,
   correr `python3 -m ej1.dae` (reentrena el final y evalúa en test) y
   `python3 -m ej1.graficos_dae`.

**Hecho cuando:** `experimento_C_*.csv` existe, la tabla y la conclusión están en el spec
(aunque no cierre la brecha: eso también es un resultado) y, si cambió el final, el test
nuevo está documentado.

---

## 4. C3 — Tamaño de lote y pérdida con su propio lr (1a)

**Problema.** El barrido del 1a concluye "lote 4 y 1 no llegan" y "MSE llega 2/5", pero los
midió con el lr elegido para lote completo + BCE. Es la misma trampa que el barrido evitó en el
paso 6 + 7 (optimizador y lr juntos). Es la conclusión más atacable del 1a.

**Qué hacer.** Reusar `ej1/verificacion.py` como modelo (lee `elecciones.json`, no toca las
elecciones, escribe en `resultados/ej1/verificacion/`):

1. Grillas, con la config vigente en cada paso (`config_del_paso`):
   - paso 9: lote {32, 8, 4, 1} × lr {0,001; 0,003; 0,01}, 5 semillas;
   - paso 10: pérdida {BCE, MSE} × lr {0,003; 0,01; 0,03; 0,1}, 5 semillas.
2. Comparar por **actualizaciones** hasta llegar (columna `actualizaciones`), no por épocas.
   Para lote 1 poner un tope de actualizaciones (por ejemplo 50.000) para que no tarde
   ~4 min por corrida.
3. Tests como en `tests/test_ej1_verificacion.py`.

**Hecho cuando:** el spec del 1a dice, para lote y pérdida, si la conclusión original se
sostiene con su mejor lr. Cómputo: ~20–40 min (casi todo lote 1).

---

## 5. A4 — Latente y β juntos en el VAE (ej. 2)

**Problema.** El barrido elige el latente con β = 1 y después cambia β. Latente y β
interactúan (con β alto se apagan dimensiones), así que el orden puede haber descartado el
latente 8 o 16. Además el ELBO de validación se mide con una sola muestra de z: es ruidoso y
la regla de "a 1 desvío, el más chico" casi siempre elige el latente más chico.

**Qué hacer.**

1. Script nuevo (o modo `--fase grilla` en `ej2/vae_exp.py`): grilla latente {4, 8, 16} ×
   β {0,1; 1; 5}, 5 semillas, con capacidad y lr del ganador actual, en
   `resultados/ej2/<dataset>/grilla_latente_beta.csv`. Reusar `_entrenar` y `_correr_tareas`.
2. Para cada celda reportar `val_recon`, KL, unidades activas y, con las muestras de N(0, I),
   las métricas de `ej2.graficos.metricas_generacion` (hay que guardar las muestras o calcular
   la métrica en `_entrenar`).
3. Figura tipo mapa de calor (latente × β) por dataset para `val_recon` y para % fuera de
   [0, 1].
4. Solo si un latente distinto de 4 gana con claridad: archivar `resultados/ej2/<dataset>/`
   en otra carpeta, cambiar `ej2/config.json` y rehacer barrido y final.

**Hecho cuando:** la grilla y la conclusión están en `ej2-vae-fix.md`. Cómputo: ~30–40 min
por dataset de imagen con 5 workers.

---

## 6. Detalles para la presentación (sin código o casi)

- **1a:** para "letra nueva", elegir 2–3 glifos de `05a_grilla_latente.png` que se lean como
  letra (el criterio "a ≥ 2 píxeles de todas" también cuenta manchas). Usar los huecos del
  latente del AE como puente al VAE.
- **1a:** decir que He/Xavier y tanh/leaky ReLU/ReLU empatan (ver "Verificación con 20
  semillas" en el spec); la regla desempató.
- **1b:** el latente "satura desde 4" (8 y 16 empatan), no "gana 8".
- **1b:** p_train = 0 corre 10.000 épocas sin cortar y es casi el AE del 1a con latente 8.
- **Ej. 2:** el latente 16 reconstruye peor que antes porque β más alto apaga dimensiones, y
  el número viejo de banderas (0,0103) venía en parte de pesos del warm-up.
- **Ej. 2:** β = 1 con MSE equivale a un decoder gaussiano con σ² = ½; la equivalencia
  general es β = 2σ² (`ej2-kl-fix.md` §4 dice "β = 1 recupera el ELBO" sin esa aclaración).
- **Ej. 2, F1:** el split no es disjunto por piloto y solo se deduplicó por hash exacto: hay
  casi-duplicados entre train y test. Decirlo.
- **Ej. 2, banderas:** validación de 7 y test de 6 filas; las elecciones ahí son ruidosas.
- **Ej. 2, `00_resumen.png`:** la fila de muestras queda bajo nombres de jugadores/países con
  los que no tiene relación; aclararlo en el pie o separar la fila.
- **Figuras del 1a:** se generaron con la fuente Segoe UI; las 04–06 se regeneraron sin ella.
  Si se quiere fuente pareja, regenerar todo con `python3 -m ej1.graficos` en una máquina
  con Segoe UI (Windows).

## 7. Repo

- `.git` pesa ~480 MB por los pesos y muestras de `resultados/`. No borrar nada sin acuerdo
  del equipo; si molesta, evaluar ignorar los `.npz` regenerables a futuro.
- En `main` hay un commit con mensaje `fix: lebrooonnnn jamesss` que ve la cátedra. Cambiarlo
  implica reescribir historia publicada: solo con acuerdo de todos.
