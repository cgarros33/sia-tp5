# Por qué colapsa el VAE y cómo se corrige la normalización del KL

Documento teórico de la corrección de `ej2-vae.md` §2. Acompaña a
`ej2-vae-fix.md`, que contiene el plan de implementación.

## 1. El objetivo del VAE

Para una muestra `x ∈ R^d`, un encoder `q_φ(z|x) = N(μ_φ(x), diag(exp(s_φ(x))))` y un
decoder `p_θ(x|z)`, el ELBO por muestra es

```text
ELBO_n = E_{q_φ(z|x_n)}[ log p_θ(x_n|z) ] − KL( q_φ(z|x_n) || N(0, I) )
```

y se maximiza `(1/n) Σ_n ELBO_n`. La forma β-VAE introducida por Higgins et al. (2017)
pondera el KL:

```text
J = −(1/n) Σ_n E_q[log p_θ(x_n|z)] + β · (1/n) Σ_n KL_n
```

Con un decoder gaussiano de varianza fija `σ²`, la log-verosimilitud es

```text
log p_θ(x|z) = −||x − x'||² / (2σ²) − (d/2) log(2πσ²)
```

El segundo término no depende de los parámetros y el primero es una **suma sobre los `d`
píxeles**, no un promedio. Ese detalle es el que se perdió en la primera implementación.

## 2. Cómo calcula las pérdidas el motor

`core/losses.py` devuelve un promedio sobre todos los elementos de la matriz:

```text
L_mean = (1 / (n·d)) Σ_{n=1..N} Σ_{j=1..d} ℓ(x_nj, x'_nj)
```

`core/vae.py` calcula el KL como suma sobre el latente y promedio sobre las muestras:

```text
R_mean = (1/n) Σ_{n=1..N} Σ_{k=1..K} KL_nk
```

La primera corrida usó

```text
J_mal = L_mean + β · R_mean
```

## 3. El desbalance de gradientes

Sea `r(θ) = L_mean` y `k(φ) = R_mean`. Los parámetros del decoder `θ` reciben
`∂r/∂θ`; los del encoder `φ` reciben

```text
∂J_mal/∂φ = ∂r/∂φ + β · ∂k/∂φ
```

El gradiente de reconstrucción que llega al encoder está multiplicado por `1/d` porque `r`
promedia sobre `d` píxeles. El del KL no: `k` ya está promediado solo sobre `n`. Por lo
tanto, la presión del KL sobre el encoder es `β·d` veces la de reconstrucción. Con `d = 1024`
y `β = 0.1`, el KL pesa ~100 veces más de lo que su valor numérico sugiere.

El óptimo de ese objetivo no es “usar poco el latente”: es **no usarlo**. Si el decoder
ignora `z`, puede predecir la imagen media; el encoder puede llevar `μ → 0` y `s → 0`, y
el KL tiende a cero. Es un mínimo local estable: la reconstrucción no mejora porque el
decoder ya no depende de `z`, y el KL no sube porque el encoder ya está en la prior. Ese es
el **colapso del posterior**.

## 4. La corrección

Para una reconstrucción promediada por píxel, el KL debe normalizarse por la misma cantidad
de features:

```text
J_bien = L_mean + (β / d) · R_mean
       = (1/d) · [ Σ_{n,j} ℓ(x_nj, x'_nj)/n + β · R_mean ]
```

El factor global `1/d` no cambia el minimizador ni el orden de magnitud relativo de los dos
términos; lo que cambia es el peso relativo, que ahora es `β`. Con `β = 1` se recupera el
VAE estándar (el ELBO sin ponderar) cuando la reconstrucción es una suma por muestra.

Equivalentemente, si se define la reconstrucción sumada por muestra,

```text
L_sum = (1/n) Σ_n Σ_j ℓ(x_nj, x'_nj) = d · L_mean
```

entonces

```text
J_bien = (1/d) · [ L_sum + β · R_mean ]
```

que es la forma habitual de β-VAE: **reconstrucción sumada sobre píxeles + β · KL**. Las dos
formas son idénticas salvo el factor global `1/d`.

### 4.1 Por qué β deja de depender de la resolución

Si se cambia la resolución de 16×16 (`d = 256`) a 32×32 (`d = 1024`), `L_mean` mide el mismo
error medio por píxel, pero el gradiente de reconstrucción que llega al encoder se divide por
un `d` cuatro veces mayor. Sin la corrección, el mismo `β` que funcionaba a 16×16 colapsa a
32×32. Con `β/d`, el peso relativo es `β` en ambas resoluciones: el mismo grid de β sirve
para todas las entradas y para banderas (`d = 12`).

### 4.2 Por qué no alcanza con “bajar β”

Bajar β a `1/d` en la fórmula vieja también equilibra los gradientes, pero deja un
hiperparámetro que depende de `d` y hace incomparables los datasets. La corrección en el
código es preferible porque el valor de β conserva su significado (β = 1 es el VAE estándar)
y el barrido es el mismo para los cuatro datasets.

## 5. Warm-up del KL

Aun con la normalización correcta, el colapso puede ocurrir al principio del entrenamiento:
el decoder es malo, el KL empuja la posterior a la prior antes de que el decoder aprenda a
usar `z`, y el sistema queda en el mínimo “ignorar z”. Una solución estándar es el
**KL cost annealing** (Bowman et al., 2016): empezar con β = 0 y subirlo gradualmente.

```text
β(t) = β_final · min(1, t / T)
```

Con `T = 200` épocas (20 % de 1000), las primeras épocas son un autoencoder determinístico:
el decoder aprende a reconstruir. Después el KL se introduce y poda las dimensiones que no
aportan. El objetivo final para `t ≥ T` es exactamente el mismo; solo cambia el camino de
optimización.

El early stopping tiene que monitorear `val_recon`, no `val_loss`, y no debe cortar ni
elegir la mejor época durante el warm-up: se le pasa `start_epoch = calentamiento + 1`, porque
los pesos de esas épocas se entrenaron con un β menor al final. Si se monitorea `val_loss`, el KL creciente del warm-up puede parecer un
empeoramiento y cortar antes de que el modelo use el latente.

## 6. Free bits

Si después del warm-up y con β bajo todavía hay dimensiones muertas, se puede imponer un
piso por dimensión (Kingma et al., 2016):

```text
KL_free = Σ_k max( KL_k , λ )
```

donde `KL_k` es la KL promedio de la dimensión `k` sobre el batch. La interpretación es
“cada dimensión puede almacenar al menos λ nats sin costo”. El gradiente de una dimensión con
`KL_k ≤ λ` se anula, así que el encoder no sigue empujándola a la prior. λ = 0.05–0.1 nats
por dimensión es un valor típico. Free bits es un resguardo, no un reemplazo de la
normalización: con `J_mal` el KL se anula en todas las dimensiones y el piso apenas cambia el
resultado.

## 7. Unidades activas y selección de modelos

Para cada dimensión latente `k` se mide

```text
KL_k = mean_n [ ½ (exp(s_nk) + μ_nk² − 1 − s_nk) ]
```

y se dice que la dimensión está **activa** si `KL_k > 0.01`. Un VAE colapsado tiene 0
unidades activas.

El ELBO solo no sirve para elegir: un modelo colapsado tiene KL ≈ 0 y, por lo tanto, un
ELBO bajo que parece bueno. La primera corrida eligió latente 2 porque el colapso tenía el
ELBO más bajo. Por eso la selección debe incluir las restricciones

```text
activas ≥ K/2    y    KL_medio > 0.01
```

en todos los pasos, no solo en el paso de β. Si ninguna corrida las cumple, se elige la de
más unidades activas y se registra el paso como colapsado; después se sube de nivel de
reintento.

## 8. Evidencia de la primera corrida

Con `J_mal`:

| dataset | activas | KL medio | recon MSE | baseline imagen media |
|---|---:|---:|---:|---:|
| NBA | 0 | 0.00009 | 0.0498 | 0.0496 |
| F1 | 0 | 0.0014 | 0.0472 | 0.0470 |
| Chino | 0 | ~0 | 0.0010 | 0.0010 |
| Banderas | 0.2 | 0.0109 | 0.1056 | — |

La reconstrucción es indistinguible de la imagen media. Los AE determinísticos sí usan el
latente (NBA AE16: 0.0135), lo que confirma que el problema es el objetivo β-VAE, no los
datos ni el motor.

Con `J_bien` se espera `KL` de orden 1 y `activas` de al menos la mitad de `K`. El criterio
de aceptación no es un número exacto de MSE sino: **activas ≥ K/2, KL > 0.01 y
reconstrucción claramente mejor que la imagen media**.

## 9. Tests que fijan la corrección

- `evaluate` con `kl_weight=1` menos `evaluate` con `kl_weight=0` es exactamente `KL / d`,
  para `d ∈ {6, 12}`. Esto detecta cualquier vuelta a la normalización vieja.
- `free_bits=0` reproduce el comportamiento anterior; con `free_bits=λ`, la diferencia de
  pérdida es `Σ max(KL_k, λ)/n`.
- El callback de warm-up deja `kl_weight = 0` al inicio, `β·t/T` durante el calentamiento y
  `β` al final.
- El filtro de selección rechaza una corrida con `activas < K/2`.

## 10. Referencias

- Kingma, D. P. y Welling, M. (2014). *Auto-Encoding Variational Bayes*.
- Higgins, I. et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained
  Variational Framework*.
- Bowman, S. R. et al. (2016). *Generating Sentences from a Continuous Space* (KL cost
  annealing).
- Kingma, D. P. et al. (2016). *Improved Variational Inference with Inverse Autoregressive
  Flow* (free bits).
