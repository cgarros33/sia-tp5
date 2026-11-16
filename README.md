# sia-tp5

TP5 de Sistemas de Inteligencia Artificial (ITBA): autoencoders, denoising autoencoder y
autoencoder variacional, implementados con NumPy (sin bibliotecas de ML).

## Instalación

Requiere Python 3.11 o más nuevo.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`python3 check-cicd.py` hace lo mismo si falta `.venv` y además corre todos los chequeos:
`pip check`, `ruff check .`, `ruff format --check .` y `pytest -q`. Son los mismos que corre
GitHub Actions en cada push.

## Cómo correr cada ejercicio

Los scripts de entrenamiento escriben en `resultados/`; los de gráficos solo leen de ahí. Los
resultados ya están en el repo, así que los gráficos se pueden regenerar sin entrenar.

### 1a — Autoencoder básico sobre `font.h`

```bash
python3 -m ej1.barrido     # barrido secuencial, ~45 min; retoma los pasos que ya existen
python3 -m ej1.graficos    # figuras en resultados/ej1/figuras/
```

### 1b — Denoising autoencoder

Usa los pesos finales del 1a como referencia.

```bash
python3 -m ej1.dae           # experimentos A y B y test final, ~55 min; reanudable
python3 -m ej1.graficos_dae  # figuras en resultados/ej1b/figuras/
```

### 2 — Autoencoder variacional

Los datasets procesados ya están en `data/procesadas/`. Solo hace falta prepararlos de nuevo si
se tienen las imágenes originales (que no están en el repo):

```bash
python3 utils/preparar_ej2.py --todo
```

Por dataset (`nba`, `f1`, `banderas` o `chino`):

```bash
python3 -m ej2.vae_exp --dataset nba --fase barrido --trabajadores 5
python3 -m ej2.vae_exp --dataset nba --fase final --trabajadores 5
python3 -m ej2.graficos --dataset nba
```

Los entrenamientos no pisan resultados: los del ej. 1 saltean lo que ya existe y los del
ej. 2 solo rehacen con `--forzar`. Los scripts de gráficos sí sobrescriben las figuras.

## Dónde está cada cosa

| Carpeta | Contenido |
|---|---|
| `core/` | motor genérico: capas, activaciones, optimizadores, autoencoder y VAE |
| `ej1/`, `ej2/` | scripts de cada ejercicio |
| `tests/` | tests de `core/` y de cada ejercicio |
| `data/` | `font.h`, banderas y datasets procesados del ej. 2 |
| `resultados/ej1/`, `resultados/ej1b/`, `resultados/ej2/<dataset>/` | CSV, pesos y `figuras/` |
| `resultados/ej2_old/`, `resultados/ej2_beta001/` | corridas anteriores del ej. 2, archivadas |
| `docs/context/` | consigna, specs de cada ejercicio con decisiones y resultados |

La estructura detallada está en [docs/structure.md](docs/structure.md).
