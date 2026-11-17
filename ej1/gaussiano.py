"""Evaluación del DAE del 1b con ruido gaussiano (solo lectura de pesos).

    python -m ej1.gaussiano [--salida resultados/ej1b/gaussiano] [--forzar]

No entrena nada: carga el DAE final de resultados/ej1b/final/ y el AE del 1a,
y los evalúa sobre copias con ruido gaussiano recortado a [0, 1]. El DAE se
entrenó solo con bit-flip; esto mide cuánto generaliza a otra clase de ruido.
Las referencias son devolver la entrada binarizada en 0.5 y el vecino más
cercano sobre esa entrada binarizada.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from core.autoencoder import Autoencoder
from ej1.barrido import cargar_modelo
from ej1.dae import (
    COLUMNAS_NIVEL,
    RAIZ,
    _escribir_csv,
    _filas_por_nivel,
    cargar_ae_1a,
    conjuntos_gaussianos,
    evaluar,
    leer_filas_dae,
    vecino_mas_cercano,
)
from ej1.datos import cargar_font

RUTA_FINAL = RAIZ / "resultados" / "ej1b" / "final"
RUTA_SALIDA = RAIZ / "resultados" / "ej1b" / "gaussiano"
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
COPIAS = 50
SEMILLA = 3000


def _binarizar(X: np.ndarray) -> np.ndarray:
    """Entrada con el mismo umbral que usa la métrica de píxeles mal."""
    return (X >= 0.5).astype(float)


def cargar_dae_final(ruta: Path = RUTA_FINAL) -> tuple[dict, dict[int, Autoencoder]]:
    """Config final y {semilla: DAE} con los pesos guardados por ej1.dae."""
    config = json.loads((ruta / "config.json").read_text(encoding="utf-8"))
    modelos = {}
    for archivo in sorted(ruta.glob("pesos_semilla_*.npz")):
        semilla = int(archivo.stem.removeprefix("pesos_semilla_"))
        with np.load(archivo) as pesos:
            n_entradas = pesos["W0"].shape[0]
        modelos[semilla] = cargar_modelo(config, archivo, n_entradas)
    return config, modelos


def evaluar_gaussiano(
    X: np.ndarray,
    modelos_dae: dict[int, Autoencoder],
    modelos_1a: dict[int, Autoencoder],
    sigmas: list[float] = SIGMAS,
    copias: int = COPIAS,
    semilla: int = SEMILLA,
) -> list[dict]:
    """Filas (modelo, semilla, sigma, medidas) del DAE, el AE del 1a y las referencias."""
    conjuntos = conjuntos_gaussianos(X, sigmas, copias, semilla)
    evaluados = [
        ("nada", None, evaluar(_binarizar, conjuntos, X)),
        ("vecino", None, evaluar(lambda X_r: vecino_mas_cercano(_binarizar(X_r), X), conjuntos, X)),
        *[("dae", s, evaluar(ae.predict, conjuntos, X)) for s, ae in modelos_dae.items()],
        *[("ae_1a", s, evaluar(ae.predict, conjuntos, X)) for s, ae in modelos_1a.items()],
    ]
    return _filas_por_nivel(evaluados)


def correr_gaussiano(
    salida: Path = RUTA_SALIDA, forzar: bool = False, escribir=print
) -> list[dict]:
    """Evalúa y escribe salida/test.csv; si ya existe, lo lee en vez de recalcular."""
    ruta = salida / "test.csv"
    if ruta.exists() and not forzar:
        escribir(f"Ya existe {ruta}: se lee en vez de recalcular.")
        return leer_filas_dae(ruta)
    X, _ = cargar_font()
    _, modelos_dae = cargar_dae_final()
    modelos_1a = cargar_ae_1a()
    filas = evaluar_gaussiano(X, modelos_dae, modelos_1a)
    salida.mkdir(parents=True, exist_ok=True)
    _escribir_csv(ruta, COLUMNAS_NIVEL, filas)
    escribir(f"{ruta}: {len(filas)} filas.")
    return leer_filas_dae(ruta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del 1b con ruido gaussiano.")
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    parser.add_argument("--forzar", action="store_true")
    args = parser.parse_args()
    correr_gaussiano(args.salida, args.forzar, escribir=lambda texto: print(texto, flush=True))


if __name__ == "__main__":
    main()
