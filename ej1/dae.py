"""Ejercicio 1b: denoising autoencoder sobre font.h (docs/context/ej1b-denoising.md).

Por ahora tiene las piezas de evaluación: los conjuntos de letras ruidosas
(validación y test), la medición contra la letra limpia y las referencias
(no hacer nada, vecino más cercano y AE del 1a).
"""

import json
from pathlib import Path

import numpy as np

from core.augmentation import BitFlip
from core.autoencoder import Autoencoder
from core.metrics import pixel_errors
from ej1.barrido import cargar_modelo

RAIZ = Path(__file__).resolve().parent.parent
RUTA_FINAL_1A = RAIZ / "resultados" / "ej1" / "final"

# Una letra está recuperada si le quedan a lo sumo estos píxeles mal (el umbral del 1a).
UMBRAL_RECUPERADA = 1


def conjuntos_ruidosos(
    X: np.ndarray, niveles: list[float], k: int, semilla: int
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """k copias ruidosas (bit-flip) de cada letra de X para cada nivel p.

    Devuelve {p: (X_ruidoso, idx)}: X_ruidoso es (n·k, n_pixeles) e idx (n·k,)
    dice de qué letra limpia sale cada fila (X[idx] es el objetivo). Un solo
    generador, default_rng(semilla), recorre los niveles en el orden dado.
    """
    rng = np.random.default_rng(semilla)
    idx = np.tile(np.arange(X.shape[0]), k)
    return {p: (BitFlip(p)(X[idx], rng), idx) for p in niveles}


def medir_limpieza(X: np.ndarray, idx: np.ndarray, X_rec: np.ndarray) -> dict:
    """Píxeles mal de cada reconstrucción contra su letra limpia X[idx].

    pixeles_mal: promedio por letra; peor_letra: el máximo; recuperadas:
    fracción de letras con <= UMBRAL_RECUPERADA píxeles mal.
    """
    errores = pixel_errors(X[idx], X_rec)
    return {
        "pixeles_mal": float(errores.mean()),
        "peor_letra": int(errores.max()),
        "recuperadas": float(np.mean(errores <= UMBRAL_RECUPERADA)),
    }


def vecino_mas_cercano(X_ruidoso: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Para cada fila de X_ruidoso, la letra de X a menor distancia de Hamming.

    Datos en {0, 1}. En un empate gana la letra de menor índice.
    """
    distancias = X_ruidoso @ (1.0 - X).T + (1.0 - X_ruidoso) @ X.T  # (m, n_letras)
    return X[np.argmin(distancias, axis=1)]


def cargar_ae_1a(ruta: Path = RUTA_FINAL_1A) -> dict[int, Autoencoder]:
    """{semilla: autoencoder} de la config final del 1a, con los pesos que guardó el barrido."""
    config = json.loads((ruta / "config.json").read_text(encoding="utf-8"))
    modelos = {}
    for archivo in sorted(ruta.glob("pesos_semilla_*.npz")):
        semilla = int(archivo.stem.removeprefix("pesos_semilla_"))
        with np.load(archivo) as pesos:
            n_entradas = pesos["W0"].shape[0]
        modelos[semilla] = cargar_modelo(config, archivo, n_entradas)
    return modelos
