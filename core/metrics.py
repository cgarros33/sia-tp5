"""Métricas de reconstrucción de imágenes binarias: píxeles mal.

Convención de formas: y_true e y_pred son (n_muestras, n_pixeles). La salida
de la red se binariza con un umbral (>= threshold → 1) y se cuentan los
píxeles distintos a los de y_true, también binarizado con el mismo umbral.
max_pixel_errors y mean_pixel_errors tienen la firma de Metric, así que se
pueden pasar en el `metrics` de fit.
"""

import numpy as np


def pixel_errors(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Píxeles mal de cada muestra, forma (n_muestras,).

    Un NaN (red divergida) no supera el umbral, así que se lee como 0.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Formas distintas: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return np.count_nonzero((y_pred >= threshold) != (y_true >= threshold), axis=1)


def max_pixel_errors(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Píxeles mal de la peor muestra."""
    return float(np.max(pixel_errors(y_true, y_pred)))


def mean_pixel_errors(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Promedio de píxeles mal por muestra."""
    return float(np.mean(pixel_errors(y_true, y_pred)))
