"""Funciones de pérdida y sus gradientes respecto de la predicción.

Convención de formas: y_true e y_pred son (n_muestras, n_salidas) y deben tener
exactamente la misma forma. grad devuelve dL/dy_pred con esa misma forma.

Todas las pérdidas se promedian sobre las muestras, así que el gradiente ya
incluye el factor 1/n_muestras: la capa solo tiene que sumar sobre el lote.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

# Cota para recortar probabilidades antes del log: evita log(0) y divisiones
# por cero cuando la salida satura en 0 o 1.
PROB_EPS = 1e-12


class Loss(Protocol):
    def value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pérdida media del lote. Ambos arrays de forma (n_muestras, n_salidas)."""
        ...

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """dL/dy_pred, forma (n_muestras, n_salidas)."""
        ...


def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    # Sin este chequeo, (n,) contra (n, 1) se broadcastea a (n, n) y la
    # pérdida da un número razonable pero incorrecto.
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Formas distintas: y_true {y_true.shape} vs y_pred {y_pred.shape}")


class MSE:
    """L = mean((y − ŷ)²), promediando sobre todas las muestras y salidas.

    Con una sola salida coincide con el error cuadrático medio usual.
    """

    def value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _check_shapes(y_true, y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        _check_shapes(y_true, y_pred)
        return 2.0 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy:
    """L = −mean(y log ŷ + (1 − y) log(1 − ŷ)), con y ∈ {0, 1} y ŷ ∈ (0, 1).

    Pensada para salida Sigmoid. Promedia sobre todas las muestras y salidas.
    """

    def value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _check_shapes(y_true, y_pred)
        p = np.clip(y_pred, PROB_EPS, 1.0 - PROB_EPS)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        _check_shapes(y_true, y_pred)
        p = np.clip(y_pred, PROB_EPS, 1.0 - PROB_EPS)
        return (p - y_true) / (p * (1.0 - p)) / y_true.size


class CategoricalCrossEntropy:
    """L = −(1/n) Σ_i Σ_k y_ik log ŷ_ik, con y one-hot y cada fila de ŷ sumando 1.

    Pensada para salida Softmax. Se suma sobre clases y se promedia sobre
    muestras (no sobre clases), que es la definición usual.
    """

    def value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _check_shapes(y_true, y_pred)
        p = np.clip(y_pred, PROB_EPS, 1.0)
        return float(-np.sum(y_true * np.log(p)) / y_true.shape[0])

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        _check_shapes(y_true, y_pred)
        p = np.clip(y_pred, PROB_EPS, 1.0)
        return -y_true / p / y_true.shape[0]

    def softmax_delta(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """dL/dz cuando ŷ = softmax(z): el par acoplado da (ŷ − y) / n.

        Es lo que usa la red cuando la capa de salida es Softmax, en vez de
        componer grad con la jacobiana de Softmax: es más barato y no divide
        por ŷ, así que no pierde precisión cuando alguna probabilidad es ~0.
        Forma (n_muestras, n_clases).
        """
        _check_shapes(y_true, y_pred)
        return (y_pred - y_true) / y_true.shape[0]


LOSSES: Mapping[str, type] = MappingProxyType(
    {
        "mse": MSE,
        "binary_crossentropy": BinaryCrossEntropy,
        "categorical_crossentropy": CategoricalCrossEntropy,
    }
)


def get_loss(name: str, **kwargs: Any) -> Loss:
    """Construye una pérdida a partir de su nombre, p. ej. get_loss("mse").

    Un nombre o parámetro desconocido es un error, no se ignora.
    """
    try:
        cls = LOSSES[name]
    except KeyError:
        raise ValueError(f"Pérdida desconocida: {name!r}. Opciones: {sorted(LOSSES)}") from None
    return cls(**kwargs)
