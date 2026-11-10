"""Inicializadores de pesos.

Cada inicializador devuelve la matriz W de una capa densa, de forma
(n_in, n_out). Los bias no pasan por acá: la capa los inicializa en cero.
La aleatoriedad sale siempre del Generator recibido, nunca del estado global
de NumPy, para que la corrida sea reproducible a partir de la semilla.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


class Initializer(Protocol):
    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """Devuelve W de forma (n_in, n_out)."""
        ...


class Uniform:
    """W ~ U(low, high), independiente del tamaño de la capa.

    Es la inicialización clásica del perceptrón simple. En redes profundas
    conviene Xavier o He, que escalan con la cantidad de entradas.
    """

    def __init__(self, low: float = -0.5, high: float = 0.5) -> None:
        if low >= high:
            raise ValueError(f"Se requiere low < high, recibido low={low}, high={high}")
        self.low = low
        self.high = high

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=(n_in, n_out))


class Xavier:
    """Glorot uniforme: W ~ U(−r, r) con r = sqrt(6 / (n_in + n_out)).

    Mantiene la varianza de las activaciones parecida entre capas cuando la
    activación es simétrica y aproximadamente lineal cerca de 0 (tanh, sigmoid).
    """

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        r = np.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-r, r, size=(n_in, n_out))


class He:
    """He normal: W ~ N(0, 2 / n_in).

    El factor 2 compensa que ReLU anula la mitad de las pre-activaciones, así
    que es la elección natural para capas ocultas ReLU.
    """

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))


INITIALIZERS: Mapping[str, type] = MappingProxyType(
    {
        "uniform": Uniform,
        "xavier": Xavier,
        "he": He,
    }
)


def get_initializer(name: str, **kwargs: Any) -> Initializer:
    """Construye un inicializador a partir de su nombre, p. ej. get_initializer("uniform", low=-1).

    Un nombre o parámetro desconocido es un error, no se ignora.
    """
    try:
        cls = INITIALIZERS[name]
    except KeyError:
        raise ValueError(
            f"Inicializador desconocido: {name!r}. Opciones: {sorted(INITIALIZERS)}"
        ) from None
    return cls(**kwargs)
