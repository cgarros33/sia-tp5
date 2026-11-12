"""Funciones de activación y sus derivadas.

Convención de formas: todos los arrays son (n_muestras, n_features). Salvo
Softmax, las activaciones operan elemento a elemento, así que forward y
backward devuelven un array de la misma forma que recibieron.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


class Activation(Protocol):
    def forward(self, z: np.ndarray) -> np.ndarray:
        """Aplica la activación a la pre-activación z, forma (n_muestras, n_features)."""
        ...

    def backward(self, z: np.ndarray) -> np.ndarray:
        """Derivada de la activación evaluada en z, misma forma que z."""
        ...


class Step:
    """f(z) = 1 si z >= 0, -1 si no. Salida del perceptrón simple escalón.

    No es derivable, así que no se entrena con backprop sino con la regla del
    perceptrón (Δw = η (y - ŷ) x), que va en un trainer aparte.
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z >= 0.0, 1.0, -1.0)

    def backward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Step no es derivable: usar la regla del perceptrón en lugar de backprop."
        )


class Identity:
    """f(z) = z. Salida de un perceptrón simple lineal."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def backward(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)


class Tanh:
    """f(z) = tanh(beta * z), con imagen en (-1, 1).

    beta controla la pendiente: valores altos hacen la transición más abrupta.
    Con beta = 1 es la tangente hiperbólica clásica.
    """

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(self.beta * z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        # d/dz tanh(beta*z) = beta * (1 - tanh(beta*z)^2)
        return self.beta * (1.0 - np.tanh(self.beta * z) ** 2)


class Sigmoid:
    """f(z) = 1 / (1 + exp(-2 * beta * z)), con imagen en (0, 1).

    Logística con la convención de la cátedra (apuntes §8.2.4): el 2 hace
    que f(z) = ½ (1 + tanh(beta * z)), así que los β que se reportan son los
    de la teoría.
    """

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def forward(self, z: np.ndarray) -> np.ndarray:
        bz = 2.0 * self.beta * z
        # Versión estable: exp(-|bz|) está siempre en (0, 1] y nunca desborda.
        # Para bz >= 0 es la fórmula usual; para bz < 0 se multiplica arriba y
        # abajo por exp(bz), que da e / (1 + e).
        e = np.exp(-np.abs(bz))
        return np.where(bz >= 0.0, 1.0 / (1.0 + e), e / (1.0 + e))

    def backward(self, z: np.ndarray) -> np.ndarray:
        # d/dz sigmoid(2*beta*z) = 2 * beta * s * (1 - s)
        s = self.forward(z)
        return 2.0 * self.beta * s * (1.0 - s)


class ReLU:
    """f(z) = max(0, z). En z = 0 no es derivable; por convención f'(0) = 0."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0.0, 1.0, 0.0)


class LeakyReLU:
    """f(z) = z si z > 0, alpha * z si no. Como ReLU, pero con pendiente alpha a la izquierda.

    La pendiente no nula evita que una neurona quede "muerta" (derivada 0 para
    todas las entradas). En z = 0 no es derivable; por convención f'(0) = alpha.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0.0, z, self.alpha * z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0.0, 1.0, self.alpha)


class Softmax:
    """f(z)_i = exp(z_i) / sum_j exp(z_j), por fila. Cada fila suma 1.

    No es elemento a elemento: cada salida depende de toda la fila, así que su
    derivada es una matriz jacobiana por muestra, no un array de la forma de z.
    Por eso solo se usa como salida junto con CategoricalCrossEntropy: la red
    detecta ese par y calcula el delta combinado dL/dz = ŷ − y directamente,
    que además es más estable numéricamente que componer las dos derivadas.
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        # Restar el máximo de cada fila no cambia el resultado (se cancela en
        # el cociente) y evita el overflow de exp con valores grandes.
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def backward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Softmax solo se usa con CategoricalCrossEntropy, cuyo delta combinado calcula la red."
        )


ACTIVATIONS: Mapping[str, type] = MappingProxyType(
    {
        "step": Step,
        "identity": Identity,
        "tanh": Tanh,
        "sigmoid": Sigmoid,
        "relu": ReLU,
        "leaky_relu": LeakyReLU,
        "softmax": Softmax,
    }
)


def get_activation(name: str, **kwargs: Any) -> Activation:
    """Construye una activación a partir de su nombre, p. ej. get_activation("tanh", beta=2.0).

    Un nombre o parámetro desconocido es un error, no se ignora.
    """
    try:
        cls = ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"Activación desconocida: {name!r}. Opciones: {sorted(ACTIVATIONS)}"
        ) from None
    return cls(**kwargs)
