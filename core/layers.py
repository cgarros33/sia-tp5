"""Capa densa (fully connected).

Convención de formas: la entrada x es (n_muestras, n_in) y la salida es
(n_muestras, n_out). Los pesos son W (n_in, n_out) y el bias b (1, n_out),
que se suma a cada fila por broadcasting; el dataset nunca lleva columna de unos.
"""

import numpy as np

from core.activations import Activation
from core.initializers import Initializer


class Dense:
    """a = activation(x @ W + b).

    forward guarda x y z = x @ W + b porque el backward los necesita: la
    derivada de la activación se evalúa en z, y el gradiente de W usa x.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: Activation,
        initializer: Initializer,
        rng: np.random.Generator,
    ) -> None:
        if n_in < 1 or n_out < 1:
            raise ValueError(f"Tamaños de capa inválidos: n_in={n_in}, n_out={n_out}")
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.W: np.ndarray = initializer(n_in, n_out, rng)
        self.b: np.ndarray = np.zeros((1, n_out))
        # Cache del último forward, para el backward.
        self.x: np.ndarray | None = None
        self.z: np.ndarray | None = None
        # Gradientes del último backward, misma forma que W y b.
        self.grad_W: np.ndarray = np.zeros_like(self.W)
        self.grad_b: np.ndarray = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (n_muestras, n_in). Devuelve la activación, (n_muestras, n_out)."""
        if x.ndim != 2 or x.shape[1] != self.n_in:
            raise ValueError(f"Se esperaba entrada (n_muestras, {self.n_in}), llegó {x.shape}")
        self.x = x
        self.z = x @ self.W + self.b
        return self.activation.forward(self.z)

    def backward(self, grad_a: np.ndarray) -> np.ndarray:
        """grad_a: dL/da de esta capa, (n_muestras, n_out).

        Pasa a dL/dz multiplicando por la derivada de la activación, evaluada
        en z (no en a), y sigue como backward_z. Devuelve dL/dx, (n_muestras, n_in).
        """
        if self.z is None:
            raise RuntimeError("backward antes de forward: no hay x ni z guardados.")
        return self.backward_z(grad_a * self.activation.backward(self.z))

    def backward_z(self, delta: np.ndarray) -> np.ndarray:
        """delta: dL/dz ya calculado, (n_muestras, n_out).

        Entrada directa para cuando la red calcula dL/dz sin pasar por la
        derivada de la activación (el par Softmax + CCE). Guarda grad_W y
        grad_b y devuelve dL/dx, (n_muestras, n_in).
        """
        if self.x is None:
            raise RuntimeError("backward antes de forward: no hay x ni z guardados.")
        # Las pérdidas ya dividen por n_muestras, así que acá solo se suma
        # sobre el lote: x.T @ delta suma los aportes de cada muestra.
        self.grad_W = self.x.T @ delta  # (n_in, n_out)
        self.grad_b = delta.sum(axis=0, keepdims=True)  # (1, n_out)
        return delta @ self.W.T  # (n_muestras, n_in)

    @property
    def params(self) -> list[np.ndarray]:
        """[W, b]. Son referencias: el optimizador los actualiza in place."""
        return [self.W, self.b]

    @property
    def grads(self) -> list[np.ndarray]:
        """[grad_W, grad_b], en el mismo orden que params."""
        return [self.grad_W, self.grad_b]

    @property
    def n_params(self) -> int:
        """Cantidad de parámetros entrenables: n_in * n_out pesos más n_out bias."""
        return self.W.size + self.b.size
