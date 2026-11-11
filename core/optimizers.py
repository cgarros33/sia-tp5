"""Optimizadores: cómo se convierte un gradiente en un paso sobre los parámetros.

Un optimizador recibe las listas paralelas params y grads de la red
(Network.params y Network.grads) y actualiza cada parámetro in place. No
calcula gradientes ni los modifica: eso es responsabilidad del backward.

Fórmulas: apuntes de la cátedra, §7.4 y §9.3. El estado (velocidad, promedios de g y g², t) es
por tensor de parámetros, indexado por su posición en la lista, y se crea en
el primer step. lr es un atributo mutable: AdaptiveEta lo cambia entre épocas.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


class Optimizer(Protocol):
    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None:
        """Actualiza cada params[i] in place usando grads[i], de la misma forma."""
        ...


class _BaseOptimizer:
    """Parte común: validación de lr, contador de pasos, estado y state_dict.

    Cada subclase declara su nombre (kind), sus hiperparámetros además de lr
    (hparams), los buffers de estado que guarda por tensor (buffers), y
    redefine _update con el paso de un tensor.
    """

    kind: str = ""
    hparams: tuple[str, ...] = ()
    buffers: tuple[str, ...] = ()

    def __init__(self, lr: float) -> None:
        self.lr = lr
        self.t = 0  # pasos dados (Adam lo usa para corregir el sesgo)
        # buffer -> una lista con un array por tensor de parámetros
        self._state: dict[str, list[np.ndarray]] = {}
        self._validate()

    def _validate(self) -> None:
        if not self.lr > 0:
            raise ValueError(f"lr tiene que ser positivo, llegó {self.lr}")

    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None:
        if len(params) != len(grads):
            raise ValueError(f"{len(params)} parámetros y {len(grads)} gradientes")
        if self.buffers and not self._state:
            self._state = {name: [np.zeros_like(p) for p in params] for name in self.buffers}
        for name, arrays in self._state.items():
            if len(arrays) != len(params):
                raise ValueError(
                    f"El estado {name!r} tiene {len(arrays)} tensores y llegaron {len(params)}"
                )
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            self._update(i, p, g)

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> None:
        """Paso in place sobre el tensor i-ésimo, p, con gradiente g."""
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Hiperparámetros, t y buffers en un dict plano.

        Los buffers van como "<buffer>_<i>" (un array por tensor), así el dict se
        puede guardar con np.savez(**state) sin pickle.
        """
        state: dict[str, Any] = {"kind": self.kind, "lr": self.lr, "t": self.t}
        for name in self.hparams:
            state[name] = getattr(self, name)
        for name, arrays in self._state.items():
            for i, a in enumerate(arrays):
                state[f"{name}_{i}"] = a.copy()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restaura lo que guardó state_dict (también leído de un np.load)."""
        kind = str(state.get("kind"))
        if kind != self.kind:
            raise ValueError(f"El estado es de un optimizador {kind!r}, no {self.kind!r}")
        self.lr = float(state["lr"])
        self.t = int(state["t"])
        for name in self.hparams:
            setattr(self, name, float(state[name]))
        self._validate()
        self._state = {}
        for name in self.buffers:
            n = sum(1 for k in state if k.startswith(f"{name}_"))
            if n:
                self._state[name] = [np.array(state[f"{name}_{i}"], dtype=float) for i in range(n)]


def _check_unit(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} tiene que estar en [0, 1), llegó {value}")


def _check_eps(eps: float) -> None:
    if not eps > 0:
        raise ValueError(f"eps tiene que ser positivo, llegó {eps}")


class GD(_BaseOptimizer):
    """Descenso por gradiente: θ ← θ − lr · g (apuntes §7.4.2).

    Si es batch completo, estocástico o mini-batch lo decide fit con el tamaño
    del lote; el paso es el mismo.
    """

    kind = "gd"

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> None:
        # In place: p es la misma matriz que guarda la capa.
        p -= self.lr * g


class Momentum(_BaseOptimizer):
    """Momentum de la cátedra (apuntes §9.3.1).

    Δθ(t+1) = −lr · g + alpha · Δθ(t);  θ ← θ + Δθ(t+1).
    """

    kind = "momentum"
    hparams = ("alpha",)
    buffers = ("velocity",)

    def __init__(self, lr: float = 0.01, alpha: float = 0.9) -> None:
        self.alpha = alpha
        super().__init__(lr)

    def _validate(self) -> None:
        super()._validate()
        _check_unit("alpha", self.alpha)

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> None:
        v = self._state["velocity"][i]
        v *= self.alpha
        v -= self.lr * g
        p += v


class RMSProp(_BaseOptimizer):
    """RMSProp de la cátedra (apuntes §9.3.3).

    S = gamma · S + (1 − gamma) · g²;  θ ← θ − lr · g / sqrt(S + eps).
    """

    kind = "rmsprop"
    hparams = ("gamma", "eps")
    buffers = ("sq_avg",)

    def __init__(self, lr: float = 0.001, gamma: float = 0.9, eps: float = 1e-8) -> None:
        self.gamma = gamma
        self.eps = eps
        super().__init__(lr)

    def _validate(self) -> None:
        super()._validate()
        _check_unit("gamma", self.gamma)
        _check_eps(self.eps)

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> None:
        s = self._state["sq_avg"][i]
        s *= self.gamma
        s += (1.0 - self.gamma) * g * g
        p -= self.lr * g / np.sqrt(s + self.eps)


class Adam(_BaseOptimizer):
    """Adam (Kingma & Ba), apuntes §9.3.4.

    m = beta1 · m + (1 − beta1) · g;  v = beta2 · v + (1 − beta2) · g²;
    m̂ = m / (1 − beta1^t);  v̂ = v / (1 − beta2^t);  θ ← θ − lr · m̂ / (sqrt(v̂) + eps).
    """

    kind = "adam"
    hparams = ("beta1", "beta2", "eps")
    buffers = ("m", "v")

    def __init__(
        self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        super().__init__(lr)

    def _validate(self) -> None:
        super()._validate()
        _check_unit("beta1", self.beta1)
        _check_unit("beta2", self.beta2)
        _check_eps(self.eps)

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> None:
        m, v = self._state["m"][i], self._state["v"][i]
        m *= self.beta1
        m += (1.0 - self.beta1) * g
        v *= self.beta2
        v += (1.0 - self.beta2) * g * g
        m_hat = m / (1.0 - self.beta1**self.t)
        v_hat = v / (1.0 - self.beta2**self.t)
        p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdaGrad(_BaseOptimizer):
    """AdaGrad (apuntes §7.4.8).

    G = G + g²;  θ ← θ − lr · g / (sqrt(G) + eps).
    """

    kind = "adagrad"
    hparams = ("eps",)
    buffers = ("sq_sum",)

    def __init__(self, lr: float = 0.01, eps: float = 1e-8) -> None:
        self.eps = eps
        super().__init__(lr)

    def _validate(self) -> None:
        super()._validate()
        _check_eps(self.eps)

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> None:
        s = self._state["sq_sum"][i]
        s += g * g
        p -= self.lr * g / (np.sqrt(s) + self.eps)


OPTIMIZERS: Mapping[str, type] = MappingProxyType(
    {
        "gd": GD,
        "momentum": Momentum,
        "rmsprop": RMSProp,
        "adam": Adam,
        "adagrad": AdaGrad,
    }
)


def get_optimizer(name: str, **kwargs: Any) -> Optimizer:
    """Construye un optimizador a partir de su nombre, p. ej. get_optimizer("gd", lr=0.1).

    Un nombre o parámetro desconocido es un error, no se ignora.
    """
    try:
        cls = OPTIMIZERS[name]
    except KeyError:
        raise ValueError(
            f"Optimizador desconocido: {name!r}. Opciones: {sorted(OPTIMIZERS)}"
        ) from None
    return cls(**kwargs)
