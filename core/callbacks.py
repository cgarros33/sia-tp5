"""Callbacks de entrenamiento: la única forma de observar o cortar un fit.

fit llama a on_train_begin una vez, a on_epoch_end al final de cada época con
los logs de esa época, y a on_train_end al terminar (por épocas agotadas o por
corte temprano). Si algún on_epoch_end devuelve True, el entrenamiento se corta.

Los callbacks con estado (EarlyStopping, AdaptiveEta) tienen state_dict y
load_state_dict para reanudar un entrenamiento cortado. on_train_begin
reinicia el estado, así que load_state_dict se aplica después de él.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from core.network import Network
    from core.optimizers import Optimizer


class Callback:
    """Base con métodos vacíos: cada callback redefine solo los que usa."""

    def on_train_begin(self, network: "Network") -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: "Network") -> bool:
        """Devuelve True para cortar el entrenamiento."""
        return False

    def on_train_end(self, network: "Network") -> None:
        pass


class PrintProgress(Callback):
    """Reporta los logs cada `every` épocas, y siempre la primera.

    write es obligatorio: el motor nunca imprime por su cuenta, así que quien
    llama a fit decide adónde va el texto (print, un logger, una lista).
    """

    def __init__(self, every: int, write: Callable[[str], None]) -> None:
        if every < 1:
            raise ValueError(f"every tiene que ser >= 1, llegó {every}")
        self.every = every
        self.write = write

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: "Network") -> bool:
        if epoch == 1 or epoch % self.every == 0:
            parts = [f"{k}={v:.6g}" for k, v in logs.items() if k != "epoch"]
            self.write(f"epoch {epoch}: " + " ".join(parts))
        return False


class EarlyStopping(Callback):
    """Corta si `monitor` no mejora durante `patience` épocas seguidas.

    mode="min" para pérdidas, "max" para métricas como accuracy. Una mejora
    tiene que superar min_delta. Con restore_best=True, al terminar la red
    vuelve a los pesos de la mejor época, haya cortado o no.
    """

    def __init__(
        self,
        patience: int,
        monitor: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience tiene que ser >= 1, llegó {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode tiene que ser 'min' o 'max', llegó {mode!r}")
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best
        self._reset()

    def _reset(self) -> None:
        self.best: float = np.inf if self.mode == "min" else -np.inf
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None
        self._wait = 0
        self._best_params: list[np.ndarray] | None = None

    def on_train_begin(self, network: "Network") -> None:
        self._reset()

    def state_dict(self) -> dict[str, Any]:
        """best, best_epoch, stopped_epoch, épocas sin mejora y copia de los mejores pesos.

        Todo es escalar de Python salvo best_params (lista de arrays o None).
        """
        return {
            "best": float(self.best),
            "best_epoch": self.best_epoch,
            "stopped_epoch": self.stopped_epoch,
            "wait": self._wait,
            "best_params": (
                None if self._best_params is None else [p.copy() for p in self._best_params]
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restaura lo que guardó state_dict. Va después de on_train_begin, que reinicia."""
        self.best = float(state["best"])
        self.best_epoch = None if state["best_epoch"] is None else int(state["best_epoch"])
        self.stopped_epoch = None if state["stopped_epoch"] is None else int(state["stopped_epoch"])
        self._wait = int(state["wait"])
        best_params = state["best_params"]
        self._best_params = (
            None if best_params is None else [np.array(p, dtype=float) for p in best_params]
        )

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: "Network") -> bool:
        if self.monitor not in logs:
            raise ValueError(
                f"EarlyStopping monitorea {self.monitor!r}, que no está en los logs "
                f"({sorted(logs)}). ¿Falta pasar validation o metric a fit?"
            )
        value = logs[self.monitor]
        # Con NaN (entrenamiento divergido) ninguna comparación da True: no
        # cuenta como mejora y se termina cortando por paciencia.
        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.best_epoch = epoch
            self._wait = 0
            if self.restore_best:
                self._best_params = [p.copy() for p in network.params]
            return False

        self._wait += 1
        if self._wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    def on_train_end(self, network: "Network") -> None:
        if self.restore_best and self._best_params is not None:
            for p, best in zip(network.params, self._best_params, strict=True):
                # In place: la capa tiene que seguir apuntando al mismo array.
                p[...] = best


class AdaptiveEta(Callback):
    """η adaptativo de la cátedra (apuntes §9.3.2), combinable con cualquier optimizador.

    Mira `monitor` al final de cada época y lo compara con la época anterior:
    tras k bajas seguidas suma `a` a optimizer.lr; tras k_prime subas seguidas
    lo multiplica por (1 − b), sin bajar nunca de min_lr. Una suba corta la
    racha de bajas y viceversa; un empate (o NaN) corta las dos. Después de
    cada cambio de η los contadores vuelven a cero.

    El η nuevo se usa desde la época siguiente, así que la columna lr del
    historial muestra el cambio una época después de que se decidió.
    """

    def __init__(
        self,
        optimizer: "Optimizer",
        a: float,
        b: float,
        k: int,
        k_prime: int,
        monitor: str = "train_loss",
        min_lr: float = 1e-8,
    ) -> None:
        if not a > 0:
            raise ValueError(f"a tiene que ser positivo, llegó {a}")
        if not 0 < b < 1:
            raise ValueError(f"b tiene que estar en (0, 1), llegó {b}")
        if k < 1 or k_prime < 1:
            raise ValueError(f"k y k_prime tienen que ser >= 1, llegaron {k} y {k_prime}")
        if min_lr < 0:
            raise ValueError(f"min_lr no puede ser negativo, llegó {min_lr}")
        if not hasattr(optimizer, "lr"):
            raise ValueError(f"{type(optimizer).__name__} no tiene un atributo lr que ajustar")
        self.optimizer = optimizer
        self.a = a
        self.b = b
        self.k = k
        self.k_prime = k_prime
        self.monitor = monitor
        self.min_lr = min_lr
        self._reset()

    def _reset(self) -> None:
        self._previous: float | None = None
        self._decreases = 0
        self._increases = 0

    def on_train_begin(self, network: "Network") -> None:
        self._reset()

    def state_dict(self) -> dict[str, Any]:
        """Valor anterior del monitor y contadores de bajas y subas (escalares de Python).

        El η actual no va acá: vive en optimizer.lr y se guarda con el optimizador.
        """
        return {
            "previous": None if self._previous is None else float(self._previous),
            "decreases": self._decreases,
            "increases": self._increases,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restaura lo que guardó state_dict. Va después de on_train_begin, que reinicia."""
        self._previous = None if state["previous"] is None else float(state["previous"])
        self._decreases = int(state["decreases"])
        self._increases = int(state["increases"])

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: "Network") -> bool:
        if self.monitor not in logs:
            raise ValueError(
                f"AdaptiveEta monitorea {self.monitor!r}, que no está en los logs ({sorted(logs)})."
            )
        value = logs[self.monitor]
        previous, self._previous = self._previous, value
        if previous is None:
            return False

        if value < previous:
            self._decreases += 1
            self._increases = 0
        elif value > previous:
            self._increases += 1
            self._decreases = 0
        else:
            self._decreases = self._increases = 0

        if self._decreases >= self.k:
            self.optimizer.lr += self.a
            self._decreases = self._increases = 0
        elif self._increases >= self.k_prime:
            self.optimizer.lr = max(self.optimizer.lr * (1.0 - self.b), self.min_lr)
            self._decreases = self._increases = 0
        return False


class StopAtTarget(Callback):
    """Corta apenas `monitor` alcanza `target` y guarda esa época en reached_epoch.

    mode="min" llega con monitor <= target (p. ej. una pérdida o la cantidad de
    errores); mode="max" llega con monitor >= target (p. ej. accuracy). Si nunca
    llega, reached_epoch queda en None. Un NaN no cuenta como llegar.
    """

    def __init__(self, monitor: str, target: float, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode tiene que ser 'min' o 'max', llegó {mode!r}")
        self.monitor = monitor
        self.target = target
        self.mode = mode
        self.reached_epoch: int | None = None

    def on_train_begin(self, network: "Network") -> None:
        self.reached_epoch = None

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: "Network") -> bool:
        if self.monitor not in logs:
            raise ValueError(
                f"StopAtTarget monitorea {self.monitor!r}, que no está en los logs "
                f"({sorted(logs)})."
            )
        value = logs[self.monitor]
        if self.mode == "min":
            reached = value <= self.target
        else:
            reached = value >= self.target
        if reached:
            self.reached_epoch = epoch
        return bool(reached)


def _check_has_lr(optimizer: "Optimizer") -> None:
    if not hasattr(optimizer, "lr"):
        raise ValueError(f"{type(optimizer).__name__} no tiene un atributo lr que ajustar")


class ExponentialDecay(Callback):
    """Decaimiento exponencial de η: al final de cada época, optimizer.lr ← lr · rate.

    La época e entrena con lr₀ · rate^(e−1). Como en AdaptiveEta, el η nuevo se
    usa desde la época siguiente.
    """

    def __init__(self, optimizer: "Optimizer", rate: float) -> None:
        if not 0 < rate < 1:
            raise ValueError(f"rate tiene que estar en (0, 1), llegó {rate}")
        _check_has_lr(optimizer)
        self.optimizer = optimizer
        self.rate = rate

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: "Network") -> bool:
        self.optimizer.lr *= self.rate
        return False


class StepDecay(Callback):
    """Decaimiento escalonado de η: al final de las épocas every, 2·every, …, lr ← lr · factor.

    Como en AdaptiveEta, el η nuevo se usa desde la época siguiente.
    """

    def __init__(self, optimizer: "Optimizer", factor: float, every: int) -> None:
        if not 0 < factor < 1:
            raise ValueError(f"factor tiene que estar en (0, 1), llegó {factor}")
        if every < 1:
            raise ValueError(f"every tiene que ser >= 1, llegó {every}")
        _check_has_lr(optimizer)
        self.optimizer = optimizer
        self.factor = factor
        self.every = every

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: "Network") -> bool:
        if epoch % self.every == 0:
            self.optimizer.lr *= self.factor
        return False
