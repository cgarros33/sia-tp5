"""Data augmentation: variantes aleatorias de los lotes de entrenamiento.

fit aplica la augmentation solo a los lotes de train, con
un generador propio, en cada época; nunca en evaluate ni predict, así que la
validación y el test no se tocan. Convención de formas: X es
(n_muestras, n_features) y la salida tiene la misma forma. Ninguna
augmentation modifica X in place: siempre devuelven un array nuevo.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


class Augmentation(Protocol):
    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Devuelve una versión nueva de X, (n_muestras, n_features), sin modificar X."""
        ...


class GaussianNoise:
    """x' = x + N(0, sigma²), elemento a elemento, recortado a clip si se pasa.

    clip = (lo, hi): escalares o arrays por feature (forma (n_features,)), por
    ejemplo el rango de cada columna después de normalizar.
    """

    def __init__(self, sigma: float, clip: tuple[Any, Any] | None = None) -> None:
        if sigma < 0:
            raise ValueError(f"sigma no puede ser negativo, llegó {sigma}")
        if clip is not None:
            lo, hi = clip
            if np.any(np.asarray(lo) > np.asarray(hi)):
                raise ValueError(f"clip tiene que ser (lo, hi) con lo <= hi, llegó {clip}")
            clip = (lo, hi)
        self.sigma = sigma
        self.clip = clip

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = X + rng.normal(0.0, self.sigma, size=X.shape)
        if self.clip is not None:
            np.clip(out, *self.clip, out=out)
        return out


class RandomShift:
    """Traslada cada imagen un entero aleatorio en [−max_px, max_px] en x y en y.

    Cada fila de X se interpreta como una imagen de forma image_shape, p. ej.
    (8, 8). Lo que queda descubierto se rellena con el mínimo de la fila (el
    fondo) y lo que se sale de la imagen se pierde. Cada muestra tiene su
    propio desplazamiento: el loop es sobre los (2·max_px + 1)² desplazamientos
    posibles, nunca sobre las muestras.
    """

    def __init__(self, max_px: int, image_shape: Sequence[int]) -> None:
        image_shape = tuple(int(s) for s in image_shape)
        if len(image_shape) != 2 or min(image_shape) < 1:
            raise ValueError(f"image_shape tiene que ser (alto, ancho) > 0, llegó {image_shape}")
        if not 0 <= max_px < min(image_shape):
            raise ValueError(
                f"max_px tiene que estar en [0, {min(image_shape)}) para {image_shape}, "
                f"llegó {max_px}"
            )
        self.max_px = int(max_px)
        self.image_shape = image_shape

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = self.image_shape
        n = X.shape[0]
        if X.shape[1] != h * w:
            raise ValueError(
                f"X tiene {X.shape[1]} features y image_shape {self.image_shape} necesita {h * w}"
            )
        images = X.reshape(n, h, w)
        shifts = rng.integers(-self.max_px, self.max_px + 1, size=(n, 2))
        background = X.min(axis=1)
        out = np.broadcast_to(background[:, None, None], (n, h, w)).copy()

        for dy in range(-self.max_px, self.max_px + 1):
            dst_y, src_y = _shift_slices(dy, h)
            for dx in range(-self.max_px, self.max_px + 1):
                rows = np.flatnonzero((shifts[:, 0] == dy) & (shifts[:, 1] == dx))
                if rows.size == 0:
                    continue
                dst_x, src_x = _shift_slices(dx, w)
                out[rows, dst_y, dst_x] = images[rows, src_y, src_x]
        return out.reshape(n, h * w)


def _shift_slices(d: int, size: int) -> tuple[slice, slice]:
    """(destino, origen) para trasladar un eje de largo size en d posiciones, con |d| < size."""
    if d >= 0:
        return slice(d, size), slice(0, size - d)
    return slice(0, size + d), slice(-d, size)


class BitFlip:
    """Invierte cada elemento (x' = 1 − x) con probabilidad p; pensado para datos en {0, 1}.

    p es un float (el mismo nivel para todas las filas) o un par (p_min, p_max):
    en ese caso cada fila sortea su propio p ~ U(p_min, p_max), así un mismo
    lote mezcla niveles de ruido.
    """

    def __init__(self, p: float | Sequence[float]) -> None:
        if isinstance(p, Sequence):
            if len(p) != 2:
                raise ValueError(f"p tiene que ser un float o un par (p_min, p_max), llegó {p}")
            p = (float(p[0]), float(p[1]))
            if not 0.0 <= p[0] <= p[1] <= 1.0:
                raise ValueError(f"p tiene que cumplir 0 <= p_min <= p_max <= 1, llegó {p}")
        elif not 0.0 <= p <= 1.0:
            raise ValueError(f"p tiene que estar en [0, 1], llegó {p}")
        self.p = p

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if isinstance(self.p, tuple):
            p = rng.uniform(*self.p, size=(X.shape[0], 1))
        else:
            p = self.p
        invertir = rng.random(X.shape) < p
        return np.where(invertir, 1.0 - X, X)


class Compose:
    """Aplica varias augmentations en orden, todas con el mismo generador."""

    def __init__(self, augmentations: Sequence[Augmentation]) -> None:
        self.augmentations = list(augmentations)

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = X.copy()
        for aug in self.augmentations:
            out = aug(out, rng)
        return out


AUGMENTATIONS: Mapping[str, type] = MappingProxyType(
    {
        "gaussian_noise": GaussianNoise,
        "random_shift": RandomShift,
        "bit_flip": BitFlip,
    }
)


def get_augmentation(config: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Augmentation:
    """Construye una augmentation desde la config del experimento.

    Un dict {"kind": nombre, **parámetros}, p. ej.
    {"kind": "gaussian_noise", "sigma": 0.05, "clip": [0, 1]}, o una lista de
    esos dicts, que se aplican en orden (Compose). Un kind o parámetro
    desconocido es un error, no se ignora.
    """
    if not isinstance(config, Mapping):
        return Compose([get_augmentation(c) for c in config])
    params = dict(config)
    if "kind" not in params:
        raise ValueError(f"Falta 'kind' en la config de augmentation: {params}")
    kind = params.pop("kind")
    try:
        cls = AUGMENTATIONS[kind]
    except KeyError:
        raise ValueError(
            f"Augmentation desconocida: {kind!r}. Opciones: {sorted(AUGMENTATIONS)}"
        ) from None
    if "clip" in params and params["clip"] is not None:
        params["clip"] = tuple(params["clip"])
    return cls(**params)
