"""Registro época a época de un entrenamiento.

Cada época es un dict con las claves de COLUMNS que correspondan: val_loss y
val_metric solo aparecen si fit recibió validación, las métricas solo si
recibió una función de métrica, y lr solo si el optimizador tiene ese
atributo. Además se aceptan columnas dinámicas train_<x> / val_<x>, una por
cada métrica de `metrics` en fit. Al exportar, lo que falta queda vacío.
"""

import csv
from os import PathLike

import numpy as np

# Columnas fijas de history.csv, en este orden.
COLUMNS = ("epoch", "train_loss", "val_loss", "train_metric", "val_metric", "lr", "elapsed_s")

# Prefijos de las columnas dinámicas: train_<métrica> y val_<métrica>.
DYNAMIC_PREFIXES = ("train_", "val_")


def is_valid_column(name: str) -> bool:
    """True si name es una columna fija o una dinámica train_<x> / val_<x> con x no vacío."""
    if name in COLUMNS:
        return True
    return any(name.startswith(p) and len(name) > len(p) for p in DYNAMIC_PREFIXES)


class History:
    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []
        # "diverged" si fit cortó porque train_loss dejó de ser finito.
        self.status = "ok"

    def append(self, logs: dict[str, float]) -> None:
        unknown = [k for k in logs if not is_valid_column(k)]
        if unknown:
            raise ValueError(f"Claves desconocidas en el historial: {sorted(unknown)}")
        self.records.append(dict(logs))

    def column(self, name: str) -> np.ndarray:
        """Valores de una columna por época, con NaN donde no se registró. Forma (n_epocas,)."""
        if not is_valid_column(name):
            raise ValueError(
                f"Columna desconocida: {name!r}. Opciones: {list(COLUMNS)} o train_<x> / val_<x>"
            )
        return np.array([r.get(name, np.nan) for r in self.records], dtype=float)

    @property
    def dynamic_columns(self) -> list[str]:
        """Columnas train_<x> / val_<x> registradas en alguna época, en orden alfabético."""
        return sorted({k for r in self.records for k in r if k not in COLUMNS})

    def to_csv(self, path: str | PathLike) -> None:
        """Escribe una fila por época: COLUMNS y después las columnas dinámicas."""
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[*COLUMNS, *self.dynamic_columns], restval="")
            writer.writeheader()
            writer.writerows(self.records)

    def __len__(self) -> int:
        return len(self.records)
