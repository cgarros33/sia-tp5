"""Carga de los datasets procesados del ejercicio 2.

Los archivos .npz los escribe utils/preparar_ej2.py. Acá no se construye
ninguna imagen de banderas: ese dataset es siempre una matriz (n, 12).
"""

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "procesadas"
RUTA_RESULTADOS = RAIZ / "resultados" / "ej2"

ARCHIVOS_IMAGEN = {
    "f1": "f1_20x20.npz",
    "nba": "nba_20x24.npz",
    "chino": "chino_16x16.npz",
}
SPLITS = ("train", "val", "test")

# Columnas de banderas que se escalan (proporcion y aspecto). El resto ya está en [0, 1].
COLUMNAS_ESCALADAS = (10, 11)


def _leer_npz(ruta: Path) -> dict:
    with np.load(ruta, allow_pickle=False) as archivo:
        return {clave: archivo[clave] for clave in archivo.files}


def _validar_split(split: np.ndarray, n: int) -> None:
    if split.shape != (n,):
        raise ValueError(f"split tiene forma {split.shape}; se esperaba ({n},)")
    desconocidos = set(map(str, np.unique(split))) - set(SPLITS)
    if desconocidos:
        raise ValueError(f"Etiquetas de split desconocidas: {sorted(desconocidos)}")


def cargar_imagenes(nombre: str, ruta: Path | None = None) -> dict:
    """Devuelve X, y, split, meta y las columnas extra del npz indicado."""
    if nombre not in ARCHIVOS_IMAGEN:
        raise ValueError(f"Dataset desconocido: {nombre!r}. Opciones: {sorted(ARCHIVOS_IMAGEN)}")
    ruta = Path(ruta) if ruta is not None else RUTA_DATOS / ARCHIVOS_IMAGEN[nombre]
    datos = _leer_npz(ruta)
    X = datos["X"]
    if X.ndim != 2:
        raise ValueError(f"X tiene que ser (n, d), llegó {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contiene NaN o Inf")
    if X.min() < 0.0 or X.max() > 1.0:
        raise ValueError(f"X fuera de [0, 1]: min={X.min()}, max={X.max()}")
    _validar_split(datos["split"], len(X))
    datos["meta"] = json.loads(str(datos["meta"].item()))
    return datos


def partes(datos: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Separa (X, y) por split, en el orden fijo train/val/test."""
    split = datos["split"]
    return {s: (datos["X"][split == s], datos["y"][split == s]) for s in SPLITS}


def cargar_banderas(ruta: Path | None = None) -> dict:
    """Carga el CSV procesado: X crudo (n, 12), nombres, split y columnas."""
    ruta = Path(ruta) if ruta is not None else RUTA_DATOS / "flags.npz"
    datos = _leer_npz(ruta)
    X = datos["X"]
    if X.ndim != 2 or X.shape[1] != 12:
        raise ValueError(f"Las banderas tienen que ser (n, 12), llegó {X.shape}")
    if len(datos["columnas"]) != 12:
        raise ValueError("columnas tiene que tener 12 nombres")
    _validar_split(datos["split"], len(X))
    datos["meta"] = json.loads(str(datos["meta"].item()))
    return datos


def escalar_banderas(X: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, dict]:
    """Escala proporcion y aspecto con mínimo/rango de train; el resto queda igual."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 12:
        raise ValueError(f"X tiene que ser (n, 12), llegó {X.shape}")
    if "train" not in set(map(str, split)):
        raise ValueError("No hay filas de train para ajustar el escalador")
    entrenamiento = X[split == "train"]
    minimo = np.zeros(X.shape[1], dtype=np.float64)
    rango = np.ones(X.shape[1], dtype=np.float64)
    for columna in COLUMNAS_ESCALADAS:
        minimo[columna] = entrenamiento[:, columna].min()
        rango[columna] = max(float(entrenamiento[:, columna].max()) - float(minimo[columna]), 1e-8)
    escalado = (X - minimo) / rango
    return escalado.astype(np.float32), {"minimo": minimo.tolist(), "rango": rango.tolist()}


def aplicar_escalador(X: np.ndarray, escalador: dict) -> np.ndarray:
    minimo = np.asarray(escalador["minimo"], dtype=np.float64)
    rango = np.asarray(escalador["rango"], dtype=np.float64)
    return ((np.asarray(X, dtype=np.float64) - minimo) / rango).astype(np.float32)


def desescalar_banderas(X: np.ndarray, escalador: dict) -> np.ndarray:
    minimo = np.asarray(escalador["minimo"], dtype=np.float64)
    rango = np.asarray(escalador["rango"], dtype=np.float64)
    return (np.asarray(X, dtype=np.float64) * rango + minimo).astype(np.float32)
