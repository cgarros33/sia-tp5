"""Banderas tricolores: parseo del CSV y render para figuras.

El VAE nunca recibe una imagen: su entrada es el vector de 12 features que
arma vector_bandera. render_flag existe solo para las figuras y no se importa
desde datos.py ni desde vae_exp.py.
"""

import csv
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RUTA_BANDERAS = RAIZ / "data" / "flags_tricolor.csv"


def leer_banderas(ruta: Path | None = None) -> list[dict]:
    """Parsea el CSV tolerando las comas internas de los triples RGB."""
    ruta = Path(ruta) if ruta is not None else RUTA_BANDERAS
    filas = []
    with open(ruta, newline="", encoding="utf-8") as archivo:
        lector = csv.reader(archivo, delimiter=";")
        next(lector)
        for fila in lector:
            if not fila:
                continue
            nombre, *resto = fila
            numeros = list(map(int, ",".join(resto[:-3]).split(",")))
            if len(numeros) != 9:
                raise ValueError(f"{nombre}: se esperaban 9 valores RGB, llegaron {numeros}")
            colores = [numeros[i : i + 3] for i in range(0, 9, 3)]
            rotada, proporcion, aspecto = int(resto[-3]), float(resto[-2]), float(resto[-1])
            filas.append(
                {
                    "nombre": nombre,
                    "colores": colores,
                    "rotada": rotada,
                    "proporcion": proporcion,
                    "aspecto": aspecto,
                }
            )
    return filas


def vector_bandera(fila: dict) -> np.ndarray:
    """Vector de 12 columnas en el orden del dataset: 9 colores, rotada, proporcion, aspecto."""
    valores = [canal / 255.0 for color in fila["colores"] for canal in color]
    valores.extend([float(fila["rotada"]), fila["proporcion"], fila["aspecto"]])
    return np.asarray(valores, dtype=np.float32)


def render_flag(vector: np.ndarray, ancho: int = 180) -> np.ndarray:
    """Dibuja una bandera tricolor a partir del vector de 12 features.

    Solo para figuras. proporcion y aspecto tienen que venir en unidades
    originales (si vienen escaladas, hay que desescalarlas antes). Devuelve
    una imagen RGB uint8 de forma (alto, ancho, 3).
    """
    vector = np.asarray(vector, dtype=np.float64).ravel()
    if vector.shape != (12,):
        raise ValueError(f"Se esperaba un vector de 12 features, llegó {vector.shape}")
    colores = np.clip(vector[:9].reshape(3, 3), 0.0, 1.0)
    colores = (colores * 255.0).round().astype(np.uint8)
    rotada = bool(vector[9] >= 0.5)
    proporcion = float(np.clip(vector[10], 0.01, 10.0))
    aspecto = float(np.clip(vector[11], 0.5, 4.0))

    alto = max(1, int(round(ancho / aspecto)))
    largo = ancho if rotada else alto
    fino = largo / (2.0 + proporcion)
    b1 = int(np.clip(round(fino), 1, max(1, largo - 2)))
    b2 = int(np.clip(round(fino * (1.0 + proporcion)), b1 + 1, max(b1 + 1, largo - 1)))
    anchos = [b1, b2 - b1, largo - b2]

    if rotada:
        fila = np.concatenate(
            [np.repeat(colores[i][None, :], anchos[i], axis=0) for i in range(3)], axis=0
        )
        return np.repeat(fila[None, :, :], alto, axis=0)
    franjas = np.concatenate(
        [np.repeat(colores[i][None, :], anchos[i], axis=0) for i in range(3)], axis=0
    )
    return np.repeat(franjas[:, None, :], ancho, axis=1)
