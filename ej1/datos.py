"""Carga de data/font.h: 32 caracteres de 7×5 píxeles.

Cada carácter es una fila de 7 enteros de 5 bits, uno por renglón de la
imagen; el bit más significativo es la columna de la izquierda. El comentario
de cada fila dice el código y el carácter ("// 0x61, a"), que se usa como
etiqueta.
"""

import re
from pathlib import Path

import numpy as np

RUTA_FONT = Path(__file__).resolve().parent.parent / "data" / "font.h"
ALTO, ANCHO = 7, 5

# {0x00, 0x0e, ...},   // 0x61, a
_FILA = re.compile(r"\{([^}]*)\},?\s*//\s*0x[0-9a-fA-F]+,\s*(.+?)\s*$")


def cargar_font(ruta: Path = RUTA_FONT) -> tuple[np.ndarray, list[str]]:
    """X de forma (32, 35) con valores en {0, 1} y las 32 etiquetas, en el orden del archivo.

    Cada fila de X es la imagen de 7×5 aplanada por renglones.
    """
    renglones, etiquetas = [], []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        m = _FILA.search(linea)
        if m:
            renglones.append([int(v, 16) for v in m.group(1).split(",")])
            etiquetas.append(m.group(2))
    codigos = np.array(renglones)  # (32, 7)
    # Bit 4 (el más significativo) es la columna 0.
    bits = (codigos[:, :, None] >> np.arange(ANCHO - 1, -1, -1)) & 1  # (32, 7, 5)
    return bits.reshape(len(etiquetas), ALTO * ANCHO).astype(float), etiquetas
