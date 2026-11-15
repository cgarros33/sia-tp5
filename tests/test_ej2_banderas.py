"""Tests del parser y del render de banderas (solo figuras)."""

from pathlib import Path

import numpy as np
import pytest

from ej2.banderas import leer_banderas, render_flag, vector_bandera

FILA = "Alemania;0;0,0;208,0,0;255,206,0;0;1;1.6667"


def test_parseo_tolera_comas_internas(tmp_path):
    ruta = tmp_path / "banderas.csv"
    ruta.write_text(
        "nombre;r1;g1;b1;r2;g2;b2;r3;g3;b3;rotada;proporcion;aspecto\n" + FILA + "\n",
        encoding="utf-8",
    )
    filas = leer_banderas(ruta)
    assert len(filas) == 1
    fila = filas[0]
    assert fila["nombre"] == "Alemania"
    assert fila["colores"] == [[0, 0, 0], [208, 0, 0], [255, 206, 0]]
    assert fila["rotada"] == 0
    assert fila["proporcion"] == 1.0
    assert fila["aspecto"] == pytest.approx(1.6667)


def test_vector_tiene_doce_columnas_en_orden(tmp_path):
    ruta = tmp_path / "banderas.csv"
    ruta.write_text(
        "nombre;r1;g1;b1;r2;g2;b2;r3;g3;b3;rotada;proporcion;aspecto\n" + FILA + "\n",
        encoding="utf-8",
    )
    vector = vector_bandera(leer_banderas(ruta)[0])
    assert vector.shape == (12,)
    np.testing.assert_allclose(vector[:9], [0, 0, 0, 208 / 255, 0, 0, 1, 206 / 255, 0])
    assert vector[9] == 0.0
    assert vector[10] == 1.0
    assert vector[11] == pytest.approx(1.6667)


def _vector(rotada: int, aspecto: float = 1.5, proporcion: float = 1.0) -> np.ndarray:
    colores = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    valores = [canal for color in colores for canal in color]
    valores.extend([rotada, proporcion, aspecto])
    return np.asarray(valores, dtype=np.float32)


def test_render_horizontal_tiene_tres_franjas():
    imagen = render_flag(_vector(rotada=0), ancho=180)
    assert imagen.shape == (120, 180, 3)
    assert imagen.dtype == np.uint8
    np.testing.assert_array_equal(imagen[0, 90], [255, 0, 0])
    np.testing.assert_array_equal(imagen[60, 90], [0, 255, 0])
    np.testing.assert_array_equal(imagen[-1, 90], [0, 0, 255])


def test_render_vertical_tiene_tres_franjas():
    imagen = render_flag(_vector(rotada=1), ancho=180)
    assert imagen.shape == (120, 180, 3)
    np.testing.assert_array_equal(imagen[60, 0], [255, 0, 0])
    np.testing.assert_array_equal(imagen[60, 90], [0, 255, 0])
    np.testing.assert_array_equal(imagen[60, -1], [0, 0, 255])


def test_render_es_deterministico_y_valida_forma():
    vector = _vector(rotada=1)
    np.testing.assert_array_equal(render_flag(vector), render_flag(vector))
    with pytest.raises(ValueError, match="12 features"):
        render_flag(np.zeros(11))


def test_render_acepta_vector_de_doce(tmp_path):
    ruta = tmp_path / "banderas.csv"
    ruta.write_text(
        "nombre;r1;g1;b1;r2;g2;b2;r3;g3;b3;rotada;proporcion;aspecto\n" + FILA + "\n",
        encoding="utf-8",
    )
    imagen = render_flag(vector_bandera(leer_banderas(ruta)[0]))
    assert imagen.ndim == 3 and imagen.shape[2] == 3
    assert Path(ruta).exists()
