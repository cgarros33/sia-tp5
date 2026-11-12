import numpy as np
import pytest

from core.losses import MSE
from core.metrics import pixel_errors
from core.pca import PCA
from ej1.datos import ALTO, ANCHO, cargar_font


def test_forma_y_valores():
    X, etiquetas = cargar_font()
    assert X.shape == (32, 35)
    assert (ALTO, ANCHO) == (7, 5)
    assert set(np.unique(X)) == {0.0, 1.0}
    assert X.dtype == float


def test_etiquetas_son_los_32_caracteres_en_orden():
    _, etiquetas = cargar_font()
    assert len(etiquetas) == 32
    assert etiquetas[0] == "`"
    assert etiquetas[1:27] == [chr(c) for c in range(ord("a"), ord("z") + 1)]
    assert etiquetas[27:] == ["{", "|", "}", "~", "DEL"]


def test_letra_o_conocida():
    X, etiquetas = cargar_font()
    o = X[etiquetas.index("o")].reshape(ALTO, ANCHO)
    esperada = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ]
    )
    np.testing.assert_array_equal(o, esperada)


def test_bit_mas_significativo_es_la_columna_izquierda():
    # 'f' fila 0 = 0x06 = 00110: los dos píxeles encendidos están en las columnas 2 y 3.
    X, etiquetas = cargar_font()
    f = X[etiquetas.index("f")].reshape(ALTO, ANCHO)
    np.testing.assert_array_equal(f[0], [0, 0, 1, 1, 0])


def test_del_tiene_todos_los_pixeles_encendidos():
    X, etiquetas = cargar_font()
    assert X[etiquetas.index("DEL")].sum() == 35


def test_las_32_letras_son_distintas():
    X, _ = cargar_font()
    assert len({tuple(fila) for fila in X}) == 32


# --- Números de referencia: PCA 2D sobre font.h ---


def test_numeros_de_referencia_sobre_font():
    # Medidos por el equipo: MSE 0,1491; 7,19 píxeles mal por letra; peor 15; 0 letras con <= 1.
    X, _ = cargar_font()
    pca = PCA(2).fit(X)
    X_rec = pca.inverse_transform(pca.transform(X))
    errores = pixel_errors(X, X_rec)
    assert MSE().value(X, X_rec) == pytest.approx(0.1491, abs=5e-5)
    assert errores.mean() == pytest.approx(7.19, abs=5e-3)
    assert errores.max() == 15
    assert np.count_nonzero(errores <= 1) == 0
