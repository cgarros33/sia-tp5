import numpy as np
import pytest

from ej1.dae import (
    cargar_ae_1a,
    conjuntos_ruidosos,
    medir_limpieza,
    vecino_mas_cercano,
)
from ej1.datos import cargar_font

NIVELES = [0.0, 0.05, 0.1, 0.2, 0.3]


# --- Conjuntos ruidosos ---


def test_forma_e_indices_de_cada_nivel():
    X, _ = cargar_font()
    conjuntos = conjuntos_ruidosos(X, NIVELES, k=3, semilla=1000)
    assert list(conjuntos) == NIVELES
    for X_r, idx in conjuntos.values():
        assert X_r.shape == (32 * 3, 35)
        assert set(np.unique(X_r)) <= {0.0, 1.0}
        # Cada letra aparece k veces.
        np.testing.assert_array_equal(np.bincount(idx), np.full(32, 3))


def test_nivel_cero_son_las_letras_limpias():
    X, _ = cargar_font()
    X_r, idx = conjuntos_ruidosos(X, [0.0], k=2, semilla=1000)[0.0]
    np.testing.assert_array_equal(X_r, X[idx])


def test_tasa_de_ruido_cercana_a_p():
    X, _ = cargar_font()
    for p, (X_r, idx) in conjuntos_ruidosos(X, [0.1, 0.3], k=50, semilla=1000).items():
        assert (X_r != X[idx]).mean() == pytest.approx(p, abs=0.01)


def test_reproducible_y_distinto_con_otra_semilla():
    X, _ = cargar_font()
    a = conjuntos_ruidosos(X, [0.1], k=5, semilla=1000)[0.1][0]
    b = conjuntos_ruidosos(X, [0.1], k=5, semilla=1000)[0.1][0]
    c = conjuntos_ruidosos(X, [0.1], k=5, semilla=2000)[0.1][0]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# --- Medición ---


def test_medir_limpieza_a_mano():
    X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    idx = np.array([0, 0, 1])
    X_rec = np.array(
        [
            [0.1, 0.2, 0.4],  # 0 mal
            [0.9, 0.6, 0.0],  # 2 mal
            [1.0, 0.7, 0.2],  # 1 mal
        ]
    )
    m = medir_limpieza(X, idx, X_rec)
    assert m["pixeles_mal"] == pytest.approx(1.0)
    assert m["peor_letra"] == 2
    assert m["recuperadas"] == pytest.approx(2 / 3)


# --- Referencias ---


def test_vecino_mas_cercano_a_mano():
    X = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    X_r = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],  # a 1 de la 0
            [1.0, 1.0, 1.0, 0.0],  # a 1 de la 1 y de la 2: empate, gana la 1 (menor índice)
            [1.0, 0.0, 0.0, 0.0],  # a 1 de la 0 y de la 2: empate, gana la 0
        ]
    )
    np.testing.assert_array_equal(vecino_mas_cercano(X_r, X), X[[0, 1, 0]])


def test_vecino_mas_cercano_sin_ruido_devuelve_cada_letra():
    X, _ = cargar_font()
    np.testing.assert_array_equal(vecino_mas_cercano(X, X), X)


def test_ae_1a_carga_las_5_semillas_y_reconstruye_las_letras():
    X, _ = cargar_font()
    modelos = cargar_ae_1a()
    assert sorted(modelos) == [0, 1, 2, 3, 4]
    for ae in modelos.values():
        assert ae.latent_dim == 2
        # La config final del 1a llega: la peor letra tiene <= 1 píxel mal.
        assert medir_limpieza(X, np.arange(32), ae.predict(X))["peor_letra"] <= 1
