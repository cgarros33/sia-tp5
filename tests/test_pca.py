import numpy as np
import pytest

from core.losses import MSE
from core.pca import PCA


def _datos(n=60, d=6, seed=0):
    # Escalas distintas por columna: autovalores bien separados.
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) @ np.diag([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])[:d, :d] + 3.0


def test_formas():
    X = _datos()
    pca = PCA(2).fit(X)
    assert pca.components.shape == (2, 6)
    assert pca.mean.shape == (6,)
    assert pca.explained_variance.shape == (2,)
    Z = pca.transform(X)
    assert Z.shape == (60, 2)
    assert pca.inverse_transform(Z).shape == X.shape


def test_coincide_con_los_autovectores_de_la_covarianza():
    X = _datos()
    k = 3
    pca = PCA(k).fit(X)
    autovalores, autovectores = np.linalg.eigh(np.cov(X, rowvar=False))
    orden = np.argsort(autovalores)[::-1][:k]
    np.testing.assert_allclose(pca.explained_variance, autovalores[orden], rtol=1e-10)
    # Cada componente es el autovector correspondiente, salvo el signo.
    productos = np.abs(np.sum(pca.components * autovectores[:, orden].T, axis=1))
    np.testing.assert_allclose(productos, np.ones(k), rtol=1e-10)


def test_componentes_ortonormales():
    pca = PCA(4).fit(_datos())
    np.testing.assert_allclose(pca.components @ pca.components.T, np.eye(4), atol=1e-12)


def test_proyeccion_centrada_y_varianza_explicada():
    X = _datos()
    pca = PCA(2).fit(X)
    Z = pca.transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.var(axis=0, ddof=1), pca.explained_variance, rtol=1e-10)


def test_error_de_reconstruccion_es_la_varianza_descartada():
    X = _datos()
    n, d = X.shape
    autovalores = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    pca = PCA(2).fit(X)
    error = MSE().value(X, pca.inverse_transform(pca.transform(X)))
    # mean((X − X')²) = Σ_{i > k} λ_i · (n − 1) / (n · d)
    assert error == pytest.approx(autovalores[2:].sum() * (n - 1) / (n * d), rel=1e-10)


def test_con_todas_las_componentes_reconstruye_exacto():
    X = _datos()
    pca = PCA(6).fit(X)
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(X)), X, atol=1e-10)


def test_signo_determinista():
    # La entrada de mayor módulo de cada componente queda positiva.
    pca = PCA(3).fit(_datos())
    idx = np.argmax(np.abs(pca.components), axis=1)
    assert np.all(pca.components[np.arange(3), idx] > 0)
    otra = PCA(3).fit(_datos())
    np.testing.assert_array_equal(pca.components, otra.components)


@pytest.mark.parametrize("k", [0, 7])
def test_cantidad_de_componentes_invalida_falla(k):
    with pytest.raises(ValueError):
        PCA(k).fit(_datos())
