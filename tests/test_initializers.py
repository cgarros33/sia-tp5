import numpy as np
import pytest

from core.initializers import INITIALIZERS, He, Uniform, Xavier, get_initializer

ALL = [
    pytest.param(Uniform(), id="uniform"),
    pytest.param(Xavier(), id="xavier"),
    pytest.param(He(), id="he"),
]


@pytest.mark.parametrize("init", ALL)
def test_forma_es_n_in_por_n_out(init):
    W = init(3, 5, np.random.default_rng(0))
    assert W.shape == (3, 5)


@pytest.mark.parametrize("init", ALL)
def test_misma_semilla_mismos_pesos(init):
    W1 = init(4, 6, np.random.default_rng(42))
    W2 = init(4, 6, np.random.default_rng(42))
    np.testing.assert_array_equal(W1, W2)


@pytest.mark.parametrize("init", ALL)
def test_semillas_distintas_pesos_distintos(init):
    W1 = init(4, 6, np.random.default_rng(1))
    W2 = init(4, 6, np.random.default_rng(2))
    assert not np.array_equal(W1, W2)


@pytest.mark.parametrize("init", ALL)
def test_no_usa_el_estado_global(init):
    # Consumir el generador global no puede cambiar los pesos.
    W1 = init(4, 6, np.random.default_rng(7))
    np.random.rand(100)
    W2 = init(4, 6, np.random.default_rng(7))
    np.testing.assert_array_equal(W1, W2)


def test_uniform_respeta_limites():
    W = Uniform(-0.2, 0.3)(50, 50, np.random.default_rng(0))
    assert W.min() >= -0.2 and W.max() < 0.3


def test_uniform_limites_invalidos_fallan():
    with pytest.raises(ValueError):
        Uniform(1.0, 1.0)


def test_xavier_respeta_limite():
    n_in, n_out = 30, 20
    r = np.sqrt(6.0 / (n_in + n_out))
    W = Xavier()(n_in, n_out, np.random.default_rng(0))
    assert np.abs(W).max() <= r


def test_he_varianza_aproximada():
    n_in = 200
    W = He()(n_in, 500, np.random.default_rng(0))
    # 100 000 muestras: la varianza empírica queda a menos de 2% de 2/n_in.
    assert W.var() == pytest.approx(2.0 / n_in, rel=0.02)
    assert W.mean() == pytest.approx(0.0, abs=5e-3)


def test_registro_construye_cada_inicializador():
    for name, cls in INITIALIZERS.items():
        assert isinstance(get_initializer(name), cls)


def test_registro_pasa_parametros():
    init = get_initializer("uniform", low=-1.0, high=1.0)
    assert (init.low, init.high) == (-1.0, 1.0)


def test_registro_nombre_desconocido_falla():
    with pytest.raises(ValueError):
        get_initializer("glorot")


def test_registro_parametro_desconocido_falla():
    with pytest.raises(TypeError):
        get_initializer("xavier", gain=2.0)
