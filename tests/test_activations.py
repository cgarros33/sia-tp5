import numpy as np
import pytest

from core.activations import (
    ACTIVATIONS,
    Identity,
    ReLU,
    Sigmoid,
    Softmax,
    Step,
    Tanh,
    get_activation,
)

EPS = 1e-5
TOL = 1e-7

# Negativos, cero y valores grandes, en forma (n_muestras, n_features).
Z = np.array(
    [
        [-50.0, -3.0, -1.0, -1e-3, 0.0],
        [1e-3, 0.5, 1.0, 3.0, 50.0],
    ]
)


def numerical_derivative(f, z: np.ndarray) -> np.ndarray:
    """Diferencia central (f(z+ε) − f(z−ε)) / 2ε, elemento a elemento."""
    return (f(z + EPS) - f(z - EPS)) / (2 * EPS)


@pytest.mark.parametrize(
    "activation",
    [
        pytest.param(Identity(), id="identity"),
        pytest.param(Tanh(), id="tanh"),
        pytest.param(Tanh(beta=0.5), id="tanh_beta0.5"),
        pytest.param(Tanh(beta=2.0), id="tanh_beta2"),
        pytest.param(Sigmoid(), id="sigmoid"),
        pytest.param(Sigmoid(beta=0.5), id="sigmoid_beta0.5"),
        pytest.param(Sigmoid(beta=2.0), id="sigmoid_beta2"),
    ],
)
def test_derivada_analitica_coincide_con_numerica(activation):
    analytic = activation.backward(Z)
    numeric = numerical_derivative(activation.forward, Z)
    assert analytic.shape == Z.shape
    np.testing.assert_allclose(analytic, numeric, rtol=0, atol=TOL)


def test_relu_derivada_lejos_del_cero():
    # En z = 0 la ReLU no es derivable: la numérica da 0.5, así que se excluye.
    z = np.array([[-3.0, -1e-3, 1e-3, 3.0]])
    relu = ReLU()
    np.testing.assert_allclose(
        relu.backward(z), numerical_derivative(relu.forward, z), rtol=0, atol=TOL
    )


def test_relu_derivada_en_cero_es_cero():
    assert ReLU().backward(np.array([[0.0]]))[0, 0] == 0.0


def test_step_devuelve_mas_menos_uno():
    z = np.array([[-2.0, 0.0, 3.0]])
    np.testing.assert_array_equal(Step().forward(z), [[-1.0, 1.0, 1.0]])


@pytest.mark.parametrize("activation", [Step(), Softmax()], ids=["step", "softmax"])
def test_backward_no_definido_falla(activation):
    with pytest.raises(NotImplementedError):
        activation.backward(Z)


def test_sigmoid_no_desborda():
    z = np.array([[-1000.0, 1000.0]])
    with np.errstate(over="raise"):
        out = Sigmoid().forward(z)
    np.testing.assert_allclose(out, [[0.0, 1.0]])


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_sigmoid_es_la_logistica_de_la_catedra_con_2beta(beta):
    # 1 / (1 + exp(−2βz)) = ½ (1 + tanh(βz))  (apuntes §8.2.4)
    # atol: en z = −50 la logística da ~1e−44 y ½(1 + tanh) redondea a 0 exacto.
    expected = 0.5 * (1.0 + np.tanh(beta * Z))
    np.testing.assert_allclose(Sigmoid(beta=beta).forward(Z), expected, rtol=1e-12, atol=1e-15)


def test_softmax_valor_conocido():
    # exp(0) = 1, exp(ln 2) = 2  ->  [1/3, 2/3]
    z = np.array([[0.0, np.log(2.0)]])
    np.testing.assert_allclose(Softmax().forward(z), [[1 / 3, 2 / 3]])


def test_softmax_filas_suman_uno():
    out = Softmax().forward(Z)
    assert out.shape == Z.shape
    np.testing.assert_allclose(out.sum(axis=1), np.ones(Z.shape[0]))


def test_softmax_estable_con_valores_grandes():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    with np.errstate(over="raise"):
        out = Softmax().forward(z)
    np.testing.assert_allclose(out, Softmax().forward(z - 1000.0))


def test_registro_construye_cada_activacion():
    for name, cls in ACTIVATIONS.items():
        assert isinstance(get_activation(name), cls)


def test_registro_pasa_parametros():
    assert get_activation("tanh", beta=2.0).beta == 2.0


def test_registro_nombre_desconocido_falla():
    with pytest.raises(ValueError):
        get_activation("tangente")


def test_registro_parametro_desconocido_falla():
    with pytest.raises(TypeError):
        get_activation("relu", beta=2.0)
