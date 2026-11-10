import numpy as np
import pytest

from core.activations import Identity, Softmax, Tanh
from core.initializers import Uniform, Xavier
from core.layers import Dense
from core.network import Network


def red_221_con_pesos_fijos() -> Network:
    """[2, 2, 1] con tanh oculta, identidad a la salida y pesos elegidos a mano."""
    net = Network([2, 2, 1], hidden_activation="tanh", output_activation="identity",
                  rng=np.random.default_rng(0))  # fmt: skip
    hidden, output = net.layers
    hidden.W = np.array([[0.5, -0.5], [0.25, 1.0]])
    hidden.b = np.array([[0.0, 0.5]])
    output.W = np.array([[2.0], [1.0]])
    output.b = np.array([[0.1]])
    return net


def test_forward_221_calculado_a_mano():
    # x = (1, −1)
    # z1 = x @ W1 + b1 = (1·0.5 − 1·0.25 + 0,  1·(−0.5) − 1·1.0 + 0.5) = (0.25, −1.0)
    # a1 = tanh(z1)    = (0.2449186624, −0.7615941560)
    # z2 = a1 @ W2 + b2 = 2·0.2449186624 + 1·(−0.7615941560) + 0.1 = −0.1717568312
    net = red_221_con_pesos_fijos()
    out = net.predict(np.array([[1.0, -1.0]]))

    np.testing.assert_allclose(net.layers[0].z, [[0.25, -1.0]])
    np.testing.assert_allclose(out, [[-0.1717568312]], atol=1e-9)


def test_forward_por_lote_equivale_a_muestra_por_muestra():
    # Segunda muestra x = (0, 2):
    # z1 = (0.5, 2.5), a1 = (0.4621171573, 0.9866142982)
    # z2 = 2·0.4621171573 + 0.9866142982 + 0.1 = 2.0108486128
    net = red_221_con_pesos_fijos()
    X = np.array([[1.0, -1.0], [0.0, 2.0]])
    out = net.predict(X)

    assert out.shape == (2, 1)
    np.testing.assert_allclose(out, [[-0.1717568312], [2.0108486128]], atol=1e-9)


def test_dense_guarda_x_y_z_para_el_backward():
    layer = Dense(3, 2, Tanh(), Xavier(), np.random.default_rng(0))
    x = np.random.default_rng(1).normal(size=(4, 3))
    a = layer.forward(x)

    assert layer.x is x
    np.testing.assert_allclose(layer.z, x @ layer.W + layer.b)
    np.testing.assert_allclose(a, np.tanh(layer.z))


def test_dense_formas_de_parametros_y_bias_en_cero():
    layer = Dense(3, 5, Identity(), Uniform(), np.random.default_rng(0))
    assert layer.W.shape == (3, 5)
    assert layer.b.shape == (1, 5)
    np.testing.assert_array_equal(layer.b, 0.0)


def test_dense_entrada_con_forma_incorrecta_falla():
    layer = Dense(3, 2, Identity(), Uniform(), np.random.default_rng(0))
    with pytest.raises(ValueError):
        layer.forward(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        layer.forward(np.zeros(3))  # vector 1-D en vez de (1, 3)


@pytest.mark.parametrize("sizes", [[2, 0], [0, 3]])
def test_dense_tamanos_invalidos_fallan(sizes):
    with pytest.raises(ValueError):
        Dense(*sizes, Identity(), Uniform(), np.random.default_rng(0))


def test_network_arma_una_capa_por_par_de_tamanos():
    net = Network([4, 8, 6, 3], rng=np.random.default_rng(0))
    assert [(layer.n_in, layer.n_out) for layer in net.layers] == [(4, 8), (8, 6), (6, 3)]
    # 4·8+8 + 8·6+6 + 6·3+3
    assert net.n_params == 40 + 54 + 21


def test_network_activacion_oculta_y_de_salida():
    net = Network([2, 3, 3, 4], hidden_activation="relu", output_activation="softmax",
                  rng=np.random.default_rng(0))  # fmt: skip
    kinds = [type(layer.activation).__name__ for layer in net.layers]
    assert kinds == ["ReLU", "ReLU", "Softmax"]

    out = net.predict(np.random.default_rng(1).normal(size=(5, 2)))
    assert out.shape == (5, 4)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_perceptron_simple_es_una_sola_capa():
    net = Network([3, 1], output_activation="identity", rng=np.random.default_rng(0))
    layer = net.layers[0]
    X = np.random.default_rng(1).normal(size=(6, 3))
    np.testing.assert_allclose(net.predict(X), X @ layer.W + layer.b)


def test_network_acepta_objetos_con_parametros():
    net = Network([2, 1], output_activation=Tanh(beta=3.0), initializer=Uniform(-1.0, 1.0),
                  rng=np.random.default_rng(0))  # fmt: skip
    assert net.layers[0].activation.beta == 3.0
    assert np.abs(net.layers[0].W).max() <= 1.0


def test_network_misma_semilla_misma_salida():
    X = np.random.default_rng(1).normal(size=(5, 3))
    out1 = Network([3, 4, 2], rng=np.random.default_rng(42)).predict(X)
    out2 = Network([3, 4, 2], rng=np.random.default_rng(42)).predict(X)
    np.testing.assert_array_equal(out1, out2)


def test_network_capas_distintas_reciben_pesos_distintos():
    # El rng se consume capa a capa: dos capas iguales no pueden arrancar iguales.
    net = Network([3, 3, 3], rng=np.random.default_rng(0))
    assert not np.array_equal(net.layers[0].W, net.layers[1].W)


def test_network_sin_rng_falla():
    with pytest.raises(TypeError):
        Network([2, 1])


@pytest.mark.parametrize("sizes", [[], [3]])
def test_network_necesita_entrada_y_salida(sizes):
    with pytest.raises(ValueError):
        Network(sizes, rng=np.random.default_rng(0))


def test_network_nombre_desconocido_falla():
    with pytest.raises(ValueError):
        Network([2, 1], output_activation="tangente", rng=np.random.default_rng(0))


def test_softmax_de_salida_no_desborda():
    net = Network([2, 3], output_activation=Softmax(), rng=np.random.default_rng(0))
    with np.errstate(over="raise"):
        out = net.predict(np.array([[1e4, -1e4]]))
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
