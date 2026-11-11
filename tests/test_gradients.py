"""Gradient check numérico: el test más importante del motor.

Para cada parámetro θ de la red compara el gradiente del backward contra
(L(θ+ε) − L(θ−ε)) / 2ε, con el error relativo
|g_a − g_n| / max(|g_a|, |g_n|, 1e-8) < 1e-6.
"""

import numpy as np
import pytest

from core.activations import Tanh
from core.initializers import Xavier
from core.layers import Dense
from core.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, get_loss
from core.network import Network

EPS = 1e-5
REL_TOL = 1e-6
N_SAMPLES = 5


def make_targets(loss_name: str, n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Targets válidos para cada pérdida, forma (n, m)."""
    if loss_name == "categorical_crossentropy":
        return np.eye(m)[rng.integers(0, m, size=n)]
    if loss_name == "binary_crossentropy":
        return rng.integers(0, 2, size=(n, m)).astype(float)
    return rng.normal(size=(n, m))


def max_relative_error(
    net: Network, X: np.ndarray, y: np.ndarray, loss, l2: float = 0.0, l2_in_backward=None
) -> float:
    """Corre forward + backward y devuelve el peor error relativo entre todos los parámetros.

    El objetivo numérico es loss.value + net.l2_penalty(l2). l2_in_backward es
    el λ que recibe el backward (por default el mismo l2): pasarle otro simula
    un bug en el término L2 del gradiente.
    """
    l2_in_backward = l2 if l2_in_backward is None else l2_in_backward
    net.backward(y, net.predict(X), loss, l2=l2_in_backward)
    analytic = [g.copy() for g in net.grads]

    def objective() -> float:
        return loss.value(y, net.predict(X)) + net.l2_penalty(l2)

    worst = 0.0
    for param, g_a in zip(net.params, analytic, strict=True):
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + EPS
            plus = objective()
            param[idx] = original - EPS
            minus = objective()
            param[idx] = original

            g_n = (plus - minus) / (2 * EPS)
            rel = abs(g_a[idx] - g_n) / max(abs(g_a[idx]), abs(g_n), 1e-8)
            worst = max(worst, rel)
    return worst


def build(layers, hidden, output, seed=0):
    rng = np.random.default_rng(seed)
    net = Network(layers, hidden_activation=hidden, output_activation=output, rng=rng)
    X = rng.normal(size=(N_SAMPLES, layers[0]))
    return net, X, rng


# Las combinaciones que exige el plan: cada activación oculta con MSE y con CCE.
REQUIRED = [
    (hidden, output, loss)
    for hidden in ["tanh", "sigmoid", "relu"]
    for output, loss in [("identity", "mse"), ("softmax", "categorical_crossentropy")]
]


@pytest.mark.parametrize(
    "hidden, output, loss_name", REQUIRED, ids=[f"{h}-{lo}" for h, _, lo in REQUIRED]
)
def test_gradient_check_combinaciones_del_plan(hidden, output, loss_name):
    net, X, rng = build([3, 4, 2], hidden, output)
    y = make_targets(loss_name, N_SAMPLES, 2, rng)
    assert max_relative_error(net, X, y, get_loss(loss_name)) < REL_TOL


@pytest.mark.parametrize(
    "output, loss_name",
    [
        ("tanh", "mse"),  # perceptrón no lineal con salida tanh
        ("sigmoid", "mse"),
        ("sigmoid", "binary_crossentropy"),
    ],
)
def test_gradient_check_otras_salidas(output, loss_name):
    net, X, rng = build([3, 4, 2], "tanh", output)
    y = make_targets(loss_name, N_SAMPLES, 2, rng)
    if output == "tanh":
        y = np.tanh(y)  # target dentro de la imagen de tanh
    assert max_relative_error(net, X, y, get_loss(loss_name)) < REL_TOL


@pytest.mark.parametrize(
    "layers", [[3, 1], [3, 2], [3, 5, 4, 2], [2, 3, 2, 1]], ids=lambda s: str(s)
)
def test_gradient_check_arquitecturas(layers):
    # Sin capas ocultas (perceptrón simple) y con varias: verifica el encadenado.
    net, X, rng = build(layers, "tanh", "identity")
    y = make_targets("mse", N_SAMPLES, layers[-1], rng)
    assert max_relative_error(net, X, y, MSE()) < REL_TOL


@pytest.mark.parametrize("layers", [[3, 5, 4, 3], [4, 6, 3]], ids=lambda s: str(s))
def test_gradient_check_softmax_con_varias_capas(layers):
    net, X, rng = build(layers, "relu", "softmax", seed=3)
    y = make_targets("categorical_crossentropy", N_SAMPLES, layers[-1], rng)
    assert max_relative_error(net, X, y, CategoricalCrossEntropy()) < REL_TOL


# --- L2 / weight decay ---


@pytest.mark.parametrize("l2", [0.01, 0.5])
def test_gradient_check_con_l2(l2):
    # Gradiente de loss.value + l2/2 · Σ‖W‖² contra el analítico del backward con l2.
    net, X, rng = build([3, 4, 2], "tanh", "identity")
    y = make_targets("mse", N_SAMPLES, 2, rng)
    assert max_relative_error(net, X, y, MSE(), l2=l2) < REL_TOL


def test_l2_no_se_aplica_a_los_bias():
    net, X, rng = build([3, 4, 2], "tanh", "identity")
    y = make_targets("mse", N_SAMPLES, 2, rng)
    y_pred = net.predict(X)
    net.backward(y, y_pred, MSE())
    plain = [g.copy() for g in net.grads]
    net.backward(y, y_pred, MSE(), l2=0.5)
    for layer, g_W, g_b in zip(net.layers, plain[0::2], plain[1::2], strict=True):
        np.testing.assert_allclose(layer.grad_W, g_W + 0.5 * layer.W)
        np.testing.assert_array_equal(layer.grad_b, g_b)


def test_gradient_check_detecta_l2_mal_aplicado():
    # Sin el término (o con el factor mal) el chequeo contra la pérdida penalizada falla.
    net, X, rng = build([3, 4, 2], "tanh", "identity")
    y = make_targets("mse", N_SAMPLES, 2, rng)
    assert max_relative_error(net, X, y, MSE(), l2=0.5, l2_in_backward=0.0) > 1e-3
    assert max_relative_error(net, X, y, MSE(), l2=0.5, l2_in_backward=1.0) > 1e-3


# --- El gradient check tiene que fallar con los errores típicos de backprop ---
# Si alguno de estos pasara, el test de arriba no probaría nada.

# Referencia al método real, tomada antes de que monkeypatch lo reemplace.
_backward_z_original = Dense.backward_z


def _derivada_evaluada_en_a(self, grad_a):
    # Bug: derivada de la activación evaluada en a = f(z) en vez de en z.
    return self.backward_z(grad_a * self.activation.backward(self.activation.forward(self.z)))


def _bias_sin_sumar_sobre_el_lote(self, delta):
    # Bug: se toma solo la primera muestra en vez de sumar sobre el lote.
    dx = _backward_z_original(self, delta)
    self.grad_b = delta[:1]
    return dx


def _signo_invertido(self, delta):
    dx = _backward_z_original(self, delta)
    self.grad_W = -self.grad_W
    return dx


def _sin_derivada_de_activacion(self, grad_a):
    return self.backward_z(grad_a)


@pytest.mark.parametrize(
    "method, bug",
    [
        ("backward", _derivada_evaluada_en_a),
        ("backward_z", _bias_sin_sumar_sobre_el_lote),
        ("backward_z", _signo_invertido),
        ("backward", _sin_derivada_de_activacion),
    ],
    ids=["derivada_en_a", "bias_sin_sumar", "signo_invertido", "sin_derivada"],
)
def test_gradient_check_detecta_bugs(monkeypatch, method, bug):
    monkeypatch.setattr(Dense, method, bug)
    net, X, rng = build([3, 4, 2], "tanh", "identity")
    y = make_targets("mse", N_SAMPLES, 2, rng)
    assert max_relative_error(net, X, y, MSE()) > 1e-3


# --- Contrato del backward ---


def test_dense_backward_devuelve_grad_de_la_entrada():
    layer = Dense(3, 2, Tanh(), Xavier(), np.random.default_rng(0))
    x = np.random.default_rng(1).normal(size=(4, 3))
    layer.forward(x)
    dx = layer.backward(np.ones((4, 2)))
    assert dx.shape == (4, 3)
    assert layer.grad_W.shape == layer.W.shape
    assert layer.grad_b.shape == layer.b.shape


def test_backward_antes_de_forward_falla():
    layer = Dense(3, 2, Tanh(), Xavier(), np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        layer.backward(np.ones((4, 2)))
    with pytest.raises(RuntimeError):
        layer.backward_z(np.ones((4, 2)))


def test_softmax_con_otra_perdida_falla():
    net, X, rng = build([3, 2], "tanh", "softmax")
    y = make_targets("categorical_crossentropy", N_SAMPLES, 2, rng)
    with pytest.raises(ValueError):
        net.backward(y, net.predict(X), MSE())


def test_step_no_se_entrena_con_backprop():
    net, X, rng = build([3, 1], "tanh", "step")
    y = np.ones((N_SAMPLES, 1))
    with pytest.raises(NotImplementedError):
        net.backward(y, net.predict(X), MSE())


def test_params_son_referencias_a_los_pesos_de_las_capas():
    net, _, _ = build([3, 4, 2], "tanh", "identity")
    params = net.params
    assert len(params) == 2 * len(net.layers)
    params[0] += 1.0  # in place
    np.testing.assert_array_equal(params[0], net.layers[0].W)


def test_grads_tienen_la_forma_de_params():
    net, X, rng = build([3, 4, 2], "sigmoid", "sigmoid")
    y = make_targets("binary_crossentropy", N_SAMPLES, 2, rng)
    net.backward(y, net.predict(X), BinaryCrossEntropy())
    assert [g.shape for g in net.grads] == [p.shape for p in net.params]
