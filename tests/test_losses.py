import numpy as np
import pytest

from core.activations import Softmax
from core.losses import (
    LOSSES,
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    get_loss,
)

EPS = 1e-5
TOL = 1e-7

RNG = np.random.default_rng(0)

# Regresión: valores arbitrarios, incluidos negativos.
Y_REG = RNG.normal(size=(5, 3))
P_REG = RNG.normal(size=(5, 3))

# Binaria: etiquetas {0, 1} y probabilidades lejos de 0 y 1.
Y_BIN = RNG.integers(0, 2, size=(5, 3)).astype(float)
P_BIN = RNG.uniform(0.1, 0.9, size=(5, 3))

# Categórica: one-hot y filas de probabilidad que suman 1. Logits chicos para
# que ŷ quede lejos de 0: cerca de 0, la derivada tercera de log ŷ crece como
# 1/ŷ³ y el error de truncamiento de la diferencia central supera TOL.
Y_CAT = np.eye(4)[RNG.integers(0, 4, size=5)]
Z_CAT = 0.5 * RNG.normal(size=(5, 4))
P_CAT = Softmax().forward(Z_CAT)


def numerical_grad(f, x: np.ndarray) -> np.ndarray:
    """Gradiente de un escalar f(x) por diferencia central, un elemento a la vez."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[idx] += EPS
        minus[idx] -= EPS
        grad[idx] = (f(plus) - f(minus)) / (2 * EPS)
    return grad


@pytest.mark.parametrize(
    "loss, y_true, y_pred",
    [
        pytest.param(MSE(), Y_REG, P_REG, id="mse"),
        pytest.param(BinaryCrossEntropy(), Y_BIN, P_BIN, id="bce"),
        pytest.param(CategoricalCrossEntropy(), Y_CAT, P_CAT, id="cce"),
    ],
)
def test_gradiente_analitico_coincide_con_numerico(loss, y_true, y_pred):
    analytic = loss.grad(y_true, y_pred)
    numeric = numerical_grad(lambda p: loss.value(y_true, p), y_pred)
    assert analytic.shape == y_pred.shape
    np.testing.assert_allclose(analytic, numeric, rtol=0, atol=TOL)


def test_softmax_delta_coincide_con_derivada_respecto_de_z():
    # dL/dz de L(softmax(z)), que es lo que la red usa en la capa de salida.
    cce, softmax = CategoricalCrossEntropy(), Softmax()
    analytic = cce.softmax_delta(Y_CAT, softmax.forward(Z_CAT))
    numeric = numerical_grad(lambda z: cce.value(Y_CAT, softmax.forward(z)), Z_CAT)
    np.testing.assert_allclose(analytic, numeric, rtol=0, atol=TOL)


def test_mse_valor_conocido():
    y_true = np.array([[1.0], [2.0]])
    y_pred = np.array([[0.0], [4.0]])
    # (1² + 2²) / 2
    assert MSE().value(y_true, y_pred) == pytest.approx(2.5)


def test_bce_valor_conocido():
    y_true = np.array([[1.0], [0.0]])
    y_pred = np.array([[0.8], [0.4]])
    expected = -(np.log(0.8) + np.log(0.6)) / 2
    assert BinaryCrossEntropy().value(y_true, y_pred) == pytest.approx(expected)


def test_cce_valor_conocido():
    y_true = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    y_pred = np.array([[0.2, 0.5, 0.3], [0.25, 0.25, 0.5]])
    expected = -(np.log(0.5) + np.log(0.25)) / 2
    assert CategoricalCrossEntropy().value(y_true, y_pred) == pytest.approx(expected)


@pytest.mark.parametrize("loss", [MSE(), BinaryCrossEntropy(), CategoricalCrossEntropy()])
def test_perdida_nula_con_prediccion_perfecta(loss):
    y = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert loss.value(y, y) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "loss", [BinaryCrossEntropy(), CategoricalCrossEntropy()], ids=["bce", "cce"]
)
def test_entropia_cruzada_finita_con_probabilidades_saturadas(loss):
    # Predicción completamente equivocada con probabilidades exactamente 0 y 1.
    y_true = np.array([[1.0, 0.0]])
    y_pred = np.array([[0.0, 1.0]])
    with np.errstate(divide="raise", invalid="raise"):
        assert np.isfinite(loss.value(y_true, y_pred))
        assert np.all(np.isfinite(loss.grad(y_true, y_pred)))


@pytest.mark.parametrize("loss", [MSE(), BinaryCrossEntropy(), CategoricalCrossEntropy()])
def test_formas_distintas_fallan(loss):
    # (n,) contra (n, 1) se broadcastearía a (n, n) sin avisar.
    with pytest.raises(ValueError):
        loss.value(np.zeros(4), np.zeros((4, 1)))
    with pytest.raises(ValueError):
        loss.grad(np.zeros(4), np.zeros((4, 1)))


def test_registro_construye_cada_perdida():
    for name, cls in LOSSES.items():
        assert isinstance(get_loss(name), cls)


def test_registro_nombre_desconocido_falla():
    with pytest.raises(ValueError):
        get_loss("mae")


def test_registro_parametro_desconocido_falla():
    with pytest.raises(TypeError):
        get_loss("mse", reduction="sum")
