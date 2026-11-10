import logging

import numpy as np
import pytest

from core.callbacks import Callback
from core.losses import MSE
from core.network import Network
from core.optimizers import GD, OPTIMIZERS, AdaGrad, Adam, Momentum, RMSProp, get_optimizer


def test_gd_paso_calculado_a_mano():
    # θ ← θ − lr · g = (1, −2) − 0.1 · (4, −6) = (0.6, −1.4)
    theta = np.array([[1.0, -2.0]])
    GD(lr=0.1).step([theta], [np.array([[4.0, -6.0]])])
    np.testing.assert_allclose(theta, [[0.6, -1.4]])


@pytest.mark.parametrize("make", [GD, Momentum, RMSProp, Adam, AdaGrad], ids=lambda c: c.__name__)
def test_actualiza_los_pesos_de_la_red_in_place(make):
    net = Network([2, 3, 1], rng=np.random.default_rng(0))
    W_before = net.layers[0].W
    copy_before = W_before.copy()

    X = np.random.default_rng(1).normal(size=(4, 2))
    y = np.zeros((4, 1))
    net.backward(y, net.predict(X), MSE())
    make(lr=0.1).step(net.params, net.grads)

    assert net.layers[0].W is W_before  # mismo objeto, no una copia
    assert not np.array_equal(net.layers[0].W, copy_before)


def test_gd_baja_la_perdida_en_y_igual_a_x():
    rng = np.random.default_rng(0)
    net = Network([1, 1], output_activation="identity", rng=rng)
    X = rng.uniform(-1, 1, size=(50, 1))
    y = X.copy()
    loss, opt = MSE(), GD(lr=0.1)

    losses = []
    for _ in range(100):
        y_pred = net.predict(X)
        losses.append(loss.value(y, y_pred))
        net.backward(y, y_pred, loss)
        opt.step(net.params, net.grads)

    assert all(b <= a for a, b in zip(losses, losses[1:], strict=False))
    assert losses[-1] < 1e-2 * losses[0]


def test_gd_listas_de_distinto_largo_fallan():
    with pytest.raises(ValueError):
        GD().step([np.zeros(2), np.zeros(2)], [np.zeros(2)])


@pytest.mark.parametrize("lr", [0.0, -0.1])
def test_gd_lr_no_positivo_falla(lr):
    with pytest.raises(ValueError):
        GD(lr=lr)


def test_registro_construye_cada_optimizador():
    for name, cls in OPTIMIZERS.items():
        assert isinstance(get_optimizer(name), cls)


def test_registro_pasa_parametros():
    assert get_optimizer("gd", lr=0.5).lr == 0.5


def test_registro_nombre_desconocido_falla():
    with pytest.raises(ValueError):
        get_optimizer("sgd")


# --- Pasos a mano (apuntes §7.4 y §9.3) con un escalar: g = 0.5, η = 0.1 ---

G, LR = 0.5, 0.1


def one_param_steps(opt, n_steps: int) -> list[float]:
    """Aplica n_steps pasos con gradiente constante G desde θ = 0; devuelve θ tras cada paso."""
    theta = np.zeros((1, 1))
    out = []
    for _ in range(n_steps):
        opt.step([theta], [np.full((1, 1), G)])
        out.append(float(theta[0, 0]))
    return out


def test_momentum_dos_pasos_a_mano():
    # Δθ(1) = −η g = −0.05                      → θ = −0.05
    # Δθ(2) = −η g + α Δθ(1) = −0.05 − 0.045    → θ = −0.05 − 0.095 = −0.145
    thetas = one_param_steps(Momentum(lr=LR, alpha=0.9), 2)
    np.testing.assert_allclose(thetas, [-0.05, -0.145])


def test_rmsprop_un_paso_a_mano():
    # S = (1 − γ) g² = 0.1 · 0.25 = 0.025;  Δθ = −η g / sqrt(S + ε)
    thetas = one_param_steps(RMSProp(lr=LR, gamma=0.9, eps=1e-8), 1)
    np.testing.assert_allclose(thetas, [-LR * G / np.sqrt(0.025 + 1e-8)])


def test_adam_primer_paso_es_eta_por_signo():
    # En t = 1: m̂ = g y v̂ = g², así que el paso es −η g / (|g| + ε) ≈ −η sign(g).
    thetas = one_param_steps(Adam(lr=LR), 1)
    np.testing.assert_allclose(thetas, [-LR * G / (G + 1e-8)])
    assert thetas[0] == pytest.approx(-LR)
    # Con gradiente negativo el paso es +η, sin importar la escala del gradiente.
    theta = np.zeros((1, 1))
    Adam(lr=LR).step([theta], [np.full((1, 1), -1e-3)])
    assert theta[0, 0] == pytest.approx(LR, rel=1e-4)


def test_adagrad_un_paso_a_mano():
    # G = g² = 0.25;  Δθ = −η g / (sqrt(G) + ε)
    thetas = one_param_steps(AdaGrad(lr=LR, eps=1e-8), 1)
    np.testing.assert_allclose(thetas, [-LR * G / (0.5 + 1e-8)])


# --- Convergencia en una cuadrática mal condicionada ---

# lr elegidos por optimizador (con los defaults Adam no llega en 2000 pasos:
# con lr = 1e-3 avanza a lo sumo ~1e-3 por paso). Pasos necesarios medidos al
# escribir el test: GD 66, Momentum 129, RMSProp 132, Adam 276, AdaGrad 215.
QUADRATIC_CASES = [
    GD(lr=0.1),
    Momentum(lr=0.01, alpha=0.9),
    RMSProp(lr=0.01),
    Adam(lr=0.01),
    AdaGrad(lr=0.1),
]


@pytest.mark.parametrize("opt", QUADRATIC_CASES, ids=lambda o: type(o).__name__)
def test_minimiza_cuadratica_mal_condicionada(opt):
    # L(θ) = ½ θᵀ diag(1, 10) θ, gradiente diag(1, 10) θ, mínimo en 0.
    curvature = np.array([[1.0, 10.0]])
    theta = np.ones((1, 2))
    for _ in range(2000):
        opt.step([theta], [curvature * theta])
        if np.linalg.norm(theta) < 1e-3:
            break
    assert np.linalg.norm(theta) < 1e-3


# --- state_dict: cortar y seguir da la misma trayectoria ---


def run_steps(opt, theta, b, n_steps):
    """n_steps pasos sobre dos tensores (θ con curvatura por elemento y un vector b).

    Modifica theta y b in place y devuelve una fila por paso con sus valores.
    """
    curvature = np.array([[1.0, 10.0], [3.0, 0.5]])
    out = []
    for _ in range(n_steps):
        opt.step([theta, b], [curvature * theta, b.copy()])
        out.append(np.concatenate([theta.ravel(), b]))
    return np.array(out)


@pytest.mark.parametrize(
    "make",
    [
        lambda: GD(lr=0.05),
        lambda: Momentum(lr=0.05, alpha=0.8),
        lambda: RMSProp(lr=0.01, gamma=0.8),
        lambda: Adam(lr=0.01, beta1=0.8, beta2=0.99),
        lambda: AdaGrad(lr=0.1),
    ],
    ids=["GD", "Momentum", "RMSProp", "Adam", "AdaGrad"],
)
def test_state_dict_reanuda_la_misma_trayectoria(make, tmp_path):
    theta0, b0 = np.array([[1.0, -1.0], [0.5, 2.0]]), np.array([0.2, -0.1])
    full = run_steps(make(), theta0.copy(), b0.copy(), 20)

    theta, b = theta0.copy(), b0.copy()
    first = make()
    head = run_steps(first, theta, b, 8)
    # Ida y vuelta por disco: el estado tiene que ser serializable con np.savez.
    np.savez(tmp_path / "opt.npz", **first.state_dict())
    with np.load(tmp_path / "opt.npz") as data:
        state = {k: data[k] for k in data.files}
    # Hiperparámetros distintos a propósito: load_state_dict los restaura.
    second = type(first)(lr=0.123)
    second.load_state_dict(state)
    tail = run_steps(second, theta, b, 12)

    np.testing.assert_array_equal(np.vstack([head, tail]), full)


def test_state_dict_antes_del_primer_paso_y_lr_mutable():
    opt = Adam(lr=0.01)
    state = opt.state_dict()
    assert state["lr"] == 0.01 and state["t"] == 0
    opt.lr = 0.5  # AdaptiveEta modifica este atributo
    assert opt.state_dict()["lr"] == 0.5


def test_load_state_dict_de_otro_optimizador_falla():
    with pytest.raises(ValueError):
        Adam().load_state_dict(Momentum().state_dict())


# --- Validación de parámetros ---


@pytest.mark.parametrize(
    "make",
    [
        lambda: Momentum(lr=0.0),
        lambda: Momentum(alpha=1.0),
        lambda: Momentum(alpha=-0.1),
        lambda: RMSProp(gamma=1.0),
        lambda: RMSProp(eps=0.0),
        lambda: Adam(lr=-1.0),
        lambda: Adam(beta1=1.0),
        lambda: Adam(beta2=-0.5),
        lambda: Adam(eps=0.0),
        lambda: AdaGrad(lr=0.0),
        lambda: AdaGrad(eps=-1e-8),
    ],
)
def test_parametros_fuera_de_rango_fallan(make):
    with pytest.raises(ValueError):
        make()


def test_registro_incluye_los_optimizadores_nuevos():
    assert {"gd", "momentum", "rmsprop", "adam", "adagrad"} == set(OPTIMIZERS)
    assert get_optimizer("adam", lr=0.02, beta1=0.8).beta1 == 0.8
    with pytest.raises(TypeError):
        get_optimizer("momentum", beta1=0.9)  # parámetro de otro optimizador


# --- XOR [2, 2, 1] con cada optimizador: dato para la presentación ---

X_XOR = np.array([[-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]])
Y_XOR = np.array([[1.0], [1.0], [-1.0], [-1.0]])


class StopWhenSolved(Callback):
    """Corta en la primera época en que las 4 muestras de XOR quedan bien clasificadas."""

    def on_train_begin(self, network):
        self.solved_at = None

    def on_epoch_end(self, epoch, logs, network):
        if np.all(np.sign(network.predict(X_XOR)) == Y_XOR):
            self.solved_at = epoch
            return True
        return False


def test_xor_epocas_hasta_resolver_por_optimizador(caplog):
    # No se asserta un orden entre optimizadores: el test solo registra el dato.
    # Para verlo: pytest tests/test_optimizers.py -k xor --log-cli-level=INFO
    optimizers = {
        "GD(lr=0.1)": lambda: GD(lr=0.1),
        "Momentum(lr=0.1, alpha=0.9)": lambda: Momentum(lr=0.1, alpha=0.9),
        "Adam(lr=0.01)": lambda: Adam(lr=0.01),
    }
    log = logging.getLogger(__name__)
    caplog.set_level(logging.INFO)
    for name, make in optimizers.items():
        solved = []
        for seed in range(20):
            net = Network([2, 2, 1], hidden_activation="tanh", output_activation="tanh",
                          initializer="xavier", rng=np.random.default_rng(seed))  # fmt: skip
            stopper = StopWhenSolved()
            net.fit(X_XOR, Y_XOR, make(), MSE(), epochs=2000, callbacks=[stopper])
            if stopper.solved_at is not None:
                solved.append(stopper.solved_at)
        assert solved, f"{name} no resolvió XOR con ninguna semilla"
        log.info(
            "XOR [2,2,1] %s: resuelve %d/20 semillas en <= 2000 epocas, "
            "mediana de epocas (entre las resueltas) = %g",
            name,
            len(solved),
            np.median(solved),
        )
