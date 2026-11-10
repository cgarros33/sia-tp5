import csv

import numpy as np
import pytest

from core.augmentation import GaussianNoise
from core.callbacks import AdaptiveEta, Callback, EarlyStopping, PrintProgress
from core.history import COLUMNS, History
from core.losses import MSE
from core.network import Network
from core.optimizers import GD, Adam


def linear_data(n: int = 50, seed: int = 0):
    """y = x con n muestras en [-1, 1], forma (n, 1)."""
    X = np.random.default_rng(seed).uniform(-1, 1, size=(n, 1))
    return X, X.copy()


def linear_net(seed: int = 0) -> Network:
    return Network([1, 1], output_activation="identity", rng=np.random.default_rng(seed))


def mean_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# --- Criterio de aceptación de la Etapa 4 ---


def test_aceptacion_y_igual_x_mse_baja_monotonamente_y_termina_bajo_1e4():
    X, y = linear_data(50)
    history = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=300)

    loss = history.column("train_loss")
    assert np.all(np.diff(loss) <= 0), "el MSE subió en alguna época"
    assert loss[-1] < 1e-4


@pytest.mark.parametrize("batch_size", [1, 8], ids=["estocastico", "mini_batch"])
def test_estocastico_y_mini_batch_tambien_convergen(batch_size):
    X, y = linear_data(50)
    history = linear_net().fit(X, y, GD(lr=0.05), MSE(), epochs=100, batch_size=batch_size)
    assert history.column("train_loss")[-1] < 1e-4


# --- Semántica de fit ---


def test_fit_lote_completo_equivale_al_paso_manual():
    X, y = linear_data(20)
    by_fit, by_hand = linear_net(), linear_net()
    by_fit.fit(X, y, GD(lr=0.1), MSE(), epochs=5)

    loss, opt = MSE(), GD(lr=0.1)
    for _ in range(5):
        by_hand.backward(y, by_hand.predict(X), loss)
        opt.step(by_hand.params, by_hand.grads)

    for a, b in zip(by_fit.params, by_hand.params, strict=True):
        np.testing.assert_array_equal(a, b)


def test_batch_size_mayor_o_igual_a_n_es_lote_completo():
    X, y = linear_data(20)
    h_none = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=5)
    h_n = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=5, batch_size=20)
    h_big = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=5, batch_size=1000)
    np.testing.assert_array_equal(h_none.column("train_loss"), h_n.column("train_loss"))
    np.testing.assert_array_equal(h_none.column("train_loss"), h_big.column("train_loss"))


def test_cantidad_de_pasos_por_epoca():
    class CountingGD(GD):
        steps = 0

        def step(self, params, grads):
            CountingGD.steps += 1
            super().step(params, grads)

    X, y = linear_data(10)
    linear_net().fit(X, y, CountingGD(lr=0.1), MSE(), epochs=3, batch_size=4)
    # 10 muestras en lotes de 4: 4 + 4 + 2 = 3 pasos por época.
    assert CountingGD.steps == 9


def test_misma_semilla_mismo_entrenamiento():
    X, y = linear_data(30)
    h1 = linear_net(7).fit(X, y, GD(lr=0.05), MSE(), epochs=10, batch_size=4)
    h2 = linear_net(7).fit(X, y, GD(lr=0.05), MSE(), epochs=10, batch_size=4)
    np.testing.assert_array_equal(h1.column("train_loss"), h2.column("train_loss"))


def test_rng_explicito_cambia_el_orden_de_los_lotes():
    X, y = linear_data(30)
    h1 = linear_net().fit(X, y, GD(lr=0.05), MSE(), epochs=3, batch_size=4,
                          rng=np.random.default_rng(1))  # fmt: skip
    h2 = linear_net().fit(X, y, GD(lr=0.05), MSE(), epochs=3, batch_size=4,
                          rng=np.random.default_rng(2))  # fmt: skip
    assert not np.array_equal(h1.column("train_loss"), h2.column("train_loss"))


def test_logs_con_validacion_y_metrica():
    X, y = linear_data(40)
    X_val, y_val = linear_data(10, seed=1)
    history = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=3,
                               validation=(X_val, y_val), metric=mean_abs_error)  # fmt: skip

    assert len(history) == 3
    record = history.records[-1]
    assert list(record) == list(COLUMNS)
    assert record["epoch"] == 3

    assert np.all(np.isfinite(history.column("val_loss")))
    assert np.all(np.isfinite(history.column("val_metric")))
    assert np.all(np.diff(history.column("elapsed_s")) >= 0)


def test_logs_sin_validacion_ni_metrica():
    X, y = linear_data(10)
    history = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=2)
    # "lr" se agregó en F04: GD tiene atributo lr y fit loguea el η usado en cada época.
    assert set(history.records[0]) == {"epoch", "train_loss", "lr", "elapsed_s"}
    assert np.all(np.isnan(history.column("val_loss")))


def test_la_perdida_registrada_es_sobre_el_conjunto_completo():
    X, y = linear_data(30)
    net = linear_net()
    history = net.fit(X, y, GD(lr=0.05), MSE(), epochs=4, batch_size=7)
    # Con los pesos finales, la pérdida sobre todo X coincide con la última registrada.
    assert history.column("train_loss")[-1] == pytest.approx(net.evaluate(X, y, MSE())["loss"])


def test_evaluate_no_modifica_los_pesos():
    X, y = linear_data(10)
    net = linear_net()
    before = [p.copy() for p in net.params]
    result = net.evaluate(X, y, MSE(), metric=mean_abs_error)
    assert set(result) == {"loss", "metric"}
    for a, b in zip(net.params, before, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [{"epochs": 0}, {"epochs": 5, "batch_size": 0}],
    ids=["epochs_0", "batch_size_0"],
)
def test_parametros_invalidos_fallan(kwargs):
    X, y = linear_data(10)
    with pytest.raises(ValueError):
        linear_net().fit(X, y, GD(), MSE(), **kwargs)


def test_x_e_y_con_distinta_cantidad_de_muestras_falla():
    X, y = linear_data(10)
    with pytest.raises(ValueError):
        linear_net().fit(X, y[:9], GD(), MSE(), epochs=1)


# --- Callbacks ---


def test_callbacks_se_llaman_en_orden():
    calls = []

    class Recorder(Callback):
        def on_train_begin(self, network):
            calls.append("begin")

        def on_epoch_end(self, epoch, logs, network):
            calls.append(epoch)
            return False

        def on_train_end(self, network):
            calls.append("end")

    X, y = linear_data(10)
    linear_net().fit(X, y, GD(), MSE(), epochs=3, callbacks=[Recorder()])
    assert calls == ["begin", 1, 2, 3, "end"]


def test_un_callback_que_devuelve_true_corta_y_todos_ven_la_epoca():
    seen = []

    class StopAt2(Callback):
        def on_epoch_end(self, epoch, logs, network):
            return epoch == 2

    class Watcher(Callback):
        def on_epoch_end(self, epoch, logs, network):
            seen.append(epoch)
            return False

    X, y = linear_data(10)
    history = linear_net().fit(X, y, GD(), MSE(), epochs=10, callbacks=[StopAt2(), Watcher()])
    assert len(history) == 2
    assert seen == [1, 2]


def test_print_progress_escribe_la_primera_y_cada_n_epocas():
    lines: list[str] = []
    X, y = linear_data(10)
    linear_net().fit(X, y, GD(), MSE(), epochs=7,
                     callbacks=[PrintProgress(every=3, write=lines.append)])  # fmt: skip
    assert [line.split(":")[0] for line in lines] == ["epoch 1", "epoch 3", "epoch 6"]
    assert "train_loss=" in lines[0]


def test_print_progress_exige_donde_escribir():
    with pytest.raises(TypeError):
        PrintProgress(every=1)


class FakeLosses(Callback):
    """Reemplaza val_loss por una secuencia fija y cambia los pesos cada época.

    Así se sabe exactamente qué época es la mejor y qué pesos tenía la red.
    """

    def __init__(self, values):
        self.values = values

    def on_epoch_end(self, epoch, logs, network):
        logs["val_loss"] = self.values[epoch - 1]
        for p in network.params:
            p[...] = epoch  # los pesos de la época k valen k
        return False


def test_early_stopping_corta_por_paciencia_y_restaura_la_mejor_epoca():
    stopper = EarlyStopping(patience=2, monitor="val_loss")
    X, y = linear_data(10)
    net = linear_net()
    history = net.fit(X, y, GD(), MSE(), epochs=10,
                      callbacks=[FakeLosses([1.0, 0.5, 0.6, 0.7, 0.1]), stopper])  # fmt: skip

    assert len(history) == 4  # sin mejora en las épocas 3 y 4
    assert stopper.best_epoch == 2
    assert stopper.stopped_epoch == 4
    for p in net.params:
        np.testing.assert_array_equal(p, 2.0)


def test_early_stopping_restaura_aunque_no_corte():
    stopper = EarlyStopping(patience=5)
    X, y = linear_data(10)
    net = linear_net()
    net.fit(X, y, GD(), MSE(), epochs=3, callbacks=[FakeLosses([0.3, 0.1, 0.2]), stopper])
    assert stopper.stopped_epoch is None
    for p in net.params:
        np.testing.assert_array_equal(p, 2.0)


def test_early_stopping_sin_restaurar_deja_los_ultimos_pesos():
    stopper = EarlyStopping(patience=2, restore_best=False)
    X, y = linear_data(10)
    net = linear_net()
    net.fit(X, y, GD(), MSE(), epochs=10, callbacks=[FakeLosses([1.0, 0.5, 0.6, 0.7]), stopper])
    for p in net.params:
        np.testing.assert_array_equal(p, 4.0)


def test_early_stopping_modo_max_y_min_delta():
    stopper = EarlyStopping(patience=1, monitor="val_loss", mode="max", min_delta=0.05)
    X, y = linear_data(10)
    history = linear_net().fit(X, y, GD(), MSE(), epochs=10,
                               callbacks=[FakeLosses([0.5, 0.9, 0.92, 0.99]), stopper])  # fmt: skip
    # 0.92 no supera 0.9 + 0.05, así que la época 3 no cuenta como mejora.
    assert stopper.best_epoch == 2
    assert len(history) == 3


def test_early_stopping_restaura_in_place():
    stopper = EarlyStopping(patience=1)
    X, y = linear_data(10)
    net = linear_net()
    W = net.layers[0].W
    net.fit(X, y, GD(), MSE(), epochs=5, callbacks=[FakeLosses([0.1, 0.2]), stopper])
    assert net.layers[0].W is W


def test_early_stopping_corta_si_diverge():
    stopper = EarlyStopping(patience=2)
    X, y = linear_data(10)
    history = linear_net().fit(X, y, GD(), MSE(), epochs=10,
                               callbacks=[FakeLosses([1.0, np.nan, np.nan]), stopper])  # fmt: skip
    assert len(history) == 3
    assert stopper.best_epoch == 1


def test_early_stopping_sin_validacion_falla_con_mensaje_claro():
    X, y = linear_data(10)
    with pytest.raises(ValueError, match="val_loss"):
        linear_net().fit(X, y, GD(), MSE(), epochs=3, callbacks=[EarlyStopping(patience=2)])


@pytest.mark.parametrize(
    "kwargs", [{"patience": 0}, {"patience": 2, "mode": "maximo"}], ids=["patience", "mode"]
)
def test_early_stopping_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        EarlyStopping(**kwargs)


# --- AdaptiveEta (apuntes §9.3.2) ---


def lr_after_each_epoch(eta: AdaptiveEta, losses, monitor="train_loss") -> list[float]:
    """Pasa una secuencia sintética de pérdidas al callback; devuelve lr tras cada época."""
    eta.on_train_begin(None)
    out = []
    for epoch, value in enumerate(losses, start=1):
        eta.on_epoch_end(epoch, {monitor: value}, None)
        out.append(eta.optimizer.lr)
    return out


def test_adaptive_eta_sube_tras_k_bajas_y_baja_tras_k_prime_subas():
    eta = AdaptiveEta(GD(lr=1.0), a=0.1, b=0.5, k=2, k_prime=2)
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Época 3: segunda baja seguida → +a. Reinicia, época 5: otra vez +a.
    # Época 7: segunda suba seguida → ×(1 − b). Reinicia, época 9: otra vez.
    expected = [1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 0.6, 0.6, 0.3]
    np.testing.assert_allclose(lr_after_each_epoch(eta, losses), expected)


def test_adaptive_eta_racha_interrumpida_o_empate_reinicia_contadores():
    eta = AdaptiveEta(GD(lr=1.0), a=0.1, b=0.5, k=3, k_prime=2)
    # Bajas: 1, (suba), 1, 2, (empate), 1, 2 → nunca 3 seguidas.
    # Subas: la única racha es de 1.
    losses = [1.0, 0.9, 1.0, 0.9, 0.8, 0.8, 0.7, 0.6]
    assert lr_after_each_epoch(eta, losses) == [1.0] * len(losses)


def test_adaptive_eta_respeta_min_lr():
    eta = AdaptiveEta(GD(lr=1e-3), a=0.1, b=0.9, k=5, k_prime=1, min_lr=5e-4)
    lrs = lr_after_each_epoch(eta, [1.0, 2.0, 3.0, 4.0])
    assert lrs == [1e-3, 5e-4, 5e-4, 5e-4]


def test_adaptive_eta_monitor_configurable_y_ausente_falla():
    eta = AdaptiveEta(GD(lr=1.0), a=0.5, b=0.5, k=1, k_prime=1, monitor="val_loss")
    assert lr_after_each_epoch(eta, [1.0, 0.5], monitor="val_loss") == [1.0, 1.5]
    with pytest.raises(ValueError, match="val_loss"):
        lr_after_each_epoch(eta, [1.0], monitor="train_loss")


def test_adaptive_eta_funciona_con_cualquier_optimizador():
    opt = Adam(lr=0.01)
    lrs = lr_after_each_epoch(AdaptiveEta(opt, a=0.01, b=0.5, k=1, k_prime=1), [1.0, 0.5, 0.6])
    np.testing.assert_allclose(lrs, [0.01, 0.02, 0.01])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a": 0.0},
        {"a": -0.1},
        {"b": 0.0},
        {"b": 1.0},
        {"k": 0},
        {"k_prime": 0},
        {"min_lr": -1.0},
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_adaptive_eta_parametros_invalidos(kwargs):
    params = {"a": 0.1, "b": 0.5, "k": 2, "k_prime": 2} | kwargs
    with pytest.raises(ValueError):
        AdaptiveEta(GD(), **params)


class FakeTrainLosses(Callback):
    """Reemplaza train_loss en los logs (lo que ven los callbacks siguientes) por una secuencia."""

    def __init__(self, values):
        self.values = values

    def on_epoch_end(self, epoch, logs, network):
        logs["train_loss"] = self.values[epoch - 1]
        return False


def test_adaptive_eta_en_fit_la_columna_lr_refleja_los_cambios():
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9, 1.0]
    opt = GD(lr=0.01)
    eta = AdaptiveEta(opt, a=0.001, b=0.5, k=2, k_prime=2)
    X, y = linear_data(10)
    history = linear_net().fit(X, y, opt, MSE(), epochs=len(losses),
                               callbacks=[FakeTrainLosses(losses), eta])  # fmt: skip
    # Cada época registra el η con el que entrenó: el cambio decidido al final
    # de la época e aparece en la fila e + 1.
    expected = [0.01, 0.01, 0.01, 0.011, 0.011, 0.012, 0.012, 0.006, 0.006]
    np.testing.assert_allclose(history.column("lr"), expected)
    assert opt.lr == pytest.approx(0.003)


# --- Extensiones de fit (F04-T3) ---


def test_columna_lr_con_optimizador_fijo():
    X, y = linear_data(10)
    history = linear_net().fit(X, y, GD(lr=0.05), MSE(), epochs=3)
    np.testing.assert_array_equal(history.column("lr"), [0.05, 0.05, 0.05])


def test_optimizador_sin_lr_no_registra_la_columna():
    class NoLr:
        def step(self, params, grads):
            for p, g in zip(params, grads, strict=True):
                p -= 0.1 * g

    X, y = linear_data(10)
    history = linear_net().fit(X, y, NoLr(), MSE(), epochs=2)
    assert "lr" not in history.records[0]


def test_metrics_registra_train_y_val_en_records_y_csv(tmp_path):
    X, y = linear_data(40)
    X_val, y_val = linear_data(10, seed=1)
    net = linear_net()
    history = net.fit(X, y, GD(lr=0.1), MSE(), epochs=3, validation=(X_val, y_val),
                      metric=mean_abs_error, metrics={"mae": mean_abs_error})  # fmt: skip

    record = history.records[-1]
    # Convive con metric: train_metric sigue estando, y coincide con train_mae.
    assert record["train_mae"] == pytest.approx(record["train_metric"])
    val_by_hand = net.evaluate(X_val, y_val, MSE(), mean_abs_error)["metric"]
    assert record["val_mae"] == pytest.approx(val_by_hand)

    path = tmp_path / "history.csv"
    history.to_csv(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == [*COLUMNS, "train_mae", "val_mae"]
    assert float(rows[-1]["val_mae"]) == pytest.approx(record["val_mae"])


def test_metrics_sin_validacion_solo_registra_train():
    X, y = linear_data(10)
    history = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=2, metrics={"mae": mean_abs_error})
    assert "train_mae" in history.records[0] and "val_mae" not in history.records[0]
    assert np.all(np.isnan(history.column("val_mae")))


@pytest.mark.parametrize("name", ["loss", "metric", ""])
def test_metrics_con_nombre_reservado_falla(name):
    X, y = linear_data(10)
    with pytest.raises(ValueError):
        linear_net().fit(X, y, GD(), MSE(), epochs=1, metrics={name: mean_abs_error})


def mlp_regression(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 3))
    y = np.tanh(X @ np.array([[1.5], [-2.0], [0.5]]))
    net = Network([3, 8, 1], hidden_activation="tanh", rng=np.random.default_rng(seed))
    return net, X, y


def weight_norm(net: Network) -> float:
    return float(np.sqrt(sum(np.sum(layer.W**2) for layer in net.layers)))


def test_l2_grande_achica_la_norma_de_los_pesos():
    net0, X, y = mlp_regression()
    net_l2, _, _ = mlp_regression()
    net0.fit(X, y, GD(lr=0.1), MSE(), epochs=200)
    net_l2.fit(X, y, GD(lr=0.1), MSE(), epochs=200, l2=0.1)
    assert weight_norm(net_l2) < 0.7 * weight_norm(net0)


def test_l2_cero_es_el_entrenamiento_de_siempre():
    net_a, X, y = mlp_regression()
    net_b, _, _ = mlp_regression()
    h_a = net_a.fit(X, y, GD(lr=0.1), MSE(), epochs=5)
    h_b = net_b.fit(X, y, GD(lr=0.1), MSE(), epochs=5, l2=0.0)
    np.testing.assert_array_equal(h_a.column("train_loss"), h_b.column("train_loss"))


def test_l2_penalty_valor_a_mano():
    net = linear_net()
    net.layers[0].W[...] = 3.0
    net.layers[0].b[...] = 100.0  # los bias no se penalizan
    assert net.l2_penalty(0.5) == pytest.approx(0.5 / 2 * 9.0)
    assert net.l2_penalty(0.0) == 0.0


def test_l2_negativo_falla():
    X, y = linear_data(10)
    with pytest.raises(ValueError):
        linear_net().fit(X, y, GD(), MSE(), epochs=1, l2=-0.1)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # overflow esperado de NumPy
def test_divergencia_corta_y_marca_el_historial():
    ended = []

    class EndRecorder(Callback):
        def on_train_end(self, network):
            ended.append(True)

    X, y = linear_data(50)
    history = linear_net().fit(X, y, GD(lr=1e6), MSE(), epochs=1000, callbacks=[EndRecorder()])
    assert history.status == "diverged"
    assert len(history) < 1000
    loss = history.column("train_loss")
    assert not np.isfinite(loss[-1]) and np.all(np.isfinite(loss[:-1]))
    assert ended == [True]


def test_sin_divergencia_el_estado_es_ok():
    X, y = linear_data(10)
    assert linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=3).status == "ok"


class RecordingNoise:
    """GaussianNoise que además anota cuántas muestras recibe en cada llamada."""

    def __init__(self, sigma):
        self.noise = GaussianNoise(sigma)
        self.sizes = []

    def __call__(self, X, rng):
        self.sizes.append(X.shape[0])
        return self.noise(X, rng)


def test_augmentation_solo_en_los_lotes_de_train():
    X, y = linear_data(10)
    X_val, y_val = linear_data(7, seed=1)
    aug = RecordingNoise(sigma=1e6)
    net = linear_net()
    history = net.fit(X, y, GD(lr=1e-12), MSE(), epochs=2, batch_size=4,
                      validation=(X_val, y_val), augment=aug)  # fmt: skip
    # 10 muestras en lotes de 4: 4 + 4 + 2 por época; nunca las 7 de validación.
    assert aug.sizes == [4, 4, 2, 4, 4, 2]
    # Con sigma enorme, las pérdidas registradas son sobre los datos limpios:
    # coinciden con evaluate llamado a mano sobre X y X_val.
    train_by_hand = net.evaluate(X, y, MSE())["loss"]
    val_by_hand = net.evaluate(X_val, y_val, MSE())["loss"]
    assert history.column("train_loss")[-1] == pytest.approx(train_by_hand)
    assert history.column("val_loss")[-1] == pytest.approx(val_by_hand)


def test_augmentation_no_cambia_el_orden_de_los_lotes():
    # Ruido 0 = identidad: si el generador de la augmentation compartiera el
    # stream con el barajado, el orden de los lotes cambiaría.
    X, y = linear_data(30)
    h_plain = linear_net().fit(X, y, GD(lr=0.05), MSE(), epochs=4, batch_size=4)
    h_aug = linear_net().fit(X, y, GD(lr=0.05), MSE(), epochs=4, batch_size=4,
                             augment=GaussianNoise(sigma=0.0))  # fmt: skip
    np.testing.assert_array_equal(h_plain.column("train_loss"), h_aug.column("train_loss"))


def test_augmentation_reproducible_y_cambia_el_entrenamiento():
    X, y = linear_data(30)

    def run(aug):
        return linear_net().fit(X, y, GD(lr=0.05), MSE(), epochs=4, batch_size=4,
                                augment=aug).column("train_loss")  # fmt: skip

    np.testing.assert_array_equal(run(GaussianNoise(0.3)), run(GaussianNoise(0.3)))
    assert not np.array_equal(run(GaussianNoise(0.3)), run(None))


# --- History ---


def test_history_to_csv(tmp_path):
    history = History()
    history.append({"epoch": 1, "train_loss": 0.5, "elapsed_s": 0.01})
    history.append({"epoch": 2, "train_loss": 0.25, "val_loss": 0.3, "elapsed_s": 0.02})
    path = tmp_path / "history.csv"
    history.to_csv(path)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(COLUMNS)
    assert rows[0]["val_loss"] == ""
    assert float(rows[1]["val_loss"]) == 0.3


def test_history_clave_desconocida_falla():
    with pytest.raises(ValueError):
        History().append({"epoch": 1, "accuracy": 0.9})


def test_history_columna_desconocida_falla():
    with pytest.raises(ValueError):
        History().column("accuracy")


def test_history_columnas_dinamicas(tmp_path):
    history = History()
    assert history.status == "ok"
    history.append({"epoch": 1, "train_loss": 0.5, "val_f1": 0.2, "train_f1": 0.3})
    history.append({"epoch": 2, "train_loss": 0.4, "train_acc": 0.9})
    np.testing.assert_array_equal(history.column("train_f1"), [0.3, np.nan])
    np.testing.assert_array_equal(history.column("val_mae"), [np.nan, np.nan])

    path = tmp_path / "history.csv"
    history.to_csv(path)
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    # COLUMNS primero y después las dinámicas en orden alfabético.
    assert header == [*COLUMNS, "train_acc", "train_f1", "val_f1"]


@pytest.mark.parametrize("key", ["accuracy", "train_", "test_acc"])
def test_history_solo_acepta_prefijos_train_y_val(key):
    with pytest.raises(ValueError):
        History().append({"epoch": 1, key: 0.9})
    with pytest.raises(ValueError):
        History().column(key)
