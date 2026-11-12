import numpy as np
import pytest

from core.losses import MSE
from core.metrics import max_pixel_errors, mean_pixel_errors, pixel_errors
from core.network import Network
from core.optimizers import GD

Y = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
)


def test_salida_perfecta_no_tiene_errores():
    np.testing.assert_array_equal(pixel_errors(Y, Y), [0, 0, 0])


def test_cuenta_por_muestra_tras_binarizar():
    y_pred = np.array(
        [
            [0.9, 0.1, 0.4, 0.6],  # 2 mal: 0.4 -> 0 y 0.6 -> 1
            [0.49, 0.2, 0.0, 0.3],  # 0 mal
            [0.0, 0.1, 0.2, 0.3],  # 4 mal
        ]
    )
    errores = pixel_errors(Y, y_pred)
    assert errores.shape == (3,)
    np.testing.assert_array_equal(errores, [2, 0, 4])


def test_umbral_medio_redondea_hacia_arriba():
    # 0.5 exacto se binariza a 1 (>= 0.5).
    y_true = np.array([[1.0, 0.0]])
    np.testing.assert_array_equal(pixel_errors(y_true, np.array([[0.5, 0.5]])), [1])


def test_umbral_configurable():
    y_true = np.array([[1.0, 0.0]])
    y_pred = np.array([[0.7, 0.7]])
    assert pixel_errors(y_true, y_pred)[0] == 1
    assert pixel_errors(y_true, y_pred, threshold=0.8)[0] == 1
    assert pixel_errors(y_true, np.array([[0.9, 0.7]]), threshold=0.8)[0] == 0


def test_max_y_media():
    y_pred = np.array([[0.9, 0.1, 0.4, 0.6], [0.0] * 4, [0.0] * 4])
    # Errores por muestra: 2, 0, 4.
    assert max_pixel_errors(Y, y_pred) == 4.0
    assert mean_pixel_errors(Y, y_pred) == pytest.approx(2.0)
    assert isinstance(max_pixel_errors(Y, y_pred), float)
    assert isinstance(mean_pixel_errors(Y, y_pred), float)


def test_nan_cuenta_como_cero():
    # Una red divergida da NaN: no supera el umbral, así que se lee como 0.
    y_true = np.array([[1.0, 0.0]])
    assert pixel_errors(y_true, np.array([[np.nan, np.nan]]))[0] == 1


def test_formas_distintas_fallan():
    with pytest.raises(ValueError):
        pixel_errors(Y, Y[:, :3])


def test_sirven_como_metrics_de_fit():
    X = np.random.default_rng(0).uniform(0, 1, size=(5, 3))
    y = (X > 0.5).astype(float)
    net = Network([3, 3], output_activation="sigmoid", rng=np.random.default_rng(0))
    metrics = {"peor": max_pixel_errors, "media": mean_pixel_errors}
    history = net.fit(X, y, GD(lr=0.1), MSE(), epochs=2, metrics=metrics)
    assert history.column("train_peor")[-1] == max_pixel_errors(y, net.predict(X))
    assert history.column("train_media")[-1] == mean_pixel_errors(y, net.predict(X))
