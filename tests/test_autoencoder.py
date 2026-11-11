import numpy as np
import pytest

from core.augmentation import GaussianNoise
from core.autoencoder import Autoencoder
from core.losses import MSE, BinaryCrossEntropy
from core.network import Network
from core.optimizers import Adam


def _autoencoder(seed=0, n_features=6, hidden=5, latent=2, latent_act="identity", out="sigmoid"):
    rng = np.random.default_rng(seed)
    encoder = Network([n_features, hidden, latent], output_activation=latent_act, rng=rng)
    decoder = Network([latent, hidden, n_features], output_activation=out, rng=rng)
    return Autoencoder(encoder, decoder)


def _datos(n=8, n_features=6, seed=1):
    return np.random.default_rng(seed).uniform(0.1, 0.9, size=(n, n_features))


# --- Estructura ---


def test_formas_de_encode_decode_y_predict():
    ae = _autoencoder()
    X = _datos()
    Z = ae.encode(X)
    assert Z.shape == (8, 2)
    assert ae.decode(Z).shape == X.shape
    assert ae.latent_dim == 2
    assert ae.layer_sizes == [6, 5, 2, 5, 6]


def test_predict_es_decode_de_encode():
    ae = _autoencoder()
    X = _datos()
    np.testing.assert_allclose(ae.predict(X), ae.decode(ae.encode(X)))


def test_comparte_los_parametros_con_encoder_y_decoder():
    ae = _autoencoder()
    esperados = ae.encoder.params + ae.decoder.params
    assert len(ae.params) == len(esperados)
    assert all(p is q for p, q in zip(ae.params, esperados, strict=True))
    assert ae.n_params == ae.encoder.n_params + ae.decoder.n_params


def test_latente_tiene_que_coincidir_con_la_entrada_del_decoder():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="encoder sale con 2"):
        Autoencoder(Network([6, 2], rng=rng), Network([3, 6], rng=rng))


def test_salida_tiene_que_coincidir_con_la_entrada():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="decoder sale con 5"):
        Autoencoder(Network([6, 2], rng=rng), Network([2, 5], rng=rng))


# --- Gradientes ---


@pytest.mark.parametrize(
    ("loss", "latent_act", "out"),
    [
        (MSE(), "identity", "identity"),
        (MSE(), "tanh", "sigmoid"),
        (BinaryCrossEntropy(), "identity", "sigmoid"),
    ],
)
def test_gradiente_coincide_con_diferencias_finitas(loss, latent_act, out):
    ae = _autoencoder(latent_act=latent_act, out=out)
    X = _datos()
    ae.backward(X, ae.predict(X), loss)
    analiticos = [g.copy() for g in ae.grads]

    eps = 1e-6
    for p, g in zip(ae.params, analiticos, strict=True):
        numerico = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            original = p[idx]
            p[idx] = original + eps
            mas = loss.value(X, ae.predict(X))
            p[idx] = original - eps
            menos = loss.value(X, ae.predict(X))
            p[idx] = original
            numerico[idx] = (mas - menos) / (2 * eps)
        np.testing.assert_allclose(g, numerico, rtol=1e-5, atol=1e-8)


# --- Entrenamiento ---


def test_fit_sin_y_reconstruye_la_entrada():
    X = _datos()
    a, b = _autoencoder(seed=3), _autoencoder(seed=3)
    h_a = a.fit(X, optimizer=Adam(lr=0.01), loss=MSE(), epochs=20)
    h_b = b.fit(X, X, optimizer=Adam(lr=0.01), loss=MSE(), epochs=20)
    np.testing.assert_array_equal(h_a.column("train_loss"), h_b.column("train_loss"))
    for p, q in zip(a.params, b.params, strict=True):
        np.testing.assert_array_equal(p, q)


def test_fit_baja_el_error_de_reconstruccion():
    ae = _autoencoder(seed=4)
    X = _datos()
    history = ae.fit(X, optimizer=Adam(lr=0.01), loss=MSE(), epochs=300)
    loss = history.column("train_loss")
    assert loss[-1] < 0.5 * loss[0]


def test_fit_actualiza_encoder_y_decoder():
    ae = _autoencoder(seed=5)
    antes = [p.copy() for p in ae.encoder.params + ae.decoder.params]
    ae.fit(_datos(), optimizer=Adam(lr=0.01), loss=MSE(), epochs=3)
    despues = ae.encoder.params + ae.decoder.params
    assert all(not np.array_equal(p, q) for p, q in zip(antes, despues, strict=True))


def test_aprende_patrones_one_hot_con_latente_de_dos_dimensiones():
    X = np.eye(4)
    ae = _autoencoder(seed=6, n_features=4, hidden=8)
    ae.fit(X, optimizer=Adam(lr=0.01), loss=BinaryCrossEntropy(), epochs=3000)
    np.testing.assert_array_equal(np.round(ae.predict(X)), X)


def test_fit_con_objetivo_distinto_de_la_entrada():
    limpio = _datos()
    ruidoso = limpio + np.random.default_rng(7).normal(0.0, 0.05, size=limpio.shape)
    ae = _autoencoder(seed=8)
    history = ae.fit(ruidoso, limpio, optimizer=Adam(lr=0.01), loss=MSE(), epochs=300)
    assert history.column("train_loss")[-1] == pytest.approx(
        MSE().value(limpio, ae.predict(ruidoso))
    )


def test_fit_acepta_augment_y_validacion():
    X = _datos()
    ae = _autoencoder(seed=9)
    history = ae.fit(
        X,
        optimizer=Adam(lr=0.01),
        loss=MSE(),
        epochs=5,
        batch_size=4,
        validation=(X, X),
        augment=GaussianNoise(0.1),
    )
    assert len(history) == 5
    assert np.all(np.isfinite(history.column("val_loss")))


# --- Autoencoder lineal (apuntes §19.1.2) ---


def test_autoencoder_lineal_alcanza_el_error_de_pca():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(40, 2)) @ rng.normal(size=(2, 5)) + 0.05 * rng.normal(size=(40, 5))
    X = X - X.mean(axis=0)

    # Mejor reconstrucción de rango 2: la proyección sobre las dos primeras componentes.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    error_pca = MSE().value(X, (U[:, :2] * S[:2]) @ Vt[:2])

    encoder = Network([5, 2], output_activation="identity", rng=rng)
    decoder = Network([2, 5], output_activation="identity", rng=rng)
    ae = Autoencoder(encoder, decoder)
    ae.fit(X, optimizer=Adam(lr=0.01), loss=MSE(), epochs=8000)

    error_ae = MSE().value(X, ae.predict(X))
    assert error_ae >= error_pca - 1e-12
    assert error_ae == pytest.approx(error_pca, rel=0.02)
