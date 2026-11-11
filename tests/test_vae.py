import numpy as np
import pytest

from core.autoencoder import Autoencoder
from core.losses import MSE, BinaryCrossEntropy
from core.network import Network
from core.optimizers import Adam
from core.vae import VariationalAutoencoder


def _vae(seed=0, n_features=6, hidden=5, latent=2, out="sigmoid", kl_weight=1.0):
    rng = np.random.default_rng(seed)
    encoder = Network([n_features, hidden, 2 * latent], output_activation="identity", rng=rng)
    decoder = Network([latent, hidden, n_features], output_activation=out, rng=rng)
    return VariationalAutoencoder(encoder, decoder, kl_weight=kl_weight)


def _datos(n=8, n_features=6, seed=1):
    return np.random.default_rng(seed).uniform(0.1, 0.9, size=(n, n_features))


def _fijar_ruido(vae, eps):
    """Reemplaza ε por un valor fijo: la red pasa a ser una función determinística."""
    vae._noise = lambda shape: np.broadcast_to(eps, shape).copy()


# --- Estructura ---


def test_es_una_extension_del_autoencoder():
    vae = _vae()
    assert isinstance(vae, Autoencoder)
    assert vae.latent_dim == 2
    assert all(
        p is q for p, q in zip(vae.params, vae.encoder.params + vae.decoder.params, strict=True)
    )


def test_encoder_tiene_que_sacar_media_y_varianza_por_dimension_latente():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="encoder sale con 2"):
        VariationalAutoencoder(Network([6, 2], rng=rng), Network([2, 6], rng=rng))


def test_kl_weight_negativo_es_un_error():
    with pytest.raises(ValueError, match="kl_weight"):
        _vae(kl_weight=-1.0)


def test_formas():
    vae = _vae()
    X = _datos()
    mu, log_var = vae.latent_distribution(X)
    assert mu.shape == log_var.shape == (8, 2)
    assert vae.encode(X).shape == (8, 2)
    assert vae.predict(X).shape == X.shape
    assert vae.sample(5).shape == (5, 6)


def test_encode_es_la_media_y_no_muestrea():
    vae = _vae()
    X = _datos()
    np.testing.assert_array_equal(vae.encode(X), vae.encode(X))
    np.testing.assert_array_equal(vae.encode(X), vae.encoder.predict(X)[:, :2])


# --- Nodo estocástico (apuntes §19.3.1) ---


def test_predict_decodifica_mu_mas_eps_por_sigma():
    vae = _vae()
    X = _datos()
    eps = np.random.default_rng(2).normal(size=(8, 2))
    _fijar_ruido(vae, eps)
    mu, log_var = vae.latent_distribution(X)
    esperado = vae.decode(mu + eps * np.sqrt(np.exp(log_var)))
    np.testing.assert_allclose(vae.predict(X), esperado)


def test_predict_es_estocastico_y_reproducible_por_semilla():
    X = _datos()
    a, b = _vae(seed=3), _vae(seed=3)
    primero = a.predict(X)
    assert not np.allclose(primero, a.predict(X))
    np.testing.assert_array_equal(primero, b.predict(X))


def test_muestrear_no_consume_el_rng_de_los_lotes():
    vae = _vae()
    antes = vae.rng.bit_generator.state
    vae.predict(_datos())
    vae.sample(3)
    assert vae.rng.bit_generator.state == antes


def test_z_muestreado_tiene_la_media_y_el_desvio_del_encoder():
    vae = _vae(seed=4)
    x = _datos(n=1)
    mu, log_var = vae.latent_distribution(x)
    vae.predict(np.repeat(x, 20000, axis=0))
    z = vae.decoder.layers[0].x
    np.testing.assert_allclose(z.mean(axis=0), mu[0], atol=0.03)
    np.testing.assert_allclose(z.std(axis=0), np.exp(0.5 * log_var[0]), rtol=0.03)


# --- Término de regularización (apuntes §19.3.3) ---


def test_kl_coincide_con_la_formula_cerrada_entre_normales():
    vae = _vae()
    X = _datos()
    mu, log_var = vae.latent_distribution(X)
    var = np.exp(log_var)
    # ½ {traza(Σ) + μᵀμ − k − log|Σ|}, por muestra.
    por_muestra = 0.5 * (var.sum(axis=1) + (mu**2).sum(axis=1) - 2 - np.log(var.prod(axis=1)))
    assert vae.kl_divergence(X) == pytest.approx(por_muestra.mean())
    assert np.all(por_muestra >= 0)


def test_kl_es_cero_cuando_q_es_la_normal_estandar():
    vae = _vae()
    for p in vae.encoder.layers[-1].params:
        p[...] = 0.0
    assert vae.kl_divergence(_datos()) == 0.0


def test_loss_es_reconstruccion_mas_lambda_por_kl():
    vae = _vae(kl_weight=0.5)
    X = _datos()
    _fijar_ruido(vae, np.random.default_rng(5).normal(size=(8, 2)))
    total = vae.evaluate(X, X, MSE())["loss"]
    reconstruccion = MSE().value(X, vae.predict(X))
    # Por muestra: ‖x − x'‖² + λ·KL, dividido por la cantidad de features como MSE.
    assert total == pytest.approx(reconstruccion + 0.5 * vae.kl_divergence(X) / X.shape[1])


# --- Gradientes ---


@pytest.mark.parametrize(
    ("loss", "out", "kl_weight", "l2"),
    [
        (MSE(), "identity", 1.0, 0.0),
        (MSE(), "sigmoid", 0.0, 0.0),
        (MSE(), "sigmoid", 0.3, 0.01),
        (BinaryCrossEntropy(), "sigmoid", 1.0, 0.0),
    ],
)
def test_gradiente_coincide_con_diferencias_finitas(loss, out, kl_weight, l2):
    vae = _vae(out=out, kl_weight=kl_weight)
    X = _datos()
    _fijar_ruido(vae, np.random.default_rng(6).normal(size=(8, 2)))

    def objetivo():
        return vae.evaluate(X, X, loss)["loss"] + vae.l2_penalty(l2)

    vae.backward(X, vae.predict(X), loss, l2=l2)
    analiticos = [g.copy() for g in vae.grads]

    eps = 1e-6
    for p, g in zip(vae.params, analiticos, strict=True):
        numerico = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            original = p[idx]
            p[idx] = original + eps
            mas = objetivo()
            p[idx] = original - eps
            menos = objetivo()
            p[idx] = original
            numerico[idx] = (mas - menos) / (2 * eps)
        np.testing.assert_allclose(g, numerico, rtol=1e-5, atol=1e-8)


def test_backward_antes_de_predict_es_un_error():
    X = _datos()
    with pytest.raises(RuntimeError, match="backward antes de predict"):
        _vae().backward(X, X, MSE())


# --- Entrenamiento ---


def test_fit_baja_la_funcion_objetivo():
    vae = _vae(seed=7, kl_weight=0.1)
    X = _datos()
    history = vae.fit(X, optimizer=Adam(lr=0.01), loss=MSE(), epochs=300)
    loss = history.column("train_loss")
    assert loss[-50:].mean() < loss[:50].mean()


def test_fit_es_reproducible_por_semilla():
    X = _datos()
    a, b = _vae(seed=8), _vae(seed=8)
    for vae in (a, b):
        vae.fit(X, optimizer=Adam(lr=0.01), loss=MSE(), epochs=10, batch_size=4)
    for p, q in zip(a.params, b.params, strict=True):
        np.testing.assert_array_equal(p, q)


def test_mas_kl_weight_acerca_q_a_la_normal_estandar():
    X = _datos()
    kl = {}
    for kl_weight in (0.0, 5.0):
        vae = _vae(seed=9, kl_weight=kl_weight)
        vae.fit(X, optimizer=Adam(lr=0.01), loss=MSE(), epochs=500)
        kl[kl_weight] = vae.kl_divergence(X)
    assert kl[5.0] < 0.1 * kl[0.0]


def test_sample_decodifica_ruido_normal_estandar():
    vae = _vae()
    z = np.random.default_rng(10).normal(size=(4, 2))
    _fijar_ruido(vae, z)
    np.testing.assert_array_equal(vae.sample(4), vae.decode(z))
