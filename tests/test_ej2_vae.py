"""Tests de los constructores y del entrenamiento mínimo de ej2."""

import numpy as np
import pytest

from core.autoencoder import Autoencoder
from core.losses import MSE, BinaryCrossEntropy
from core.optimizers import Adam
from core.vae import VariationalAutoencoder
from ej2.modelos import (
    construir_ae_imagen,
    construir_ae_tabular,
    construir_vae_imagen,
    construir_vae_tabular,
)


@pytest.mark.parametrize("latente", [2, 16])
def test_vae_imagen_tiene_las_formas_esperadas(latente):
    vae = construir_vae_imagen(40, latente, [32, 16], salida="sigmoid", semilla=0)
    assert isinstance(vae, VariationalAutoencoder)
    assert vae.latent_dim == latente
    assert vae.encoder.layer_sizes == [40, 32, 16, 2 * latente]
    assert vae.decoder.layer_sizes == [latente, 16, 32, 40]
    X = np.random.default_rng(0).uniform(0.1, 0.9, size=(6, 40))
    assert vae.encode(X).shape == (6, latente)
    assert vae.sample(4).shape == (4, 40)


def test_ae_imagen_es_deterministico():
    ae = construir_ae_imagen(40, 2, [32, 16], salida="sigmoid", semilla=0)
    assert isinstance(ae, Autoencoder)
    assert ae.encoder.layer_sizes == [40, 32, 16, 2]
    assert ae.decoder.layer_sizes == [2, 16, 32, 40]


def test_vae_tabular_tiene_doce_entradas():
    vae = construir_vae_tabular(2, [32, 16], semilla=0)
    assert vae.encoder.layer_sizes == [12, 32, 16, 4]
    assert vae.decoder.layer_sizes == [2, 16, 32, 12]
    ae = construir_ae_tabular(2, [32, 16], semilla=0)
    assert ae.encoder.layer_sizes == [12, 32, 16, 2]


def test_capacidad_invalida_es_un_error():
    with pytest.raises(ValueError, match="dos anchos"):
        construir_vae_imagen(10, 2, [32], semilla=0)
    with pytest.raises(ValueError, match="Salida desconocida"):
        construir_vae_imagen(10, 2, [32, 16], salida="tanh", semilla=0)


def test_smoke_training_vae_imagen():
    rng = np.random.default_rng(1)
    X = (rng.random((8, 20)) > 0.5).astype(float)
    vae = construir_vae_imagen(20, 2, [8, 4], semilla=2)
    historia = vae.fit(
        X, optimizer=Adam(lr=0.01), loss=BinaryCrossEntropy(), epochs=2, batch_size=4
    )
    assert len(historia) == 2
    assert np.all(np.isfinite(historia.column("train_loss")))


def test_smoke_training_vae_tabular():
    rng = np.random.default_rng(2)
    X = rng.random((12, 12))
    vae = construir_vae_tabular(16, [8, 4], semilla=3)
    historia = vae.fit(X, optimizer=Adam(lr=0.01), loss=MSE(), epochs=2, batch_size=6)
    assert len(historia) == 2
    assert vae.sample(3).shape == (3, 12)


def test_entrenamiento_reproducible_por_semilla():
    rng = np.random.default_rng(3)
    X = rng.random((10, 20))
    a = construir_vae_imagen(20, 2, [8, 4], salida="identity", semilla=4)
    b = construir_vae_imagen(20, 2, [8, 4], salida="identity", semilla=4)
    for vae in (a, b):
        vae.fit(X, optimizer=Adam(lr=0.01), loss=MSE(), epochs=3, batch_size=5)
    for p, q in zip(a.params, b.params, strict=True):
        np.testing.assert_array_equal(p, q)


def test_beta_se_pasa_como_peso_del_kl():
    vae = construir_vae_imagen(40, 2, [8, 4], beta=0.5, semilla=0)
    assert vae.kl_weight == 0.5
    tabular = construir_vae_tabular(16, [8, 4], beta=5.0, semilla=0)
    assert tabular.kl_weight == 5.0
