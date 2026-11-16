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
from ej2.vae_exp import CalentamientoKL, _aplicar_nivel, _elegir


@pytest.mark.parametrize("latente", [4, 8, 16])
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
    vae = construir_vae_imagen(40, 4, [8, 4], beta=0.5, semilla=0)
    assert vae.kl_weight == 0.5
    tabular = construir_vae_tabular(16, [8, 4], beta=5.0, semilla=0)
    assert tabular.kl_weight == 5.0


def test_free_bits_se_pasa_al_vae():
    vae = construir_vae_imagen(40, 4, [8, 4], beta=0.5, free_bits=0.1, semilla=0)
    assert vae.free_bits == 0.1
    tabular = construir_vae_tabular(4, [8, 4], beta=2.0, free_bits=0.2, semilla=0)
    assert tabular.free_bits == 0.2


def test_calentamiento_kl_sube_de_cero_a_beta():
    class Red:
        kl_weight = 1.0

    red = Red()
    calentamiento = CalentamientoKL(beta=2.0, epocas=4)
    calentamiento.on_train_begin(red)
    assert red.kl_weight == 0.0
    for epoca, esperado in [(1, 0.5), (2, 1.0), (3, 1.5), (4, 2.0), (5, 2.0)]:
        calentamiento.on_epoch_end(epoca, {}, red)
        assert red.kl_weight == pytest.approx(esperado)


def _fila_barrrido(valor, semilla, elbo, recon, activas, kl):
    return {
        "paso": "1_latente",
        "valor": valor,
        "semilla": str(semilla),
        "val_elbo": str(elbo),
        "val_recon": str(recon),
        "unidades_activas": str(activas),
        "kl_medio": str(kl),
        "valido": "1",
        "epocas": "10",
        "segundos": "1",
    }


def test_elegir_descarta_el_latente_colapsado():
    paso = {"id": "1_latente", "valores": [4, 8, 16]}
    filas = []
    for valor, elbo, activas, kl in (
        ("4", 0.1, 0, 0.0),
        ("8", 0.5, 8, 1.0),
        ("16", 0.5, 16, 1.0),
    ):
        for semilla in range(2):
            filas.append(_fila_barrrido(valor, semilla, elbo, 0.3, activas, kl))
    ganador, detalle = _elegir(paso, filas, {}, {"perdidas": ["mse"]})
    assert ganador == "8"
    assert detalle["colapsado"] is False


def test_elegir_marca_colapso_si_todos_los_latentes_estan_muertos():
    paso = {"id": "1_latente", "valores": [4, 8, 16]}
    filas = []
    for valor, elbo in (("4", 0.1), ("8", 0.2), ("16", 0.3)):
        for semilla in range(2):
            filas.append(_fila_barrrido(valor, semilla, elbo, 0.3, 0, 0.0))
    _ganador, detalle = _elegir(paso, filas, {}, {"perdidas": ["mse"]})
    assert detalle["colapsado"] is True


def _filas_beta(valores):
    return [
        _fila_barrrido(valor, semilla, 0.5, recon, activas, 1.0)
        for valor, recon, activas in valores
        for semilla in range(2)
    ]


def test_elegir_beta_prefiere_el_mayor_dentro_de_la_tolerancia_de_reconstruccion():
    paso = {"id": "3_beta", "valores": [0.01, 0.1, 1.0, 5.0]}
    filas = _filas_beta(
        [("0.01", 0.0200, 4), ("0.1", 0.0205, 4), ("1.0", 0.0208, 4), ("5.0", 0.0250, 4)]
    )
    ganador, detalle = _elegir(paso, filas, {"latente": 4}, {"perdidas": ["mse"]})
    # 0.0208 está a 4 % del mejor (0.0200); 0.0250 queda afuera.
    assert ganador == "1.0"
    assert detalle["colapsado"] is False


def test_elegir_beta_no_elige_uno_colapsado_aunque_reconstruya_igual():
    paso = {"id": "3_beta", "valores": [0.01, 0.1, 1.0]}
    filas = _filas_beta([("0.01", 0.0200, 4), ("0.1", 0.0205, 4), ("1.0", 0.0205, 0)])
    ganador, _detalle = _elegir(paso, filas, {"latente": 4}, {"perdidas": ["mse"]})
    assert ganador == "0.1"


def test_aplicar_nivel_cambia_beta_free_bits_y_epocas():
    from ej2.vae_exp import _leer_config

    cfg = _leer_config()
    cfg_nivel, ds_nivel = _aplicar_nivel(cfg, cfg["datasets"]["banderas"], 2)

    betas = next(p["valores"] for p in cfg_nivel["pasos"] if p["id"].startswith("3"))
    capacidades = next(p["valores"] for p in cfg_nivel["pasos"] if p["id"].startswith("2"))
    assert betas == [0.001, 0.01, 0.1]
    assert capacidades == [[256, 128], [512, 256]]
    assert cfg_nivel["free_bits"] == 0.1
    assert cfg_nivel["calentamiento_frac"] == 0.5
    assert ds_nivel["max_epocas"] == 1500
    assert ds_nivel["paciencia"] == 300
    # El original no se muta.
    assert cfg["free_bits"] == 0.0
