"""Constructores de VAE y de AE determinístico para el ejercicio 2."""

from collections.abc import Sequence

import numpy as np

from core.autoencoder import Autoencoder
from core.network import Network
from core.vae import VariationalAutoencoder


def _capacidad(capacidad: Sequence[int]) -> tuple[int, int]:
    if len(capacidad) != 2:
        raise ValueError(f"capacidad tiene que tener dos anchos, llegó {list(capacidad)}")
    c1, c2 = int(capacidad[0]), int(capacidad[1])
    if c1 < 1 or c2 < 1:
        raise ValueError(f"Los anchos de capacidad tienen que ser positivos, llegaron {c1}, {c2}")
    return c1, c2


def _redes(d: int, latente: int, capacidad: Sequence[int], salida: str, semilla: int):
    c1, c2 = _capacidad(capacidad)
    if d < 1 or latente < 1:
        raise ValueError(f"d y latente tienen que ser positivos, llegaron {d}, {latente}")
    if salida not in ("sigmoid", "identity"):
        raise ValueError(f"Salida desconocida: {salida!r}")
    rng = np.random.default_rng(semilla)
    encoder = Network(
        [d, c1, c2, 2 * latente],
        hidden_activation="tanh",
        output_activation="identity",
        initializer="xavier",
        rng=rng,
    )
    decoder = Network(
        [latente, c2, c1, d],
        hidden_activation="tanh",
        output_activation=salida,
        initializer="xavier",
        rng=rng,
    )
    return encoder, decoder


def construir_vae_imagen(
    d: int,
    latente: int,
    capacidad: Sequence[int],
    salida: str = "sigmoid",
    semilla: int = 0,
    beta: float = 1.0,
    free_bits: float = 0.0,
) -> VariationalAutoencoder:
    encoder, decoder = _redes(d, latente, capacidad, salida, semilla)
    return VariationalAutoencoder(encoder, decoder, kl_weight=beta, free_bits=free_bits)


def construir_ae_imagen(
    d: int,
    latente: int,
    capacidad: Sequence[int],
    salida: str = "sigmoid",
    semilla: int = 0,
) -> Autoencoder:
    c1, c2 = _capacidad(capacidad)
    rng = np.random.default_rng(semilla)
    encoder = Network(
        [d, c1, c2, latente],
        hidden_activation="tanh",
        output_activation="identity",
        initializer="xavier",
        rng=rng,
    )
    decoder = Network(
        [latente, c2, c1, d],
        hidden_activation="tanh",
        output_activation=salida,
        initializer="xavier",
        rng=rng,
    )
    return Autoencoder(encoder, decoder)


def construir_vae_tabular(
    latente: int,
    capacidad: Sequence[int],
    semilla: int = 0,
    beta: float = 1.0,
    free_bits: float = 0.0,
) -> VariationalAutoencoder:
    return construir_vae_imagen(
        12,
        latente,
        capacidad,
        salida="identity",
        semilla=semilla,
        beta=beta,
        free_bits=free_bits,
    )


def construir_ae_tabular(latente: int, capacidad: Sequence[int], semilla: int = 0) -> Autoencoder:
    return construir_ae_imagen(12, latente, capacidad, salida="identity", semilla=semilla)
