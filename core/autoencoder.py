"""Autoencoder: dos redes multicapa encadenadas (apuntes §19.1.1).

    Z  = f(X) = encoder(X)        código latente, (n_muestras, latent_dim)
    X' = g(Z) = decoder(Z)        reconstrucción, (n_muestras, n_features)

El codificador y el decodificador son Network comunes. El autoencoder es la
red que resulta de poner las capas de uno a continuación de las del otro, así
que predict, backward, fit, los optimizadores y los callbacks son los del
MLP. No sabe nada del problema: qué se reconstruye, con qué pérdida y con qué
ruido lo decide quien lo entrena.
"""

from typing import Any

import numpy as np

from core.history import History
from core.network import Network


class Autoencoder(Network):
    """Encadena encoder y decoder y los entrena como una sola red.

    Las capas se comparten, no se copian: entrenar el autoencoder actualiza
    los pesos de encoder y decoder, que después se pueden usar por separado
    (el decoder solo es el modelo generativo de §19.2). El orden de los lotes
    sale del rng del encoder.
    """

    # Salidas del encoder por cada dimensión del espacio latente.
    encoder_outputs_per_latent = 1

    def __init__(self, encoder: Network, decoder: Network) -> None:
        expected = self.encoder_outputs_per_latent * decoder.layer_sizes[0]
        if encoder.layer_sizes[-1] != expected:
            raise ValueError(
                f"El encoder sale con {encoder.layer_sizes[-1]} dimensiones y se esperaban "
                f"{expected} para un decoder con entrada de {decoder.layer_sizes[0]}"
            )
        if decoder.layer_sizes[-1] != encoder.layer_sizes[0]:
            raise ValueError(
                f"El decoder sale con {decoder.layer_sizes[-1]} dimensiones y la entrada "
                f"del encoder tiene {encoder.layer_sizes[0]}"
            )
        self.encoder = encoder
        self.decoder = decoder
        self.layer_sizes = encoder.layer_sizes + decoder.layer_sizes[1:]
        self.layers = encoder.layers + decoder.layers
        self.rng = encoder.rng

    @property
    def latent_dim(self) -> int:
        return self.decoder.layer_sizes[0]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """X: (n_muestras, n_features). Devuelve Z, (n_muestras, latent_dim)."""
        return self.encoder.predict(X)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Z: (n_muestras, latent_dim). Devuelve X', (n_muestras, n_features)."""
        return self.decoder.predict(Z)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, *args: Any, **kwargs: Any) -> History:
        """Network.fit con y opcional: sin y, el objetivo es reconstruir X.

        Pasar y distinto de X entrena a reconstruir y a partir de X (por
        ejemplo X con ruido e y limpio). El resto de los argumentos son los de
        Network.fit.
        """
        return super().fit(X, X if y is None else y, *args, **kwargs)
