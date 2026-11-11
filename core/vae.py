"""Autoencoder variacional (apuntes §19.3): un Autoencoder con capa latente estocástica.

El encoder ya no devuelve un código sino los parámetros de q(z/x) = N(μ(x), Σ(x))
con Σ diagonal: sus salidas son [μ, s], con s = log Σ (la sustitución
Σ = exp(s) de §19.3.3). El código se muestrea con el truco de la
reparametrización (ec. 19.6):

    z = μ + ε ⊙ σ,    ε ~ N(0, I),    σ = exp(s / 2)

y la función a minimizar es J = L + λR (ec. 19.12), por muestra:

    L = pérdida de reconstrucción entre x y x' = decoder(z)
    R = KL(q(z/x) || N(0, I)) = ½ Σ_k (exp(s_k) + μ_k² − 1 − s_k)

Las pérdidas del motor promedian sobre muestras y features, así que R se
normaliza igual: con MSE, J = (1 / (n·d)) Σ_muestras [‖x − x'‖² + λ·KL], que es
la expresión de los apuntes salvo el factor constante 1/d.
"""

import numpy as np

from core.autoencoder import Autoencoder
from core.losses import Loss
from core.network import Metric, Network


class VariationalAutoencoder(Autoencoder):
    """Autoencoder cuyo encoder parametriza una normal por cada entrada.

    encoder: salida de 2 · latent_dim, [μ, log Σ], con activación identidad
    (los dos son valores reales sin cota). decoder: entrada de latent_dim.
    kl_weight es λ. El ruido ε sale de un generador hijo del rng del encoder,
    así que muestrear no cambia el orden de los lotes.
    """

    encoder_outputs_per_latent = 2

    def __init__(self, encoder: Network, decoder: Network, kl_weight: float = 1.0) -> None:
        super().__init__(encoder, decoder)
        if kl_weight < 0:
            raise ValueError(f"kl_weight no puede ser negativo, llegó {kl_weight}")
        self.kl_weight = kl_weight
        self.noise_rng = encoder.rng.spawn(1)[0]
        # Del último predict, para el backward.
        self._mu: np.ndarray | None = None
        self._log_var: np.ndarray | None = None
        self._eps: np.ndarray | None = None

    def latent_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(μ, log Σ) de q(z/x), cada uno (n_muestras, latent_dim)."""
        out = self.encoder.predict(X)
        return out[:, : self.latent_dim], out[:, self.latent_dim :]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Media μ(x) de q(z/x), (n_muestras, latent_dim). No muestrea."""
        return self.latent_distribution(X)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Reconstrucción estocástica: muestrea z ~ q(z/x) y lo decodifica.

        La reconstrucción sin ruido es decode(encode(X)).
        """
        mu, log_var = self.latent_distribution(X)
        eps = self._noise(mu.shape)
        self._mu, self._log_var, self._eps = mu, log_var, eps
        return self.decoder.predict(mu + eps * np.exp(0.5 * log_var))

    def sample(self, n: int) -> np.ndarray:
        """Genera n muestras nuevas: z ~ p(z) = N(0, I) por el decoder."""
        return self.decode(self._noise((n, self.latent_dim)))

    def kl_divergence(self, X: np.ndarray) -> float:
        """KL(q(z/x) || N(0, I)) promedio por muestra, sumada sobre el espacio latente."""
        mu, log_var = self.latent_distribution(X)
        return _kl_sum(mu, log_var) / X.shape[0]

    def backward(
        self, y_true: np.ndarray, y_pred: np.ndarray, loss: Loss, l2: float = 0.0
    ) -> np.ndarray:
        """Gradiente de J = L + λR; y_pred tiene que salir del predict inmediatamente anterior.

        El decoder retropropaga L hasta z. En el nodo estocástico (§19.3.3):
        ∂z/∂μ = 1 y ∂z/∂σ = ε, es decir ∂z/∂s = ε·σ/2; R no pasa por el
        decoder y aporta ∂R/∂μ = μ y ∂R/∂s = ½ (exp(s) − 1). Con eso sigue el
        backpropagation por el encoder.
        """
        if self._mu is None:
            raise RuntimeError("backward antes de predict: no hay μ, s ni ε guardados.")
        mu, log_var, eps = self._mu, self._log_var, self._eps
        grad_z = self.decoder.backward(y_true, y_pred, loss, l2=l2)
        scale = self.kl_weight / y_true.size
        grad_mu = grad_z + scale * mu
        grad_log_var = grad_z * eps * 0.5 * np.exp(0.5 * log_var) + scale * 0.5 * (
            np.exp(log_var) - 1.0
        )
        grad = np.hstack([grad_mu, grad_log_var])
        for layer in reversed(self.encoder.layers):
            grad = layer.backward(grad)
        if l2:
            for layer in self.encoder.layers:
                layer.grad_W += l2 * layer.W
        return grad

    def _scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss: Loss,
        metric: Metric | None,
        metrics: dict[str, Metric] | None,
    ) -> dict[str, float]:
        """Como en Network, pero "loss" es J = L + λR con una muestra de z por entrada."""
        result = super()._scores(X, y, loss, metric, metrics)
        result["loss"] += self.kl_weight * _kl_sum(self._mu, self._log_var) / y.size
        return result

    def _noise(self, shape: tuple[int, ...]) -> np.ndarray:
        return self.noise_rng.standard_normal(shape)


def _kl_sum(mu: np.ndarray, log_var: np.ndarray) -> float:
    """Σ sobre muestras y dimensiones latentes de ½ (exp(s) + μ² − 1 − s)."""
    return float(0.5 * np.sum(np.exp(log_var) + mu**2 - 1.0 - log_var))
