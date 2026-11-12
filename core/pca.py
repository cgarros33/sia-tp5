"""Análisis de componentes principales (PCA), con NumPy.

Convención de formas: X es (n_muestras, n_features). Las componentes son las
direcciones de máxima varianza de X centrado, es decir los autovectores de su
matriz de covarianza con los k autovalores más grandes. Se calculan con la SVD
de X centrado, Xc = U S Vᵀ: las filas de Vᵀ son esos autovectores y
S² / (n − 1) sus autovalores, sin armar la covarianza.

Un autoencoder lineal entrenado con MSE alcanza el mismo error de
reconstrucción que PCA con k componentes (apuntes §19.1.2).
"""

import numpy as np


class PCA:
    """Proyección sobre las n_components primeras componentes principales.

    Después de fit: mean (n_features,), components (n_components, n_features)
    con filas ortonormales, y explained_variance (n_components,), la varianza
    de los datos a lo largo de cada componente, de mayor a menor. El signo de
    cada componente es arbitrario; se fija dejando positiva su entrada de mayor
    módulo, para que el resultado no dependa de la implementación de la SVD.
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.explained_variance: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Ajusta media y componentes a X, (n_muestras, n_features). Devuelve self."""
        n, d = X.shape
        k = self.n_components
        if not 1 <= k <= min(n, d):
            raise ValueError(f"n_components tiene que estar en [1, {min(n, d)}], llegó {k}")
        self.mean = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        components = Vt[:k]
        largest = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(k), largest])
        self.components = components * signs[:, None]
        self.explained_variance = S[:k] ** 2 / (n - 1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Coordenadas de X en las componentes, (n_muestras, n_components)."""
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstrucción a partir de las coordenadas Z, (n_muestras, n_features)."""
        return Z @ self.components + self.mean
