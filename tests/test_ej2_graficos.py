"""Tests de las utilidades de figuras del ejercicio 2."""

from pathlib import Path

import numpy as np
import pytest

from ej2 import graficos


def test_mosaico_acepta_imagenes_de_distinto_tamano():
    imagenes = [
        np.full((10, 12), 0.25),
        np.full((14, 8), 0.5),
        np.full((9, 9), 0.75),
        np.zeros((11, 10)),
    ]
    mosaico = graficos._mosaico(imagenes, columnas=2, separacion=1)
    assert mosaico.shape == (14 * 2 + 1, 12 * 2 + 1)
    assert mosaico.min() >= 0.0 and mosaico.max() <= 1.0


def test_render_normaliza_a_cero_uno():
    escalador = {"minimo": [0.0] * 12, "rango": [1.0] * 12}
    vector = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7, 0.9, 0.0, 1.0, 1.5])
    imagen = graficos._render(vector, escalador)
    assert imagen.ndim == 3 and imagen.shape[2] == 3
    assert 0.0 <= imagen.min() <= imagen.max() <= 1.0


def test_falta_artefacto_indica_la_fase(tmp_path, monkeypatch):
    monkeypatch.setattr(graficos, "RESULTADOS", tmp_path)
    with pytest.raises(SystemExit, match="Corré primero"):
        graficos._carpeta("nba", "final")


def test_grilla_decodificada_tiene_el_tamano_de_la_grilla():
    from ej2.modelos import construir_vae_tabular

    modelo = construir_vae_tabular(2, [4, 3], semilla=0)
    salida = graficos._grilla_puntos(modelo, np.zeros(2), 1.0)
    assert salida.shape == (graficos.LADO_GRILLA**2, 12)
    assert Path(__file__).exists()
