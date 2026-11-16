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


def test_resumen_incluye_los_fijos_de_nba():
    etiquetas = np.array(
        ["a", "lebron", "Luka-Doncic", "Nikola-Jokic", "Stephen-Curry", "b", "c", "d", "e"]
    )
    indices = graficos._elegir_resumen("nba", etiquetas)
    assert len(indices) == 8
    assert {1, 2, 3, 4}.issubset(indices)


def test_resumen_chino_elige_ocho_clases_distintas():
    etiquetas = np.array([f"clase{i}" for i in range(15) for _ in range(2)])
    indices = graficos._elegir_resumen("chino", etiquetas)
    assert len(indices) == 8
    assert len({etiquetas[i] for i in indices}) == 8


def test_resumen_banderas_usa_todas_las_muestras_si_hay_pocas():
    etiquetas = np.array(["a", "b", "c"])
    indices = graficos._elegir_resumen("banderas", etiquetas)
    assert sorted(indices) == [0, 1, 2]


def test_render_auxiliar_de_banderas_usa_el_generador(tmp_path):
    escalador = {"minimo": [0.0] * 12, "rango": [1.0] * 12}
    vector = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.5])
    imagenes = graficos._render_auxiliares("prueba", np.array([vector]), escalador, tmp_path)
    assert len(imagenes) == 1
    assert imagenes[0].shape == (160, 240, 3)
    assert imagenes[0].min() >= 0.0 and imagenes[0].max() <= 1.0
    assert (tmp_path / "prueba_00.png").exists()


def test_reconstruccion_nba_pone_lebron_primero():
    conjunto = {
        "split": np.array(["train"] * 4 + ["test"] * 3),
        "y": np.array(["a", "b", "lebron", "c", "d", "e", "f"]),
    }
    indices = graficos._indices_reconstruccion("nba", conjunto, "train")
    assert indices[0] == 2
    assert len(indices) == 4


def test_reconstruccion_otro_dataset_no_cambia_el_orden():
    conjunto = {
        "split": np.array(["train"] * 3 + ["test"] * 2),
        "y": np.array(["a", "b", "c", "d", "e"]),
    }
    indices = graficos._indices_reconstruccion("f1", conjunto, "test")
    np.testing.assert_array_equal(indices, [3, 4])
