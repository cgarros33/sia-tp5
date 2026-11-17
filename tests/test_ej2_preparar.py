"""Tests de preparación de los datasets: chino invertido y banderas auxiliares."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent


def _cargar(nombre: str):
    ruta = RAIZ / "utils" / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(f"{nombre}_test", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _fila_china(posicion: int, caracter: str) -> list[str]:
    pixeles = np.zeros(4096, dtype=np.int16)
    pixeles[posicion] = 255
    return [str(v) for v in pixeles] + ["0", caracter]


def test_leer_chino_invierte_normaliza_y_binariza(tmp_path):
    preparador = _cargar("preparar_ej2")
    ruta = tmp_path / "chino.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"pixel_{i}" for i in range(4096)] + ["label", "character"])
        for clase in range(15):
            writer.writerow(_fila_china(clase * 256, chr(0x4E00 + clase)))

    X, y, clases = preparador._leer_chino(ruta, por_clase=1)

    assert X.shape == (15, 1024)
    assert y.tolist() == list(range(15))
    assert len(clases) == 15
    # Todo queda en {0, 1}: el trazo más intenso en 0 y el fondo en 1.
    assert set(np.unique(X)) == {0.0, 1.0}
    plano = X[0].reshape(32, 32)
    assert plano[0, 0] == pytest.approx(0.0)
    assert plano[5, 5] == pytest.approx(1.0)
    # El carácter 1 tiene su trazo en la fila 2 del plano, no en la 0.
    assert X[1].reshape(32, 32)[0, 0] == pytest.approx(1.0)
    assert X[1].reshape(32, 32)[2, 0] == pytest.approx(0.0)


def test_generar_vector_escribe_una_bandera(tmp_path):
    generador = _cargar("generar_banderas")
    vector = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.5]

    generador.generar_vector("prueba", vector, ancho=60, salida=tmp_path)

    with Image.open(tmp_path / "prueba.png") as imagen:
        assert imagen.size == (60, 40)
        assert imagen.getpixel((30, 0)) == (255, 0, 0)
        assert imagen.getpixel((30, 20)) == (0, 255, 0)
        assert imagen.getpixel((30, 39)) == (0, 0, 255)


def test_generar_vector_rechaza_forma_incorrecta(tmp_path):
    generador = _cargar("generar_banderas")
    with pytest.raises(ValueError, match="12 features"):
        generador.generar_vector("malo", [0.0] * 11, salida=tmp_path)
