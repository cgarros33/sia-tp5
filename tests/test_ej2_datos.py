"""Tests de carga y escalado de los datasets del ejercicio 2."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ej2.datos import (
    cargar_banderas,
    cargar_imagenes,
    desescalar_banderas,
    escalar_banderas,
    partes,
)

RAIZ = Path(__file__).resolve().parent.parent


def _npz_imagen(ruta: Path, n: int = 12, d: int = 20) -> Path:
    X = np.linspace(0.0, 1.0, n * d, dtype=np.float32).reshape(n, d)
    split = np.array(["train"] * 8 + ["val"] * 2 + ["test"] * 2)
    meta = {"dataset": "sintetico", "conteos": {"train": 8, "val": 2, "test": 2}}
    np.savez_compressed(
        ruta,
        X=X,
        y=np.arange(n),
        split=split,
        meta=np.asarray(json.dumps(meta)),
    )
    return ruta


def _npz_banderas(ruta: Path) -> Path:
    X = np.zeros((10, 12), dtype=np.float32)
    X[:, :9] = 0.5
    X[:, 9] = np.arange(10) % 2
    X[:, 10] = np.linspace(0.5, 3.0, 10)
    X[:, 11] = np.linspace(1.15, 2.0, 10)
    split = np.array(["train"] * 6 + ["val"] * 2 + ["test"] * 2)
    np.savez_compressed(
        ruta,
        X=X,
        nombres=np.asarray([f"pais{i}" for i in range(10)]),
        split=split,
        columnas=np.asarray(
            [
                "r1",
                "g1",
                "b1",
                "r2",
                "g2",
                "b2",
                "r3",
                "g3",
                "b3",
                "rotada",
                "proporcion",
                "aspecto",
            ]
        ),
        meta=np.asarray(json.dumps({"dataset": "banderas"})),
    )
    return ruta


def test_cargar_imagenes_formas_y_partes(tmp_path):
    ruta = _npz_imagen(tmp_path / "nba.npz")
    datos = cargar_imagenes("nba", ruta=ruta)
    assert datos["X"].shape == (12, 20)
    assert datos["meta"]["dataset"] == "sintetico"
    divisiones = partes(datos)
    assert set(divisiones) == {"train", "val", "test"}
    assert [len(divisiones[s][0]) for s in ("train", "val", "test")] == [8, 2, 2]


def test_cargar_imagenes_rechaza_rango_invalido(tmp_path):
    ruta = tmp_path / "malo.npz"
    _npz_imagen(ruta)
    datos = np.load(ruta)
    np.savez_compressed(
        ruta,
        X=datos["X"] + 2.0,
        y=datos["y"],
        split=datos["split"],
        meta=datos["meta"],
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cargar_imagenes("nba", ruta=ruta)


def test_banderas_son_doce_columnas_sin_imagenes(tmp_path):
    ruta = _npz_banderas(tmp_path / "flags.npz")
    datos = cargar_banderas(ruta=ruta)
    assert datos["X"].shape == (10, 12)
    assert len(datos["columnas"]) == 12
    assert "imagenes" not in datos
    assert "pixeles" not in datos


def test_escalador_se_ajusta_solo_con_train(tmp_path):
    ruta = _npz_banderas(tmp_path / "flags.npz")
    datos = cargar_banderas(ruta=ruta)
    X = datos["X"].copy()
    split = datos["split"]
    escalado, escalador = escalar_banderas(X, split)

    X_alterado = X.copy()
    X_alterado[split != "train", 10] = 999.0
    X_alterado[split != "train", 11] = -999.0
    _, escalador_alterado = escalar_banderas(X_alterado, split)

    assert escalador == escalador_alterado
    assert escalador["minimo"][10] == pytest.approx(X[split == "train", 10].min())
    assert escalador["rango"][10] == pytest.approx(
        X[split == "train", 10].max() - X[split == "train", 10].min()
    )
    assert escalado[split == "train", 10].min() == pytest.approx(0.0)
    assert escalado[split == "train", 10].max() == pytest.approx(1.0)
    # Las columnas que ya estaban en [0, 1] no se tocan.
    np.testing.assert_allclose(escalado[:, :10], X[:, :10])
    np.testing.assert_allclose(desescalar_banderas(escalado, escalador), X, rtol=1e-6)


def test_datos_no_importa_banderas():
    codigo = (
        "import sys; import ej2.datos; "
        "assert 'ej2.banderas' not in sys.modules, 'datos importó banderas'"
    )
    subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, check=True)


def test_archivos_imagen_apuntan_a_las_resoluciones_nuevas():
    from ej2.datos import ARCHIVOS_IMAGEN

    assert ARCHIVOS_IMAGEN == {
        "f1": "f1_32x32.npz",
        "nba": "nba_30x36.npz",
        "chino": "chino_32x32.npz",
    }
