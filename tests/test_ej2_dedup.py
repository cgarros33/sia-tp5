"""Tests de deduplicación de F1 (utils/preparar_ej2.py)."""

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent


def _cargar_preparador():
    ruta = RAIZ / "utils" / "preparar_ej2.py"
    spec = importlib.util.spec_from_file_location("preparar_ej2_test", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _imagen(ruta: Path, valor: int) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((20, 20, 3), valor, dtype=np.uint8)).save(ruta)


def _fixture(raiz: Path) -> Path:
    headshots = raiz / "headshots"
    for temporada in ("2020", "2021"):
        (headshots / temporada / "by_ref").mkdir(parents=True, exist_ok=True)
        for codigo, valor in (("AAA", 100), ("BBB", 150)):
            _imagen(headshots / temporada / f"{codigo}.png", valor)
            shutil.copyfile(
                headshots / temporada / f"{codigo}.png",
                headshots / temporada / "by_ref" / f"X{codigo}01.png",
            )
        _imagen(headshots / temporada / "CCC.png", 200)
    # AAA de 2021 es idéntico a AAA de 2020: el dedup por hash tiene que quedarse
    # con la temporada más reciente.
    shutil.copyfile(headshots / "2020" / "AAA.png", headshots / "2021" / "AAA.png")
    return headshots


def _configurar(modulo, raiz: Path) -> Path:
    headshots = raiz / "headshots"
    modulo.RAIZ = raiz
    modulo.HEADSHOTS = headshots
    modulo.PROCESADAS = raiz / "procesadas"
    modulo.RESULTADOS = raiz / "resultados"
    return headshots


def test_dedup_borra_by_ref_y_conserva_una_por_hash(tmp_path):
    preparador = _cargar_preparador()
    headshots = _configurar(preparador, tmp_path)
    _fixture(tmp_path)

    resumen = preparador.deduplicar_headshots(headshots)

    assert resumen == {"borrados": 4, "conservados": 3}
    assert not list(headshots.glob("*/by_ref"))
    manifest = (tmp_path / "procesadas" / "f1_manifest.csv").read_text().splitlines()
    assert len(manifest) == 4
    assert "2021/AAA.png" in "\n".join(manifest)
    assert "2020/AAA.png" not in "\n".join(manifest)
    assert len((tmp_path / "resultados" / "dedup_hash.csv").read_text().splitlines()) == 4
    assert len((tmp_path / "resultados" / "dedup_headshots.csv").read_text().splitlines()) == 5


def test_dedup_aborta_si_un_by_ref_no_tiene_duplicado(tmp_path):
    preparador = _cargar_preparador()
    headshots = _configurar(preparador, tmp_path)
    _fixture(tmp_path)
    _imagen(headshots / "2020" / "by_ref" / "HUERFANO.png", 7)

    with pytest.raises(RuntimeError, match="sin duplicado exacto"):
        preparador.deduplicar_headshots(headshots)

    assert list(headshots.glob("*/by_ref"))
    assert not (tmp_path / "procesadas" / "f1_manifest.csv").exists()


def test_dedup_puede_preservar_huerfanos(tmp_path):
    preparador = _cargar_preparador()
    headshots = _configurar(preparador, tmp_path)
    _fixture(tmp_path)
    huerfano = headshots / "2020" / "by_ref" / "HUERFANO.png"
    _imagen(huerfano, 7)

    resumen = preparador.deduplicar_headshots(headshots, permitir_huerfanos=True)

    assert resumen["borrados"] == 5
    assert not list(headshots.glob("*/by_ref"))
    copias = list((tmp_path / "procesadas" / "f1_byref_sin_par").glob("*.png"))
    assert len(copias) == 1
    assert np.asarray(Image.open(copias[0])).mean() == 7
