"""Tests de C3: lote y pérdida del 1a con su propio lr."""

import copy
import json

from ej1 import verificacion_c3
from ej1.barrido import aplicar, leer_filas
from ej1.verificacion_c3 import (
    combinaciones,
    elegir_actualizaciones,
    max_epocas,
    mejor_por_nivel,
    verificar_c3,
)

CONFIG = json.loads(open("ej1/config.json", encoding="utf-8").read())


def _elecciones(config_barrido):
    base = config_barrido["base"]
    pasos = []
    config = copy.deepcopy(base)
    for paso in config_barrido["pasos"]:
        config = aplicar(config, paso["valores"][0]["cambios"])
        pasos.append(
            {"id": paso["id"], "ganador": paso["valores"][0]["etiqueta"], "config": config}
        )
    return {
        "config_barrido": config_barrido,
        "pasos": pasos,
        "config_final": config,
        "verificacion_final": {"paso": "3", "ganador": "tanh"},
    }


def test_combinaciones_de_los_dos_pasos():
    paso_9 = combinaciones("9")
    paso_10 = combinaciones("10")
    assert len(paso_9) == 12
    assert len(paso_10) == 8
    assert {c["valor"] for c in paso_9} == {
        f"{lote}/{lr:g}" for lote in (32, 8, 4, 1) for lr in (0.001, 0.003, 0.01)
    }
    assert {c["cambios"]["perdida"] for c in paso_10} == {"binary_crossentropy", "mse"}


def test_solo_el_lote_uno_se_corta_en_50_mil_actualizaciones():
    assert max_epocas({"max_epocas": 10000, "batch_size": 1}) == 1562
    assert max_epocas({"max_epocas": 10000, "batch_size": 4}) == 10000
    assert max_epocas({"max_epocas": 10000, "batch_size": 32}) == 10000


def _r(pct, actualizaciones, peor=1.0):
    return {
        "pct_llegan": pct,
        "actualizaciones_media": actualizaciones,
        "peor_letra_media": peor,
    }


def test_elegir_actualizaciones_usa_actualizaciones_no_epocas():
    resumen = {"a": _r(100, 1000), "b": _r(100, 800)}
    assert elegir_actualizaciones(resumen, "a") == ("b", "actualizaciones")
    resumen = {"a": _r(100, 1000), "b": _r(80, 100)}
    assert elegir_actualizaciones(resumen, "a") == ("a", "llegan")
    resumen = {"a": _r(0, None, 5), "b": _r(0, None, 3)}
    assert elegir_actualizaciones(resumen, None) == ("b", "peor_letra")


def test_mejor_por_nivel_agrupa_por_lote():
    resumen = {
        "32/0.001": _r(100, 1000),
        "32/0.01": _r(100, 800),
        "8/0.001": _r(50, 500),
    }
    por_nivel = mejor_por_nivel(resumen)
    assert set(por_nivel) == {"32", "8"}
    assert por_nivel["32"]["ganador"] == "32/0.01"
    assert por_nivel["8"]["ganador"] == "8/0.001"


def test_verificar_c3_escribe_los_csv_y_se_puede_retomar(tmp_path, monkeypatch):
    config_barrido = copy.deepcopy(CONFIG)
    config_barrido["base"]["max_epocas"] = 2
    elecciones = _elecciones(config_barrido)
    resultado = verificar_c3(config_barrido, elecciones, tmp_path, [0], max_epocas_tope=2)

    paso_9 = leer_filas(tmp_path / "paso_09_tamano_lote.csv")
    paso_10 = leer_filas(tmp_path / "paso_10_perdida.csv")
    assert len(paso_9) == 12
    assert len(paso_10) == 8
    assert {f["semilla"] for f in paso_9} == {0}
    assert resultado["conclusion"]["paso_9"]["ganador_original"] == "32"
    assert resultado["conclusion"]["paso_10"]["ganador_original"] == "bce"
    assert json.loads((tmp_path / "c3.json").read_text(encoding="utf-8")) == resultado

    # Con los CSV escritos no se vuelve a entrenar y el resumen sale igual.
    def no_entrenar(*args, **kwargs):
        raise AssertionError("no tendría que entrenar")

    monkeypatch.setattr(verificacion_c3, "entrenar", no_entrenar)
    assert verificar_c3(config_barrido, elecciones, tmp_path, [0], max_epocas_tope=2) == resultado
