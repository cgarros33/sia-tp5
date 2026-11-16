"""Tests de la verificación con más semillas del barrido del 1a."""

import copy
import json

from ej1.barrido import aplicar, leer_filas
from ej1.verificacion import config_del_paso, empatan_con_el_ganador, verificar

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


def test_config_del_paso_es_la_vigente_antes_de_elegir():
    elecciones = _elecciones(CONFIG)
    paso, config = config_del_paso(elecciones, "5")
    assert paso["nombre"] == "inicializacion"
    assert config == elecciones["pasos"][3]["config"]
    paso, config = config_del_paso(elecciones, "1")
    assert config == CONFIG["base"]


def test_config_del_paso_final_repite_el_paso_con_la_config_final():
    elecciones = _elecciones(CONFIG)
    paso, config = config_del_paso(elecciones, "final")
    assert paso["id"] == "3"
    assert config == elecciones["config_final"]


def _r(pct, media, desvio):
    return {"pct_llegan": pct, "epocas_media": media, "epocas_desvio": desvio}


def test_empatan_dentro_de_un_desvio_del_ganador():
    resumen = {
        "a": _r(100, 100, 10),
        "b": _r(100, 108, 30),
        "c": _r(100, 115, 5),
        "d": _r(80, 100, 1),
    }
    assert empatan_con_el_ganador(resumen, "a") == ["a", "b"]


def test_verificar_escribe_un_csv_por_paso_y_se_puede_retomar(tmp_path):
    config_barrido = copy.deepcopy(CONFIG)
    config_barrido["base"]["max_epocas"] = 20
    elecciones = _elecciones(config_barrido)
    resumen = verificar(config_barrido, elecciones, tmp_path, [0, 1], pasos=("final",))

    filas = leer_filas(tmp_path / "paso_final_activacion_ocultas.csv")
    assert len(filas) == 4 * 2
    assert {f["semilla"] for f in filas} == {0, 1}
    assert resumen["final"]["ganador_original"] == "tanh"
    assert json.loads((tmp_path / "verificacion.json").read_text(encoding="utf-8")) == resumen
    # Con el CSV escrito no se vuelve a entrenar y el resumen sale igual.
    assert verificar(config_barrido, elecciones, tmp_path, [0, 1], pasos=("final",)) == resumen
