import copy
import csv
import json

import numpy as np
import pytest

from core.activations import LeakyReLU, Sigmoid, Tanh
from core.initializers import Uniform
from core.metrics import pixel_errors
from ej1 import barrido
from ej1.barrido import (
    RUTA_CONFIG,
    aplicar,
    armar_autoencoder,
    cargar_modelo,
    correr_barrido,
    elegir,
    entrenar,
    leer_filas,
)
from ej1.datos import cargar_font

CONFIG = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
BASE = CONFIG["base"]

COLUMNAS = {
    "paso",
    "valor",
    "semilla",
    "llego",
    "epoca_llegada",
    "epocas",
    "peor_letra",
    "pixeles_mal",
    "mse",
    "segundos",
    "actualizaciones",
    "estado",
    "config",
}


# --- aplicar ---


def test_aplicar_reemplaza_sin_tocar_el_original():
    original = copy.deepcopy(BASE)
    nueva = aplicar(BASE, {"lr": 0.1, "ocultas": [5]})
    assert nueva["lr"] == 0.1 and nueva["ocultas"] == [5]
    assert BASE == original


def test_aplicar_factor_ancho_redondea_mitades_hacia_arriba():
    config = aplicar(BASE, {"ocultas": [30, 23, 17, 10]})
    assert aplicar(config, {"factor_ancho": 0.5})["ocultas"] == [15, 12, 9, 5]
    assert aplicar(config, {"factor_ancho": 2})["ocultas"] == [60, 46, 34, 20]
    assert aplicar(config, {"factor_ancho": 1}) == config


def test_aplicar_factor_ancho_sin_ocultas_y_minimo_uno():
    assert aplicar(aplicar(BASE, {"ocultas": []}), {"factor_ancho": 0.5})["ocultas"] == []
    assert aplicar(aplicar(BASE, {"ocultas": [1]}), {"factor_ancho": 0.1})["ocultas"] == [1]


def test_aplicar_factor_lr_multiplica_el_vigente():
    config = aplicar(BASE, {"lr": 0.02})
    nueva = aplicar(config, {"perdida": "mse", "factor_lr": 5})
    assert nueva["perdida"] == "mse"
    assert nueva["lr"] == pytest.approx(0.1)
    assert "factor_lr" not in nueva


# --- elegir ---


def _resumen(pct, epocas=None, peor=5.0):
    return {"pct_llegan": pct, "epocas_media": epocas, "peor_letra_media": peor}


def test_elegir_gana_el_que_mas_llega():
    resumen = {"a": _resumen(60, 100), "b": _resumen(100, 900), "c": _resumen(0)}
    assert elegir(resumen, actual="a") == ("b", "llegan")


def test_elegir_empate_en_llegan_desempata_por_epocas():
    resumen = {"a": _resumen(100, 500), "b": _resumen(100, 400), "c": _resumen(40, 10)}
    assert elegir(resumen, actual="a") == ("b", "epocas")


def test_elegir_si_ninguno_llega_gana_la_menor_peor_letra():
    resumen = {"a": _resumen(0, peor=4.0), "b": _resumen(0, peor=2.4), "c": _resumen(0, peor=3)}
    assert elegir(resumen, actual="a") == ("b", "peor_letra")


def test_elegir_empate_total_se_queda_el_actual():
    resumen = {"a": _resumen(100, 400), "b": _resumen(100, 400), "c": _resumen(100, 400)}
    assert elegir(resumen, actual="b") == ("b", "empate")
    resumen = {"a": _resumen(0, peor=3.0), "b": _resumen(0, peor=3.0)}
    assert elegir(resumen, actual="b") == ("b", "empate")


def test_elegir_empate_sin_actual_entre_los_empatados_gana_el_primero():
    resumen = {"a": _resumen(100, 400), "b": _resumen(100, 400), "c": _resumen(0)}
    assert elegir(resumen, actual="c") == ("a", "empate")


# --- armar_autoencoder y entrenar ---


def test_armar_autoencoder_espeja_el_decoder():
    ae = armar_autoencoder(BASE, 35, np.random.default_rng(0))
    assert ae.layer_sizes == [35, 30, 20, 10, 2, 10, 20, 30, 35]
    assert isinstance(ae.layers[0].activation, Tanh)
    assert isinstance(ae.layers[-1].activation, Sigmoid)


def test_armar_autoencoder_sin_ocultas_y_specs_con_parametros():
    config = aplicar(
        BASE,
        {
            "ocultas": [],
            "activacion_ocultas": {"kind": "leaky_relu", "alpha": 0.2},
            "inicializacion": {"kind": "uniform", "low": -0.1, "high": 0.1},
        },
    )
    ae = armar_autoencoder(config, 35, np.random.default_rng(0))
    assert ae.layer_sizes == [35, 2, 35]
    assert isinstance(ae.encoder.initializer, Uniform)
    assert np.abs(ae.layers[0].W).max() <= 0.1
    con_ocultas = armar_autoencoder(aplicar(config, {"ocultas": [4]}), 35, np.random.default_rng(0))
    assert isinstance(con_ocultas.layers[0].activation, LeakyReLU)
    assert con_ocultas.layers[0].activation.alpha == 0.2


def test_entrenar_registra_epocas_y_actualizaciones():
    X, _ = cargar_font()
    config = aplicar(BASE, {"batch_size": 8, "max_epocas": 3})
    ae, historia, res = entrenar(config, X, semilla=0, objetivo=1)
    assert len(historia) == res["epocas"] == 3
    assert res["actualizaciones"] == 3 * 4
    assert not res["llego"] and res["epoca_llegada"] is None
    assert res["estado"] == "ok"
    errores = pixel_errors(X, ae.predict(X))
    np.testing.assert_array_equal(res["errores"], errores)
    assert res["peor_letra"] == errores.max()
    assert res["pixeles_mal"] == pytest.approx(errores.mean())


def test_entrenar_misma_semilla_mismo_resultado():
    X, _ = cargar_font()
    config = aplicar(BASE, {"max_epocas": 5, "batch_size": 4})
    _, h1, r1 = entrenar(config, X, semilla=3, objetivo=1)
    _, h2, r2 = entrenar(config, X, semilla=3, objetivo=1)
    np.testing.assert_array_equal(h1.column("train_loss"), h2.column("train_loss"))
    assert r1["mse"] == r2["mse"]


def test_entrenar_corta_al_llegar():
    X, _ = cargar_font()
    # Objetivo trivial (35 píxeles mal): llega en la primera época.
    _, historia, res = entrenar(BASE, X, semilla=0, objetivo=35)
    assert res["llego"] and res["epoca_llegada"] == 1 and len(historia) == 1


def test_entrenar_divergencia_queda_registrada():
    X, _ = cargar_font()
    config = aplicar(
        BASE,
        {
            "ocultas": [],
            "activacion_salida": "identity",
            "perdida": "mse",
            "optimizador": "gd",
            "lr": 1e6,
            "max_epocas": 50,
        },
    )
    _, _, res = entrenar(config, X, semilla=0, objetivo=1)
    assert res["estado"] == "diverged"
    assert not res["llego"]


# --- Smoke test del barrido ---


def _config_mini():
    """1 semilla, pocas épocas y 2 valores por paso. El AE lineal conserva sus 5000 épocas."""
    mini = copy.deepcopy(CONFIG)
    mini["semillas"] = [0]
    mini["base"]["max_epocas"] = 30
    for paso in mini["pasos"]:
        paso["valores"] = paso["valores"][:2]
    return mini


def test_barrido_mini_escribe_todo_y_se_puede_retomar(tmp_path, monkeypatch):
    config = _config_mini()
    salida = tmp_path / "ej1"
    mensajes = []
    elecciones = correr_barrido(config, salida, escribir=mensajes.append)

    # Un CSV por paso (líneas de base, 10 pasos y la verificación final).
    csvs = sorted(p.name for p in salida.glob("paso_*.csv"))
    assert len(csvs) == 1 + len(config["pasos"]) + 1
    assert "paso_00_lineas_de_base.csv" in csvs
    assert "paso_06_07_optimizador_y_lr.csv" in csvs
    assert "paso_final_activacion_ocultas.csv" in csvs

    # Columnas pedidas y una fila por valor y semilla.
    with open(salida / "paso_01_profundidad.csv", encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f)
        columnas = set(lector.fieldnames)
        assert len(list(lector)) == 2
    assert COLUMNAS <= columnas
    assert len([c for c in columnas if c.startswith("px_")]) == 32

    base = leer_filas(salida / "paso_00_lineas_de_base.csv")
    assert [f["valor"] for f in base] == ["pca", "ae_lineal", "sin_ocultas"]
    assert base[0]["semilla"] is None and base[0]["peor_letra"] == 15
    assert elecciones["lineas_de_base"]["verificacion_lineal"]["dentro_de_tolerancia"]

    # Una elección por paso, y la config final es la de aplicar los ganadores en orden.
    assert [p["id"] for p in elecciones["pasos"]] == [p["id"] for p in config["pasos"]]
    esperada = copy.deepcopy(config["base"])
    for paso, eleccion in zip(config["pasos"], elecciones["pasos"], strict=True):
        cambios = {v["etiqueta"]: v["cambios"] for v in paso["valores"]}
        assert eleccion["ganador"] in cambios
        esperada = aplicar(esperada, cambios[eleccion["ganador"]])
    assert elecciones["config_final"] == esperada
    assert elecciones["verificacion_final"]["paso"] == "3"
    assert json.loads((salida / "elecciones.json").read_text(encoding="utf-8")) == elecciones

    # Config final: pesos e historia por semilla, y los pesos reproducen la corrida guardada.
    final = salida / "final"
    assert (final / "historia_semilla_0.csv").exists()
    assert json.loads((final / "config.json").read_text(encoding="utf-8")) == esperada
    fila = leer_filas(final / "corridas.csv")[0]
    X, _ = cargar_font()
    ae = cargar_modelo(esperada, final / "pesos_semilla_0.npz", X.shape[1])
    np.testing.assert_array_equal(pixel_errors(X, ae.predict(X)), fila["errores"])
    pca = np.load(final / "pca.npz")
    assert pca["Z"].shape == (32, 2)

    # Retomar: con todo escrito no se entrena nada y las elecciones salen iguales.
    def no_entrenar(*args, **kwargs):
        raise AssertionError("no tendría que entrenar")

    monkeypatch.setattr(barrido, "entrenar", no_entrenar)
    assert correr_barrido(config, salida, escribir=mensajes.append) == elecciones
