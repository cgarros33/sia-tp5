import copy
import json

import numpy as np
import pytest

from core.metrics import pixel_errors
from ej1 import dae
from ej1.barrido import aplicar
from ej1.dae import (
    RUTA_CONFIG,
    RUTA_FINAL_1A,
    _normalizar_config,
    cargar_ae_1a,
    conjuntos_gaussianos,
    conjuntos_ruidosos,
    correr_dae,
    elegir_dae,
    entrenar_dae,
    leer_filas_dae,
    medir_limpieza,
    validacion_para,
    vecino_mas_cercano,
)
from ej1.datos import cargar_font
from ej1.graficos_dae import curvas_por_modelo, fig_gaussiano, generar_figuras

CONFIG = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
BASE = CONFIG["base"]

NIVELES = [0.0, 0.05, 0.1, 0.2, 0.3]


# --- Conjuntos ruidosos ---


def test_forma_e_indices_de_cada_nivel():
    X, _ = cargar_font()
    conjuntos = conjuntos_ruidosos(X, NIVELES, k=3, semilla=1000)
    assert list(conjuntos) == NIVELES
    for X_r, idx in conjuntos.values():
        assert X_r.shape == (32 * 3, 35)
        assert set(np.unique(X_r)) <= {0.0, 1.0}
        # Cada letra aparece k veces.
        np.testing.assert_array_equal(np.bincount(idx), np.full(32, 3))


def test_nivel_cero_son_las_letras_limpias():
    X, _ = cargar_font()
    X_r, idx = conjuntos_ruidosos(X, [0.0], k=2, semilla=1000)[0.0]
    np.testing.assert_array_equal(X_r, X[idx])


def test_tasa_de_ruido_cercana_a_p():
    X, _ = cargar_font()
    for p, (X_r, idx) in conjuntos_ruidosos(X, [0.1, 0.3], k=50, semilla=1000).items():
        assert (X_r != X[idx]).mean() == pytest.approx(p, abs=0.01)


def test_reproducible_y_distinto_con_otra_semilla():
    X, _ = cargar_font()
    a = conjuntos_ruidosos(X, [0.1], k=5, semilla=1000)[0.1][0]
    b = conjuntos_ruidosos(X, [0.1], k=5, semilla=1000)[0.1][0]
    c = conjuntos_ruidosos(X, [0.1], k=5, semilla=2000)[0.1][0]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# --- Conjuntos gaussianos ---


def test_gaussiano_sigma_cero_deja_las_letras_limpias():
    X, _ = cargar_font()
    X_r, idx = conjuntos_gaussianos(X, [0.0], k=2, semilla=3000)[0.0]
    np.testing.assert_array_equal(X_r, X[idx])


def test_gaussiano_mismas_formas_e_indices_que_bit_flip():
    X, _ = cargar_font()
    sigmas = [0.1, 0.3]
    gaussianos = conjuntos_gaussianos(X, sigmas, k=3, semilla=3000)
    ruidosos = conjuntos_ruidosos(X, sigmas, k=3, semilla=3000)
    assert list(gaussianos) == sigmas
    for s in sigmas:
        X_g, idx_g = gaussianos[s]
        X_r, idx_r = ruidosos[s]
        assert X_g.shape == X_r.shape == (32 * 3, 35)
        np.testing.assert_array_equal(idx_g, idx_r)


def test_gaussiano_queda_en_rango_sin_binarizar_y_es_reproducible():
    X, _ = cargar_font()
    X_r, _ = conjuntos_gaussianos(X, [0.3], k=5, semilla=3000)[0.3]
    assert X_r.min() >= 0.0 and X_r.max() <= 1.0
    # No se binariza: hay valores intermedios.
    assert np.any((X_r > 0.0) & (X_r < 1.0))
    a = conjuntos_gaussianos(X, [0.1], k=5, semilla=3000)[0.1][0]
    b = conjuntos_gaussianos(X, [0.1], k=5, semilla=3000)[0.1][0]
    c = conjuntos_gaussianos(X, [0.1], k=5, semilla=4000)[0.1][0]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# --- Medición ---


def test_medir_limpieza_a_mano():
    X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    idx = np.array([0, 0, 1])
    X_rec = np.array(
        [
            [0.1, 0.2, 0.4],  # 0 mal
            [0.9, 0.6, 0.0],  # 2 mal
            [1.0, 0.7, 0.2],  # 1 mal
        ]
    )
    m = medir_limpieza(X, idx, X_rec)
    assert m["pixeles_mal"] == pytest.approx(1.0)
    assert m["peor_letra"] == 2
    assert m["recuperadas"] == pytest.approx(2 / 3)


# --- Referencias ---


def test_vecino_mas_cercano_a_mano():
    X = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    X_r = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],  # a 1 de la 0
            [1.0, 1.0, 1.0, 0.0],  # a 1 de la 1 y de la 2: empate, gana la 1 (menor índice)
            [1.0, 0.0, 0.0, 0.0],  # a 1 de la 0 y de la 2: empate, gana la 0
        ]
    )
    np.testing.assert_array_equal(vecino_mas_cercano(X_r, X), X[[0, 1, 0]])


def test_vecino_mas_cercano_sin_ruido_devuelve_cada_letra():
    X, _ = cargar_font()
    np.testing.assert_array_equal(vecino_mas_cercano(X, X), X)


def test_ae_1a_carga_las_5_semillas_y_reconstruye_las_letras():
    X, _ = cargar_font()
    modelos = cargar_ae_1a()
    assert sorted(modelos) == [0, 1, 2, 3, 4]
    for ae in modelos.values():
        assert ae.latent_dim == 2
        # La config final del 1a llega: la peor letra tiene <= 1 píxel mal.
        assert medir_limpieza(X, np.arange(32), ae.predict(X))["peor_letra"] <= 1


# --- Config ---


def test_base_es_la_config_final_del_1a():
    final_1a = json.loads((RUTA_FINAL_1A / "config.json").read_text(encoding="utf-8"))
    assert final_1a.pop("decaimiento") is None
    propias = {
        k: BASE[k] for k in BASE if k not in ("paciencia", "ruido", "copias_epoca", "monitor")
    }
    assert propias == final_1a


# --- Validación para fit ---


def test_validacion_para_nivel_fijo_y_mixto():
    X, _ = cargar_font()
    val = conjuntos_ruidosos(X, NIVELES, k=2, semilla=1000)
    X_v, y_v = validacion_para(0.1, val, X)
    np.testing.assert_array_equal(X_v, val[0.1][0])
    np.testing.assert_array_equal(y_v, X[val[0.1][1]])
    # Mixto: la unión de todos los niveles con ruido (p > 0).
    X_v, y_v = validacion_para([0.0, 0.3], val, X)
    assert X_v.shape == y_v.shape == (4 * 64, 35)
    np.testing.assert_array_equal(X_v[:64], val[0.05][0])


# --- elegir_dae ---


def _r(pixeles, recuperadas=0.5):
    return {"pixeles_mal_media": pixeles, "recuperadas_media": recuperadas}


def test_elegir_dae_mas_recuperadas_aunque_tenga_mas_pixeles_mal():
    resumen = {"a": _r(2.0, 0.6), "b": _r(3.0, 0.7), "c": _r(1.5, 0.65)}
    assert elegir_dae(resumen, actual="a") == ("b", "recuperadas")


def test_elegir_dae_empate_desempata_por_pixeles_mal():
    resumen = {"a": _r(1.5, 0.8), "b": _r(1.0, 0.8), "c": _r(0.5, 0.6)}
    assert elegir_dae(resumen, actual="a") == ("b", "pixeles_mal")


def test_elegir_dae_empate_total_actual_o_primero():
    resumen = {"a": _r(1.0), "b": _r(1.0), "c": _r(2.0)}
    assert elegir_dae(resumen, actual="b") == ("b", "empate")
    assert elegir_dae(resumen, actual="c") == ("a", "empate")


# --- entrenar_dae ---


def _validacion(X):
    return validacion_para(0.1, conjuntos_ruidosos(X, [0.1], k=2, semilla=1000), X)


def test_entrenar_dae_registra_validacion_y_es_reproducible():
    X, _ = cargar_font()
    config = aplicar(BASE, {"max_epocas": 5})
    _, h1, info = entrenar_dae(config, X, semilla=0, validacion=_validacion(X))
    _, h2, _ = entrenar_dae(config, X, semilla=0, validacion=_validacion(X))
    assert len(h1) == info["epocas"] == 5
    assert 1 <= info["mejor_epoca"] <= 5
    assert info["estado"] == "ok"
    np.testing.assert_array_equal(h1.column("val_loss"), h2.column("val_loss"))


def test_entrenar_dae_el_ruido_cambia_el_entrenamiento():
    X, _ = cargar_font()
    sin = aplicar(BASE, {"max_epocas": 3, "ruido": 0.0})
    con = aplicar(BASE, {"max_epocas": 3, "ruido": [0.0, 0.3]})
    _, h_sin, _ = entrenar_dae(sin, X, semilla=0, validacion=_validacion(X))
    _, h_con, _ = entrenar_dae(con, X, semilla=0, validacion=_validacion(X))
    assert h_sin.column("train_loss")[0] != h_con.column("train_loss")[0]


def test_entrenar_dae_corta_por_paciencia_y_restaura_lo_mejor():
    X, _ = cargar_font()
    # lr enorme: la validación empeora enseguida y corta a las mejor_epoca + paciencia.
    config = aplicar(BASE, {"max_epocas": 200, "paciencia": 3, "lr": 1.0})
    _, historia, info = entrenar_dae(config, X, semilla=0, validacion=_validacion(X))
    assert info["epocas"] == info["mejor_epoca"] + 3 < 200


def test_copias_epoca_triplica_las_actualizaciones(monkeypatch):
    from core.optimizers import Adam

    original = Adam.step
    llamadas = []

    def contar(self, params, grads):
        llamadas.append(1)
        return original(self, params, grads)

    monkeypatch.setattr(Adam, "step", contar)
    X, _ = cargar_font()
    config = aplicar(BASE, {"max_epocas": 1, "paciencia": 10})
    entrenar_dae(config, X, semilla=0, validacion=_validacion(X))
    una = len(llamadas)
    llamadas.clear()
    entrenar_dae(aplicar(config, {"copias_epoca": 3}), X, semilla=0, validacion=_validacion(X))
    assert una == 1
    assert len(llamadas) == 3 * una


def test_corte_por_recuperadas_restaura_la_mejor_epoca():
    X, _ = cargar_font()
    config = aplicar(
        BASE, {"max_epocas": 40, "paciencia": 3, "monitor": "val_recuperadas", "lr": 0.02}
    )
    validacion = _validacion(X)
    ae, historia, info = entrenar_dae(config, X, semilla=0, validacion=validacion)
    recuperadas_epoca = historia.column("val_recuperadas")
    assert np.isfinite(recuperadas_epoca).all()
    mejor = info["mejor_epoca"]
    assert recuperadas_epoca[mejor - 1] == np.max(recuperadas_epoca)
    X_v, y_v = validacion
    final = float(np.mean(pixel_errors(y_v, ae.predict(X_v)) <= dae.UMBRAL_RECUPERADA))
    assert final == pytest.approx(recuperadas_epoca[mejor - 1])


def test_copias_epoca_uno_y_val_loss_reproducen_el_entrenamiento_anterior():
    from core.augmentation import BitFlip
    from core.callbacks import EarlyStopping
    from core.losses import get_loss
    from core.optimizers import get_optimizer
    from ej1.barrido import armar_autoencoder

    X, _ = cargar_font()
    config = aplicar(BASE, {"max_epocas": 5})
    validacion = _validacion(X)
    ae_nuevo, historia_nueva, _ = entrenar_dae(config, X, semilla=0, validacion=validacion)

    ae_viejo = armar_autoencoder(config, X.shape[1], np.random.default_rng(0))
    corte = EarlyStopping(patience=config["paciencia"], monitor="val_loss", restore_best=True)
    historia_vieja = ae_viejo.fit(
        X,
        X,
        optimizer=get_optimizer(config["optimizador"], lr=config["lr"]),
        loss=get_loss(config["perdida"]),
        epochs=config["max_epocas"],
        batch_size=config["batch_size"],
        validation=validacion,
        callbacks=[corte],
        l2=config["l2"],
        augment=BitFlip(config["ruido"]),
    )
    np.testing.assert_array_equal(
        historia_nueva.column("val_loss"), historia_vieja.column("val_loss")
    )
    for nuevo, viejo in zip(ae_nuevo.params, ae_viejo.params, strict=True):
        np.testing.assert_array_equal(nuevo, viejo)


def test_normalizar_config_completa_los_defaults_del_experimento_c():
    vieja = {k: v for k, v in BASE.items() if k not in ("copias_epoca", "monitor")}
    assert _normalizar_config(vieja) == _normalizar_config(BASE)


def test_experimento_c_tiene_las_nueve_combinaciones():
    exp = next(e for e in CONFIG["experimentos"] if e["id"] == "C")
    ocultas = {tuple(v["cambios"]["ocultas"]) for v in exp["valores"]}
    copias = {v["cambios"]["copias_epoca"] for v in exp["valores"]}
    assert ocultas == {(60, 40, 20), (128, 64), (128, 128, 64)}
    assert copias == {1, 8, 16}
    assert len(exp["valores"]) == 9


def test_fig_gaussiano_dibuja_pixeles_mal_y_recuperadas():
    sigmas = [0.0, 0.1, 0.2]
    filas = []
    for modelo, base in (("dae", 0.5), ("nada", 1.0), ("vecino", 0.1), ("ae_1a", 2.0)):
        semillas = (0, 1) if modelo in ("dae", "ae_1a") else (None,)
        for semilla in semillas:
            for s in sigmas:
                filas.append(
                    {
                        "modelo": modelo,
                        "semilla": semilla,
                        "nivel": s,
                        "pixeles_mal": base + s,
                        "peor_letra": 3,
                        "recuperadas": max(0.0, 1.0 - base - s),
                    }
                )
    curvas = {m: curvas_por_modelo(filas, sigmas, m) for m in ("pixeles_mal", "recuperadas")}
    fig = fig_gaussiano(curvas, sigmas, 2)
    assert len(fig.axes) == 2


# --- Smoke test ---


def _config_mini():
    """1 semilla, 2 copias por letra, pocas épocas y 2 valores por experimento."""
    mini = copy.deepcopy(CONFIG)
    mini["semillas"] = [0]
    mini["copias"] = 2
    mini["base"]["max_epocas"] = 15
    mini["base"]["paciencia"] = 5
    mini["experimentos"][0]["valores"] = mini["experimentos"][0]["valores"][:2]
    mini["experimentos"][1]["valores"] = mini["experimentos"][1]["valores"][-2:]
    mini["experimentos"][2]["valores"] = mini["experimentos"][2]["valores"][:2]
    return mini


def test_dae_mini_escribe_todo_y_se_puede_retomar(tmp_path, monkeypatch):
    config = _config_mini()
    salida = tmp_path / "ej1b"
    mensajes = []
    elecciones = correr_dae(config, salida, escribir=mensajes.append)

    a = leer_filas_dae(salida / "experimento_A_latente.csv")
    b = leer_filas_dae(salida / "experimento_B_ruido.csv")
    c = leer_filas_dae(salida / "experimento_C_arquitectura_y_copias.csv")
    assert [f["valor"] for f in a] == ["2", "4"]
    assert [f["valor"] for f in b] == ["0.3", "mixto"]
    assert [f["valor"] for f in c] == ["60-40-20 k1", "60-40-20 k8"]
    for nivel in ("0", "0.05", "0.1", "0.2", "0.3"):
        assert f"val_pixeles_mal_{nivel}" in a[0]
    assert b[1]["config"]["ruido"] == [0.0, 0.3]

    # La config final es la de aplicar los ganadores en orden.
    esperada = copy.deepcopy(config["base"])
    for exp, eleccion in zip(config["experimentos"], elecciones["experimentos"], strict=True):
        cambios = {v["etiqueta"]: v["cambios"] for v in exp["valores"]}
        esperada = aplicar(esperada, cambios[eleccion["ganador"]])
    assert elecciones["config_final"] == esperada
    assert json.loads((salida / "elecciones.json").read_text(encoding="utf-8")) == elecciones

    # Referencias en validación y test final: 4 modelos × 5 niveles (dae y ae_1a por semilla).
    ref = leer_filas_dae(salida / "referencias_validacion.csv")
    assert {f["modelo"] for f in ref} == {"nada", "vecino", "ae_1a"}
    final = salida / "final"
    test = leer_filas_dae(final / "test.csv")
    assert {f["modelo"] for f in test} == {"dae", "nada", "vecino", "ae_1a"}
    assert len([f for f in test if f["modelo"] == "dae"]) == 5
    assert len([f for f in test if f["modelo"] == "ae_1a"]) == 5 * 5
    assert (final / "historia_semilla_0.csv").exists()

    # Los pesos finales reproducen la corrida del experimento que fijó la config final.
    X, _ = cargar_font()
    ae = dae.cargar_modelo(esperada, final / "pesos_semilla_0.npz", X.shape[1])
    fila_final = next(
        f for f in (*a, *b, *c) if _normalizar_config(f["config"]) == _normalizar_config(esperada)
    )
    X_v, idx = conjuntos_ruidosos(X, NIVELES, k=2, semilla=1000)[0.1]
    assert medir_limpieza(X, idx, ae.predict(X_v))["pixeles_mal"] == pytest.approx(
        fila_final["val_pixeles_mal_0.1"]
    )

    # Pesos del latente 2 de A: se guardan y reproducen su fila; los del latente 4, no.
    pesos_a = salida / "pesos" / "A_2_semilla_0.npz"
    assert pesos_a.exists() and not (salida / "pesos" / "A_4_semilla_0.npz").exists()
    fila_a = next(f for f in a if f["valor"] == "2")
    ae_a = dae.cargar_modelo(fila_a["config"], pesos_a, X.shape[1])
    assert medir_limpieza(X, idx, ae_a.predict(X_v))["pixeles_mal"] == pytest.approx(
        fila_a["val_pixeles_mal_0.1"]
    )

    # Figuras: solo leen lo escrito.
    escritas = generar_figuras(salida)
    nombres = {r.name for r in escritas}
    assert {"01_latente.png", "02_ruido_entrenamiento.png", "03_final_test.png"} <= nombres
    assert {"04_ejemplos.png", "05_latente_dae.png"} <= nombres
    assert all(r.stat().st_size > 0 for r in escritas)

    # Retomar: con todo escrito no se entrena nada y las elecciones salen iguales.
    def no_entrenar(*args, **kwargs):
        raise AssertionError("no tendría que entrenar")

    monkeypatch.setattr(dae, "entrenar_dae", no_entrenar)
    assert correr_dae(config, salida, escribir=mensajes.append) == elecciones
