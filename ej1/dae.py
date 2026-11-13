"""Ejercicio 1b: denoising autoencoder sobre font.h (docs/context/ej1b-denoising.md).

    python -m ej1.dae [--config ej1/config_dae.json] [--salida resultados/ej1b]

Entrena con la letra ruidosa (bit-flip) como entrada y la limpia como objetivo,
y mide contra la letra limpia. Recorre los experimentos en orden (A: latente,
B: ruido de entrenamiento); en cada uno elige con la validación y deja fijo el
ganador. Solo escribe en la carpeta de salida:

    experimento_A_latente.csv, ...   una fila por corrida (valor × semilla)
    referencias_validacion.csv       nada, vecino más cercano y AE del 1a
    pesos/<exp>_<valor>_semilla_<s>.npz   pesos de los valores en "guardar_pesos"
                                     (el latente 2 de A, para la figura 5)
    elecciones.json                  qué ganó en cada experimento y por qué
    final/                           config elegida: config.json,
                                     pesos_semilla_<s>.npz, historia_semilla_<s>.csv
                                     y test.csv (DAE y referencias en test)

Si el CSV de un experimento ya existe no se vuelve a correr: se lee para
elegir. Si se corrió con otra config, es un error: hay que borrarlo, igual que
final/. El test se lee solo para escribir final/test.csv.
"""

import argparse
import copy
import csv
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from core.augmentation import BitFlip
from core.autoencoder import Autoencoder
from core.callbacks import EarlyStopping
from core.history import History
from core.losses import get_loss
from core.metrics import pixel_errors
from core.optimizers import get_optimizer
from ej1.barrido import (
    _construir,
    _media_desvio,
    _valor_actual,
    aplicar,
    armar_autoencoder,
    cargar_modelo,
)
from ej1.datos import cargar_font

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CONFIG = RAIZ / "ej1" / "config_dae.json"
RUTA_SALIDA = RAIZ / "resultados" / "ej1b"
RUTA_FINAL_1A = RAIZ / "resultados" / "ej1" / "final"

MEDIDAS = ("pixeles_mal", "peor_letra", "recuperadas")

# Una letra está recuperada si le quedan a lo sumo estos píxeles mal (el umbral del 1a).
UMBRAL_RECUPERADA = 1


def conjuntos_ruidosos(
    X: np.ndarray, niveles: list[float], k: int, semilla: int
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """k copias ruidosas (bit-flip) de cada letra de X para cada nivel p.

    Devuelve {p: (X_ruidoso, idx)}: X_ruidoso es (n·k, n_pixeles) e idx (n·k,)
    dice de qué letra limpia sale cada fila (X[idx] es el objetivo). Un solo
    generador, default_rng(semilla), recorre los niveles en el orden dado.
    """
    rng = np.random.default_rng(semilla)
    idx = np.tile(np.arange(X.shape[0]), k)
    return {p: (BitFlip(p)(X[idx], rng), idx) for p in niveles}


def medir_limpieza(X: np.ndarray, idx: np.ndarray, X_rec: np.ndarray) -> dict:
    """Píxeles mal de cada reconstrucción contra su letra limpia X[idx].

    pixeles_mal: promedio por letra; peor_letra: el máximo; recuperadas:
    fracción de letras con <= UMBRAL_RECUPERADA píxeles mal.
    """
    errores = pixel_errors(X[idx], X_rec)
    return {
        "pixeles_mal": float(errores.mean()),
        "peor_letra": int(errores.max()),
        "recuperadas": float(np.mean(errores <= UMBRAL_RECUPERADA)),
    }


def vecino_mas_cercano(X_ruidoso: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Para cada fila de X_ruidoso, la letra de X a menor distancia de Hamming.

    Datos en {0, 1}. En un empate gana la letra de menor índice.
    """
    distancias = X_ruidoso @ (1.0 - X).T + (1.0 - X_ruidoso) @ X.T  # (m, n_letras)
    return X[np.argmin(distancias, axis=1)]


def cargar_ae_1a(ruta: Path = RUTA_FINAL_1A) -> dict[int, Autoencoder]:
    """{semilla: autoencoder} de la config final del 1a, con los pesos que guardó el barrido."""
    config = json.loads((ruta / "config.json").read_text(encoding="utf-8"))
    modelos = {}
    for archivo in sorted(ruta.glob("pesos_semilla_*.npz")):
        semilla = int(archivo.stem.removeprefix("pesos_semilla_"))
        with np.load(archivo) as pesos:
            n_entradas = pesos["W0"].shape[0]
        modelos[semilla] = cargar_modelo(config, archivo, n_entradas)
    return modelos


# --- Entrenamiento ---


def validacion_para(
    ruido: float | list[float], val: dict[float, tuple[np.ndarray, np.ndarray]], X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(X_val, y_val) que mira el corte: el nivel de entrenamiento, o todos los p > 0 si es mixto.

    Con ruido 0 es el nivel 0: las letras limpias.
    """
    niveles = [p for p in val if p > 0] if isinstance(ruido, list) else [ruido]
    X_v = np.concatenate([val[p][0] for p in niveles])
    idx = np.concatenate([val[p][1] for p in niveles])
    return X_v, X[idx]


def entrenar_dae(
    config: dict, X: np.ndarray, semilla: int, validacion: tuple[np.ndarray, np.ndarray]
) -> tuple[Autoencoder, History, dict]:
    """Entrena con X ruidoso (BitFlip(config["ruido"])) como entrada y X limpio como objetivo.

    Corta con EarlyStopping sobre val_loss (paciencia de la config) y vuelve a
    los pesos de la mejor época. Devuelve el autoencoder, su historia y
    {epocas, mejor_epoca, segundos, estado}.
    """
    ae = armar_autoencoder(config, X.shape[1], np.random.default_rng(semilla))
    optimizer = _construir(get_optimizer, config["optimizador"], lr=config["lr"])
    corte = EarlyStopping(patience=config["paciencia"], monitor="val_loss", restore_best=True)
    # Una corrida que diverge desborda antes de que fit la corte: no es un error.
    with np.errstate(over="ignore", invalid="ignore"):
        inicio = time.perf_counter()
        historia = ae.fit(
            X,
            X,
            optimizer=optimizer,
            loss=get_loss(config["perdida"]),
            epochs=config["max_epocas"],
            batch_size=config["batch_size"],
            validation=validacion,
            callbacks=[corte],
            l2=config["l2"],
            augment=BitFlip(config["ruido"]),
        )
        segundos = time.perf_counter() - inicio
    return (
        ae,
        historia,
        {
            "epocas": len(historia),
            "mejor_epoca": corte.best_epoch,
            "segundos": segundos,
            "estado": historia.status,
        },
    )


def evaluar(
    reconstruir: Callable[[np.ndarray], np.ndarray],
    conjuntos: dict[float, tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
) -> dict[float, dict]:
    """{p: medir_limpieza} de reconstruir sobre cada nivel de conjuntos."""
    with np.errstate(over="ignore", invalid="ignore"):
        return {p: medir_limpieza(X, idx, reconstruir(X_r)) for p, (X_r, idx) in conjuntos.items()}


# --- CSV ---


def _nivel(p: float) -> str:
    """0.05 → "0.05", 0.0 → "0": sufijo de las columnas de cada nivel."""
    return f"{p:g}"


def _columnas_val(niveles: list[float]) -> list[str]:
    return [f"val_{m}_{_nivel(p)}" for p in niveles for m in MEDIDAS]


def _escribir_csv(ruta: Path, columnas: list[str], filas: list[dict]) -> None:
    """Se escribe a un temporal y se renombra: un CSV a medias nunca queda con su nombre."""
    temporal = ruta.with_suffix(".tmp")
    with open(temporal, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(
                {
                    c: "" if fila[c] is None else json.dumps(fila[c]) if c == "config" else fila[c]
                    for c in columnas
                }
            )
    os.replace(temporal, ruta)


def _convertir(columna: str, texto: str) -> Any:
    if texto == "":
        return None
    if columna == "config":
        return json.loads(texto)
    if columna in ("experimento", "valor", "modelo", "estado"):
        return texto
    if columna in ("semilla", "epocas", "mejor_epoca") or "peor_letra" in columna:
        return int(texto)
    return float(texto)


def leer_filas_dae(ruta: Path) -> list[dict]:
    """Inversa de _escribir_csv: config como dict, números como int o float, None si vacío."""
    with open(ruta, encoding="utf-8", newline="") as f:
        return [{c: _convertir(c, v) for c, v in crudo.items()} for crudo in csv.DictReader(f)]


# --- Elección ---


def resumir_dae(filas: list[dict], niveles: list[float]) -> dict[str, dict]:
    """Por valor: media ± desvío sobre semillas del promedio sobre los p > 0 de validación."""
    con_ruido = [_nivel(p) for p in niveles if p > 0]
    grupos: dict[str, list[dict]] = {}
    for fila in filas:
        grupos.setdefault(fila["valor"], []).append(fila)
    resumen = {}
    for valor, grupo in grupos.items():
        r: dict[str, Any] = {"corridas": len(grupo)}
        for medida in ("pixeles_mal", "recuperadas"):
            por_corrida = [np.mean([f[f"val_{medida}_{n}"] for n in con_ruido]) for f in grupo]
            r[f"{medida}_media"], r[f"{medida}_desvio"] = _media_desvio(por_corrida)
        r["epocas_media"], r["epocas_desvio"] = _media_desvio([f["epocas"] for f in grupo])
        r["segundos_media"] = float(np.mean([f["segundos"] for f in grupo]))
        r["divergidas"] = sum(f["estado"] == "diverged" for f in grupo)
        resumen[valor] = r
    return resumen


def elegir_dae(resumen: dict[str, dict], actual: str | None) -> tuple[str, str]:
    """Ganador y criterio: "pixeles_mal", "recuperadas" o "empate".

    1. Menos píxeles mal medios. 2. Si empatan: más letras recuperadas.
    3. Si siguen empatados: el valor actual, o el primero de la grilla.
    """
    minimo = min(r["pixeles_mal_media"] for r in resumen.values())
    candidatos = [v for v, r in resumen.items() if r["pixeles_mal_media"] == minimo]
    if len(candidatos) == 1:
        return candidatos[0], "pixeles_mal"
    maximo = max(resumen[v]["recuperadas_media"] for v in candidatos)
    candidatos = [v for v in candidatos if resumen[v]["recuperadas_media"] == maximo]
    if len(candidatos) == 1:
        return candidatos[0], "recuperadas"
    return (actual if actual in candidatos else candidatos[0]), "empate"


# --- Corrida completa ---


def _filas_referencias(
    conjuntos: dict, X: np.ndarray, modelos_1a: dict[int, Autoencoder]
) -> list[dict]:
    """Filas (modelo, semilla, nivel, medidas) de nada, vecino más cercano y AE del 1a."""
    evaluados = [
        ("nada", None, evaluar(lambda X_r: X_r, conjuntos, X)),
        ("vecino", None, evaluar(lambda X_r: vecino_mas_cercano(X_r, X), conjuntos, X)),
        *[("ae_1a", s, evaluar(ae.predict, conjuntos, X)) for s, ae in modelos_1a.items()],
    ]
    return _filas_por_nivel(evaluados)


def _filas_por_nivel(evaluados: list[tuple[str, int | None, dict]]) -> list[dict]:
    return [
        {"modelo": modelo, "semilla": semilla, "nivel": p, **medidas}
        for modelo, semilla, por_nivel in evaluados
        for p, medidas in por_nivel.items()
    ]


COLUMNAS_NIVEL = ["modelo", "semilla", "nivel", *MEDIDAS]


def correr_dae(config_dae: dict, salida: Path, escribir: Callable[[str], None] = print) -> dict:
    """Corre (o retoma) los experimentos y el test final.

    Devuelve lo que escribe en elecciones.json.
    """
    X, _ = cargar_font()
    salida.mkdir(parents=True, exist_ok=True)
    niveles = config_dae["niveles"]
    semillas = config_dae["semillas"]
    k = config_dae["copias"]
    val = conjuntos_ruidosos(X, niveles, k, config_dae["semilla_validacion"])
    modelos_1a = cargar_ae_1a()

    ruta_ref = salida / "referencias_validacion.csv"
    if not ruta_ref.exists():
        _escribir_csv(ruta_ref, COLUMNAS_NIVEL, _filas_referencias(val, X, modelos_1a))

    columnas = [
        "experimento", "valor", "semilla", "epocas", "mejor_epoca", "segundos", "estado",
        *_columnas_val(niveles), "config",
    ]  # fmt: skip

    def correr_experimento(exp: dict, config: dict) -> list[dict]:
        ruta = salida / f"experimento_{exp['id']}_{exp['nombre']}.csv"
        esperadas = {v["etiqueta"]: aplicar(config, v["cambios"]) for v in exp["valores"]}
        if ruta.exists():
            escribir(f"[{exp['id']}] ya existe {ruta.name}: se lee en vez de correr.")
            filas = leer_filas_dae(ruta)
            for fila in filas:
                if fila["config"] != esperadas.get(fila["valor"]):
                    raise RuntimeError(
                        f"{ruta.name} se corrió con otra config ({fila['valor']}): "
                        "borralo para volver a correr ese experimento."
                    )
            return filas

        filas = []
        guardar = exp.get("guardar_pesos", [])
        if guardar:
            (salida / "pesos").mkdir(exist_ok=True)
        for etiqueta, config_valor in esperadas.items():
            validacion = validacion_para(config_valor["ruido"], val, X)
            for semilla in semillas:
                ae, _, info = entrenar_dae(config_valor, X, semilla, validacion)
                if etiqueta in guardar:
                    ruta_pesos = salida / "pesos" / f"{exp['id']}_{etiqueta}_semilla_{semilla}.npz"
                    _guardar_pesos(ae, ruta_pesos)
                fila = {"experimento": exp["id"], "valor": etiqueta, "semilla": semilla, **info}
                for p, medidas in evaluar(ae.predict, val, X).items():
                    for m in MEDIDAS:
                        fila[f"val_{m}_{_nivel(p)}"] = medidas[m]
                fila["config"] = config_valor
                filas.append(fila)
                escribir(
                    f"[{exp['id']}] {etiqueta} semilla {semilla}: {info['epocas']} épocas "
                    f"(mejor {info['mejor_epoca']}), {info['segundos']:.1f} s"
                )
        _escribir_csv(ruta, columnas, filas)
        return leer_filas_dae(ruta)

    elecciones: dict[str, Any] = {"config_dae": config_dae, "experimentos": []}
    config = copy.deepcopy(config_dae["base"])
    for exp in config_dae["experimentos"]:
        resumen = resumir_dae(correr_experimento(exp, config), niveles)
        actual = _valor_actual(exp["valores"], config)
        ganador, criterio = elegir_dae(resumen, actual)
        cambios = next(v["cambios"] for v in exp["valores"] if v["etiqueta"] == ganador)
        config = aplicar(config, cambios)
        elecciones["experimentos"].append(
            {
                "id": exp["id"],
                "nombre": exp["nombre"],
                "valor_anterior": actual,
                "ganador": ganador,
                "criterio": criterio,
                "resumen": resumen,
                "config": config,
            }
        )
        r = resumen[ganador]
        escribir(
            f"[{exp['id']}] gana {ganador} ({criterio}): {r['pixeles_mal_media']:.3f} ± "
            f"{r['pixeles_mal_desvio']:.3f} píxeles mal en validación"
        )
    elecciones["config_final"] = config

    _guardar_final(config, X, val, semillas, config_dae, modelos_1a, salida / "final", escribir)
    texto = json.dumps(elecciones, indent=2, ensure_ascii=False)
    (salida / "elecciones.json").write_text(texto, encoding="utf-8")
    return json.loads(texto)


def _guardar_pesos(ae: Autoencoder, ruta: Path) -> None:
    """W<i> y b<i> de cada capa, como los lee cargar_modelo."""
    pesos = {}
    for i, capa in enumerate(ae.layers):
        pesos[f"W{i}"], pesos[f"b{i}"] = capa.W, capa.b
    np.savez(ruta, **pesos)


def _guardar_final(
    config: dict,
    X: np.ndarray,
    val: dict,
    semillas: list[int],
    config_dae: dict,
    modelos_1a: dict[int, Autoencoder],
    carpeta: Path,
    escribir: Callable[[str], None],
) -> None:
    """Reentrena la config elegida con cada semilla y la evalúa, con las referencias, en test.

    test.csv se escribe último: si existe, todo lo demás ya está.
    """
    ruta_test = carpeta / "test.csv"
    if ruta_test.exists():
        guardada = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        if guardada != config:
            raise RuntimeError(f"{carpeta} es de otra config final: borrala para regenerarla.")
        escribir("[final] ya existe final/test.csv: no se vuelve a entrenar.")
        return

    carpeta.mkdir(exist_ok=True)
    texto = json.dumps(config, indent=2, ensure_ascii=False)
    (carpeta / "config.json").write_text(texto, encoding="utf-8")
    validacion = validacion_para(config["ruido"], val, X)
    modelos = {}
    for semilla in semillas:
        ae, historia, info = entrenar_dae(config, X, semilla, validacion)
        _guardar_pesos(ae, carpeta / f"pesos_semilla_{semilla}.npz")
        historia.to_csv(carpeta / f"historia_semilla_{semilla}.csv")
        modelos[semilla] = ae
        escribir(
            f"[final] semilla {semilla}: {info['epocas']} épocas (mejor {info['mejor_epoca']})"
        )

    # Recién acá se genera y se lee el test.
    test = conjuntos_ruidosos(
        X, config_dae["niveles"], config_dae["copias"], config_dae["semilla_test"]
    )
    evaluados = [("dae", s, evaluar(ae.predict, test, X)) for s, ae in modelos.items()]
    filas = _filas_por_nivel(evaluados) + _filas_referencias(test, X, modelos_1a)
    _escribir_csv(ruta_test, COLUMNAS_NIVEL, filas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoising autoencoder del ejercicio 1b.")
    parser.add_argument("--config", type=Path, default=RUTA_CONFIG)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    correr_dae(config, args.salida, escribir=lambda texto: print(texto, flush=True))


if __name__ == "__main__":
    main()
