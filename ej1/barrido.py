"""Barrido secuencial del ejercicio 1a (docs/context/ej1-autoencoder.md).

    python -m ej1.barrido [--config ej1/config.json] [--salida resultados/ej1]

Arranca de la config base y recorre los pasos en orden: en cada uno entrena
cada valor con todas las semillas, elige el mejor con la regla del spec y lo
deja fijo para los pasos siguientes. Solo escribe en la carpeta de salida:

    paso_00_lineas_de_base.csv     PCA, AE lineal y sin ocultas
    paso_01_profundidad.csv, ...   una fila por corrida (valor × semilla)
    paso_final_<paso>.csv          el paso repetido con la config final
    elecciones.json                qué ganó en cada paso y por qué
    final/                         config final: config.json, corridas.csv,
                                   pesos_semilla_<s>.npz, historia_semilla_<s>.csv,
                                   y pca.npz (proyección PCA, para los gráficos)

Si el CSV de un paso ya existe no se vuelve a correr: se lee para elegir, así
que un barrido cortado se retoma desde el primer paso que falta. Si ese CSV se
corrió con otra config (porque cambió un paso anterior), es un error: hay que
borrarlo, igual que final/.
"""

import argparse
import copy
import csv
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from core.activations import get_activation
from core.autoencoder import Autoencoder
from core.callbacks import ExponentialDecay, StepDecay, StopAtTarget
from core.history import History
from core.initializers import get_initializer
from core.losses import MSE, get_loss
from core.metrics import max_pixel_errors, mean_pixel_errors, pixel_errors
from core.network import Network
from core.optimizers import get_optimizer
from core.pca import PCA
from ej1.datos import cargar_font

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CONFIG = RAIZ / "ej1" / "config.json"
RUTA_SALIDA = RAIZ / "resultados" / "ej1"

# fit registra train_peor_letra (lo que mira el corte) y train_pixeles_mal por época.
METRICAS = {"peor_letra": max_pixel_errors, "pixeles_mal": mean_pixel_errors}
DECAIMIENTOS = {"exponential": ExponentialDecay, "step": StepDecay}

# La línea de base que tiene que dar el MSE de PCA (apuntes §19.1.2), con esta tolerancia.
ETIQUETA_LINEAL = "ae_lineal"
TOLERANCIA_LINEAL = 0.02

RESULTADO = (
    "llego",
    "epoca_llegada",
    "epocas",
    "peor_letra",
    "pixeles_mal",
    "mse",
    "segundos",
    "actualizaciones",
    "estado",
)
COLUMNAS = ("paso", "valor", "semilla", *RESULTADO, "config")
# Después de COLUMNAS va una columna px_<letra> por letra: sus píxeles mal al final.
PREFIJO_LETRA = "px_"


# --- Config ---


def aplicar(config: dict, cambios: dict) -> dict:
    """Config nueva con los cambios de un valor del barrido; config no se modifica.

    Dos claves son relativas a la config vigente: factor_ancho multiplica cada
    capa oculta (redondeo con las mitades hacia arriba, mínimo 1) y factor_lr
    multiplica el lr. Cualquier otra clave reemplaza el valor que había.
    """
    cambios = copy.deepcopy(cambios)
    factor_ancho = cambios.pop("factor_ancho", None)
    factor_lr = cambios.pop("factor_lr", None)
    nueva = copy.deepcopy(config) | cambios
    if factor_ancho is not None:
        nueva["ocultas"] = [max(1, math.floor(w * factor_ancho + 0.5)) for w in nueva["ocultas"]]
    if factor_lr is not None:
        nueva["lr"] = nueva["lr"] * factor_lr
    return nueva


def _construir(fabrica: Callable[..., Any], spec: str | dict, **extra: Any) -> Any:
    """spec es un nombre ("tanh") o {"kind": nombre, **parámetros}, como en get_augmentation."""
    if isinstance(spec, str):
        return fabrica(spec, **extra)
    params = dict(spec)
    return fabrica(params.pop("kind"), **params, **extra)


def armar_autoencoder(config: dict, n_entradas: int, rng: np.random.Generator) -> Autoencoder:
    """Encoder n_entradas → ocultas → latente y decoder espejado.

    El decoder recorre las mismas capas ocultas al revés: latente → ... → n_entradas.
    """
    ocultas = list(config["ocultas"])
    comunes = {
        "hidden_activation": _construir(get_activation, config["activacion_ocultas"]),
        "initializer": _construir(get_initializer, config["inicializacion"]),
        "rng": rng,
    }
    encoder = Network(
        [n_entradas, *ocultas, config["latente"]],
        output_activation=_construir(get_activation, config["activacion_latente"]),
        **comunes,
    )
    decoder = Network(
        [config["latente"], *ocultas[::-1], n_entradas],
        output_activation=_construir(get_activation, config["activacion_salida"]),
        **comunes,
    )
    return Autoencoder(encoder, decoder)


def cargar_modelo(config: dict, ruta: Path, n_entradas: int) -> Autoencoder:
    """Autoencoder de la config con los pesos que guardó el barrido en ruta (.npz)."""
    ae = armar_autoencoder(config, n_entradas, np.random.default_rng(0))
    with np.load(ruta) as pesos:
        for i, capa in enumerate(ae.layers):
            capa.W[...] = pesos[f"W{i}"]
            capa.b[...] = pesos[f"b{i}"]
    return ae


# --- Una corrida ---


def _medir(X: np.ndarray, X_rec: np.ndarray) -> dict:
    errores = pixel_errors(X, X_rec)
    return {
        "peor_letra": int(errores.max()),
        "pixeles_mal": float(errores.mean()),
        "mse": MSE().value(X, X_rec),
        "errores": errores,
    }


def entrenar(
    config: dict, X: np.ndarray, semilla: int, objetivo: float
) -> tuple[Autoencoder, History, dict]:
    """Entrena una corrida; devuelve el autoencoder, su historia por época y el resultado.

    Corta apenas la peor letra tiene <= objetivo píxeles mal. El resultado
    tiene las claves de RESULTADO (epoca_llegada es None si no llegó) y
    errores: los píxeles mal de cada letra al final, (n_muestras,).
    """
    ae = armar_autoencoder(config, X.shape[1], np.random.default_rng(semilla))
    optimizer = _construir(get_optimizer, config["optimizador"], lr=config["lr"])
    llegada = StopAtTarget(monitor="train_peor_letra", target=objetivo)
    callbacks = [llegada]
    if config["decaimiento"] is not None:
        params = dict(config["decaimiento"])
        callbacks.append(DECAIMIENTOS[params.pop("kind")](optimizer, **params))

    # Una corrida que diverge desborda antes de que fit la corte: no es un error.
    with np.errstate(over="ignore", invalid="ignore"):
        inicio = time.perf_counter()
        historia = ae.fit(
            X,
            optimizer=optimizer,
            loss=get_loss(config["perdida"]),
            epochs=config["max_epocas"],
            batch_size=config["batch_size"],
            callbacks=callbacks,
            metrics=METRICAS,
            l2=config["l2"],
        )
        segundos = time.perf_counter() - inicio
        resultado = _medir(X, ae.predict(X))

    n = X.shape[0]
    lotes_por_epoca = math.ceil(n / min(config["batch_size"], n))
    resultado |= {
        "llego": llegada.reached_epoch is not None,
        "epoca_llegada": llegada.reached_epoch,
        "epocas": len(historia),
        "segundos": segundos,
        "actualizaciones": len(historia) * lotes_por_epoca,
        "estado": historia.status,
    }
    return ae, historia, resultado


def _correr_pca(X: np.ndarray, componentes: int, objetivo: float) -> tuple[PCA, dict]:
    inicio = time.perf_counter()
    pca = PCA(componentes).fit(X)
    resultado = _medir(X, pca.inverse_transform(pca.transform(X)))
    resultado |= {
        "llego": resultado["peor_letra"] <= objetivo,
        "epoca_llegada": None,
        "epocas": None,
        "segundos": time.perf_counter() - inicio,
        "actualizaciones": 0,
        "estado": "ok",
    }
    return pca, resultado


def _describir(res: dict) -> str:
    if res["llego"]:
        texto = f"llega en {res['epoca_llegada']} épocas"
    else:
        texto = f"no llega: peor letra {res['peor_letra']} tras {res['epocas']} épocas"
    if res["estado"] != "ok":
        texto += f" [{res['estado']}]"
    return f"{texto} ({res['segundos']:.1f} s)"


# --- Filas y CSV ---


def _fila(paso: str, valor: str, config: dict, semilla: int | None, res: dict) -> dict:
    return {
        "paso": paso,
        "valor": valor,
        "semilla": semilla,
        **{k: res[k] for k in RESULTADO},
        "config": config,
        "errores": [int(e) for e in res["errores"]],
    }


def _celda(valor: Any) -> Any:
    if valor is None:
        return ""
    if isinstance(valor, bool):
        return int(valor)
    if isinstance(valor, dict):
        return json.dumps(valor, ensure_ascii=False)
    return valor


def escribir_filas(ruta: Path, filas: list[dict], etiquetas: list[str]) -> None:
    """CSV con COLUMNAS y una columna px_<letra> por letra.

    Se escribe a un temporal y se renombra al final: un paso cortado a la mitad
    no deja un CSV que el próximo barrido saltearía.
    """
    temporal = ruta.with_suffix(".tmp")
    with open(temporal, "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow([*COLUMNAS, *(PREFIJO_LETRA + e for e in etiquetas)])
        for fila in filas:
            escritor.writerow([_celda(fila[c]) for c in COLUMNAS] + fila["errores"])
    os.replace(temporal, ruta)


_TIPOS: dict[str, Callable[[str], Any]] = {
    "semilla": int,
    "llego": lambda texto: texto == "1",
    "epoca_llegada": int,
    "epocas": int,
    "peor_letra": int,
    "pixeles_mal": float,
    "mse": float,
    "segundos": float,
    "actualizaciones": int,
    "config": json.loads,
}


def leer_filas(ruta: Path) -> list[dict]:
    """Inversa de escribir_filas: tipos de Python, None en celdas vacías y errores como lista."""
    filas = []
    with open(ruta, encoding="utf-8", newline="") as f:
        for crudo in csv.DictReader(f):
            fila = {c: None if crudo[c] == "" else _TIPOS.get(c, str)(crudo[c]) for c in COLUMNAS}
            fila["errores"] = [int(v) for k, v in crudo.items() if k.startswith(PREFIJO_LETRA)]
            filas.append(fila)
    return filas


# --- Elección ---


def _media_desvio(valores: list[float]) -> tuple[float | None, float | None]:
    """Media y desvío muestral (ddof = 1); con un solo valor el desvío es 0, sin valores None."""
    if not valores:
        return None, None
    v = np.asarray(valores, dtype=float)
    desvio = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return float(v.mean()), desvio


def resumir(filas: list[dict]) -> dict[str, dict]:
    """Resumen de cada valor sobre sus semillas, en el orden en que aparecen las filas."""
    grupos: dict[str, list[dict]] = {}
    for fila in filas:
        grupos.setdefault(fila["valor"], []).append(fila)
    resumen = {}
    for valor, grupo in grupos.items():
        llegadas = [f["epoca_llegada"] for f in grupo if f["llego"]]
        epocas = _media_desvio(llegadas)
        peor = _media_desvio([f["peor_letra"] for f in grupo])
        pixeles = _media_desvio([f["pixeles_mal"] for f in grupo])
        resumen[valor] = {
            "corridas": len(grupo),
            "llegan": len(llegadas),
            "pct_llegan": 100.0 * len(llegadas) / len(grupo),
            "epocas_media": epocas[0],
            "epocas_desvio": epocas[1],
            "peor_letra_media": peor[0],
            "peor_letra_desvio": peor[1],
            "pixeles_mal_media": pixeles[0],
            "pixeles_mal_desvio": pixeles[1],
            "segundos_media": float(np.mean([f["segundos"] for f in grupo])),
            "actualizaciones_media": float(np.mean([f["actualizaciones"] for f in grupo])),
            "divergidas": sum(f["estado"] == "diverged" for f in grupo),
        }
    return resumen


def elegir(resumen: dict[str, dict], actual: str | None) -> tuple[str, str]:
    """Ganador de un paso y el criterio que lo decidió: "llegan", "epocas", "peor_letra" o "empate".

    1. Más % de semillas que llegan. 2. Si empatan: menos épocas medias hasta
    llegar (solo las semillas que llegaron). 3. Si ninguno llega: menor peor
    letra media al final. 4. Si siguen empatados: el valor actual (o el primero
    de los empatados, si el actual no está entre ellos).
    """
    mejor = max(r["pct_llegan"] for r in resumen.values())
    candidatos = [v for v, r in resumen.items() if r["pct_llegan"] == mejor]
    if len(candidatos) == 1:
        return candidatos[0], "llegan"
    clave, criterio = (
        ("epocas_media", "epocas") if mejor > 0 else ("peor_letra_media", "peor_letra")
    )
    minimo = min(resumen[v][clave] for v in candidatos)
    candidatos = [v for v in candidatos if resumen[v][clave] == minimo]
    if len(candidatos) == 1:
        return candidatos[0], criterio
    return (actual if actual in candidatos else candidatos[0]), "empate"


def _motivo(r: dict, criterio: str) -> str:
    """Por qué ganó, en una línea, a partir del resumen r del ganador."""
    llegan = f"llegan {r['llegan']}/{r['corridas']} semillas"
    if criterio == "llegan":
        return f"{llegan}, más que cualquier otro valor"
    if criterio == "epocas":
        return (
            f"empata en semillas que llegan ({llegan}) y llega en menos épocas: "
            f"{r['epocas_media']:.0f} ± {r['epocas_desvio']:.0f}"
        )
    if criterio == "peor_letra":
        return (
            f"ningún valor llega; menor peor letra media: "
            f"{r['peor_letra_media']:.2f} ± {r['peor_letra_desvio']:.2f} píxeles"
        )
    return f"empate en todos los criterios ({llegan}): se queda el valor que ya estaba"


def _valor_actual(valores: list[dict], config: dict) -> str | None:
    """Etiqueta del valor que deja la config como está, si hay uno."""
    return next((v["etiqueta"] for v in valores if aplicar(config, v["cambios"]) == config), None)


def _verificar_lineal(filas: list[dict]) -> dict:
    mse_pca = next(f["mse"] for f in filas if f["valor"] == "pca")
    mse_lineal = [f["mse"] for f in filas if f["valor"] == ETIQUETA_LINEAL]
    desvio = max(abs(m / mse_pca - 1.0) for m in mse_lineal)
    return {
        "mse_pca": mse_pca,
        "mse_ae_lineal": mse_lineal,
        "desvio_relativo_max": desvio,
        "tolerancia": TOLERANCIA_LINEAL,
        "dentro_de_tolerancia": desvio <= TOLERANCIA_LINEAL,
    }


def _id_archivo(paso_id: str) -> str:
    """ "1" → "01", "6+7" → "06_07": así los CSV se ordenan por paso."""
    return "_".join(f"{int(p):02d}" for p in paso_id.split("+"))


# --- Barrido ---


def correr_barrido(
    config_barrido: dict, salida: Path, escribir: Callable[[str], None] = print
) -> dict:
    """Corre (o retoma) el barrido completo y devuelve lo que escribe en elecciones.json."""
    X, etiquetas = cargar_font()
    salida.mkdir(parents=True, exist_ok=True)
    semillas = config_barrido["semillas"]
    objetivo = config_barrido["objetivo_peor_letra"]
    base = config_barrido["base"]

    def correr_paso(
        nombre_csv: str, paso_id: str, valores: list[dict], config: dict, con_pca: bool = False
    ) -> list[dict]:
        ruta = salida / nombre_csv
        if ruta.exists():
            escribir(f"[paso {paso_id}] ya existe {nombre_csv}: se lee en vez de correr.")
            filas = leer_filas(ruta)
            esperadas = {v["etiqueta"]: aplicar(config, v["cambios"]) for v in valores}
            for fila in filas:
                if fila["valor"] in esperadas and fila["config"] != esperadas[fila["valor"]]:
                    raise RuntimeError(
                        f"{nombre_csv} se corrió con otra config ({fila['valor']}): "
                        "borralo para volver a correr ese paso."
                    )
            return filas

        filas = []
        if con_pca:
            _, res = _correr_pca(X, base["latente"], objetivo)
            config_pca = {"modelo": "pca", "componentes": base["latente"]}
            filas.append(_fila(paso_id, "pca", config_pca, None, res))
            escribir(f"[paso {paso_id}] pca: peor letra {res['peor_letra']}")
        for valor in valores:
            config_valor = aplicar(config, valor["cambios"])
            for semilla in semillas:
                _, _, res = entrenar(config_valor, X, semilla, objetivo)
                filas.append(_fila(paso_id, valor["etiqueta"], config_valor, semilla, res))
                escribir(
                    f"[paso {paso_id}] {valor['etiqueta']} semilla {semilla}: {_describir(res)}"
                )
        escribir_filas(ruta, filas, etiquetas)
        return leer_filas(ruta)

    # Paso 0: líneas de base. No eligen nada, pero el AE lineal valida el motor.
    filas = correr_paso(
        "paso_00_lineas_de_base.csv", "0", config_barrido["lineas_de_base"], base, con_pca=True
    )
    verificacion = _verificar_lineal(filas)
    escribir(
        f"[paso 0] MSE PCA {verificacion['mse_pca']:.4f}; AE lineal se aparta hasta "
        f"{100 * verificacion['desvio_relativo_max']:.2f} %"
    )
    if not verificacion["dentro_de_tolerancia"]:
        raise RuntimeError(f"El AE lineal no da el MSE de PCA: hay un bug. {verificacion}")
    elecciones: dict[str, Any] = {
        "config_barrido": config_barrido,
        "lineas_de_base": {"resumen": resumir(filas), "verificacion_lineal": verificacion},
        "pasos": [],
    }

    config = copy.deepcopy(base)
    for paso in config_barrido["pasos"]:
        nombre_csv = f"paso_{_id_archivo(paso['id'])}_{paso['nombre']}.csv"
        resumen = resumir(correr_paso(nombre_csv, paso["id"], paso["valores"], config))
        actual = _valor_actual(paso["valores"], config)
        ganador, criterio = elegir(resumen, actual)
        motivo = _motivo(resumen[ganador], criterio)
        cambios = next(v["cambios"] for v in paso["valores"] if v["etiqueta"] == ganador)
        config = aplicar(config, cambios)
        elecciones["pasos"].append(
            {
                "id": paso["id"],
                "nombre": paso["nombre"],
                "valor_anterior": actual,
                "ganador": ganador,
                "criterio": criterio,
                "motivo": motivo,
                "resumen": resumen,
                "config": config,
            }
        )
        escribir(f"[paso {paso['id']}] gana {ganador}: {motivo}")
    elecciones["config_final"] = config

    # Final: repetir un paso con la config final. Si gana otro valor no se vuelve
    # a barrer: queda anotado como hallazgo.
    repetido = next(
        p for p in config_barrido["pasos"] if p["id"] == config_barrido["repetir_al_final"]
    )
    nombre_csv = f"paso_final_{repetido['nombre']}.csv"
    resumen = resumir(correr_paso(nombre_csv, "final", repetido["valores"], config))
    actual = _valor_actual(repetido["valores"], config)
    ganador, criterio = elegir(resumen, actual)
    elecciones["verificacion_final"] = {
        "paso": repetido["id"],
        "nombre": repetido["nombre"],
        "valor_final": actual,
        "ganador": ganador,
        "criterio": criterio,
        "motivo": _motivo(resumen[ganador], criterio),
        "hallazgo": ganador != actual,
        "resumen": resumen,
    }
    escribir(f"[final] repetido el paso {repetido['id']}: gana {ganador} (config final: {actual})")

    _guardar_final(config, X, etiquetas, semillas, objetivo, salida / "final", escribir)
    texto = json.dumps(elecciones, indent=2, ensure_ascii=False)
    (salida / "elecciones.json").write_text(texto, encoding="utf-8")
    return json.loads(texto)


def _guardar_final(
    config: dict,
    X: np.ndarray,
    etiquetas: list[str],
    semillas: list[int],
    objetivo: float,
    carpeta: Path,
    escribir: Callable[[str], None],
) -> None:
    """Config final con cada semilla: pesos, historia por época y fila; y la proyección PCA.

    corridas.csv se escribe último: si existe, todo lo demás ya está.
    """
    corridas = carpeta / "corridas.csv"
    if corridas.exists():
        guardada = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        if guardada != config:
            raise RuntimeError(f"{carpeta} es de otra config final: borrala para regenerarla.")
        escribir("[final] ya existe final/corridas.csv: no se vuelve a entrenar.")
        return

    carpeta.mkdir(exist_ok=True)
    texto = json.dumps(config, indent=2, ensure_ascii=False)
    (carpeta / "config.json").write_text(texto, encoding="utf-8")
    filas = []
    for semilla in semillas:
        ae, historia, res = entrenar(config, X, semilla, objetivo)
        pesos = {}
        for i, capa in enumerate(ae.layers):
            pesos[f"W{i}"], pesos[f"b{i}"] = capa.W, capa.b
        np.savez(carpeta / f"pesos_semilla_{semilla}.npz", **pesos)
        historia.to_csv(carpeta / f"historia_semilla_{semilla}.csv")
        filas.append(_fila("final", "final", config, semilla, res))
        escribir(f"[final] semilla {semilla}: {_describir(res)}")
    pca = PCA(config["latente"]).fit(X)
    np.savez(carpeta / "pca.npz", mean=pca.mean, components=pca.components, Z=pca.transform(X))
    escribir_filas(corridas, filas, etiquetas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Barrido secuencial del ejercicio 1a.")
    parser.add_argument("--config", type=Path, default=RUTA_CONFIG)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    correr_barrido(config, args.salida, escribir=lambda texto: print(texto, flush=True))


if __name__ == "__main__":
    main()
