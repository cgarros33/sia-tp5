"""Barrido compacto y entrenamiento final del ejercicio 2.

    python3 -m ej2.vae_exp --dataset nba --fase barrido --trabajadores 5
    python3 -m ej2.vae_exp --dataset nba --fase final --trabajadores 5
    python3 -m ej2.vae_exp --dataset nba --fase smoke

El barrido es secuencial por pasos (latente, capacidad, beta, lr, pérdida) y
cada paso se puede retomar: si el CSV existe y la config no cambió, solo se
corren las combinaciones que faltan. Todos los pasos descartan las corridas
colapsadas (menos de k/2 unidades activas o KL <= 0.01). Si el ganador del
barrido sigue colapsado, se sube por la escalera de NIVELES (más warm-up, free
bits y capacidad) hasta que la validación muestre un latente activo.

La fase final entrena el ganador y los modelos de latente 4 y 16 (VAE y AE
determinístico) y guarda pesos, historias, métricas de test y muestras para las
figuras.
"""

import argparse
import copy
import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from core.callbacks import Callback, EarlyStopping
from core.losses import MSE, BinaryCrossEntropy, get_loss
from core.optimizers import Adam
from core.vae import VariationalAutoencoder
from ej2 import datos
from ej2.modelos import (
    construir_ae_imagen,
    construir_ae_tabular,
    construir_vae_imagen,
    construir_vae_tabular,
)

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CONFIG = RAIZ / "ej2" / "config.json"
RUTA_RESULTADOS = RAIZ / "resultados" / "ej2"
MUESTRAS = 32

# Escalera de reintentos cuando el barrido sigue colapsado. El nivel 0 es el
# plan base; los siguientes alargan el warm-up, agregan free bits y capacidad.
NIVELES = (
    {
        "beta": [0.01, 0.1, 1.0, 5.0],
        "calentamiento_frac": 0.2,
        "free_bits": 0.0,
        "capacidades": [[128, 64], [256, 128]],
        "max_epocas": 1000,
        "paciencia": 200,
    },
    {
        "beta": [0.01, 0.1, 1.0],
        "calentamiento_frac": 0.4,
        "free_bits": 0.05,
        "capacidades": [[128, 64], [256, 128]],
        "max_epocas": 1200,
        "paciencia": 250,
    },
    {
        "beta": [0.001, 0.01, 0.1],
        "calentamiento_frac": 0.5,
        "free_bits": 0.1,
        "capacidades": [[256, 128], [512, 256]],
        "max_epocas": 1500,
        "paciencia": 300,
    },
    {
        "beta": [0.0001, 0.001, 0.01],
        "calentamiento_frac": 0.5,
        "free_bits": 0.2,
        "capacidades": [[256, 128], [512, 256]],
        "max_epocas": 1500,
        "paciencia": 300,
    },
)

CAMPOS_BARRIDO = [
    "paso",
    "valor",
    "semilla",
    "val_elbo",
    "val_recon",
    "kl_medio",
    "unidades_activas",
    "epocas",
    "segundos",
    "valido",
]


class CalentamientoKL(Callback):
    """Sube kl_weight de 0 a beta en las primeras `epocas` épocas.

    El decoder aprende a reconstruir como autoencoder determinístico antes de
    que el KL lo apague; después el término se introduce linealmente y poda las
    dimensiones que no aportan (Bowman et al., 2016).
    """

    def __init__(self, beta: float, epocas: int) -> None:
        self.beta = float(beta)
        self.epocas = max(1, int(epocas))

    def on_train_begin(self, network) -> None:
        network.kl_weight = 0.0

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network) -> bool:
        network.kl_weight = self.beta * min(1.0, epoch / self.epocas)
        return False


def _leer_config() -> dict:
    with open(RUTA_CONFIG, encoding="utf-8") as archivo:
        return json.load(archivo)


def _etiqueta(valor) -> str:
    if isinstance(valor, (list, tuple)):
        return "-".join(str(v) for v in valor)
    if isinstance(valor, float):
        return repr(valor)
    return str(valor)


def _mapa_valores(valores: list) -> dict[str, object]:
    return {_etiqueta(valor): valor for valor in valores}


def _cargar(nombre: str, tipo: str):
    if tipo == "imagen":
        conjunto = datos.cargar_imagenes(nombre)
        return conjunto["X"], conjunto["y"], conjunto["split"], {}
    conjunto = datos.cargar_banderas()
    X, escalador = datos.escalar_banderas(conjunto["X"], conjunto["split"])
    return (
        X,
        conjunto["nombres"],
        conjunto["split"],
        {
            "raw": conjunto["X"],
            "escalador": escalador,
        },
    )


def _construir(
    tipo: str,
    modelo: str,
    d: int,
    latente: int,
    capacidad,
    salida: str,
    semilla: int,
    beta: float = 1.0,
    free_bits: float = 0.0,
):
    if modelo == "ae":
        if tipo == "imagen":
            return construir_ae_imagen(d, latente, capacidad, salida=salida, semilla=semilla)
        return construir_ae_tabular(latente, capacidad, semilla=semilla)
    if tipo == "imagen":
        return construir_vae_imagen(
            d,
            latente,
            capacidad,
            salida=salida,
            semilla=semilla,
            beta=beta,
            free_bits=free_bits,
        )
    return construir_vae_tabular(
        latente, capacidad, semilla=semilla, beta=beta, free_bits=free_bits
    )


def _kl_por_dimension(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    return 0.5 * (np.exp(log_var) + mu**2 - 1.0 - log_var)


def _kl_medio(mu: np.ndarray, log_var: np.ndarray) -> float:
    return float(_kl_por_dimension(mu, log_var).sum(axis=1).mean())


def _unidades_activas(mu: np.ndarray, log_var: np.ndarray) -> int:
    return int(np.sum(_kl_por_dimension(mu, log_var).mean(axis=0) > 0.01))


def _nn_dist(generadas: np.ndarray, X_train: np.ndarray) -> float:
    """MSE media de cada muestra generada a su vecina más cercana de train."""
    distancias = np.empty(len(generadas))
    for i, muestra in enumerate(generadas):
        distancias[i] = np.mean((X_train - muestra) ** 2, axis=1).min()
    return float(distancias.mean())


def _desescalar_original(recon: np.ndarray, escalador: dict) -> np.ndarray:
    original = datos.desescalar_banderas(recon, escalador)
    original[:, :9] = np.round(np.clip(original[:, :9], 0.0, 1.0) * 255.0) / 255.0
    original[:, 9] = (original[:, 9] >= 0.5).astype(float)
    minimo = np.asarray(escalador["minimo"], dtype=np.float64)
    rango = np.asarray(escalador["rango"], dtype=np.float64)
    for columna in datos.COLUMNAS_ESCALADAS:
        original[:, columna] = np.clip(
            original[:, columna], minimo[columna], minimo[columna] + rango[columna]
        )
    return original


def _guardar_pesos(ruta: Path, modelo) -> None:
    np.savez_compressed(ruta, **{f"p{i}": p for i, p in enumerate(modelo.params)})


def cargar_pesos(modelo, ruta: Path) -> None:
    """Copia los pesos guardados por _guardar_pesos dentro de un modelo ya construido."""
    with np.load(ruta, allow_pickle=False) as archivo:
        for i, parametro in enumerate(modelo.params):
            parametro[...] = archivo[f"p{i}"]


def _metricas_test(modelo, X_test, X_train, extra: dict, binarizar: bool) -> dict:
    reconstruccion = modelo.decode(modelo.encode(X_test))
    fila = {}
    if isinstance(modelo, VariationalAutoencoder):
        mu, log_var = modelo.latent_distribution(X_test)
        fila["kl_medio"] = _kl_medio(mu, log_var)
        fila["unidades_activas"] = _unidades_activas(mu, log_var)
        z = modelo.noise_rng.standard_normal((MUESTRAS, modelo.latent_dim))
        generadas = modelo.decode(z)
    else:
        z = np.random.default_rng(1234).standard_normal((MUESTRAS, modelo.latent_dim))
        generadas = modelo.decode(z)
    fila["nn_dist"] = _nn_dist(generadas, X_train)

    if extra:
        fila["recon_mse"] = MSE().value(X_test, reconstruccion)
        fila["recon_mae"] = float(np.mean(np.abs(X_test - reconstruccion)))
        original = _desescalar_original(reconstruccion, extra["escalador"])
        fila["recon_mse_original"] = MSE().value(extra["raw_test"], original)
        fila["recon_mae_original"] = float(np.mean(np.abs(extra["raw_test"] - original)))
        fila["_generadas_original"] = _desescalar_original(generadas, extra["escalador"])
    else:
        recorte = np.clip(reconstruccion, 0.0, 1.0)
        fila["recon_bce"] = BinaryCrossEntropy().value(X_test, recorte)
        fila["recon_mse"] = MSE().value(X_test, recorte)
        if binarizar:
            fila["pixeles_mal"] = float(np.mean((X_test >= 0.5) != (recorte >= 0.5)))
        else:
            fila["pixeles_mal"] = float(np.mean(np.abs(X_test - recorte) > 0.5))
    fila["_z"] = z
    fila["_generadas"] = generadas
    fila["_reconstruccion"] = reconstruccion
    return fila


def _entrenar(tarea: dict) -> dict:
    nombre = tarea["dataset"]
    ds = tarea["ds"]
    cfg = tarea["cfg"]
    semilla = int(tarea["semilla"])
    X, y, split, extra = _cargar(nombre, ds["tipo"])
    if extra:
        extra["raw_test"] = extra["raw"][split == "test"]
    X_train = X[split == "train"]
    X_val = X[split == "val"]
    X_test = X[split == "test"]
    y_test = y[split == "test"]
    salida = "sigmoid" if cfg["perdida"] == "binary_crossentropy" else "identity"
    beta = float(cfg["beta"])
    free_bits = float(cfg.get("free_bits", 0.0))
    epocas = int(tarea["epocas"])
    calentamiento = max(50, int(round(float(cfg.get("calentamiento_frac", 0.2)) * epocas)))
    modelo = _construir(
        ds["tipo"],
        tarea["modelo"],
        ds["d"],
        cfg["latente"],
        cfg["capacidad"],
        salida,
        semilla,
        beta=beta,
        free_bits=free_bits,
    )
    loss = get_loss(cfg["perdida"])

    def recon_epoca(y_true, _y_pred):
        """Reconstrucción determinística (sin ruido) para el early stopping."""
        reconstruccion = modelo.decode(modelo.encode(y_true))
        if ds["tipo"] == "imagen":
            reconstruccion = np.clip(reconstruccion, 0.0, 1.0)
        return MSE().value(y_true, reconstruccion)

    metricas = {"recon": recon_epoca}
    if isinstance(modelo, VariationalAutoencoder):

        def kl_epoca(_y_true, _y_pred):
            return _kl_medio(modelo._mu, modelo._log_var)

        def activas_epoca(_y_true, _y_pred):
            return float(_unidades_activas(modelo._mu, modelo._log_var))

        metricas |= {"kl": kl_epoca, "activas": activas_epoca}
    callbacks = []
    if isinstance(modelo, VariationalAutoencoder):
        callbacks.append(CalentamientoKL(beta, calentamiento))
    callbacks.append(
        EarlyStopping(
            patience=int(ds["paciencia"]),
            monitor="val_recon",
            mode="min",
            restore_best=True,
        )
    )
    inicio = time.perf_counter()
    historia = modelo.fit(
        X_train,
        optimizer=Adam(lr=cfg["lr"]),
        loss=loss,
        epochs=epocas,
        batch_size=int(ds["batch_size"]),
        validation=(X_val, X_val),
        callbacks=callbacks,
        metrics=metricas,
    )
    segundos = time.perf_counter() - inicio
    if isinstance(modelo, VariationalAutoencoder):
        # Si el early stopping cortó durante el warm-up, el objetivo guardado
        # queda igual al final del entrenamiento.
        modelo.kl_weight = beta
    val_elbo = modelo.evaluate(X_val, X_val, loss)["loss"]
    val_recon = recon_epoca(X_val, None)
    fila = {
        "paso": tarea.get("paso", ""),
        "valor": tarea.get("valor", ""),
        "semilla": semilla,
        "val_elbo": float(val_elbo),
        "val_recon": float(val_recon),
        "epocas": len(historia),
        "segundos": segundos,
        "valido": int(np.isfinite(val_elbo) and historia.status == "ok"),
    }
    if isinstance(modelo, VariationalAutoencoder):
        mu, log_var = modelo.latent_distribution(X_val)
        fila["kl_medio"] = _kl_medio(mu, log_var)
        fila["unidades_activas"] = _unidades_activas(mu, log_var)
    else:
        fila["kl_medio"] = float("nan")
        fila["unidades_activas"] = 0

    if tarea.get("guardar"):
        carpeta = RUTA_RESULTADOS / nombre / tarea["carpeta"]
        carpeta.mkdir(parents=True, exist_ok=True)
        _guardar_pesos(carpeta / f"pesos_semilla_{semilla}.npz", modelo)
        historia.to_csv(carpeta / f"historia_semilla_{semilla}.csv")
        metricas_finales = _metricas_test(
            modelo, X_test, X_train, extra, binarizar=nombre == "chino"
        )
        z = metricas_finales.pop("_z")
        generadas = metricas_finales.pop("_generadas")
        reconstruccion = metricas_finales.pop("_reconstruccion")
        generadas_original = metricas_finales.pop("_generadas_original", None)
        np.savez_compressed(
            carpeta / f"muestras_semilla_{semilla}.npz",
            z=z,
            generadas=generadas,
            reconstrucciones=reconstruccion,
            X_test=X_test,
            y_test=y_test,
            mu_test=modelo.encode(X_test),
            **({} if generadas_original is None else {"generadas_original": generadas_original}),
        )
        fila.update(metricas_finales)
    return fila


def _escribir_csv(ruta: Path, filas: list[dict], campos: list[str]) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos, extrasaction="ignore")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({campo: fila.get(campo, "") for campo in campos})


def _leer_csv(ruta: Path) -> list[dict]:
    with open(ruta, newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def _fingerprint(dataset: str, paso: str, cfg: dict, ds: dict) -> str:
    carga = {
        "dataset": dataset,
        "paso": paso,
        "cfg": cfg,
        "semillas_barrido": ds["semillas_barrido"],
        "epocas_barrido": ds["epocas_barrido"],
        "batch_size": ds["batch_size"],
    }
    return hashlib.sha256(json.dumps(carga, sort_keys=True, default=str).encode()).hexdigest()


def _agrupar(filas: list[dict], etiqueta: str, campo: str, minimo: float | None = None) -> dict:
    grupos: dict[str, list[float]] = {}
    for fila in filas:
        if not int(fila["valido"]):
            continue
        if minimo is not None and float(fila[campo]) < minimo:
            continue
        grupos.setdefault(fila[etiqueta], []).append(float(fila[campo]))
    resumen = {}
    for valor, mediciones in grupos.items():
        arreglo = np.asarray(mediciones)
        resumen[valor] = {
            "media": float(arreglo.mean()),
            "desvio": float(arreglo.std(ddof=1)) if len(arreglo) > 1 else 0.0,
            "n": len(arreglo),
        }
    return resumen


def _elegir(paso: dict, filas: list[dict], cfg: dict, ds: dict) -> tuple[str, dict]:
    valores = paso["valores"]
    if paso["id"].startswith("5"):
        permitidas = set(ds["perdidas"])
        valores = [valor for valor in valores if valor in permitidas]
    etiquetas = [_etiqueta(valor) for valor in valores]
    elegibles = [f for f in filas if f["valor"] in etiquetas]
    elbos = _agrupar(elegibles, "valor", "val_elbo")
    recons = _agrupar(elegibles, "valor", "val_recon")
    activas = _agrupar(elegibles, "valor", "unidades_activas")
    kls = _agrupar(elegibles, "valor", "kl_medio")
    if not elbos:
        raise RuntimeError(f"No hay corridas válidas para el paso {paso['id']}")

    def _media(grupo: dict, etiqueta: str) -> float:
        return grupo.get(etiqueta, {}).get("media", float("inf"))

    def _latente(etiqueta: str) -> int:
        return int(float(etiqueta)) if paso["id"].startswith("1") else int(cfg["latente"])

    def _activa(etiqueta: str) -> bool:
        k = _latente(etiqueta)
        return (
            etiqueta in elbos
            and _media(activas, etiqueta) >= 0.5 * k
            and _media(kls, etiqueta) > 0.01
        )

    con_latente = [etiqueta for etiqueta in etiquetas if _activa(etiqueta)]
    colapsado = not con_latente
    if colapsado:
        fracciones = {
            etiqueta: _media(activas, etiqueta) / max(1, _latente(etiqueta)) for etiqueta in elbos
        }
        mejor_fraccion = max(fracciones.values())
        candidatas = [etiqueta for etiqueta in elbos if fracciones[etiqueta] == mejor_fraccion]
        aviso = (
            "ningún valor cumplió activas/KL; se usó el de mayor fracción de unidades "
            f"activas ({mejor_fraccion:.2f})"
        )
    else:
        candidatas = con_latente
        aviso = ""

    detalle = {
        "colapsado": colapsado,
        "recon": {e: recons.get(e) for e in etiquetas},
        "elbo": elbos,
        "activas": {e: activas.get(e) for e in etiquetas},
        "kl": {e: kls.get(e) for e in etiquetas},
    }

    if paso["id"].startswith("3"):
        ganador = min(candidatas, key=lambda e: (recons[e]["media"], e))
        motivo = (
            "menor reconstrucción entre los betas con latente activo y KL > 0.01"
            if not colapsado
            else f"{aviso} y, entre esos, la menor reconstrucción"
        )
        return ganador, {"motivo": motivo, **detalle}

    mejor = min(candidatas, key=lambda e: elbos[e]["media"])
    limite = elbos[mejor]["media"] + elbos[mejor]["desvio"]
    finalistas = [e for e in candidatas if elbos[e]["media"] <= limite]
    if paso["id"].startswith("1"):
        ganador = min(finalistas, key=float)
        motivo = "menor ELBO de validación; a menos de un desvío se prefiere el latente más chico"
    elif paso["id"].startswith("2"):
        orden = {etiqueta: i for i, etiqueta in enumerate(etiquetas)}
        ganador = min(finalistas, key=lambda e: orden[e])
        motivo = "menor ELBO de validación; a menos de un desvío se prefiere la capacidad más chica"
    elif paso["id"].startswith("4"):
        ganador = min(finalistas, key=float)
        motivo = "menor ELBO de validación; a menos de un desvío se prefiere el lr más chico"
    else:
        mejor_recon = min(finalistas, key=lambda e: recons[e]["media"])
        limite_recon = recons[mejor_recon]["media"] + recons[mejor_recon]["desvio"]
        finalistas_recon = [e for e in finalistas if recons[e]["media"] <= limite_recon]
        ganador = (
            "binary_crossentropy"
            if "binary_crossentropy" in finalistas_recon
            else min(finalistas_recon, key=lambda e: recons[e]["media"])
        )
        motivo = "menor error de reconstrucción determinístico; empate prefiere BCE"
    if colapsado:
        motivo = f"{aviso}; {motivo}"
    return ganador, {"motivo": motivo, **detalle}


def _aplicar_nivel(cfg_general: dict, ds: dict, nivel: int) -> tuple[dict, dict]:
    """Copia la config y aplica los overrides del nivel anticolapso."""
    if not 0 <= nivel < len(NIVELES):
        raise ValueError(f"Nivel inválido: {nivel}. Opciones: 0..{len(NIVELES) - 1}")
    overrides = NIVELES[nivel]
    cfg = copy.deepcopy(cfg_general)
    ds = copy.deepcopy(ds)
    cfg["calentamiento_frac"] = overrides["calentamiento_frac"]
    cfg["free_bits"] = overrides["free_bits"]
    for paso in cfg["pasos"]:
        if paso["id"].startswith("2"):
            paso["valores"] = [list(c) for c in overrides["capacidades"]]
        elif paso["id"].startswith("3"):
            paso["valores"] = list(overrides["beta"])
    ds["max_epocas"] = overrides["max_epocas"]
    ds["paciencia"] = overrides["paciencia"]
    return cfg, ds


def _evaluar_nivel(nombre: str, ds: dict, cfg_final: dict, filas: list[dict]) -> tuple[bool, dict]:
    """Acepta el nivel si el ganador usa el latente y mejora a la imagen media."""
    if not filas:
        return False, {"motivo": "no hay filas del ganador"}
    X, _y, split, _extra = _cargar(nombre, ds["tipo"])
    X_train = X[split == "train"]
    X_val = X[split == "val"]
    media = np.broadcast_to(np.mean(X_train, axis=0), X_val.shape)
    baseline = float(MSE().value(X_val, media))
    activas = float(np.mean([float(f["unidades_activas"]) for f in filas]))
    kl = float(np.mean([float(f["kl_medio"]) for f in filas]))
    recon = float(np.mean([float(f["val_recon"]) for f in filas]))
    k = int(cfg_final["latente"])
    cumple = activas >= 0.5 * k and kl > 0.01 and recon < baseline
    return cumple, {
        "unidades_activas": activas,
        "kl_medio": kl,
        "val_recon": recon,
        "baseline": baseline,
        "latente": k,
    }


def correr_barrido(nombre: str, trabajadores: int, forzar: bool, nivel: int = 0) -> dict:
    base_cfg = _leer_config()
    cfg_general, ds = _aplicar_nivel(base_cfg, base_cfg["datasets"][nombre], nivel)
    actual = {
        "latente": cfg_general["base"]["latente"],
        "capacidad": cfg_general["base"]["capacidad"],
        "beta": cfg_general["base"]["beta"],
        "lr": cfg_general["base"]["lr"],
        "perdida": cfg_general["base"]["perdida"],
        "free_bits": cfg_general["free_bits"],
        "calentamiento_frac": cfg_general["calentamiento_frac"],
        "nivel_anticolapso": nivel,
    }
    carpeta = (
        RUTA_RESULTADOS / nombre / "barrido"
        if nivel == 0
        else RUTA_RESULTADOS / nombre / f"barrido_nivel_{nivel}"
    )
    carpeta.mkdir(parents=True, exist_ok=True)
    elecciones = []
    semillas = list(range(int(ds["semillas_barrido"])))
    filas_ganadoras: list[dict] = []

    for paso in cfg_general["pasos"]:
        valores = paso["valores"]
        if paso["id"].startswith("5"):
            valores = [v for v in valores if v in ds["perdidas"]]
            if len(valores) == 1:
                actual["perdida"] = valores[0]
                elecciones.append(
                    {
                        "paso": paso["id"],
                        "saltado": True,
                        "ganador": _etiqueta(valores[0]),
                        "motivo": "el dataset tiene una sola pérdida posible",
                    }
                )
                continue
        csv_paso = carpeta / f"paso_{paso['id']}.csv"
        meta_paso = carpeta / f"paso_{paso['id']}.meta.json"
        huella = _fingerprint(nombre, paso["id"], actual, ds)
        if forzar:
            csv_paso.unlink(missing_ok=True)
            meta_paso.unlink(missing_ok=True)
        filas = []
        if csv_paso.exists():
            if not meta_paso.exists():
                raise RuntimeError(f"Falta {meta_paso.name}; borrá {csv_paso.name} y reintentá.")
            meta = json.loads(meta_paso.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != huella:
                raise RuntimeError(
                    f"{csv_paso.name} se corrió con otra config; borralo y volvé a correr."
                )
            filas = _leer_csv(csv_paso)
            print(f"{nombre} {paso['id']}: se retoma con {len(filas)} corridas.")
        etiquetas = [_etiqueta(v) for v in valores]
        hechas = {(f["valor"], int(f["semilla"])) for f in filas}
        tareas = []
        for etiqueta, valor in zip(etiquetas, valores, strict=True):
            for semilla in semillas:
                if (etiqueta, semilla) in hechas:
                    continue
                tareas.append(
                    {
                        "dataset": nombre,
                        "ds": ds,
                        "cfg": {**actual, paso["nombre"]: valor},
                        "modelo": "vae",
                        "semilla": semilla,
                        "epocas": int(ds["epocas_barrido"]),
                        "paso": paso["id"],
                        "valor": etiqueta,
                    }
                )
        nuevas = _correr_tareas(tareas, trabajadores)
        filas.extend(nuevas)
        orden = {etiqueta: i for i, etiqueta in enumerate(etiquetas)}
        filas.sort(key=lambda f: (orden.get(f["valor"], 99), int(f["semilla"])))
        _escribir_csv(csv_paso, filas, CAMPOS_BARRIDO)
        meta_paso.write_text(
            json.dumps({"fingerprint": huella, "corridas": len(filas)}, indent=2),
            encoding="utf-8",
        )
        ganador, detalle = _elegir(paso, filas, actual, ds)
        actual[paso["nombre"]] = _mapa_valores(valores)[ganador]
        filas_ganadoras = [f for f in filas if f["valor"] == ganador]
        elecciones.append(
            {
                "paso": paso["id"],
                "ganador": ganador,
                "nivel_anticolapso": nivel,
                **detalle,
            }
        )
        print(f"{nombre} {paso['id']}: gana {ganador} (nivel {nivel})")
        _escribir_elecciones(nombre, cfg_general, actual, elecciones, nivel=nivel)

    aceptado, validacion = _evaluar_nivel(nombre, ds, actual, filas_ganadoras)
    _escribir_elecciones(
        nombre,
        cfg_general,
        actual,
        elecciones,
        nivel=nivel,
        aceptado=aceptado,
        validacion=validacion,
    )
    print(
        f"{nombre}: nivel {nivel} {'aceptado' if aceptado else 'sigue colapsado'} "
        f"(activas={validacion.get('unidades_activas', 0):.2f}, "
        f"KL={validacion.get('kl_medio', 0):.4f}, "
        f"recon={validacion.get('val_recon', 0):.4f} vs base "
        f"{validacion.get('baseline', 0):.4f})"
    )
    return {
        "config_final": actual,
        "pasos": elecciones,
        "aceptado": aceptado,
        "validacion": validacion,
        "nivel_anticolapso": nivel,
    }


def correr_barrido_con_reintentos(nombre: str, trabajadores: int, forzar: bool) -> dict:
    """Corre niveles 0..3 hasta que el ganador tenga un latente activo."""
    ruta = RUTA_RESULTADOS / nombre / "elecciones.json"
    if not forzar and ruta.exists():
        previo = json.loads(ruta.read_text(encoding="utf-8"))
        if previo.get("aceptado") is True:
            nivel_previo = previo.get("nivel_anticolapso")
            print(f"{nombre}: el barrido ya está aceptado (nivel {nivel_previo}).")
            return previo
    resultado = {}
    for nivel in range(len(NIVELES)):
        resultado = correr_barrido(nombre, trabajadores, forzar if nivel == 0 else False, nivel)
        if resultado["aceptado"]:
            return resultado
        if nivel + 1 < len(NIVELES):
            print(f"{nombre}: se sube al nivel anticolapso {nivel + 1}.")
    return resultado


def _escribir_elecciones(
    nombre: str,
    cfg_general: dict,
    actual: dict,
    elecciones: list,
    nivel: int = 0,
    aceptado: bool | None = None,
    validacion: dict | None = None,
) -> None:
    carpeta = RUTA_RESULTADOS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    contenido = {
        "dataset": nombre,
        "semilla_split": cfg_general["semilla_split"],
        "config_final": actual,
        "nivel_anticolapso": nivel,
        "pasos": elecciones,
    }
    if aceptado is not None:
        contenido["aceptado"] = bool(aceptado)
        contenido["validacion"] = validacion
    (carpeta / "elecciones.json").write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _correr_tareas(tareas: list[dict], trabajadores: int) -> list[dict]:
    if not tareas:
        return []
    if trabajadores == 1:
        filas = []
        for i, tarea in enumerate(tareas, 1):
            fila = _entrenar(tarea)
            print(
                f"  [{i}/{len(tareas)}] {tarea['dataset']} {tarea.get('carpeta', '')} "
                f"semilla {tarea['semilla']}: val_elbo={fila['val_elbo']:.4f} "
                f"({fila['segundos']:.1f}s)",
                flush=True,
            )
            filas.append(fila)
        return filas
    contexto = mp.get_context("spawn")
    filas = []
    with contexto.Pool(trabajadores) as pool:
        for i, fila in enumerate(pool.imap_unordered(_entrenar, tareas), 1):
            filas.append(fila)
            print(
                f"  [{i}/{len(tareas)}] {fila.get('paso', '')} {fila.get('valor', '')} "
                f"semilla {fila['semilla']}: val_elbo={fila['val_elbo']:.4f} "
                f"({fila['segundos']:.1f}s)",
                flush=True,
            )
    return filas


def _config_carpeta(nombre: str, ds: dict, cfg: dict, latente: int, modelo: str, escalador):
    calentamiento = max(
        50, int(round(float(cfg.get("calentamiento_frac", 0.2)) * int(ds["max_epocas"])))
    )
    return {
        "dataset": nombre,
        "tipo": ds["tipo"],
        "d": ds["d"],
        "forma": ds["forma"],
        "modelo": modelo,
        "latente": latente,
        "capacidad": cfg["capacidad"],
        "beta": cfg["beta"],
        "free_bits": float(cfg.get("free_bits", 0.0)),
        "lr": cfg["lr"],
        "perdida": cfg["perdida"],
        "batch_size": ds["batch_size"],
        "epocas": ds["max_epocas"],
        "paciencia": ds["paciencia"],
        "calentamiento_epocas": calentamiento,
        "nivel_anticolapso": int(cfg.get("nivel_anticolapso", 0)),
        "semilla_split": _leer_config()["semilla_split"],
        "escalador": escalador,
    }


def correr_final(nombre: str, trabajadores: int, forzar: bool) -> None:
    ruta_elecciones = RUTA_RESULTADOS / nombre / "elecciones.json"
    if not ruta_elecciones.exists():
        raise RuntimeError(f"Falta {ruta_elecciones}; corré primero --fase barrido.")
    elecciones = json.loads(ruta_elecciones.read_text(encoding="utf-8"))
    cfg_final = elecciones["config_final"]
    nivel = int(cfg_final.get("nivel_anticolapso", elecciones.get("nivel_anticolapso", 0)))
    base = _leer_config()
    cfg_general, ds = _aplicar_nivel(base, base["datasets"][nombre], nivel)
    if elecciones.get("aceptado") is False:
        print(
            f"ADVERTENCIA {nombre}: el barrido no logró un latente activo; "
            "se entrena igual el mejor modelo encontrado."
        )
    escalador = None
    if ds["tipo"] == "tabular":
        conjunto = datos.cargar_banderas()
        _, escalador = datos.escalar_banderas(conjunto["X"], conjunto["split"])
        (RUTA_RESULTADOS / nombre).mkdir(parents=True, exist_ok=True)
        (RUTA_RESULTADOS / nombre / "escalador.json").write_text(
            json.dumps(escalador, indent=2), encoding="utf-8"
        )

    modelos = [
        ("final", "vae", int(cfg_final["latente"]), int(ds["semillas_final"])),
        ("latente_4", "vae", 4, int(ds["semillas_latente"])),
        ("latente_16", "vae", 16, int(ds["semillas_latente"])),
        ("ae_latente_4", "ae", 4, int(ds["semillas_latente"])),
        ("ae_latente_16", "ae", 16, int(ds["semillas_latente"])),
    ]
    for carpeta, modelo, latente, semillas in modelos:
        destino = RUTA_RESULTADOS / nombre / carpeta
        destino.mkdir(parents=True, exist_ok=True)
        config = _config_carpeta(nombre, ds, cfg_final, latente, modelo, escalador)
        (destino / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if (destino / "test.csv").exists() and not forzar:
            print(f"{nombre}/{carpeta}: ya existe test.csv, se saltea.")
            continue
        # Si el ganador ya tiene el latente pedido, se reutiliza la corrida final.
        origen = RUTA_RESULTADOS / nombre / "final"
        if (
            not forzar
            and modelo == "vae"
            and carpeta != "final"
            and latente == int(cfg_final["latente"])
            and (origen / "test.csv").exists()
        ):
            for archivo in sorted(origen.iterdir()):
                if archivo.is_file():
                    shutil.copy2(archivo, destino / archivo.name)
            print(f"{nombre}/{carpeta}: reutiliza la corrida final (latente {latente}).")
            continue
        tareas = [
            {
                "dataset": nombre,
                "ds": ds,
                "cfg": {**cfg_final, "latente": latente},
                "modelo": modelo,
                "semilla": semilla,
                "epocas": int(ds["max_epocas"]),
                "carpeta": carpeta,
                "guardar": True,
            }
            for semilla in range(semillas)
        ]
        filas = _correr_tareas(tareas, trabajadores)
        filas.sort(key=lambda f: int(f["semilla"]))
        excluir = {"paso", "valor", "segundos", "valido"}
        campos = sorted(
            {
                clave
                for fila in filas
                for clave in fila
                if not clave.startswith("_") and clave not in excluir
            }
        )
        orden = ["semilla", "recon_bce", "recon_mse", "pixeles_mal", "kl_medio"]
        campos = [c for c in orden if c in campos] + [c for c in campos if c not in orden]
        _escribir_csv(destino / "test.csv", filas, campos)
        metricas = {}
        for campo in campos:
            if campo == "semilla":
                continue
            valores = [float(f[campo]) for f in filas if f.get(campo) not in (None, "")]
            if valores:
                metricas[campo] = {
                    "media": float(np.mean(valores)),
                    "desvio": float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0,
                }
        (destino / "metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
        print(f"{nombre}/{carpeta}: {len(filas)} semillas.")


def correr_smoke(nombre: str) -> None:
    cfg_general = _leer_config()
    ds = cfg_general["datasets"][nombre]
    tarea = {
        "dataset": nombre,
        "ds": {**ds, "max_epocas": 2, "paciencia": 1},
        "cfg": {
            "latente": cfg_general["base"]["latente"],
            "capacidad": cfg_general["base"]["capacidad"],
            "beta": cfg_general["base"]["beta"],
            "free_bits": cfg_general["free_bits"],
            "calentamiento_frac": cfg_general["calentamiento_frac"],
            "lr": cfg_general["base"]["lr"],
            "perdida": ds["perdidas"][0],
        },
        "modelo": "vae",
        "semilla": 0,
        "epocas": 2,
    }
    fila = _entrenar(tarea)
    print(json.dumps(fila, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description="Barrido y modelos finales del ejercicio 2.")
    parser.add_argument("--dataset", required=True, choices=["nba", "f1", "chino", "banderas"])
    parser.add_argument("--fase", default="barrido", choices=["barrido", "final", "smoke"])
    parser.add_argument("--trabajadores", type=int, default=5)
    parser.add_argument("--hilos", type=int, default=4)
    parser.add_argument("--forzar", action="store_true")
    parser.add_argument(
        "--nivel",
        type=int,
        choices=range(len(NIVELES)),
        help="Fija un nivel anticolapso en vez de reintentar 0..3 automáticamente.",
    )
    args = parser.parse_args()
    if args.trabajadores < 1 or args.hilos < 1:
        parser.error("--trabajadores y --hilos tienen que ser >= 1")
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.hilos)
    os.environ["OMP_NUM_THREADS"] = str(args.hilos)
    os.environ["MKL_NUM_THREADS"] = str(args.hilos)
    if args.fase == "barrido":
        if args.nivel is None:
            correr_barrido_con_reintentos(args.dataset, args.trabajadores, args.forzar)
        else:
            correr_barrido(args.dataset, args.trabajadores, args.forzar, args.nivel)
    elif args.fase == "final":
        correr_final(args.dataset, args.trabajadores, args.forzar)
    else:
        correr_smoke(args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
