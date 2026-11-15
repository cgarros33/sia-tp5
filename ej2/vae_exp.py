"""Barrido compacto y entrenamiento final del ejercicio 2.

    python3 -m ej2.vae_exp --dataset nba --fase barrido --trabajadores 5
    python3 -m ej2.vae_exp --dataset nba --fase final --trabajadores 5
    python3 -m ej2.vae_exp --dataset nba --fase smoke

El barrido es secuencial por pasos (latente, capacidad, beta, lr, pérdida) y
cada paso se puede retomar: si el CSV existe y la config no cambió, solo se
corren las combinaciones que faltan. La fase final entrena el ganador y los
modelos de latente 2 y 16 (VAE y AE determinístico) y guarda pesos, historias,
métricas de test y muestras para las figuras.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from core.callbacks import EarlyStopping
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
):
    if modelo == "ae":
        if tipo == "imagen":
            return construir_ae_imagen(d, latente, capacidad, salida=salida, semilla=semilla)
        return construir_ae_tabular(latente, capacidad, semilla=semilla)
    if tipo == "imagen":
        return construir_vae_imagen(
            d, latente, capacidad, salida=salida, semilla=semilla, beta=beta
        )
    return construir_vae_tabular(latente, capacidad, semilla=semilla, beta=beta)


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
    modelo = _construir(
        ds["tipo"],
        tarea["modelo"],
        ds["d"],
        cfg["latente"],
        cfg["capacidad"],
        salida,
        semilla,
        beta=float(cfg["beta"]),
    )
    loss = get_loss(cfg["perdida"])
    metricas = None
    if isinstance(modelo, VariationalAutoencoder):

        def kl_epoca(_y_true, _y_pred):
            return _kl_medio(modelo._mu, modelo._log_var)

        def activas_epoca(_y_true, _y_pred):
            return float(_unidades_activas(modelo._mu, modelo._log_var))

        metricas = {"kl": kl_epoca, "activas": activas_epoca}
    callbacks = [
        EarlyStopping(
            patience=int(ds["paciencia"]),
            monitor="val_loss",
            mode="min",
            restore_best=True,
        )
    ]
    inicio = time.perf_counter()
    historia = modelo.fit(
        X_train,
        optimizer=Adam(lr=cfg["lr"]),
        loss=loss,
        epochs=int(tarea["epocas"]),
        batch_size=int(ds["batch_size"]),
        validation=(X_val, X_val),
        callbacks=callbacks,
        metrics=metricas,
    )
    segundos = time.perf_counter() - inicio
    val_elbo = modelo.evaluate(X_val, X_val, loss)["loss"]
    val_recon = MSE().value(X_val, modelo.decode(modelo.encode(X_val)))
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

    if paso["id"].startswith("3"):
        latente = int(cfg["latente"])
        candidatas = [
            etiqueta
            for etiqueta in etiquetas
            if etiqueta in elbos
            and _media(activas, etiqueta) >= 0.5 * latente
            and _media(kls, etiqueta) > 0.01
        ]
        if not candidatas:
            max_activas = max(_media(activas, etiqueta) for etiqueta in elbos)
            candidatas = [
                etiqueta for etiqueta in elbos if _media(activas, etiqueta) == max_activas
            ]
            motivo = (
                "ningún beta cumplió activas/KL; se eligió el de más unidades activas "
                f"({max_activas:g}) y, entre esos, la menor reconstrucción"
            )
        else:
            motivo = "menor reconstrucción entre los betas con latente activo y KL > 0.01"
        ganador = min(candidatas, key=lambda e: (recons[e]["media"], e))
        detalle = {
            "motivo": motivo,
            "recon": {e: recons.get(e) for e in etiquetas},
            "elbo": elbos,
            "activas": {e: activas.get(e) for e in etiquetas},
            "kl": {e: kls.get(e) for e in etiquetas},
        }
        return ganador, detalle

    mejor = min(elbos, key=lambda e: elbos[e]["media"])
    limite = elbos[mejor]["media"] + elbos[mejor]["desvio"]
    candidatas = [e for e in etiquetas if e in elbos and elbos[e]["media"] <= limite]
    if paso["id"].startswith("1"):
        ganador = min(candidatas, key=float)
        motivo = "menor ELBO de validación; a menos de un desvío se prefiere el latente más chico"
    elif paso["id"].startswith("2"):
        orden = {etiqueta: i for i, etiqueta in enumerate(etiquetas)}
        ganador = min(candidatas, key=lambda e: orden[e])
        motivo = "menor ELBO de validación; a menos de un desvío se prefiere la capacidad más chica"
    elif paso["id"].startswith("4"):
        ganador = min(candidatas, key=float)
        motivo = "menor ELBO de validación; a menos de un desvío se prefiere el lr más chico"
    else:
        mejor_recon = min(recons, key=lambda e: recons[e]["media"])
        limite_recon = recons[mejor_recon]["media"] + recons[mejor_recon]["desvio"]
        candidatas_recon = [
            e for e in etiquetas if e in recons and recons[e]["media"] <= limite_recon
        ]
        ganador = (
            "binary_crossentropy"
            if "binary_crossentropy" in candidatas_recon
            else min(candidatas_recon, key=lambda e: recons[e]["media"])
        )
        motivo = "menor error de reconstrucción determinístico; empate prefiere BCE"
    return ganador, {"motivo": motivo, "recon": recons, "elbo": elbos}


def correr_barrido(nombre: str, trabajadores: int, forzar: bool) -> dict:
    cfg_general = _leer_config()
    ds = cfg_general["datasets"][nombre]
    actual = {
        "latente": cfg_general["base"]["latente"],
        "capacidad": cfg_general["base"]["capacidad"],
        "beta": cfg_general["base"]["beta"],
        "lr": cfg_general["base"]["lr"],
        "perdida": cfg_general["base"]["perdida"],
    }
    carpeta = RUTA_RESULTADOS / nombre / "barrido"
    carpeta.mkdir(parents=True, exist_ok=True)
    elecciones = []
    semillas = list(range(int(ds["semillas_barrido"])))

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
        elecciones.append({"paso": paso["id"], "ganador": ganador, **detalle})
        print(f"{nombre} {paso['id']}: gana {ganador}")
        _escribir_elecciones(nombre, cfg_general, actual, elecciones)

    _escribir_elecciones(nombre, cfg_general, actual, elecciones)
    return {"config_final": actual, "pasos": elecciones}


def _escribir_elecciones(nombre: str, cfg_general: dict, actual: dict, elecciones: list) -> None:
    carpeta = RUTA_RESULTADOS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    contenido = {
        "dataset": nombre,
        "semilla_split": cfg_general["semilla_split"],
        "config_final": actual,
        "pasos": elecciones,
    }
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
    return {
        "dataset": nombre,
        "tipo": ds["tipo"],
        "d": ds["d"],
        "forma": ds["forma"],
        "modelo": modelo,
        "latente": latente,
        "capacidad": cfg["capacidad"],
        "beta": cfg["beta"],
        "lr": cfg["lr"],
        "perdida": cfg["perdida"],
        "batch_size": ds["batch_size"],
        "epocas": ds["max_epocas"],
        "paciencia": ds["paciencia"],
        "semilla_split": _leer_config()["semilla_split"],
        "escalador": escalador,
    }


def correr_final(nombre: str, trabajadores: int, forzar: bool) -> None:
    ruta_elecciones = RUTA_RESULTADOS / nombre / "elecciones.json"
    if not ruta_elecciones.exists():
        raise RuntimeError(f"Falta {ruta_elecciones}; corré primero --fase barrido.")
    elecciones = json.loads(ruta_elecciones.read_text(encoding="utf-8"))
    cfg_final = elecciones["config_final"]
    cfg_general = _leer_config()
    ds = cfg_general["datasets"][nombre]
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
        ("latente_2", "vae", 2, int(ds["semillas_latente"])),
        ("latente_16", "vae", 16, int(ds["semillas_latente"])),
        ("ae_latente_2", "ae", 2, int(ds["semillas_latente"])),
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
    args = parser.parse_args()
    if args.trabajadores < 1 or args.hilos < 1:
        parser.error("--trabajadores y --hilos tienen que ser >= 1")
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.hilos)
    os.environ["OMP_NUM_THREADS"] = str(args.hilos)
    os.environ["MKL_NUM_THREADS"] = str(args.hilos)
    if args.fase == "barrido":
        correr_barrido(args.dataset, args.trabajadores, args.forzar)
    elif args.fase == "final":
        correr_final(args.dataset, args.trabajadores, args.forzar)
    else:
        correr_smoke(args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
