"""Preparación de los cuatro datasets del ejercicio 2 (docs/context/ej2-vae.md).

    python3 utils/preparar_ej2.py --dedup-f1
    python3 utils/preparar_ej2.py --dataset f1
    python3 utils/preparar_ej2.py --dataset nba
    python3 utils/preparar_ej2.py --dataset chino
    python3 utils/preparar_ej2.py --dataset banderas
    python3 utils/preparar_ej2.py --todo

Escribe solo en data/procesadas/ y resultados/ej2/. El paso --dedup-f1 es
destructivo: borra los directorios by_ref de data/headshots después de
verificar que cada archivo es un duplicado exacto de uno del nivel superior.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data"
HEADSHOTS = DATOS / "headshots"
PROCESADAS = DATOS / "procesadas"
RESULTADOS = RAIZ / "resultados" / "ej2"

RES_F1 = (32, 32)
RES_NBA = (30, 36)
RES_CHINO = (32, 32)
POR_CLASE_CHINO = 20
SEMILLA_SPLIT = 2026


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _asegurar_dirs() -> None:
    PROCESADAS.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)


def _repartir(n: int, rng: np.random.Generator, n_train: int | None = None) -> np.ndarray:
    """Etiquetas train/val/test para n muestras, en el orden de los índices 0..n-1."""
    if n_train is None:
        n_val = int(round(0.1 * n))
        n_test = int(round(0.1 * n))
        n_train = n - n_val - n_test
    else:
        n_val = int(round(0.1 * n))
        n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"Split inválido para n={n}: {n_train}/{n_val}/{n_test}")
    orden = rng.permutation(n)
    split = np.empty(n, dtype="<U5")
    split[orden[:n_train]] = "train"
    split[orden[n_train : n_train + n_val]] = "val"
    split[orden[n_train + n_val :]] = "test"
    return split


def _componer_blanco(imagen: Image.Image) -> Image.Image:
    """Convierte cualquier modo a RGB componiendo la transparencia sobre blanco."""
    rgba = imagen.convert("RGBA")
    fondo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(fondo, rgba).convert("RGB")


def _bbox_alpha(imagen: Image.Image, umbral: int = 8) -> tuple[int, int, int, int]:
    alpha = np.asarray(imagen.convert("RGBA"))[:, :, 3]
    ys, xs = np.nonzero(alpha > umbral)
    if len(xs) == 0:
        return (0, 0, imagen.width - 1, imagen.height - 1)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _cuadrado_blanco(imagen: Image.Image, caja: tuple[int, int, int, int]) -> Image.Image:
    """Recorta caja con margen y la pega centrada en un cuadrado blanco."""
    x0, y0, x1, y1 = caja
    ancho, alto = x1 - x0 + 1, y1 - y0 + 1
    margen = int(round(0.02 * max(ancho, alto)))
    x0 = max(0, x0 - margen)
    y0 = max(0, y0 - margen)
    x1 = min(imagen.width - 1, x1 + margen)
    y1 = min(imagen.height - 1, y1 + margen)
    recorte = imagen.crop((x0, y0, x1 + 1, y1 + 1))
    lado = max(recorte.width, recorte.height)
    lienzo = Image.new("RGB", (lado, lado), (255, 255, 255))
    lienzo.paste(recorte, ((lado - recorte.width) // 2, (lado - recorte.height) // 2))
    return lienzo


def _recorte_3_4(imagen: Image.Image) -> Image.Image:
    """Recorta a 3:4 centrado en el sujeto, completando con blanco si hace falta."""
    x0, y0, x1, y1 = _bbox_alpha(imagen)
    alto = y1 - y0 + 1
    objetivo = max(1, int(round(0.75 * alto)))
    centro = (x0 + x1) // 2
    izquierda = centro - objetivo // 2
    arriba = y0
    lienzo = Image.new("RGB", (objetivo, alto), (255, 255, 255))
    recorte = imagen.crop((izquierda, arriba, izquierda + objetivo, arriba + alto))
    lienzo.paste(recorte, (0, 0))
    return lienzo


def _guardar_contacto(imagenes: np.ndarray, titulos: list[str], ruta: Path, columnas: int = 16):
    n = len(imagenes)
    filas = int(np.ceil(n / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(columnas * 1.1, filas * 1.25), dpi=110)
    ejes = np.atleast_1d(ejes).ravel()
    for eje in ejes:
        eje.axis("off")
    for i, (imagen, titulo) in enumerate(zip(imagenes, titulos)):
        ejes[i].imshow(imagen, cmap="gray", vmin=0.0, vmax=1.0)
        ejes[i].set_title(titulo, fontsize=6, pad=1)
    fig.tight_layout(pad=0.2)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


# --- Deduplicación de F1 ---


def _filas_nivel_superior(
    raiz: Path, piloto_por_hash: dict[str, str] | None = None
) -> list[dict]:
    """Manifiesto completo de los archivos top-level (sin mirar by_ref)."""
    filas = []
    for ruta in sorted(p for p in raiz.glob("*/*") if p.is_file() and "by_ref" not in p.parts):
        digest = _sha256(ruta)
        with Image.open(ruta) as imagen:
            ancho, alto = imagen.size
        filas.append(
            {
                "archivo": str(ruta.relative_to(RAIZ)),
                "codigo": ruta.stem,
                "piloto": (piloto_por_hash or {}).get(digest, ruta.stem),
                "temporada": ruta.parent.name,
                "sha256": digest,
                "ancho": ancho,
                "alto": alto,
            }
        )
    return filas


def _deduplicar_filas(filas_completo: list[dict]) -> tuple[list[dict], list[dict]]:
    """Conserva una imagen por hash: temporada más nueva, luego mayor área."""
    grupos: dict[str, list[dict]] = defaultdict(list)
    for fila in filas_completo:
        grupos[fila["sha256"]].append(fila)

    conservados, descartados = [], []
    for digest, filas in grupos.items():
        filas = sorted(
            filas,
            key=lambda x: (-int(x["temporada"]), -(int(x["ancho"]) * int(x["alto"])), x["archivo"]),
        )
        conservados.append(filas[0])
        for fila in filas[1:]:
            descartados.append(
                {
                    "ruta_conservada": filas[0]["archivo"],
                    "ruta_descartada": fila["archivo"],
                    "sha256": digest,
                    "temporada_conservada": filas[0]["temporada"],
                    "temporada_descartada": fila["temporada"],
                }
            )
    conservados.sort(key=lambda x: x["archivo"])
    descartados.sort(key=lambda x: x["ruta_descartada"])
    return conservados, descartados


def _escribir_manifiestos(
    filas_completo: list[dict],
    conservados: list[dict],
    descartados: list[dict],
    filas_borrado: list[dict],
) -> None:
    with open(PROCESADAS / "f1_manifest_completo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_completo[0]))
        writer.writeheader()
        writer.writerows(filas_completo)
    # El registro de by_ref borrados es histórico: la reconstrucción del
    # manifiesto (sin by_ref) no debe borrarlo.
    ruta_borrado = RESULTADOS / "dedup_headshots.csv"
    if filas_borrado or not ruta_borrado.exists():
        with open(ruta_borrado, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ruta_borrada", "sha256", "ruta_conservada"])
            writer.writeheader()
            writer.writerows(filas_borrado)
    with open(PROCESADAS / "f1_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(conservados[0]))
        writer.writeheader()
        writer.writerows(conservados)
    with open(RESULTADOS / "dedup_hash.csv", "w", newline="", encoding="utf-8") as f:
        campos = list(descartados[0]) if descartados else [
            "ruta_conservada",
            "ruta_descartada",
            "sha256",
            "temporada_conservada",
            "temporada_descartada",
        ]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(descartados)
    por_piloto = {}
    for fila in sorted(conservados, key=lambda x: (-int(x["temporada"]), x["archivo"])):
        por_piloto.setdefault(fila["codigo"], fila)
    with open(PROCESADAS / "f1_manifest_piloto.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(conservados[0]))
        writer.writeheader()
        writer.writerows(por_piloto.values())


def deduplicar_headshots(raiz: Path = HEADSHOTS, permitir_huerfanos: bool = False) -> dict:
    """Borra los by_ref verificados y devuelve los manifiestos.

    Si los by_ref ya no existen y data/procesadas/f1_manifest.csv está, no
    hace nada. Si no existen y el manifiesto tampoco, lo reconstruye desde los
    archivos top-level (el borrado ya se hizo).

    Un by_ref huérfano (sin duplicado exacto en el nivel superior de su
    temporada) aborta por defecto. Con permitir_huerfanos=True se copia antes
    a data/procesadas/f1_byref_sin_par/ y se deja registro en el CSV; así no
    se pierde el archivo y el dataset principal sigue teniendo 158 imágenes.
    """
    manifiesto_final = PROCESADAS / "f1_manifest.csv"
    dirs_by_ref = sorted(raiz.glob("*/by_ref"))
    if not dirs_by_ref:
        if manifiesto_final.exists():
            print("Dedup F1: los by_ref ya no existen; manifiesto presente, se saltea.")
            return {"borrados": 0, "conservados": 158}
        # El borrado ya se hizo pero falta el manifiesto (por ejemplo, se limpió
        # data/procesadas): se reconstruye desde los 172 archivos top-level.
        _asegurar_dirs()
        filas_completo = _filas_nivel_superior(raiz)
        conservados, descartados = _deduplicar_filas(filas_completo)
        _escribir_manifiestos(filas_completo, conservados, descartados, [])
        print(
            f"Dedup F1: sin by_ref; manifiesto reconstruido con {len(conservados)} imágenes "
            f"únicas de {len({f['codigo'] for f in conservados})} pilotos."
        )
        return {"borrados": 0, "conservados": len(conservados)}

    _asegurar_dirs()
    top = sorted(p for p in raiz.glob("*/*") if p.is_file() and "by_ref" not in p.parts)
    refs = sorted(p for d in dirs_by_ref for p in d.glob("*") if p.is_file())
    hash_top: dict[str, list[Path]] = defaultdict(list)
    for ruta in top:
        hash_top[_sha256(ruta)].append(ruta)

    filas_borrado, faltantes = [], []
    for ref in refs:
        temporada = ref.parent.parent.name
        digest = _sha256(ref)
        candidatos = [p for p in hash_top.get(digest, []) if p.parent.name == temporada]
        if not candidatos:
            faltantes.append(ref)
            continue
        filas_borrado.append(
            {"ruta_borrada": str(ref), "sha256": digest, "ruta_conservada": str(candidatos[0])}
        )
    if faltantes and not permitir_huerfanos:
        detalle = "\n".join(str(p) for p in faltantes)
        raise RuntimeError(
            f"Hay by_ref sin duplicado exacto; no se borra nada:\n{detalle}\n"
            "Usá --permitir-huerfanos para copiarlos antes de borrar."
        )
    for ref in faltantes:
        digest = _sha256(ref)
        carpeta = PROCESADAS / "f1_byref_sin_par"
        carpeta.mkdir(parents=True, exist_ok=True)
        conservado = carpeta / f"{ref.parent.parent.name}_{ref.stem}_{digest[:8]}.png"
        shutil.copy2(ref, conservado)
        filas_borrado.append(
            {"ruta_borrada": str(ref), "sha256": digest, "ruta_conservada": str(conservado)}
        )
        with open(RESULTADOS / "dedup_byref_sin_par.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["ruta_original", "ruta_conservada", "sha256"])
            writer.writerow([str(ref), str(conservado), digest])

    # Mapa código -> piloto, construido antes de borrar.
    piloto_por_hash = {}
    for ref in refs:
        piloto_por_hash.setdefault(_sha256(ref), ref.stem)

    filas_completo = _filas_nivel_superior(raiz, piloto_por_hash)
    conservados, descartados = _deduplicar_filas(filas_completo)
    _escribir_manifiestos(filas_completo, conservados, descartados, filas_borrado)

    for directorio in dirs_by_ref:
        shutil.rmtree(directorio)

    restantes = sorted(p for p in raiz.glob("*/*") if p.is_file() and "by_ref" not in p.parts)
    if list(raiz.glob("*/by_ref")):
        raise RuntimeError("Quedaron directorios by_ref después del borrado.")
    if len(restantes) != len(top):
        raise RuntimeError(f"Quedaron {len(restantes)} archivos top-level, se esperaban {len(top)}")

    print(
        f"Dedup F1: {len(filas_borrado)} by_ref borrados, "
        f"{len(conservados)} imágenes únicas de {len({f['codigo'] for f in conservados})} pilotos."
    )
    return {"borrados": len(filas_borrado), "conservados": len(conservados)}


# --- F1 ---


def preparar_f1(forzar: bool = False) -> dict:
    salida_npz = PROCESADAS / "f1_32x32.npz"
    if salida_npz.exists() and not forzar:
        print("F1: ya existe, se saltea.")
        return {}
    with open(PROCESADAS / "f1_manifest.csv", newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if len(filas) != 158:
        raise RuntimeError(f"Se esperaban 158 imágenes F1, hay {len(filas)}")

    carpeta = PROCESADAS / "f1_32x32"
    carpeta.mkdir(parents=True, exist_ok=True)
    X, y, pilotos, temporadas, archivos, imagenes = [], [], [], [], [], []
    for fila in filas:
        ruta = RAIZ / fila["archivo"]
        with Image.open(ruta) as original:
            compuesta = _componer_blanco(original)
            caja = _bbox_alpha(original)
            cuadrada = _cuadrado_blanco(compuesta, caja)
            chica = cuadrada.convert("L").resize(RES_F1, Image.Resampling.LANCZOS)
        destino = carpeta / f"{fila['codigo']}_{fila['temporada']}_{fila['sha256'][:8]}.png"
        chica.save(destino)
        vector = np.asarray(chica, dtype=np.float32) / 255.0
        X.append(vector.ravel())
        y.append(fila["codigo"])
        pilotos.append(fila["piloto"])
        temporadas.append(int(fila["temporada"]))
        archivos.append(fila["archivo"])
        imagenes.append(vector)

    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(SEMILLA_SPLIT)
    split = _repartir(len(X), rng)
    meta = {
        "dataset": "f1",
        "resolucion": list(RES_F1),
        "fuente": "data/headshots",
        "semilla_split": SEMILLA_SPLIT,
        "conteos": {s: int(np.sum(split == s)) for s in ("train", "val", "test")},
    }
    np.savez_compressed(
        salida_npz,
        X=X,
        y=np.asarray(y),
        piloto=np.asarray(pilotos),
        temporada=np.asarray(temporadas, dtype=np.int16),
        archivo=np.asarray(archivos),
        split=split,
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    _guardar_contacto(np.asarray(imagenes), y, RESULTADOS / "f1_contacto.png", columnas=16)
    print(f"F1: X={X.shape}, split={meta['conteos']}")
    return meta


# --- NBA ---


def preparar_nba(forzar: bool = False) -> dict:
    salida_npz = PROCESADAS / "nba_30x36.npz"
    if salida_npz.exists() and not forzar:
        print("NBA: ya existe, se saltea.")
        return {}
    rutas = sorted(p for p in (DATOS / "player_images").iterdir() if p.is_file())
    if len(rutas) != 347:
        raise RuntimeError(f"Se esperaban 347 imágenes NBA, hay {len(rutas)}")

    carpeta = PROCESADAS / "nba_30x36"
    carpeta.mkdir(parents=True, exist_ok=True)
    X, y, archivos, imagenes = [], [], [], []
    for ruta in rutas:
        with Image.open(ruta) as original:
            compuesta = _componer_blanco(original)
            if ruta.name == "lebron.png":
                compuesta = _recorte_3_4(compuesta)
            chica = compuesta.convert("L").resize(RES_NBA, Image.Resampling.LANCZOS)
        destino = carpeta / f"{ruta.stem}.png"
        chica.save(destino)
        vector = np.asarray(chica, dtype=np.float32) / 255.0
        X.append(vector.ravel())
        y.append(ruta.stem)
        archivos.append(str(ruta.relative_to(RAIZ)))
        imagenes.append(vector)

    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(SEMILLA_SPLIT)
    split = _repartir(len(X), rng)
    meta = {
        "dataset": "nba",
        "resolucion": list(RES_NBA),
        "fuente": "data/player_images",
        "semilla_split": SEMILLA_SPLIT,
        "conteos": {s: int(np.sum(split == s)) for s in ("train", "val", "test")},
    }
    np.savez_compressed(
        salida_npz,
        X=X,
        y=np.asarray(y),
        archivo=np.asarray(archivos),
        split=split,
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    _guardar_contacto(np.asarray(imagenes), y, RESULTADOS / "nba_contacto.png", columnas=20)
    print(f"NBA: X={X.shape}, split={meta['conteos']}")
    return meta


# --- Chino ---


def _leer_chino(ruta: Path, por_clase: int = POR_CLASE_CHINO) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Lee `por_clase` filas por carácter, las baja a 32×32 y las invierte.

    El CSV trae 64×64 con fondo negro (0) y trazo claro (255). Se promedia cada
    bloque 2×2, se normaliza por el máximo de la imagen (el promedio diluye el
    trazo) y se invierte: fondo blanco 1.0 y trazo negro 0.0. Sin la inversión
    los caracteres quedan casi invisibles sobre negro.
    """
    X, y, clases = [], [], []
    indice: dict[str, int] = {}
    conteo: dict[str, int] = defaultdict(int)
    with open(ruta, newline="", encoding="utf-8") as f:
        lector = csv.reader(f)
        next(lector)
        for fila in lector:
            caracter = fila[4097]
            if conteo[caracter] >= por_clase:
                continue
            if caracter not in indice:
                indice[caracter] = len(clases)
                clases.append(caracter)
            pixeles = np.asarray(fila[:4096], dtype=np.float32).reshape(64, 64)
            chico = pixeles.reshape(32, 2, 32, 2).mean(axis=(1, 3))
            maximo = float(chico.max())
            if maximo > 0.0:
                chico = chico / maximo
            chico = 1.0 - chico
            X.append(chico.ravel())
            y.append(indice[caracter])
            conteo[caracter] += 1
            if len(indice) == 15 and all(conteo[c] >= por_clase for c in indice):
                break
    if len(X) != 15 * por_clase:
        raise RuntimeError(f"Se esperaban {15 * por_clase} imágenes chinas, hay {len(X)}")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int16), clases


def preparar_chino(forzar: bool = False, guardar_png: bool = False) -> dict:
    salida_npz = PROCESADAS / "chino_32x32.npz"
    if salida_npz.exists() and not forzar:
        print("Chino: ya existe, se saltea.")
        return {}
    X, clases_idx, caracteres = _leer_chino(DATOS / "chineseMNIST.csv")
    rng = np.random.default_rng(SEMILLA_SPLIT)
    split = np.empty(len(X), dtype="<U5")
    n_val = max(1, int(round(0.1 * POR_CLASE_CHINO)))
    n_train = POR_CLASE_CHINO - 2 * n_val
    for clase in range(15):
        idx = np.flatnonzero(clases_idx == clase)
        orden = rng.permutation(len(idx))
        split[idx[orden[:n_train]]] = "train"
        split[idx[orden[n_train : n_train + n_val]]] = "val"
        split[idx[orden[n_train + n_val :]]] = "test"

    meta = {
        "dataset": "chino",
        "resolucion": list(RES_CHINO),
        "fuente": "data/chineseMNIST.csv",
        "semilla_split": SEMILLA_SPLIT,
        "por_clase": POR_CLASE_CHINO,
        "invertido": True,
        "clases": caracteres,
        "conteos": {s: int(np.sum(split == s)) for s in ("train", "val", "test")},
    }
    np.savez_compressed(
        salida_npz,
        X=X,
        y=np.asarray([caracteres[i] for i in clases_idx]),
        clase=clases_idx,
        split=split,
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )

    if guardar_png:
        carpeta = PROCESADAS / "chino_32x32"
        carpeta.mkdir(parents=True, exist_ok=True)
        for i, vector in enumerate(X):
            Image.fromarray((vector.reshape(*RES_CHINO) * 255).astype(np.uint8)).save(
                carpeta / f"chino_{i:04d}.png"
            )
    for clase, _caracter in enumerate(caracteres):
        idx = np.flatnonzero(clases_idx == clase)[:8]
        _guardar_contacto(
            X[idx].reshape(-1, *RES_CHINO),
            [f"clase {clase}"] * 8,
            RESULTADOS / f"chino_{clase:02d}.png",
            columnas=4,
        )
    print(f"Chino: X={X.shape}, split={meta['conteos']}")
    return meta


# --- Banderas ---


def leer_banderas(ruta: Path = DATOS / "flags_tricolor.csv") -> list[dict]:
    filas = []
    with open(ruta, newline="", encoding="utf-8") as f:
        lector = csv.reader(f, delimiter=";")
        next(lector)
        for fila in lector:
            if not fila:
                continue
            nombre, *resto = fila
            numeros = list(map(int, ",".join(resto[:-3]).split(",")))
            colores = [numeros[i : i + 3] for i in range(0, len(numeros), 3)]
            rotada, proporcion, aspecto = int(resto[-3]), float(resto[-2]), float(resto[-1])
            filas.append(
                {
                    "nombre": nombre,
                    "colores": colores,
                    "rotada": rotada,
                    "proporcion": proporcion,
                    "aspecto": aspecto,
                }
            )
    return filas


def vector_bandera(fila: dict) -> np.ndarray:
    valores = []
    for color in fila["colores"]:
        valores.extend([c / 255.0 for c in color])
    valores.extend([float(fila["rotada"]), fila["proporcion"], fila["aspecto"]])
    return np.asarray(valores, dtype=np.float32)


def preparar_banderas(forzar: bool = False) -> dict:
    salida_npz = PROCESADAS / "flags.npz"
    if salida_npz.exists() and not forzar:
        print("Banderas: ya existe, se saltea.")
        return {}
    filas = leer_banderas()
    if len(filas) != 67:
        raise RuntimeError(f"Se esperaban 67 banderas, hay {len(filas)}")
    X = np.asarray([vector_bandera(f) for f in filas], dtype=np.float32)
    rng = np.random.default_rng(SEMILLA_SPLIT)
    split = _repartir(len(X), rng, n_train=54)
    columnas = np.asarray(
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
    )
    meta = {
        "dataset": "banderas",
        "fuente": "data/flags_tricolor.csv",
        "semilla_split": SEMILLA_SPLIT,
        "conteos": {s: int(np.sum(split == s)) for s in ("train", "val", "test")},
    }
    np.savez_compressed(
        salida_npz,
        X=X,
        nombres=np.asarray([f["nombre"] for f in filas]),
        split=split,
        columnas=columnas,
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    print(f"Banderas: X={X.shape}, split={meta['conteos']}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara los datasets del ejercicio 2.")
    parser.add_argument("--dedup-f1", action="store_true", help="Borra los by_ref de F1.")
    parser.add_argument("--dataset", choices=["f1", "nba", "chino", "banderas"])
    parser.add_argument("--todo", action="store_true", help="Dedup y todos los datasets.")
    parser.add_argument("--forzar", action="store_true", help="Rehace las salidas existentes.")
    parser.add_argument("--guardar-png", action="store_true", help="Guarda los PNG chinos.")
    parser.add_argument(
        "--permitir-huerfanos",
        action="store_true",
        help="Copia los by_ref sin duplicado antes de borrar (no aborta).",
    )
    args = parser.parse_args()
    if not (args.dedup_f1 or args.dataset or args.todo):
        parser.error("Indicá --dedup-f1, --dataset o --todo.")

    _asegurar_dirs()
    if args.dedup_f1 or args.todo:
        deduplicar_headshots(permitir_huerfanos=args.permitir_huerfanos)
    if args.todo or args.dataset == "f1":
        preparar_f1(args.forzar)
    if args.todo or args.dataset == "nba":
        preparar_nba(args.forzar)
    if args.todo or args.dataset == "chino":
        preparar_chino(args.forzar, args.guardar_png)
    if args.todo or args.dataset == "banderas":
        preparar_banderas(args.forzar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
