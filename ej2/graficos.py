"""Figuras del ejercicio 2 (docs/context/ej2-vae.md §8 y ej2-vae-fix.md §3.6).

    python3 -m ej2.graficos [--dataset nba]

Lee los artefactos que dejó ej2.vae_exp en resultados/ej2/<dataset>/ y los
datasets procesados para mostrar las entradas. No entrena nada: los modelos se
reconstruyen desde config.json y pesos_semilla_<s>.npz.

Salidas en resultados/ej2/<dataset>/figuras/:

    00_resumen.png              original / reconstrucción / muestras, con nombres
    01_dataset.png              muestra del dataset
    02_reconstruccion.png       original vs reconstrucción (train y test)
    03_latente_4.png            dispersión de mu y grilla decodificada
    04_latente_16_pca.png       PCA de mu a 2D y grilla decodificada
    05_generacion.png           muestras de N(0, I) y distancias a train
    06_curvas.png               pérdidas, KL y unidades activas por época
    07_latente_4_vs_16.png      reconstrucciones y muestras lado a lado
    08_banderas_decodificadas.png  solo banderas

Las banderas del resumen se renderizan con utils/generar_banderas.py en
figuras/auxiliares/, para que originales, reconstrucciones y muestras usen la
misma geometría.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from core.pca import PCA  # noqa: E402
from ej2 import datos  # noqa: E402
from ej2.banderas import render_flag  # noqa: E402
from ej2.modelos import (  # noqa: E402
    construir_ae_imagen,
    construir_ae_tabular,
    construir_vae_imagen,
    construir_vae_tabular,
)

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados" / "ej2"
DATASETS = ("nba", "f1", "chino", "banderas")
CARPETAS = ("final", "latente_4", "latente_16", "ae_latente_4", "ae_latente_16")

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

ESTILO = {
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "text.color": TINTA,
    "axes.titlesize": 10,
    "axes.titlecolor": TINTA,
    "axes.labelcolor": TINTA_2,
    "axes.edgecolor": EJE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": GRILLA,
    "grid.linewidth": 0.6,
    "xtick.color": TENUE,
    "ytick.color": TENUE,
    "xtick.labelcolor": TINTA_2,
    "ytick.labelcolor": TINTA_2,
    "legend.frameon": False,
    "lines.linewidth": 2,
}

LADO_GRILLA = 8
MUESTRAS_FILA = 8


# --- Lectura de artefactos ---


def _requiere(ruta: Path, fase: str) -> Path:
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}. Corré primero {fase}.")
    return ruta


def _normalizar(nombre: str) -> None:
    if nombre not in DATASETS:
        raise SystemExit(f"Dataset desconocido: {nombre}. Opciones: {', '.join(DATASETS)}")


def _carpeta(nombre: str, sub: str) -> Path:
    _normalizar(nombre)
    return _requiere(
        RESULTADOS / nombre / sub,
        f"python3 -m ej2.vae_exp --dataset {nombre} --fase final",
    )


def _config(nombre: str, sub: str) -> dict:
    return json.loads((_carpeta(nombre, sub) / "config.json").read_text(encoding="utf-8"))


def _dataset(nombre: str) -> dict:
    cfg = _config(nombre, "final")
    if cfg["tipo"] == "imagen":
        conjunto = datos.cargar_imagenes(nombre)
        return {
            "tipo": "imagen",
            "X": conjunto["X"],
            "y": conjunto["y"],
            "split": conjunto["split"],
            # config guarda (ancho, alto); los vectores se aplanaron desde (alto, ancho).
            "forma": (cfg["forma"][1], cfg["forma"][0]),
            "archivo": conjunto.get("archivo"),
        }
    conjunto = datos.cargar_banderas()
    escalador = _requiere(
        RESULTADOS / nombre / "escalador.json",
        f"python3 -m ej2.vae_exp --dataset {nombre} --fase final",
    )
    return {
        "tipo": "tabular",
        "X": conjunto["X"],
        "y": conjunto["nombres"],
        "split": conjunto["split"],
        "escalador": json.loads(escalador.read_text(encoding="utf-8")),
        "columnas": conjunto["columnas"],
    }


def _modelo(nombre: str, sub: str) -> tuple[object, dict]:
    cfg = _config(nombre, sub)
    salida = "sigmoid" if cfg["perdida"] == "binary_crossentropy" else "identity"
    beta = float(cfg.get("beta", 1.0))
    free_bits = float(cfg.get("free_bits", 0.0))
    if cfg["modelo"] == "ae":
        if cfg["tipo"] == "imagen":
            modelo = construir_ae_imagen(
                cfg["d"], cfg["latente"], cfg["capacidad"], salida=salida, semilla=0
            )
        else:
            modelo = construir_ae_tabular(cfg["latente"], cfg["capacidad"], semilla=0)
    elif cfg["tipo"] == "imagen":
        modelo = construir_vae_imagen(
            cfg["d"],
            cfg["latente"],
            cfg["capacidad"],
            salida=salida,
            semilla=0,
            beta=beta,
            free_bits=free_bits,
        )
    else:
        modelo = construir_vae_tabular(
            cfg["latente"],
            cfg["capacidad"],
            semilla=0,
            beta=beta,
            free_bits=free_bits,
        )
    ruta = _requiere(
        _carpeta(nombre, sub) / "pesos_semilla_0.npz",
        f"python3 -m ej2.vae_exp --dataset {nombre} --fase final",
    )
    with np.load(ruta, allow_pickle=False) as archivo:
        for i, parametro in enumerate(modelo.params):
            parametro[...] = archivo[f"p{i}"]
    return modelo, cfg


def _muestras(nombre: str, sub: str, semilla: int = 0) -> dict:
    ruta = _requiere(
        _carpeta(nombre, sub) / f"muestras_semilla_{semilla}.npz",
        f"python3 -m ej2.vae_exp --dataset {nombre} --fase final",
    )
    with np.load(ruta, allow_pickle=False) as archivo:
        return {clave: archivo[clave] for clave in archivo.files}


def _historias(nombre: str, sub: str) -> list[dict[str, np.ndarray]]:
    rutas = sorted(_carpeta(nombre, sub).glob("historia_semilla_*.csv"))
    if not rutas:
        raise SystemExit(f"No hay historias en {_carpeta(nombre, sub)}.")
    historias = []
    for ruta in rutas:
        with open(ruta, newline="", encoding="utf-8") as archivo:
            filas = list(csv.DictReader(archivo))
        historias.append(
            {
                columna: np.array(
                    [float(fila[columna]) if fila[columna] else np.nan for fila in filas]
                )
                for columna in filas[0]
            }
        )
    return historias


def _test_csv(nombre: str, sub: str) -> list[dict]:
    with open(_carpeta(nombre, sub) / "test.csv", newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


# --- Utilidades de dibujo ---


def _imagenes(datos_ds: dict, vectores: np.ndarray) -> list[np.ndarray]:
    if datos_ds["tipo"] == "imagen":
        return [fila.reshape(datos_ds["forma"]) for fila in vectores]
    return [_render(fila, datos_ds["escalador"]) for fila in vectores]


def _render(vector: np.ndarray, escalador: dict) -> np.ndarray:
    original = datos.desescalar_banderas(np.asarray(vector, dtype=np.float64), escalador)
    return render_flag(original, ancho=150).astype(np.float64) / 255.0


def _mosaico(imagenes: list[np.ndarray], columnas: int, separacion: int = 2) -> np.ndarray:
    filas = int(np.ceil(len(imagenes) / columnas))
    alto = max(imagen.shape[0] for imagen in imagenes)
    ancho = max(imagen.shape[1] for imagen in imagenes)
    canales = () if imagenes[0].ndim == 2 else (3,)
    forma = (
        filas * alto + (filas - 1) * separacion,
        columnas * ancho + (columnas - 1) * separacion,
    )
    lienzo = np.ones(forma + canales, dtype=float)
    for i, imagen in enumerate(imagenes):
        f, c = divmod(i, columnas)
        y = f * (alto + separacion)
        x = c * (ancho + separacion)
        desplazamiento_y = (alto - imagen.shape[0]) // 2
        desplazamiento_x = (ancho - imagen.shape[1]) // 2
        recorte_y = slice(y + desplazamiento_y, y + desplazamiento_y + imagen.shape[0])
        recorte_x = slice(x + desplazamiento_x, x + desplazamiento_x + imagen.shape[1])
        lienzo[recorte_y, recorte_x] = np.clip(imagen, 0.0, 1.0)
    return lienzo


def _alto_fila(conjunto: dict) -> float:
    return 1.4 if conjunto["tipo"] == "imagen" else 0.95


def _mostrar_mosaico(ax, imagenes: list[np.ndarray], columnas: int, titulo: str) -> None:
    mosaico = _mosaico(imagenes, columnas)
    if mosaico.ndim == 2:
        ax.imshow(mosaico, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(mosaico)
    ax.set_title(titulo)
    ax.axis("off")
    ax.grid(visible=False)


def _colorear(ax, Z: np.ndarray, etiquetas: np.ndarray) -> None:
    unicos = list(dict.fromkeys(map(str, etiquetas)))
    if len(unicos) <= 10:
        for i, etiqueta in enumerate(unicos):
            mascara = np.array([str(e) == etiqueta for e in etiquetas])
            ax.scatter(
                Z[mascara, 0],
                Z[mascara, 1],
                s=30,
                color=SERIES[i % len(SERIES)],
                edgecolors=SUPERFICIE,
                linewidths=1.0,
                label=etiqueta,
                zorder=3,
            )
        if len(unicos) > 1:
            ax.legend(fontsize=7, loc="best")
    else:
        codigos = np.array([unicos.index(str(e)) for e in etiquetas])
        ax.scatter(
            Z[:, 0],
            Z[:, 1],
            s=28,
            c=codigos,
            cmap=colormaps["viridis"],
            edgecolors=SUPERFICIE,
            linewidths=0.6,
            zorder=3,
        )


def _etiquetas_latentes(nombre: str, datos_ds: dict, y_test: np.ndarray) -> np.ndarray:
    if nombre == "banderas":
        original = datos.cargar_banderas()["X"]
        rotada = dict(zip(map(str, datos_ds["y"]), original[:, 9], strict=True))
        return np.array(
            ["vertical" if rotada.get(str(n), 0) >= 0.5 else "horizontal" for n in y_test]
        )
    if nombre == "chino":
        orden = list(dict.fromkeys(map(str, datos_ds["y"])))
        return np.array([f"clase {orden.index(str(caracter))}" for caracter in y_test])
    return y_test


def _guardar(fig: Figure, nombre: str) -> Path:
    carpeta = RESULTADOS / nombre / "figuras"
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def _grilla_puntos(modelo, centro: np.ndarray, radio: float):
    """Decodifica una grilla cuadrada variando las dos primeras dimensiones latentes."""
    x = np.linspace(centro[0] + radio, centro[0] - radio, LADO_GRILLA)
    y = np.linspace(centro[1] + radio, centro[1] - radio, LADO_GRILLA)
    z = np.tile(np.asarray(centro, dtype=float), (LADO_GRILLA**2, 1))
    z[:, 0] = np.repeat(x, LADO_GRILLA)
    z[:, 1] = np.tile(y, LADO_GRILLA)
    return modelo.decode(z)


# --- Resumen: original, reconstrucción y muestras ---


def _cargar_generador_banderas():
    ruta = RAIZ / "utils" / "generar_banderas.py"
    spec = importlib.util.spec_from_file_location("generar_banderas_figuras", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _vector_crudo(vector: np.ndarray, escalador: dict) -> np.ndarray:
    """Desescala un vector de 12 features y lo deja listo para renderizar."""
    original = datos.desescalar_banderas(np.asarray(vector, dtype=np.float64), escalador)
    original[:9] = np.clip(original[:9], 0.0, 1.0)
    original[9] = float(original[9] >= 0.5)
    return original


def _render_auxiliares(
    prefijo: str, vectores: np.ndarray, escalador: dict, carpeta: Path
) -> list[np.ndarray]:
    """Renderiza banderas con utils/generar_banderas.py y las lee con Pillow."""
    generador = _cargar_generador_banderas()
    carpeta.mkdir(parents=True, exist_ok=True)
    imagenes = []
    for i, vector in enumerate(vectores):
        nombre = f"{prefijo}_{i:02d}"
        generador.generar_vector(
            nombre, _vector_crudo(vector, escalador), ancho=240, salida=carpeta
        )
        with Image.open(carpeta / f"{nombre}.png") as imagen:
            imagenes.append(np.asarray(imagen.convert("RGB"), dtype=np.float64) / 255.0)
    return imagenes


def _elegir_resumen(nombre: str, etiquetas: np.ndarray, n: int = 8, semilla: int = 0) -> list[int]:
    """Elige hasta n índices: fijos por dataset, el resto al azar (clases distintas en chino)."""
    textos = [str(e) for e in etiquetas]
    fijas = {
        "nba": ["lebron", "Luka-Doncic", "Nikola-Jokic", "Stephen-Curry"],
        "f1": ["ALO", "VER"],
    }.get(nombre, [])
    elegidos = []
    for fija in fijas:
        if fija in textos and textos.index(fija) not in elegidos:
            elegidos.append(textos.index(fija))
    rng = np.random.default_rng(semilla)
    if nombre == "chino":
        presentes = {textos[i] for i in elegidos}
        clases = [c for c in dict.fromkeys(textos) if c not in presentes]
        for clase in rng.permutation(clases)[: n - len(elegidos)]:
            elegidos.append(textos.index(clase))
    if len(elegidos) < n:
        disponibles = [i for i in range(len(textos)) if i not in elegidos]
        faltan = n - len(elegidos)
        elegidos.extend(int(i) for i in rng.permutation(disponibles)[:faltan])
    return elegidos[:n]


def _vectores_resumen(nombre: str, conjunto: dict, modelo) -> tuple[np.ndarray, ...]:
    """Originales, reconstrucciones y muestras del resumen, con sus etiquetas.

    Usa el dataset completo para que los nombres fijos (LeBron, ALO, etc.)
    estén siempre, aunque no hayan caído en test.
    """
    X = (
        conjunto["X"]
        if conjunto["tipo"] == "imagen"
        else datos.aplicar_escalador(conjunto["X"], conjunto["escalador"])
    )
    y = conjunto["y"]
    reconstruccion = modelo.decode(modelo.encode(X))
    generadas = _muestras(nombre, "final")["generadas"]
    indices = _elegir_resumen(nombre, y)
    etiquetas = [str(v) for v in np.asarray(y)[indices]]
    return X[indices], reconstruccion[indices], generadas[: len(indices)], etiquetas


def fig_resumen(nombre: str) -> Figure:
    """La figura principal: 8 columnas por original / reconstrucción / muestra."""
    conjunto = _dataset(nombre)
    modelo, _ = _modelo(nombre, "final")
    X, reconstruccion, generadas, etiquetas = _vectores_resumen(nombre, conjunto, modelo)
    if conjunto["tipo"] == "imagen":
        originales = _imagenes(conjunto, X)
        reconstruidas = _imagenes(conjunto, reconstruccion)
        muestras = _imagenes(conjunto, generadas)
        ancho_columna, alto = 1.15, 4.6
    else:
        carpeta = RESULTADOS / nombre / "figuras" / "auxiliares"
        originales = _render_auxiliares("original", X, conjunto["escalador"], carpeta)
        reconstruidas = _render_auxiliares(
            "reconstruccion", reconstruccion, conjunto["escalador"], carpeta
        )
        muestras = _render_auxiliares("muestra", generadas, conjunto["escalador"], carpeta)
        ancho_columna, alto = 1.35, 5.0

    columnas = max(1, len(originales))
    fig = Figure(figsize=(ancho_columna * columnas, alto), layout="constrained")
    ejes = fig.subplots(3, columnas, squeeze=False)
    filas = (
        ("originales", originales),
        ("reconstrucción", reconstruidas),
        ("muestras N(0, I)", muestras),
    )
    for fila, (titulo, imagenes) in enumerate(filas):
        for columna in range(columnas):
            ax = ejes[fila, columna]
            if columna < len(imagenes):
                imagen = imagenes[columna]
                if imagen.ndim == 2:
                    ax.imshow(imagen, cmap="gray", vmin=0.0, vmax=1.0)
                else:
                    ax.imshow(imagen)
                if fila == 0 and columna < len(etiquetas):
                    ax.set_title(etiquetas[columna], fontsize=7)
            ax.axis("off")
        ejes[fila, 0].text(
            -0.12,
            0.5,
            titulo,
            transform=ejes[fila, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=8,
        )
    fig.suptitle(f"{nombre}: original, reconstrucción y muestras de N(0, I)")
    return fig


# --- 1. Dataset ---


def fig_dataset(nombre: str) -> Figure:
    conjunto = _dataset(nombre)
    indices = np.flatnonzero(conjunto["split"] == "test")[: MUESTRAS_FILA * 4]
    if len(indices) < 4:
        indices = np.arange(min(16, len(conjunto["X"])))
    imagenes = _imagenes(conjunto, conjunto["X"][indices])
    filas = int(np.ceil(len(imagenes) / MUESTRAS_FILA))
    fig = Figure(figsize=(9, 0.5 + _alto_fila(conjunto) * filas), layout="constrained")
    ax = fig.subplots()
    _mostrar_mosaico(ax, imagenes, MUESTRAS_FILA, f"{nombre}: muestras de test")
    return fig


# --- 2. Reconstrucción ---


def _indices_reconstruccion(nombre: str, conjunto: dict, split: str) -> np.ndarray:
    """Primeros 8 del split; en NBA, LeBron primero y el resto igual."""
    indices = list(np.flatnonzero(conjunto["split"] == split)[:MUESTRAS_FILA])
    if nombre == "nba":
        lebron = int(np.flatnonzero(conjunto["y"] == "lebron")[0])
        indices = [lebron] + [i for i in indices if i != lebron]
    return np.asarray(indices)


def fig_reconstruccion(nombre: str) -> Figure:
    conjunto = _dataset(nombre)
    modelo, _ = _modelo(nombre, "final")
    filas = []
    for split in ("train", "test"):
        idx = _indices_reconstruccion(nombre, conjunto, split)
        originales = conjunto["X"][idx]
        reconstruccion = modelo.decode(modelo.encode(originales))
        filas.append((f"{split}: originales", _imagenes(conjunto, originales)))
        filas.append((f"{split}: reconstrucción", _imagenes(conjunto, reconstruccion)))
    columnas = len(filas[0][1])
    fig = Figure(
        figsize=(1.15 * columnas, 0.5 + _alto_fila(conjunto) * len(filas)),
        layout="constrained",
    )
    ejes = fig.subplots(len(filas), 1)
    for ax, (titulo, imagenes) in zip(np.atleast_1d(ejes), filas, strict=True):
        _mostrar_mosaico(ax, imagenes, columnas, titulo)
    return fig


# --- 3. Latente 4 ---


def fig_latente_4(nombre: str) -> Figure:
    conjunto = _dataset(nombre)
    muestras = _muestras(nombre, "latente_4")
    modelo, _ = _modelo(nombre, "latente_4")
    Z = muestras["mu_test"]
    etiquetas = _etiquetas_latentes(nombre, conjunto, muestras["y_test"])
    centro = np.median(Z, axis=0)
    radio = max(1.0, float(np.abs(Z - centro).max()) * 1.35)
    decodificadas = _grilla_puntos(modelo, centro, radio)
    imagenes = _imagenes(conjunto, np.asarray(decodificadas))

    fig = Figure(figsize=(9.5, 4.6), layout="constrained")
    ax_scatter, ax_grilla = fig.subplots(1, 2, width_ratios=[1, 1])
    _colorear(ax_scatter, Z, etiquetas)
    ax_scatter.set_title("μ del test en el latente 4")
    ax_scatter.set_xlabel("z₁")
    ax_scatter.set_ylabel("z₂")
    ax_scatter.set_aspect("equal", adjustable="datalim")
    _mostrar_mosaico(ax_grilla, imagenes, LADO_GRILLA, "decodificación de la grilla")
    return fig


# --- 4. Latente 16 con PCA ---


def fig_latente_16(nombre: str) -> Figure:
    conjunto = _dataset(nombre)
    muestras = _muestras(nombre, "latente_16")
    modelo, _ = _modelo(nombre, "latente_16")
    pca = PCA(2).fit(muestras["mu_test"])
    Z = pca.transform(muestras["mu_test"])
    etiquetas = _etiquetas_latentes(nombre, conjunto, muestras["y_test"])
    centro = np.median(Z, axis=0)
    radio = max(1.0, float(np.abs(Z - centro).max()) * 1.35)
    x = np.linspace(centro[0] + radio, centro[0] - radio, LADO_GRILLA)
    y = np.linspace(centro[1] + radio, centro[1] - radio, LADO_GRILLA)
    grilla_2d = np.array([[xi, yi] for yi in y for xi in x])
    grilla_16 = pca.inverse_transform(grilla_2d)
    imagenes = _imagenes(conjunto, modelo.decode(grilla_16))

    fig = Figure(figsize=(9.5, 4.6), layout="constrained")
    ax_scatter, ax_grilla = fig.subplots(1, 2)
    _colorear(ax_scatter, Z, etiquetas)
    ax_scatter.set_title("μ del test proyectado con PCA (latente 16)")
    ax_scatter.set_xlabel("PC₁")
    ax_scatter.set_ylabel("PC₂")
    ax_scatter.set_aspect("equal", adjustable="datalim")
    _mostrar_mosaico(ax_grilla, imagenes, LADO_GRILLA, "decodificación de la grilla PCA")
    return fig


# --- 5. Generación ---


def fig_generacion(nombre: str) -> Figure:
    conjunto = _dataset(nombre)
    muestras = _muestras(nombre, "final")
    generadas = _imagenes(conjunto, muestras["generadas"])
    train = conjunto["X"][conjunto["split"] == "train"]
    distancias = np.array(
        [np.mean((train - vector) ** 2, axis=1).min() for vector in muestras["generadas"]]
    )
    medias = [float(fila["nn_dist"]) for fila in _test_csv(nombre, "final") if fila.get("nn_dist")]

    fig = Figure(figsize=(9.5, 4.2), layout="constrained")
    ax_grilla, ax_hist = fig.subplots(1, 2, width_ratios=[1.35, 1])
    _mostrar_mosaico(ax_grilla, generadas, MUESTRAS_FILA, "muestras de N(0, I)")
    ax_hist.hist(distancias, bins=12, color=SERIES[0], edgecolor=SUPERFICIE)
    if medias:
        ax_hist.axvline(float(np.mean(medias)), color=SERIES[1], linestyle="--")
        for i, media in enumerate(medias):
            ax_hist.scatter(
                [media],
                [0],
                s=22,
                color=SERIES[1],
                zorder=4,
                label="media por semilla" if i == 0 else None,
            )
    ax_hist.set_title("distancia al vecino más cercano de train (MSE)")
    ax_hist.set_xlabel("MSE")
    ax_hist.set_ylabel("muestras generadas")
    if medias:
        ax_hist.legend(fontsize=7)
    return fig


# --- 6. Curvas ---


def fig_curvas(nombre: str) -> Figure:
    historias = _historias(nombre, "final")
    corto = min(len(h["epoch"]) for h in historias)
    series = {
        "loss": [h["train_loss"][:corto] for h in historias],
        "val_loss": [h["val_loss"][:corto] for h in historias],
        "kl": [h.get("train_kl", np.full(corto, np.nan))[:corto] for h in historias],
        "activas": [h.get("train_activas", np.full(corto, np.nan))[:corto] for h in historias],
    }
    ejes = [
        ("loss", "train_loss"),
        ("val_loss", "val_loss"),
        ("kl", "train_kl"),
        ("activas", "train_activas"),
    ]
    fig = Figure(figsize=(9.5, 5.4), layout="constrained")
    paneles = fig.subplots(2, 2)
    titulos = {
        "loss": "pérdida de train",
        "val_loss": "pérdida de validación",
        "kl": "KL medio de train",
        "activas": "unidades activas de train",
    }
    for ax, (clave, _) in zip(paneles.ravel(), ejes, strict=True):
        matriz = np.vstack(series[clave])
        media = np.nanmean(matriz, axis=0)
        desvio = np.nanstd(matriz, axis=0, ddof=1) if matriz.shape[0] > 1 else np.zeros_like(media)
        epocas = np.arange(1, corto + 1)
        for fila in matriz:
            ax.plot(epocas, fila, color=SERIES[0], alpha=0.22, linewidth=0.8)
        ax.plot(epocas, media, color=SERIES[0])
        ax.fill_between(
            epocas, media - desvio, media + desvio, color=SERIES[0], alpha=0.18, linewidth=0
        )
        ax.set_title(titulos[clave])
        ax.set_xlabel("época")
    return fig


# --- 7. Latente 4 vs 16 ---


def fig_latente_4_vs_16(nombre: str) -> Figure:
    conjunto = _dataset(nombre)
    m4 = _muestras(nombre, "latente_4")
    m16 = _muestras(nombre, "latente_16")
    filas = [
        ("test: originales", _imagenes(conjunto, m4["X_test"][:MUESTRAS_FILA])),
        ("latente 4: reconstrucción", _imagenes(conjunto, m4["reconstrucciones"][:MUESTRAS_FILA])),
        (
            "latente 16: reconstrucción",
            _imagenes(conjunto, m16["reconstrucciones"][:MUESTRAS_FILA]),
        ),
        ("latente 4: muestras", _imagenes(conjunto, m4["generadas"][:MUESTRAS_FILA])),
        ("latente 16: muestras", _imagenes(conjunto, m16["generadas"][:MUESTRAS_FILA])),
    ]
    fig = Figure(figsize=(9, 0.5 + _alto_fila(conjunto) * len(filas)), layout="constrained")
    ejes = fig.subplots(len(filas), 1)
    for ax, (titulo, imagenes) in zip(np.atleast_1d(ejes), filas, strict=True):
        _mostrar_mosaico(ax, imagenes, MUESTRAS_FILA, titulo)
    return fig


# --- 8. Banderas decodificadas ---


def fig_banderas(nombre: str) -> Figure:
    conjunto = _dataset(nombre)
    if conjunto["tipo"] != "tabular":
        raise SystemExit("La figura 08 es solo para banderas.")
    muestras = _muestras(nombre, "final")
    random = _imagenes(conjunto, muestras["generadas"][:MUESTRAS_FILA])
    reconstruidas = _imagenes(conjunto, muestras["reconstrucciones"][:MUESTRAS_FILA])
    originales = _imagenes(conjunto, muestras["X_test"][:MUESTRAS_FILA])
    fig = Figure(figsize=(9.5, 0.6 + 0.95 * 3), layout="constrained")
    ejes = fig.subplots(3, 1)
    for ax, (titulo, imagenes) in zip(
        ejes,
        (
            ("banderas de test", originales),
            ("reconstrucción determinística", reconstruidas),
            ("muestras de N(0, I)", random),
        ),
        strict=True,
    ):
        _mostrar_mosaico(ax, imagenes, MUESTRAS_FILA, titulo)
    return fig


FIGURAS = {
    "00_resumen.png": fig_resumen,
    "01_dataset.png": fig_dataset,
    "02_reconstruccion.png": fig_reconstruccion,
    "03_latente_4.png": fig_latente_4,
    "04_latente_16_pca.png": fig_latente_16,
    "05_generacion.png": fig_generacion,
    "06_curvas.png": fig_curvas,
    "07_latente_4_vs_16.png": fig_latente_4_vs_16,
}


def generar(nombre: str) -> list[Path]:
    rutas = []
    with matplotlib.rc_context(ESTILO):
        for archivo, funcion in FIGURAS.items():
            fig = funcion(nombre)
            ruta = _guardar(fig, nombre) / archivo
            fig.savefig(ruta)
            rutas.append(ruta)
            print(f"{nombre}: {ruta.relative_to(RAIZ)}")
        if nombre == "banderas":
            fig = fig_banderas(nombre)
            ruta = _guardar(fig, nombre) / "08_banderas_decodificadas.png"
            fig.savefig(ruta)
            rutas.append(ruta)
            print(f"{nombre}: {ruta.relative_to(RAIZ)}")
    print(f"{nombre}: figuras en {(RESULTADOS / nombre / 'figuras').resolve()}")
    return rutas


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera las figuras del ejercicio 2.")
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    args = parser.parse_args()
    for nombre in args.dataset or DATASETS:
        generar(nombre)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
