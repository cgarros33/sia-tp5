"""Figuras del ejercicio 1a (docs/context/ej1-autoencoder.md).

    python -m ej1.graficos [--resultados resultados/ej1]

Solo lee lo que escribió ej1.barrido (CSV de cada paso, elecciones.json y
final/) y guarda en <resultados>/figuras/. No entrena nada: el autoencoder
final se reconstruye con los pesos guardados y la proyección PCA viene de
final/pca.npz.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

from ej1.barrido import RUTA_SALIDA, cargar_modelo, leer_filas, resumir
from ej1.datos import ALTO, ANCHO, cargar_font

# Paleta de referencia (modo claro): superficie, tintas, grilla y series en orden fijo.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"
APAGADO = "#c3c2b7"  # valores que no ganaron
FONDO_LETRA = "#f0efec"  # píxel apagado de una letra
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

ESTILO = {
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
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

# Una letra generada es "nueva" si está a esta cantidad de píxeles o más de todas las del set.
DISTANCIA_NUEVA = 2
LADO_GRILLA = 12
PASOS_INTERPOLACION = 9
INTERPOLACIONES = 3
CONTORNO = [patheffects.withStroke(linewidth=2.5, foreground=SUPERFICIE)]


# --- Lectura ---


def _leer_historia(ruta: Path) -> dict[str, np.ndarray]:
    with open(ruta, newline="") as f:
        filas = list(csv.DictReader(f))
    return {c: np.array([float(r[c]) if r[c] else np.nan for r in filas]) for c in filas[0]}


def _binarizar(salida: np.ndarray) -> np.ndarray:
    return (salida >= 0.5).astype(float)


def _distancias(letras: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Píxeles distintos entre cada letra (n, 35) y cada una del set (32, 35): (n, 32)."""
    return np.count_nonzero(letras[:, None, :] != X[None, :, :], axis=2)


# --- 1. Un gráfico por paso ---


def fig_paso(resumen: dict, ganador: str, titulo: str, motivo: str) -> Figure:
    valores = list(resumen)
    n = len(valores)
    fig = Figure(figsize=(9.5, 1.3 + 0.32 * n), layout="constrained")
    ax_pct, ax_ep = fig.subplots(1, 2, sharey=True, width_ratios=[1, 1.7])
    y = np.arange(n)
    colores = [SERIES[0] if v == ganador else APAGADO for v in valores]

    ax_pct.barh(y, [resumen[v]["pct_llegan"] for v in valores], height=0.62, color=colores)
    for yi, v in zip(y, valores, strict=True):
        r = resumen[v]
        ax_pct.text(
            min(r["pct_llegan"], 100) + 2,
            yi,
            f"{r['llegan']}/{r['corridas']}",
            va="center",
            color=TINTA_2,
        )
    ax_pct.set_xlim(0, 125)
    ax_pct.set_xticks([0, 25, 50, 75, 100])
    ax_pct.set_xlabel("% de semillas que llegan")
    ax_pct.set_yticks(y, valores)
    ax_pct.invert_yaxis()
    ax_pct.grid(axis="y", visible=False)

    for yi, v in zip(y, valores, strict=True):
        r = resumen[v]
        if r["llegan"]:
            ax_ep.barh(
                yi,
                r["epocas_media"],
                xerr=r["epocas_desvio"],
                height=0.62,
                color=SERIES[0] if v == ganador else APAGADO,
                error_kw={"ecolor": TINTA_2, "elinewidth": 1, "capsize": 3},
            )
        else:
            texto = (
                f"no llega · peor letra al final {r['peor_letra_media']:.1f} ± "
                f"{r['peor_letra_desvio']:.1f} píxeles"
            )
            ax_ep.text(0, yi, "  " + texto, va="center", color=TINTA_2)
    ax_ep.set_xlim(left=0)
    ax_ep.set_xlabel("épocas hasta llegar (media ± desvío, solo semillas que llegan)")
    ax_ep.grid(axis="y", visible=False)

    fig.suptitle(titulo, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.01, f"Gana {ganador}: {motivo}.", ha="left", va="top", color=TINTA_2)
    return fig


# --- 2. Pasos 6 + 7: épocas vs lr por optimizador ---


def fig_optimizadores(filas: list[dict], max_epocas: int) -> Figure:
    resumen = resumir(filas)
    config_de = {f["valor"]: f["config"] for f in filas}
    curvas: dict[str, list[tuple[float, dict]]] = {}
    for valor, r in resumen.items():
        spec = config_de[valor]["optimizador"]
        nombre = spec if isinstance(spec, str) else spec["kind"]
        curvas.setdefault(nombre, []).append((config_de[valor]["lr"], r))

    fig = Figure(figsize=(8, 5), layout="constrained")
    ax = fig.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Franja de arriba: los (optimizador, lr) que no llegan con ninguna semilla.
    franja = [max_epocas * f for f in (1.5, 2.0, 2.7)]
    for k, (nombre, puntos) in enumerate(curvas.items()):
        color = SERIES[k]
        puntos.sort(key=lambda p: p[0])
        lrs = np.array([lr for lr, _ in puntos])
        media = np.array([r["epocas_media"] if r["llegan"] else np.nan for _, r in puntos])
        desvio = np.array([r["epocas_desvio"] if r["llegan"] else np.nan for _, r in puntos])
        ax.plot(lrs, media, color=color, label=nombre, zorder=2)
        ax.errorbar(lrs, media, yerr=desvio, fmt="none", ecolor=color, elinewidth=1, capsize=3)
        completos = [r["llegan"] == r["corridas"] for _, r in puntos]
        for lr, m, completo, (_, r) in zip(lrs, media, completos, puntos, strict=True):
            if np.isnan(m):
                ax.scatter(lr, franja[k], marker="x", s=40, color=color, linewidths=1.8, zorder=3)
                continue
            ax.scatter(
                lr,
                m,
                s=48,
                zorder=3,
                color=color if completo else SUPERFICIE,
                edgecolors=color,
                linewidths=1.8,
            )
            if not completo:
                ax.annotate(
                    f"{r['llegan']}/{r['corridas']}",
                    (lr, m),
                    xytext=(6, 4),
                    textcoords="offset points",
                    color=TINTA_2,
                    fontsize=8,
                )
    ax.axhline(max_epocas, color=TENUE, linewidth=0.8)
    # x en coordenadas del eje (0 = borde izquierdo), y en datos.
    borde = ax.get_yaxis_transform()
    ax.text(
        0.01,
        max_epocas * 0.93,
        f"máximo: {max_epocas} épocas",
        transform=borde,
        color=TINTA_2,
        va="top",
        fontsize=8,
    )
    ax.text(
        0.01,
        franja[-1] * 1.12,
        "× no llega con ninguna semilla",
        transform=borde,
        color=TINTA_2,
        va="bottom",
        fontsize=8,
    )
    ax.set_ylim(top=franja[-1] * 1.6)
    todos = sorted({lr for puntos in curvas.values() for lr, _ in puntos})
    ax.set_xticks(todos, [f"{lr:g}" for lr in todos])
    ax.minorticks_off()
    bajo = ax.get_ylim()[0]
    marcas = [
        m
        for m in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
        if bajo <= m <= max_epocas
    ]
    ax.set_yticks(marcas, [f"{m:g}" for m in marcas])
    ax.set_xlabel("tasa de aprendizaje (lr)")
    ax.set_ylabel("épocas hasta llegar (media ± desvío)")
    leyenda = ax.legend(loc="lower left", title="optimizador")
    leyenda.get_title().set_color(TINTA_2)
    fig.suptitle(
        "Pasos 6 + 7: optimizador y lr. Círculo lleno: llegan todas las semillas; "
        "vacío: llegan algunas",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    return fig


# --- 3. Líneas de base vs config final ---


def fig_lineas_de_base(base: dict, final: dict) -> Figure:
    nombres = {"pca": "PCA 2D", "ae_lineal": "AE lineal", "sin_ocultas": "AE sin ocultas"}
    etiquetas = [nombres.get(v, v) for v in base] + ["config final"]
    resumenes = [*base.values(), final]
    colores = [APAGADO] * len(base) + [SERIES[0]]
    x = np.arange(len(etiquetas))

    fig = Figure(figsize=(9, 3.8), layout="constrained")
    paneles = fig.subplots(1, 2)
    for ax, clave, titulo in zip(
        paneles,
        ("pixeles_mal", "peor_letra"),
        ("Píxeles mal por letra (promedio de las 32)", "Píxeles mal de la peor letra"),
        strict=True,
    ):
        medias = [r[f"{clave}_media"] for r in resumenes]
        desvios = [r[f"{clave}_desvio"] for r in resumenes]
        ax.bar(
            x,
            medias,
            yerr=desvios,
            width=0.6,
            color=colores,
            error_kw={"ecolor": TINTA_2, "elinewidth": 1, "capsize": 3},
        )
        for xi, m, d in zip(x, medias, desvios, strict=True):
            ax.text(xi, m + d, f"{m:.2f}", ha="center", va="bottom", color=TINTA_2)
        ax.set_xticks(x, etiquetas)
        ax.set_title(titulo, loc="left")
        ax.grid(axis="x", visible=False)
    paneles[1].axhline(1, color=TENUE, linewidth=0.8, linestyle="--")
    paneles[1].text(-0.45, 1.15, "objetivo ≤ 1", color=TINTA_2, va="bottom", ha="left")
    fig.suptitle(
        "Líneas de base vs config final (media ± desvío entre semillas)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    return fig


# --- 4. Espacio latente: AE final y PCA ---


def _scatter_rotulado(ax, Z: np.ndarray, etiquetas: list[str], titulo: str) -> None:
    ax.scatter(
        Z[:, 0], Z[:, 1], s=28, color=SERIES[0], edgecolors=SUPERFICIE, linewidths=1.5, zorder=3
    )
    for (z1, z2), e in zip(Z, etiquetas, strict=True):
        ax.annotate(
            e,
            (z1, z2),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=9,
            color=TINTA,
            path_effects=CONTORNO,
            zorder=4,
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(titulo, loc="left")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")


def fig_espacio_latente(
    Z_ae: np.ndarray, Z_pca: np.ndarray, etiquetas: list[str], semilla
) -> Figure:
    fig = Figure(figsize=(11, 5.2), layout="constrained")
    ax_ae, ax_pca = fig.subplots(1, 2)
    _scatter_rotulado(ax_ae, Z_ae, etiquetas, f"Autoencoder, config final (semilla {semilla})")
    _scatter_rotulado(ax_pca, Z_pca, etiquetas, "PCA 2D")
    fig.suptitle(
        "Las 32 letras en el espacio latente de 2 dimensiones",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    return fig


# --- 5. Letras nuevas ---


def _dibujar_letra(ax, letra: np.ndarray, centro, ancho, alto, nueva: bool) -> None:
    tinta = SERIES[0] if nueva else TINTA_2
    x, y = centro
    ax.imshow(
        letra.reshape(ALTO, ANCHO),
        cmap=ListedColormap([FONDO_LETRA, tinta]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=(x - ancho / 2, x + ancho / 2, y - alto / 2, y + alto / 2),
        zorder=1,
    )


def _leyenda_letras() -> list[Patch]:
    return [
        Patch(color=SERIES[0], label=f"nueva: a ≥ {DISTANCIA_NUEVA} píxeles de todas las del set"),
        Patch(color=TINTA_2, label=f"reproduce una letra del set (≤ {DISTANCIA_NUEVA - 1} píxel)"),
    ]


def letras_de_grilla(ae, Z: np.ndarray, X: np.ndarray) -> dict:
    """Decodifica una grilla de LADO_GRILLA² puntos sobre el rango de Z (con 8 % de margen)."""
    bajo, alto = Z.min(axis=0), Z.max(axis=0)
    margen = 0.08 * (alto - bajo)
    xs = np.linspace(bajo[0] - margen[0], alto[0] + margen[0], LADO_GRILLA)
    ys = np.linspace(bajo[1] - margen[1], alto[1] + margen[1], LADO_GRILLA)
    puntos = np.array([(x, y) for y in ys for x in xs])
    letras = _binarizar(ae.decode(puntos))
    d = _distancias(letras, X)
    return {
        "puntos": puntos,
        "letras": letras,
        "distancia": d.min(axis=1),
        "parecida": d.argmin(axis=1),
        "dx": xs[1] - xs[0],
        "dy": ys[1] - ys[0],
    }


def fig_grilla(grilla: dict, Z: np.ndarray, etiquetas: list[str], semilla) -> Figure:
    fig = Figure(figsize=(9, 10.4), layout="constrained")
    ax = fig.subplots()
    dx, dy = grilla["dx"], grilla["dy"]
    for punto, letra, dist in zip(
        grilla["puntos"], grilla["letras"], grilla["distancia"], strict=True
    ):
        _dibujar_letra(ax, letra, punto, 0.5 * dx, 0.62 * dy, dist >= DISTANCIA_NUEVA)
        ax.text(
            punto[0],
            punto[1] - 0.34 * dy,
            str(dist),
            ha="center",
            va="top",
            fontsize=6.5,
            color=TENUE,
        )
    ax.scatter(
        Z[:, 0], Z[:, 1], s=22, color=SERIES[1], edgecolors=SUPERFICIE, linewidths=1.2, zorder=3
    )
    for (z1, z2), e in zip(Z, etiquetas, strict=True):
        ax.annotate(
            e,
            (z1, z2),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=TINTA,
            path_effects=CONTORNO,
            zorder=4,
        )
    puntos = grilla["puntos"]
    ax.set_xlim(puntos[:, 0].min() - dx / 2, puntos[:, 0].max() + dx / 2)
    ax.set_ylim(puntos[:, 1].min() - dy / 2, puntos[:, 1].max() + dy / 2)
    ax.set_aspect("auto")
    ax.grid(False)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    marcas = [*_leyenda_letras(), Patch(color=SERIES[1], label="código de una letra del set")]
    ax.legend(handles=marcas, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    n_nuevas = int(np.count_nonzero(grilla["distancia"] >= DISTANCIA_NUEVA))
    fig.suptitle(
        f"Letras decodificadas recorriendo el espacio latente (semilla {semilla}): "
        f"{n_nuevas} de {len(puntos)} son nuevas. Número: píxeles a la letra más parecida",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    return fig


def elegir_interpolaciones(ae, Z: np.ndarray, X: np.ndarray) -> list[tuple[int, int, dict]]:
    """Los INTERPOLACIONES pares (letra, su vecina más cercana en el latente) más distintos.

    Se ordenan por píxeles distintos entre las dos letras del par, de mayor a
    menor, y solo cuentan los pares cuyo segmento pasa por al menos una letra
    nueva. Para cada par devuelve sus índices y lo decodificado en
    PASOS_INTERPOLACION puntos del segmento (extremos incluidos).
    """
    t = np.linspace(0.0, 1.0, PASOS_INTERPOLACION)
    separacion = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    np.fill_diagonal(separacion, np.inf)
    pares = {tuple(sorted((i, int(np.argmin(separacion[i]))))) for i in range(len(Z))}
    candidatos = []
    for i, j in pares:
        puntos = Z[i] + t[:, None] * (Z[j] - Z[i])
        letras = _binarizar(ae.decode(puntos))
        d = _distancias(letras, X)
        if np.any(d.min(axis=1) >= DISTANCIA_NUEVA):
            datos = {
                "t": t,
                "puntos": puntos,
                "letras": letras,
                "distancia": d.min(axis=1),
                "parecida": d.argmin(axis=1),
            }
            candidatos.append((int(np.count_nonzero(X[i] != X[j])), i, j, datos))
    candidatos.sort(key=lambda c: (-c[0], c[1], c[2]))
    return [(i, j, datos) for _, i, j, datos in candidatos[:INTERPOLACIONES]]


def fig_interpolaciones(
    interpolaciones: list[tuple[int, int, dict]], etiquetas: list[str], semilla
) -> Figure:
    filas, n = len(interpolaciones), PASOS_INTERPOLACION
    fig = Figure(figsize=(1.05 * n, 0.4 + 2.15 * filas), layout="constrained")
    ejes = fig.subplots(filas, n, squeeze=False)
    for fila, (i, j, datos) in enumerate(interpolaciones):
        for k, ax in enumerate(ejes[fila]):
            dist = int(datos["distancia"][k])
            nueva = dist >= DISTANCIA_NUEVA
            _dibujar_letra(ax, datos["letras"][k], (0.5, 0.5), 1.0, 1.4, nueva)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.2, 1.2)
            ax.set_aspect("equal")
            ax.axis("off")
            parecida = etiquetas[datos["parecida"][k]]
            titulo = f"t = {datos['t'][k]:.3g}"
            if k == 0:
                titulo = f"«{etiquetas[i]}» → «{etiquetas[j]}»    {titulo}"
            ax.set_title(titulo, fontsize=8, color=TINTA_2, loc="left" if k == 0 else "center")
            ax.text(
                0.5,
                -0.25,
                f"≈ {parecida} · {dist} px",
                ha="center",
                va="top",
                fontsize=8,
                color=SERIES[0] if nueva else TINTA_2,
                fontweight="bold" if nueva else "normal",
            )
    fig.suptitle(
        f"Interpolación en el latente entre letras vecinas (semilla {semilla}). Debajo: letra "
        f"más parecida y a cuántos píxeles; en azul, a ≥ {DISTANCIA_NUEVA} píxeles de todas",
        x=0.01,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    return fig


def _escribir_generadas(ruta: Path, grilla: dict, interpolaciones: list, etiquetas) -> None:
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["origen", "z1", "z2", "letra_mas_parecida", "distancia", "nueva"])
        tramos = [(f"{etiquetas[i]} -> {etiquetas[j]}", datos) for i, j, datos in interpolaciones]
        for origen, datos in [("grilla", grilla), *tramos]:
            for punto, parecida, dist in zip(
                datos["puntos"], datos["parecida"], datos["distancia"], strict=True
            ):
                escritor.writerow(
                    [
                        origen,
                        punto[0],
                        punto[1],
                        etiquetas[parecida],
                        int(dist),
                        int(dist >= DISTANCIA_NUEVA),
                    ]
                )


# --- 6. Curvas de la config final ---


def fig_curvas(historias: dict[int, dict]) -> Figure:
    fig = Figure(figsize=(10, 4), layout="constrained")
    ax_loss, ax_peor = fig.subplots(1, 2)
    for k, (semilla, h) in enumerate(historias.items()):
        color = SERIES[k % len(SERIES)]
        ax_loss.plot(
            h["epoch"], h["train_loss"], color=color, linewidth=1.5, label=f"semilla {semilla}"
        )
        ax_peor.plot(h["epoch"], h["train_peor_letra"], color=color, linewidth=1.5)
    ax_loss.set_yscale("log")
    ax_loss.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax_loss.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_loss.yaxis.set_minor_formatter(NullFormatter())
    ax_loss.set_title("Pérdida de entrenamiento", loc="left")
    ax_loss.set_xlabel("época")
    ax_loss.legend()
    ax_peor.axhline(1, color=TENUE, linewidth=0.8, linestyle="--")
    ax_peor.set_title("Píxeles mal de la peor letra (línea: objetivo ≤ 1)", loc="left")
    ax_peor.set_xlabel("época")
    fig.suptitle(
        "Config final: entrenamiento por época, hasta llegar",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    return fig


# --- 7. Letras más difíciles ---


def fig_letras_dificiles(filas: list[dict], etiquetas: list[str]) -> Figure:
    corridas = [f for f in filas if f["valor"] != "pca"]
    mal = np.sum([np.array(f["errores"]) > 1 for f in corridas], axis=0)
    orden = np.argsort(-mal, kind="stable")
    fig = Figure(figsize=(10, 3.8), layout="constrained")
    ax = fig.subplots()
    x = np.arange(len(etiquetas))
    ax.bar(x, mal[orden], width=0.7, color=SERIES[0])
    for xi in x[:3]:
        ax.text(xi, mal[orden][xi], str(mal[orden][xi]), ha="center", va="bottom", color=TINTA_2)
    ax.set_xticks(x, [etiquetas[k] for k in orden])
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("corridas")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(
        f"En cuántas de las {len(corridas)} corridas de autoencoder del barrido quedó mal cada "
        "letra (> 1 píxel mal al final)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    return fig


# --- Todo ---


def _guardar(fig: Figure, ruta: Path, escritas: list[Path]) -> None:
    fig.savefig(ruta)
    escritas.append(ruta)


def generar_figuras(resultados: Path) -> list[Path]:
    """Genera todas las figuras en resultados/figuras/ y devuelve las rutas escritas."""
    figuras = resultados / "figuras"
    figuras.mkdir(exist_ok=True)
    elecciones = json.loads((resultados / "elecciones.json").read_text(encoding="utf-8"))
    por_paso: dict[str, list[dict]] = {}
    nombre_de: dict[str, str] = {}  # paso -> nombre del CSV sin extensión
    for ruta in sorted(resultados.glob("paso_*.csv")):
        for f in leer_filas(ruta):
            por_paso.setdefault(f["paso"], []).append(f)
            nombre_de[f["paso"]] = ruta.stem
    filas = [f for grupo in por_paso.values() for f in grupo]
    X, etiquetas = cargar_font()
    escritas: list[Path] = []

    with matplotlib.rc_context(ESTILO):
        # 1. Un gráfico por paso, más la repetición final.
        pasos = [*elecciones["pasos"], {**elecciones["verificacion_final"], "id": "final"}]
        for paso in pasos:
            resumen = resumir(por_paso[paso["id"]])
            if paso["id"] == "final":
                titulo = (
                    f"Final: paso {paso['paso']} ({paso['nombre']}) repetido con la config final"
                )
            else:
                titulo = f"Paso {paso['id']}: {paso['nombre'].replace('_', ' ')}"
            fig = fig_paso(resumen, paso["ganador"], titulo, paso["motivo"])
            _guardar(fig, figuras / f"01_{nombre_de[paso['id']]}.png", escritas)

        # 2. Optimizador y lr.
        max_epocas = elecciones["config_barrido"]["base"]["max_epocas"]
        fig = fig_optimizadores(por_paso["6+7"], max_epocas)
        _guardar(fig, figuras / "02_optimizador_y_lr.png", escritas)

        # 3. Líneas de base vs final.
        final = leer_filas(resultados / "final" / "corridas.csv")
        base = resumir(por_paso["0"])
        fig = fig_lineas_de_base(base, resumir(final)["final"])
        _guardar(fig, figuras / "03_lineas_de_base_vs_final.png", escritas)

        # 4 y 5. Con la menor semilla de la config final que llegó (o la menor, si ninguna).
        semilla = min(
            (f["semilla"] for f in final if f["llego"]), default=min(f["semilla"] for f in final)
        )
        ae = cargar_modelo(
            elecciones["config_final"],
            resultados / "final" / f"pesos_semilla_{semilla}.npz",
            X.shape[1],
        )
        Z = ae.encode(X)
        with np.load(resultados / "final" / "pca.npz") as pca:
            Z_pca = pca["Z"]
        fig = fig_espacio_latente(Z, Z_pca, etiquetas, semilla)
        _guardar(fig, figuras / "04_espacio_latente.png", escritas)

        grilla = letras_de_grilla(ae, Z, X)
        _guardar(
            fig_grilla(grilla, Z, etiquetas, semilla), figuras / "05a_grilla_latente.png", escritas
        )
        interpolaciones = elegir_interpolaciones(ae, Z, X)
        fig = fig_interpolaciones(interpolaciones, etiquetas, semilla)
        _guardar(fig, figuras / "05b_interpolacion.png", escritas)
        ruta_generadas = figuras / "05_letras_generadas.csv"
        _escribir_generadas(ruta_generadas, grilla, interpolaciones, etiquetas)
        escritas.append(ruta_generadas)

        # 6. Curvas de la config final.
        historias = {
            f["semilla"]: _leer_historia(
                resultados / "final" / f"historia_semilla_{f['semilla']}.csv"
            )
            for f in final
        }
        _guardar(fig_curvas(historias), figuras / "06_curvas_final.png", escritas)

        # 7. Letras más difíciles.
        fig = fig_letras_dificiles(filas, etiquetas)
        _guardar(fig, figuras / "07_letras_dificiles.png", escritas)
    return escritas


def main() -> None:
    parser = argparse.ArgumentParser(description="Figuras del ejercicio 1a.")
    parser.add_argument("--resultados", type=Path, default=RUTA_SALIDA)
    args = parser.parse_args()
    for ruta in generar_figuras(args.resultados):
        print(ruta, flush=True)


if __name__ == "__main__":
    main()
