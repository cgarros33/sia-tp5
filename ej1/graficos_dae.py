"""Figuras del ejercicio 1b (docs/context/ej1b-denoising.md).

    python -m ej1.graficos_dae [--resultados resultados/ej1b]

Solo lee lo que escribió ej1.dae (CSV de cada experimento, referencias,
elecciones.json y final/) y guarda en <resultados>/figuras/. No entrena nada:
el DAE final se reconstruye con los pesos guardados, y los conjuntos ruidosos
se regeneran con sus semillas fijas (no dependen de ningún modelo).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from ej1.barrido import cargar_modelo
from ej1.dae import RUTA_SALIDA, conjuntos_ruidosos, leer_filas_dae
from ej1.datos import ALTO, ANCHO, cargar_font
from ej1.graficos import (
    APAGADO,
    ESTILO,
    FONDO_LETRA,
    SERIES,
    SUPERFICIE,
    TENUE,
    TINTA,
    TINTA_2,
    _guardar,
)

# Referencias: nombre en las figuras y estilo de línea.
REFERENCIAS = {
    "nada": ("no hacer nada (entrada ruidosa)", TENUE, ":"),
    "ae_1a": ("AE del 1a (sin ruido)", APAGADO, "--"),
    "vecino": ("vecino más cercano (techo)", TINTA, "--"),
}
# Letras de la figura 4: variadas, y la «l» y la «|», que están a 2 píxeles.
LETRAS_EJEMPLO = ["a", "e", "g", "k", "m", "s", "l", "|"]
NIVEL_LATENTE = 0.1  # nivel de ruido de las letras de la figura 5


def _titulo(fig: Figure, texto: str) -> None:
    fig.suptitle(texto, x=0.01, ha="left", fontsize=11, fontweight="bold")


def _media_desvio(valores: list[float]) -> tuple[float, float]:
    v = np.asarray(valores, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def curvas_por_valor(filas: list[dict], niveles: list[float], medida: str) -> dict[str, tuple]:
    """{valor: (medias, desvios)} de val_<medida>_<p> sobre las semillas, en el orden de niveles."""
    grupos: dict[str, list[dict]] = {}
    for f in filas:
        grupos.setdefault(f["valor"], []).append(f)
    curvas = {}
    for valor, grupo in grupos.items():
        pares = [_media_desvio([f[f"val_{medida}_{p:g}"] for f in grupo]) for p in niveles]
        curvas[valor] = tuple(np.array(x) for x in zip(*pares, strict=True))
    return curvas


def curvas_por_modelo(filas: list[dict], niveles: list[float], medida: str) -> dict[str, tuple]:
    """{modelo: (medias, desvios)} de filas (modelo, semilla, nivel, medidas) por nivel."""
    curvas = {}
    for modelo in dict.fromkeys(f["modelo"] for f in filas):
        pares = [
            _media_desvio([f[medida] for f in filas if f["modelo"] == modelo and f["nivel"] == p])
            for p in niveles
        ]
        curvas[modelo] = tuple(np.array(x) for x in zip(*pares, strict=True))
    return curvas


def _dibujar_referencias(ax, refs: dict[str, tuple], niveles: list[float], escala: float) -> None:
    for modelo, (nombre, color, estilo) in REFERENCIAS.items():
        medias, _ = refs[modelo]
        ax.plot(
            niveles, escala * medias, color=color, linestyle=estilo, linewidth=1.5, label=nombre
        )


# --- 1. Experimento A: latente ---


def fig_latente(
    curvas: dict[str, dict], refs: dict[str, dict], niveles: list[float], ganador: str, ruido
) -> Figure:
    """curvas y refs: {medida: {valor/modelo: (medias, desvios)}} en validación."""
    fig = Figure(figsize=(11, 4.2), layout="constrained")
    paneles = fig.subplots(1, 2)
    for ax, medida, escala, titulo in zip(
        paneles,
        ("pixeles_mal", "recuperadas"),
        (1, 100),
        ("Píxeles mal por letra (promedio)", "Letras recuperadas (≤ 1 píxel mal), %"),
        strict=True,
    ):
        _dibujar_referencias(ax, refs[medida], niveles, escala)
        for i, (valor, (medias, desvios)) in enumerate(curvas[medida].items()):
            ax.errorbar(
                niveles,
                escala * medias,
                yerr=escala * desvios,
                color=SERIES[i],
                marker="o",
                markersize=4,
                capsize=3,
                linewidth=2.5 if valor == ganador else 1.5,
                label=f"latente {valor}" + (" (gana)" if valor == ganador else ""),
            )
        ax.set_xticks(niveles)
        ax.set_xlabel("p de las letras de validación")
        ax.set_title(titulo, loc="left")
    paneles[0].legend(loc="upper left")
    _titulo(
        fig, f"Experimento A: dimensión del latente (validación, media ± desvío, ruido {ruido})"
    )
    return fig


# --- 2. Experimento B: ruido de entrenamiento ---


def fig_ruido(
    curvas: dict[str, tuple], niveles: list[float], ganador: str, n_semillas: int
) -> Figure:
    """Mapa de calor p_train × p_test de los píxeles mal medios en validación."""
    valores = list(curvas)
    matriz = np.array([curvas[v][0] for v in valores])
    fig = Figure(figsize=(7, 4.2), layout="constrained")
    ax = fig.subplots()
    imagen = ax.imshow(matriz, cmap="Blues", aspect="auto")
    ax.grid(visible=False)
    for i, fila in enumerate(matriz):
        for j, v in enumerate(fila):
            oscuro = v > matriz.max() * 0.6
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=SUPERFICIE if oscuro else TINTA)  # fmt: skip
    ax.set_xticks(range(len(niveles)), [f"{p:g}" for p in niveles])
    etiquetas = [f"{v} (gana)" if v == ganador else v for v in valores]
    ax.set_yticks(range(len(valores)), etiquetas)
    ax.set_xlabel("p de las letras de validación")
    ax.set_ylabel("p de entrenamiento")
    fig.colorbar(imagen, ax=ax, label="píxeles mal por letra")
    _titulo(
        fig, f"Experimento B: ruido de entrenamiento (validación, media de {n_semillas} semillas)"
    )
    return fig


# --- 3. Final en test ---


def fig_final(curvas: dict[str, dict], niveles: list[float]) -> Figure:
    """curvas: {medida: {modelo: (medias, desvios)}} en test, con "dae" y las referencias."""
    fig = Figure(figsize=(11, 4.2), layout="constrained")
    paneles = fig.subplots(1, 2)
    for ax, medida, escala, titulo in zip(
        paneles,
        ("pixeles_mal", "recuperadas"),
        (1, 100),
        ("Píxeles mal por letra (promedio)", "Letras recuperadas (≤ 1 píxel mal), %"),
        strict=True,
    ):
        _dibujar_referencias(ax, curvas[medida], niveles, escala)
        medias, desvios = curvas[medida]["dae"]
        ax.errorbar(
            niveles,
            escala * medias,
            yerr=escala * desvios,
            color=SERIES[0],
            marker="o",
            markersize=4,
            capsize=3,
            linewidth=2.5,
            label="DAE elegido",
        )
        if escala == 1:
            for p, m in zip(niveles, medias, strict=True):
                ax.annotate(f"{m:.2f}", (p, m), xytext=(-6, 8), textcoords="offset points",
                            ha="right", color=SERIES[0])  # fmt: skip
        ax.set_xticks(niveles)
        ax.set_xlabel("p de las letras de test")
        ax.set_title(titulo, loc="left")
    paneles[0].legend(loc="upper left")
    _titulo(fig, "Resultado final en test (media ± desvío entre semillas)")
    return fig


# --- 4. Ejemplos ---


def _dibujar(ax, letra: np.ndarray, tinta: str) -> None:
    ax.imshow(letra.reshape(ALTO, ANCHO), cmap=ListedColormap([FONDO_LETRA, tinta]),
              vmin=0, vmax=1, interpolation="nearest")  # fmt: skip
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
    for borde in ax.spines.values():
        borde.set_visible(False)


def fig_ejemplos(
    X: np.ndarray, columnas: list[int], ruidosas: dict, reconstruidas: dict, semilla: int
) -> Figure:
    """Una columna por letra; fila limpia y, por nivel, la ruidosa y la del DAE.

    ruidosas y reconstruidas: {p: (n_letras, 35)} binarias, en el orden de columnas.
    """
    niveles = list(ruidosas)
    filas = 1 + 2 * len(niveles)
    fig = Figure(figsize=(0.75 * len(columnas) + 1.6, 0.95 * filas), layout="constrained")
    ejes = fig.subplots(filas, len(columnas))
    for j, i_letra in enumerate(columnas):
        _dibujar(ejes[0, j], X[i_letra], TINTA_2)
        for k, p in enumerate(niveles):
            _dibujar(ejes[1 + 2 * k, j], ruidosas[p][j], TENUE)
            mal = np.count_nonzero(reconstruidas[p][j] != X[i_letra])
            _dibujar(ejes[2 + 2 * k, j], reconstruidas[p][j], SERIES[0] if mal <= 1 else SERIES[1])
            ejes[2 + 2 * k, j].set_title(f"{mal} mal", fontsize=7, color=TINTA_2, pad=2)
    ejes[0, 0].set_ylabel("limpia", rotation=0, ha="right", va="center")
    for k, p in enumerate(niveles):
        ejes[1 + 2 * k, 0].set_ylabel(f"p = {p:g}", rotation=0, ha="right", va="center")
        ejes[2 + 2 * k, 0].set_ylabel("DAE", rotation=0, ha="right", va="center")
    _titulo(
        fig,
        f"Letras de test con ruido y su reconstrucción (DAE final, semilla {semilla})\n"
        "azul: ≤ 1 píxel mal · naranja: más de 1",
    )
    return fig


# --- 5. Espacio latente del DAE (solo con latente 2) ---


def fig_latente_dae(
    Z_ruidosas: np.ndarray, idx: np.ndarray, Z_limpias: np.ndarray, etiquetas, semilla, p
) -> Figure:
    fig = Figure(figsize=(7.5, 6.5), layout="constrained")
    ax = fig.subplots()
    colores = np.array([SERIES[i % len(SERIES)] for i in range(len(etiquetas))])
    ax.scatter(Z_ruidosas[:, 0], Z_ruidosas[:, 1], s=6, c=colores[idx], alpha=0.35, linewidths=0)
    ax.scatter(Z_limpias[:, 0], Z_limpias[:, 1], s=26, color=TINTA, zorder=3)
    for (z1, z2), e in zip(Z_limpias, etiquetas, strict=True):
        ax.annotate(e, (z1, z2), xytext=(4, 3), textcoords="offset points", color=TINTA, zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    _titulo(
        fig,
        f"Latente del DAE (semilla {semilla}): letras de test con p = {p:g} (color = letra de "
        "origen) y letras limpias (negro)",
    )
    return fig


# --- Todo ---


def generar_figuras(resultados: Path) -> list[Path]:
    """Genera todas las figuras en resultados/figuras/ y devuelve las rutas escritas."""
    figuras = resultados / "figuras"
    figuras.mkdir(exist_ok=True)
    elecciones = json.loads((resultados / "elecciones.json").read_text(encoding="utf-8"))
    config_dae = elecciones["config_dae"]
    niveles = config_dae["niveles"]
    ganadores = {e["id"]: e["ganador"] for e in elecciones["experimentos"]}
    nombres = {e["id"]: e["nombre"] for e in elecciones["experimentos"]}
    filas = {i: leer_filas_dae(resultados / f"experimento_{i}_{nombres[i]}.csv") for i in nombres}
    refs_val = leer_filas_dae(resultados / "referencias_validacion.csv")
    test = leer_filas_dae(resultados / "final" / "test.csv")
    X, etiquetas = cargar_font()
    medidas = ("pixeles_mal", "recuperadas")
    escritas: list[Path] = []

    with matplotlib.rc_context(ESTILO):
        # 1. Latente.
        curvas = {m: curvas_por_valor(filas["A"], niveles, m) for m in medidas}
        refs = {m: curvas_por_modelo(refs_val, niveles, m) for m in medidas}
        ruido_a = elecciones["experimentos"][0]["config"]["ruido"]
        fig = fig_latente(curvas, refs, niveles, ganadores["A"], ruido_a)
        _guardar(fig, figuras / "01_latente.png", escritas)

        # 2. Ruido de entrenamiento.
        fig = fig_ruido(
            curvas_por_valor(filas["B"], niveles, "pixeles_mal"),
            niveles,
            ganadores["B"],
            len(config_dae["semillas"]),
        )
        _guardar(fig, figuras / "02_ruido_entrenamiento.png", escritas)

        # 3. Final en test.
        fig = fig_final({m: curvas_por_modelo(test, niveles, m) for m in medidas}, niveles)
        _guardar(fig, figuras / "03_final_test.png", escritas)

        # 4 y 5. DAE final con la menor semilla.
        config = elecciones["config_final"]
        semilla = min(config_dae["semillas"])
        ae = cargar_modelo(
            config, resultados / "final" / f"pesos_semilla_{semilla}.npz", X.shape[1]
        )
        conjuntos = conjuntos_ruidosos(X, niveles, config_dae["copias"], config_dae["semilla_test"])
        columnas = [etiquetas.index(e) for e in LETRAS_EJEMPLO]
        ruidosas, reconstruidas = {}, {}
        for p in (p for p in niveles if p > 0):
            X_r, _ = conjuntos[p]
            # idx repite 0..31: la fila i es la primera copia ruidosa de la letra i.
            ruidosas[p] = X_r[columnas]
            reconstruidas[p] = (ae.predict(X_r[columnas]) >= 0.5).astype(float)
        fig = fig_ejemplos(X, columnas, ruidosas, reconstruidas, semilla)
        _guardar(fig, figuras / "04_ejemplos.png", escritas)

        if config["latente"] == 2:
            X_r, idx = conjuntos[NIVEL_LATENTE]
            fig = fig_latente_dae(
                ae.encode(X_r), idx, ae.encode(X), etiquetas, semilla, NIVEL_LATENTE
            )
            _guardar(fig, figuras / "05_latente_dae.png", escritas)
    return escritas


def main() -> None:
    parser = argparse.ArgumentParser(description="Figuras del ejercicio 1b.")
    parser.add_argument("--resultados", type=Path, default=RUTA_SALIDA)
    args = parser.parse_args()
    for ruta in generar_figuras(args.resultados):
        print(ruta, flush=True)


if __name__ == "__main__":
    main()
