"""C3 del 1a: tamaño de lote y pérdida, cada uno con su propio lr.

    python -m ej1.verificacion_c3 [--semillas 5]

No cambia las elecciones del barrido. Repite los pasos 9 y 10 con la config
vigente cuando se eligieron, probando una grilla de lr alrededor de cada valor,
y compara por actualizaciones hasta llegar (no por épocas). Para el lote 1 se
corta en 50.000 actualizaciones. Escribe en
resultados/ej1/verificacion/c3/: un CSV por paso y c3.json.
"""

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

from ej1.barrido import (
    RUTA_CONFIG,
    RUTA_SALIDA,
    _id_archivo,
    _media_desvio,
    aplicar,
    entrenar,
    escribir_filas,
    leer_filas,
)
from ej1.datos import cargar_font
from ej1.verificacion import config_del_paso

PASOS = ("9", "10")
MAX_ACTUALIZACIONES_LOTE_1 = 50_000
LOTE_9 = [32, 8, 4, 1]
LR_9 = [0.001, 0.003, 0.01]
PERDIDAS_10 = ["binary_crossentropy", "mse"]
LR_10 = [0.003, 0.01, 0.03, 0.1]
NOMBRE_PERDIDA = {"binary_crossentropy": "bce", "mse": "mse"}


def combinaciones(paso_id: str) -> list[dict]:
    """Valores de la grilla de C3 para el paso 9 o el 10."""
    if paso_id == "9":
        return [
            {"valor": f"{lote}/{lr:g}", "cambios": {"batch_size": lote, "lr": lr}}
            for lote in LOTE_9
            for lr in LR_9
        ]
    if paso_id == "10":
        return [
            {
                "valor": f"{NOMBRE_PERDIDA[perdida]}/{lr:g}",
                "cambios": {"perdida": perdida, "lr": lr},
            }
            for perdida in PERDIDAS_10
            for lr in LR_10
        ]
    raise ValueError(f"Paso sin grilla de C3: {paso_id!r}")


def max_epocas(config: dict) -> int:
    """Tope de épocas: solo el lote 1 se corta en 50.000 actualizaciones."""
    maximo = int(config["max_epocas"])
    lote = int(config["batch_size"])
    if lote == 1:
        lotes = math.ceil(32 / lote)
        maximo = min(maximo, MAX_ACTUALIZACIONES_LOTE_1 // lotes)
    return maximo


def resumir_c3(filas: list[dict]) -> dict[str, dict]:
    """Por valor: % de semillas que llegan y actualizaciones medias de las que llegan."""
    grupos: dict[str, list[dict]] = {}
    for fila in filas:
        grupos.setdefault(fila["valor"], []).append(fila)
    resumen = {}
    for valor, grupo in grupos.items():
        llegadas = [f["actualizaciones"] for f in grupo if f["llego"]]
        actualizaciones = _media_desvio(llegadas)
        epocas = _media_desvio([f["epocas"] for f in grupo])
        peor = _media_desvio([f["peor_letra"] for f in grupo])
        pixeles = _media_desvio([f["pixeles_mal"] for f in grupo])
        resumen[valor] = {
            "corridas": len(grupo),
            "llegan": len(llegadas),
            "pct_llegan": 100.0 * len(llegadas) / len(grupo),
            "actualizaciones_media": actualizaciones[0],
            "actualizaciones_desvio": actualizaciones[1],
            "epocas_media": epocas[0],
            "epocas_desvio": epocas[1],
            "peor_letra_media": peor[0],
            "peor_letra_desvio": peor[1],
            "pixeles_mal_media": pixeles[0],
            "pixeles_mal_desvio": pixeles[1],
        }
    return resumen


def elegir_actualizaciones(resumen: dict[str, dict], actual: str | None) -> tuple[str, str]:
    """Ganador por actualizaciones hasta llegar, no por épocas.

    1. Más % de semillas que llegan. 2. Si empatan: menos actualizaciones medias
    hasta llegar (solo las semillas que llegaron). 3. Si ninguno llega: menor
    peor letra media. 4. Si siguen empatados: el valor actual o el primero.
    """
    mejor = max(r["pct_llegan"] for r in resumen.values())
    candidatos = [v for v, r in resumen.items() if r["pct_llegan"] == mejor]
    if len(candidatos) == 1:
        return candidatos[0], "llegan"
    clave, criterio = (
        ("actualizaciones_media", "actualizaciones")
        if mejor > 0
        else ("peor_letra_media", "peor_letra")
    )
    minimo = min(resumen[v][clave] for v in candidatos)
    candidatos = [v for v in candidatos if resumen[v][clave] == minimo]
    if len(candidatos) == 1:
        return candidatos[0], criterio
    return (actual if actual in candidatos else candidatos[0]), "empate"


def mejor_por_nivel(resumen: dict[str, dict]) -> dict[str, dict]:
    """Mejor lr dentro de cada lote (paso 9) o de cada pérdida (paso 10)."""
    grupos: dict[str, dict[str, dict]] = {}
    for valor, r in resumen.items():
        nivel = valor.split("/")[0]
        grupos.setdefault(nivel, {})[valor] = r
    return {
        nivel: {
            "ganador": elegir_actualizaciones(sub, None)[0],
            "criterio": elegir_actualizaciones(sub, None)[1],
            "resumen": sub,
        }
        for nivel, sub in grupos.items()
    }


def _conclusion(resultado: dict, elecciones: dict) -> dict:
    """Resume si la conclusión original se sostiene con el mejor lr de cada valor."""
    conclusion = {}
    if "9" in resultado:
        original_9 = next(p["ganador"] for p in elecciones["pasos"] if p["id"] == "9")
        por_lote = resultado["9"]["mejor_por_nivel"]
        conclusion["paso_9"] = {
            "ganador_original": original_9,
            "mejor_por_lote": {
                lote: {
                    "ganador": datos["ganador"],
                    "llegan": datos["resumen"][datos["ganador"]]["llegan"],
                    "corridas": datos["resumen"][datos["ganador"]]["corridas"],
                    "actualizaciones_media": datos["resumen"][datos["ganador"]][
                        "actualizaciones_media"
                    ],
                }
                for lote, datos in por_lote.items()
            },
        }
    if "10" in resultado:
        original_10 = next(p["ganador"] for p in elecciones["pasos"] if p["id"] == "10")
        por_perdida = resultado["10"]["mejor_por_nivel"]
        conclusion["paso_10"] = {
            "ganador_original": original_10,
            "mejor_por_perdida": {
                perdida: {
                    "ganador": datos["ganador"],
                    "llegan": datos["resumen"][datos["ganador"]]["llegan"],
                    "corridas": datos["resumen"][datos["ganador"]]["corridas"],
                    "actualizaciones_media": datos["resumen"][datos["ganador"]][
                        "actualizaciones_media"
                    ],
                }
                for perdida, datos in por_perdida.items()
            },
        }
    return conclusion


def verificar_c3(
    config_barrido: dict,
    elecciones: dict,
    salida: Path,
    semillas: list[int],
    pasos: tuple[str, ...] = PASOS,
    max_epocas_tope: int | None = None,
    escribir: Callable[[str], None] = print,
) -> dict:
    """Corre (o retoma) C3, escribe un CSV por paso y c3.json."""
    X, etiquetas = cargar_font()
    salida.mkdir(parents=True, exist_ok=True)
    objetivo = config_barrido["objetivo_peor_letra"]
    resultado: dict[str, dict] = {}
    for paso_id in pasos:
        paso, config = config_del_paso(elecciones, paso_id)
        ruta = salida / f"paso_{_id_archivo(paso['id'])}_{paso['nombre']}.csv"
        if ruta.exists():
            escribir(f"[{paso_id}] ya existe {ruta.name}: se lee en vez de correr.")
        else:
            filas = []
            for combo in combinaciones(paso_id):
                config_valor = aplicar(config, combo["cambios"])
                tope = max_epocas(config_valor)
                if max_epocas_tope is not None:
                    tope = min(tope, max_epocas_tope)
                config_valor["max_epocas"] = tope
                for semilla in semillas:
                    _, _, res = entrenar(config_valor, X, semilla, objetivo)
                    filas.append(
                        {
                            "paso": paso_id,
                            "valor": combo["valor"],
                            "semilla": semilla,
                            **res,
                            "config": config_valor,
                            "errores": [int(e) for e in res["errores"]],
                        }
                    )
                escribir(
                    f"[{paso_id}] {combo['valor']}: {len(semillas)} semillas ({tope} épocas máximo)"
                )
            escribir_filas(ruta, filas, etiquetas)
        resumen = resumir_c3(leer_filas(ruta))
        ganador, criterio = elegir_actualizaciones(resumen, None)
        resultado[paso_id] = {
            "semillas": len(semillas),
            "ganador": ganador,
            "criterio": criterio,
            "mejor_por_nivel": mejor_por_nivel(resumen),
            "resumen": resumen,
        }
        r = resumen[ganador]
        escribir(
            f"[{paso_id}] gana {ganador} ({criterio}): {r['pct_llegan']:.0f} % llegan, "
            f"{r['actualizaciones_media']} actualizaciones de media"
        )
    completo = {"resultado": resultado, "conclusion": _conclusion(resultado, elecciones)}
    texto = json.dumps(completo, indent=2, ensure_ascii=False)
    (salida / "c3.json").write_text(texto, encoding="utf-8")
    return json.loads(texto)


def main() -> None:
    parser = argparse.ArgumentParser(description="C3 del 1a: lote y pérdida con su propio lr.")
    parser.add_argument("--semillas", type=int, default=5)
    args = parser.parse_args()
    config = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
    elecciones = json.loads((RUTA_SALIDA / "elecciones.json").read_text(encoding="utf-8"))
    verificar_c3(
        config,
        elecciones,
        RUTA_SALIDA / "verificacion" / "c3",
        list(range(args.semillas)),
        escribir=lambda texto: print(texto, flush=True),
    )


if __name__ == "__main__":
    main()
