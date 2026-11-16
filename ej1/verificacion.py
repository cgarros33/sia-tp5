"""Repite pasos del barrido del 1a con más semillas, sin cambiar las elecciones.

    python -m ej1.verificacion [--semillas 20]

Cada paso se corre con la config vigente cuando se eligió (la de elecciones.json) y
"final" es la repetición del paso 3 con la config final. Escribe en
resultados/ej1/verificacion/: un CSV por paso y verificacion.json. Si el CSV de un
paso existe, se lee en vez de correr.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from ej1.barrido import (
    RUTA_CONFIG,
    RUTA_SALIDA,
    _id_archivo,
    aplicar,
    entrenar,
    escribir_filas,
    leer_filas,
    resumir,
)
from ej1.datos import cargar_font

PASOS = ("5", "final")


def config_del_paso(elecciones: dict, paso_id: str) -> tuple[dict, dict]:
    """(paso de config_barrido, config vigente antes de elegirlo)."""
    pasos = elecciones["config_barrido"]["pasos"]
    if paso_id == "final":
        repetido = elecciones["verificacion_final"]["paso"]
        return next(p for p in pasos if p["id"] == repetido), elecciones["config_final"]
    indice = next(i for i, p in enumerate(pasos) if p["id"] == paso_id)
    if indice == 0:
        return pasos[0], elecciones["config_barrido"]["base"]
    return pasos[indice], elecciones["pasos"][indice - 1]["config"]


def empatan_con_el_ganador(resumen: dict[str, dict], ganador: str) -> list[str]:
    """Valores que llegan tanto como el ganador y a menos de un desvío de sus épocas."""
    g = resumen[ganador]
    return [
        v
        for v, r in resumen.items()
        if r["pct_llegan"] == g["pct_llegan"]
        and r["epocas_media"] is not None
        and abs(r["epocas_media"] - g["epocas_media"]) <= g["epocas_desvio"]
    ]


def verificar(
    config_barrido: dict,
    elecciones: dict,
    salida: Path,
    semillas: list[int],
    pasos: tuple[str, ...] = PASOS,
    escribir: Callable[[str], None] = print,
) -> dict:
    X, etiquetas = cargar_font()
    salida.mkdir(parents=True, exist_ok=True)
    objetivo = config_barrido["objetivo_peor_letra"]
    resultado = {}
    for paso_id in pasos:
        paso, config = config_del_paso(elecciones, paso_id)
        prefijo = "final" if paso_id == "final" else _id_archivo(paso["id"])
        ruta = salida / f"paso_{prefijo}_{paso['nombre']}.csv"
        if ruta.exists():
            escribir(f"[{paso_id}] ya existe {ruta.name}: se lee en vez de correr.")
        else:
            filas = []
            for valor in paso["valores"]:
                config_valor = aplicar(config, valor["cambios"])
                for semilla in semillas:
                    _, _, res = entrenar(config_valor, X, semilla, objetivo)
                    filas.append(
                        {
                            "paso": paso_id,
                            "valor": valor["etiqueta"],
                            "semilla": semilla,
                            **res,
                            "config": config_valor,
                            "errores": [int(e) for e in res["errores"]],
                        }
                    )
                escribir(f"[{paso_id}] {valor['etiqueta']}: {len(semillas)} semillas")
            escribir_filas(ruta, filas, etiquetas)
        resumen = resumir(leer_filas(ruta))
        if paso_id == "final":
            ganador = elecciones["verificacion_final"]["ganador"]
        else:
            ganador = next(p["ganador"] for p in elecciones["pasos"] if p["id"] == paso_id)
        resultado[paso_id] = {
            "semillas": len(semillas),
            "ganador_original": ganador,
            "empatan_con_el_ganador": empatan_con_el_ganador(resumen, ganador),
            "resumen": resumen,
        }
    texto = json.dumps(resultado, indent=2, ensure_ascii=False)
    (salida / "verificacion.json").write_text(texto, encoding="utf-8")
    return json.loads(texto)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verificación con más semillas del 1a.")
    parser.add_argument("--semillas", type=int, default=20)
    args = parser.parse_args()
    config = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
    elecciones = json.loads((RUTA_SALIDA / "elecciones.json").read_text(encoding="utf-8"))
    verificar(
        config,
        elecciones,
        RUTA_SALIDA / "verificacion",
        list(range(args.semillas)),
        escribir=lambda texto: print(texto, flush=True),
    )


if __name__ == "__main__":
    main()
