import argparse
import csv
import struct
import zlib
from pathlib import Path


def escribir_png(ruta, ancho, alto, filas):
    """Escribe un PNG RGB de 8 bits sin dependencias externas."""
    def chunk(tipo, datos):
        return struct.pack(">I", len(datos)) + tipo + datos + struct.pack(">I", zlib.crc32(tipo + datos))

    crudo = b"".join(b"\x00" + fila for fila in filas)
    ruta.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", ancho, alto, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(crudo, 9))
        + chunk(b"IEND", b"")
    )


def generar(fila, ancho, salida):
    nombre, *resto = fila
    # Los colores vienen como triples separados por comas; unirlos tolera el ';' de más.
    numeros = list(map(int, ",".join(resto[:-3]).split(",")))
    colores = [bytes(numeros[i:i + 3]) for i in range(0, len(numeros), 3)]
    rotada, proporcion, aspecto = int(resto[-3]), float(resto[-2]), float(resto[-1])

    alto = round(ancho / aspecto)
    largo = ancho if rotada else alto
    fino = largo / (2 + proporcion)

    bordes = [0, round(fino), round(fino * (1 + proporcion)), largo]

    if rotada:
        franja = b"".join(colores[i] * (bordes[i + 1] - bordes[i]) for i in range(3))
        filas = [franja] * alto
    else:
        filas = [colores[i] * ancho for i in range(3) for _ in range(bordes[i + 1] - bordes[i])]

    escribir_png(salida / f"{nombre}.png", ancho, alto, filas)


def main():
    parser = argparse.ArgumentParser(description="Genera banderas PNG a partir de un CSV.")
    parser.add_argument("csv", nargs="?", default="data/flags_tricolor.csv")
    parser.add_argument("--ancho", type=int, default=200)
    parser.add_argument("--salida", type=Path, default=Path("data/banderas"))
    args = parser.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    with open(args.csv, newline="", encoding="utf-8") as archivo:
        lector = csv.reader(archivo, delimiter=";")
        next(lector)  # encabezado
        for fila in lector:
            generar(fila, args.ancho, args.salida)


if __name__ == "__main__":
    main()
