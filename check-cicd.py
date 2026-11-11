#!/usr/bin/env python3
"""Ejecuta localmente los mismos chequeos que el CI de GitHub.

El script se puede correr con el Python del sistema o con el del entorno
virtual. Si hace falta, crea `.venv`, se vuelve a ejecutar dentro de él,
actualiza las dependencias cuando cambia `requirements.txt` y corre:

1. `pip check`
2. `ruff check .`
3. `ruff format --check .`
4. `pytest -q`
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
RUTA_VENV = RAIZ / ".venv"
RUTA_REQUISITOS = RAIZ / "requirements.txt"
SELLO_REQUISITOS = RUTA_VENV / ".requirements.sha256"
PYTHON_MINIMO = (3, 11)
MODULOS_REQUERIDOS = ("numpy", "pytest", "ruff")


def python_venv() -> Path:
    """Devuelve la ruta al Python del entorno virtual."""
    if os.name == "nt":
        return RUTA_VENV / "Scripts" / "python.exe"
    return RUTA_VENV / "bin" / "python"


def bin_venv() -> Path:
    """Devuelve la carpeta de ejecutables del entorno virtual."""
    return python_venv().parent


def en_venv() -> bool:
    """Indica si el proceso actual corre dentro de `.venv`."""
    return Path(sys.prefix).resolve() == RUTA_VENV.resolve()


def hash_requisitos() -> str:
    """Calcula el hash del archivo de dependencias."""
    return hashlib.sha256(RUTA_REQUISITOS.read_bytes()).hexdigest()


def dependencias_al_dia() -> bool:
    """Comprueba el sello de requirements y que los módulos estén instalados."""
    if not SELLO_REQUISITOS.exists():
        return False
    if SELLO_REQUISITOS.read_text(encoding="utf-8").strip() != hash_requisitos():
        return False
    return all(importlib.util.find_spec(modulo) is not None for modulo in MODULOS_REQUERIDOS)


def ejecutar(comando: list[str]) -> None:
    """Ejecuta un comando desde la raíz del repositorio."""
    print(f"$ {' '.join(comando)}")
    subprocess.run(comando, cwd=RAIZ, check=True)


def crear_venv() -> None:
    """Crea el entorno virtual si no existe."""
    if RUTA_VENV.exists() and not python_venv().exists():
        raise RuntimeError(
            "La carpeta .venv existe pero no tiene un Python válido. "
            "Borrala y volvé a ejecutar el script."
        )

    if not RUTA_VENV.exists():
        if sys.version_info < PYTHON_MINIMO:
            raise RuntimeError("Se necesita Python 3.11 o superior para crear el entorno virtual.")
        print("No existe .venv. Creando el entorno virtual...")
        subprocess.run(
            [sys.executable, "-m", "venv", str(RUTA_VENV)],
            cwd=RAIZ,
            check=True,
        )


def instalar_requisitos() -> None:
    """Actualiza pip e instala las versiones fijadas en requirements.txt."""
    print("Actualizando pip...")
    ejecutar([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    print("Instalando requirements.txt...")
    ejecutar(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "-r",
            str(RUTA_REQUISITOS),
        ]
    )

    SELLO_REQUISITOS.write_text(hash_requisitos() + "\n", encoding="utf-8")


def ejecutar_en_venv() -> int:
    """Crea el venv si hace falta y ejecuta el script dentro de él."""
    crear_venv()

    entorno = os.environ.copy()
    entorno["VIRTUAL_ENV"] = str(RUTA_VENV)
    entorno["PATH"] = str(bin_venv()) + os.pathsep + entorno.get("PATH", "")

    print("Ejecutando los chequeos dentro de .venv...")
    resultado = subprocess.run(
        [
            str(python_venv()),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        cwd=RAIZ,
        env=entorno,
        check=False,
    )
    return resultado.returncode


def ejecutar_checks() -> int:
    """Instala dependencias si hace falta y corre los chequeos del CI."""
    if sys.version_info < PYTHON_MINIMO:
        raise RuntimeError("Se necesita Python 3.11 o superior dentro de .venv.")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Entorno: {sys.prefix}")

    if not dependencias_al_dia():
        instalar_requisitos()
    else:
        print("Las dependencias ya están al día.")

    comandos = [
        ("Dependencias", [sys.executable, "-m", "pip", "check"]),
        ("Lint", [sys.executable, "-m", "ruff", "check", "."]),
        ("Formato", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("Tests", [sys.executable, "-m", "pytest", "-q"]),
    ]

    for nombre, comando in comandos:
        print(f"\n== {nombre} ==")
        try:
            ejecutar(comando)
        except subprocess.CalledProcessError as error:
            print(f"\nFALLÓ: {nombre} (código {error.returncode}).")
            return error.returncode or 1

    print("\nTodo en verde: lint, formato y tests.")
    return 0


def main() -> int:
    """Punto de entrada del script."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    try:
        if not en_venv():
            return ejecutar_en_venv()
        return ejecutar_checks()
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
