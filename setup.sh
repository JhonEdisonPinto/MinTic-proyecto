#!/usr/bin/env bash
# Script de setup para Mac/Linux
# Ejecutar: `bash setup.sh` (o `./setup.sh` tras dar permisos)

set -e

echo "Creando entorno virtual..."
python3 -m venv .venv
echo "Activando entorno virtual..."
source .venv/bin/activate
echo "Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Instalando hooks de pre-commit..."
pre-commit install

echo "Listo. Activar el entorno con: source .venv/bin/activate"
