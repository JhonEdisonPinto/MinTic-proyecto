@echo off
REM Script de setup para Windows (PowerShell recomendado)
echo Creando entorno virtual...
python -m venv .venv
echo Activando entorno virtual...
call .venv\Scripts\activate.bat
echo Instalando dependencias...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Instalando pre-commit hooks...
pre-commit install

echo Listo. Para activar el entorno en PowerShell: .\.venv\Scripts\Activate.ps1
