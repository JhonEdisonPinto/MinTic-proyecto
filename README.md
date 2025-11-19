# MinTIC - Proyecto de Analítica de Datos Abiertos

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-ready-brightgreen.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Proyecto colaborativo (equipo 3-4 personas) para analizar datasets abiertos del gobierno colombiano (datos.gov.co).

Descripción
- Stack: Python, Pandas, Streamlit/Flask, LangChain
- Volumen objetivo: 50,000+ registros
- Deploy objetivo: Streamlit Cloud o Render

Características
- Código modular empaquetado como paquete Python en `src/mintic_project`.
- App de visualización en `app/streamlit_app.py`.
- Notebooks para exploración en `notebooks/`.
- CI: GitHub Actions para lint y tests.

Setup rápido
1. Clonar el repositorio:
	```powershell
	git clone <repo-url>
	cd MinTic-proyecto
	```
2. Copiar variables de entorno y editar:
	```powershell
	copy .env.example .env
	# editar .env con la URL del dataset y claves
	```
3. Usar script de plataforma:
	- Windows (PowerShell): `setup.bat`
	- Mac/Linux: `bash setup.sh`

Instalación manual (opcional)
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1  # PowerShell
pip install -r requirements.txt
pre-commit install
```

Estructura del proyecto
- `app/streamlit_app.py`: Entrada de la app Streamlit.
- `src/mintic_project/`: Código fuente empaquetado.
- `notebooks/`: Notebooks Jupyter para análisis exploratorio.
- `tests/`: Pruebas unitarias con `pytest`.
- `.github/workflows/ci.yml`: CI para lint y tests.

Guía de contribución
- Leer `CONTRIBUTING.md` para el flujo de trabajo Git.
- Usar `black` y `flake8` localmente antes de enviar PRs.

Licencia
MIT

