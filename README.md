# MinTIC - Proyecto de Analítica de Datos Abiertos

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-ready-brightgreen.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-blue.svg)](https://github.com/features/actions)

## 📋 Descripción

Proyecto colaborativo (equipo 3-4 personas) para analizar datasets abiertos del gobierno colombiano (datos.gov.co).

**Stack tecnológico:**
- Python 3.11+
- Pandas (manipulación de datos)
- Streamlit (visualización interactiva)
- Flask (API REST opcional)
- LangChain (integración con LLMs)
- Jupyter (exploración y análisis)

**Objetivos:**
- Procesar 50,000+ registros de datos.gov.co
- Trabajo colaborativo con Git
- Deploy automatizado (Streamlit Cloud / Render)

## ✨ Características

- ✅ Código modular empaquetado como paquete Python en `src/mintic_project/`
- ✅ App de visualización con Streamlit en `app/streamlit_app.py`
- ✅ Notebooks Jupyter para exploración en `notebooks/`
- ✅ CI/CD automático con GitHub Actions
- ✅ Pre-commit hooks para asegurar calidad de código (Black, Flake8)
- ✅ Tests unitarios con PyTest

## 🚀 Setup rápido

### Opción 1: Scripts automáticos (recomendado)

**Windows (PowerShell):**
```powershell
# 1. Navega al directorio del proyecto
cd C:\Users\Jhon\Documents\GitHub\MinTic-proyecto

# 2. Ejecuta el script de setup
.\setup.bat

# 3. Copia y edita las variables de entorno
copy .env.example .env

# 4. Activa el entorno (ya debería estar activado)
.\.venv\Scripts\Activate.ps1

# 5. Ejecuta la app Streamlit
streamlit run app\streamlit_app.py
```

**Mac/Linux (Bash):**
```bash
cd ~/GitHub/MinTic-proyecto
bash setup.sh
cp .env.example .env
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

### Opción 2: Setup manual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pre-commit install
copy .env.example .env
```

## 📁 Estructura del proyecto

```
MinTic-proyecto/
├── app/
│   └── streamlit_app.py          # Aplicación Streamlit principal
├── src/
│   └── mintic_project/
│       ├── __init__.py
│       └── processor.py           # Funciones de procesamiento
├── notebooks/
│   └── 01_exploracion.ipynb       # Exploración con Jupyter
├── tests/
│   └── test_processor.py          # Tests unitarios
├── .github/
│   ├── workflows/ci.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE/
├── .vscode/settings.json
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── requirements.txt
├── setup.py
├── setup.sh
├── setup.bat
├── CONTRIBUTING.md
└── README.md
```

## 🧪 Desarrollo local

### Formateo y linting

```powershell
black .                    # Formatear con Black
flake8 src tests          # Linting con Flake8
black --check .           # Verificar sin modificar
```

### Ejecutar tests

```powershell
pytest                    # Tests básicos
pytest -v               # Verbose
pytest --cov=src        # Con coverage
```

### Instalar en modo desarrollo

```powershell
pip install -e .
```

## 🔧 Variables de entorno

Crear `.env` (copiar de `.env.example`):

```env
OPENAI_API_KEY="tu-clave-aqui"
DATA_SOURCE_URL="https://www.datos.gov.co/resource/xxxx-xxxx.csv"
STREAMLIT_SERVER_PORT=8501
ENV=development
LOG_LEVEL=INFO
```

## 📝 Guía de contribución

Ver [`CONTRIBUTING.md`](CONTRIBUTING.md) para:
- Flujo Git (ramas, commits, PRs)
- Estándares de código
- Proceso de revisión

**Resumen rápido:**
1. `git checkout -b feat/mi-caracteristica`
2. Hacer cambios
3. `black .` y `flake8 src tests`
4. `pytest`
5. Push y abrir PR

## 🌐 Deploy

### Streamlit Cloud
1. Push a GitHub
2. Conectar repo en Streamlit Cloud
3. Seleccionar `app/streamlit_app.py`
4. Agregar secrets en Settings

### Render.com
1. Conectar GitHub
2. Build: `pip install -r requirements.txt`
3. Start: `streamlit run app/streamlit_app.py --server.port=$PORT`
4. Agregar env vars

## 📚 Recursos

- [Streamlit Docs](https://docs.streamlit.io/)
- [Pandas](https://pandas.pydata.org/docs/)
- [LangChain](https://python.langchain.com/)
- [datos.gov.co](https://www.datos.gov.co/)
- [Black](https://black.readthedocs.io/)
- [Flake8](https://flake8.pycqa.org/)

## 📄 Licencia

MIT License

## 👥 Equipo

Proyecto colaborativo MinTIC (3-4 personas)

