# Sistema de Predicción de Siniestros Viales - MinTIC

## 📊 Descripción del Sistema

Este sistema procesa datos abiertos de siniestros viales en Palmira desde **datos.gov.co**, limpia los datos, realiza ingeniería de características y entrena modelos predictivos con integración a LangChain para crear un **sistema multiagente de predicción**.

## 🎯 Objectivos

1. ✅ Descargar datos de dos APIs públicas de datos.gov.co
2. ✅ Limpiar y validar datos (2000+ registros)
3. ✅ Ingeniería de características para predicción
4. ✅ Entrenar modelos de ML (Random Forest, etc.)
5. ✅ Integración con LangChain para RAG (Retrieval-Augmented Generation)
6. ✅ Sistema multiagente para análisis y predicción

## 📁 Estructura de Archivos

```
MinTic-proyecto/
├── src/mintic_project/
│   ├── data_loader.py          # Descarga y limpieza de datos
│   ├── feature_engineering.py  # Ingeniería de características
│   └── processor.py            # Utilidades de procesamiento
├── notebooks/
│   ├── 01_exploracion.ipynb                  # Setup inicial
│   ├── 02_analisis_siniestros.ipynb          # EDA y visualizaciones
│   └── 03_multiagente_langchain.ipynb        # Predicción con ML + LangChain
├── scripts/
│   └── descargar_datos.py      # Script ejecutable para descargar datos
├── tests/
│   └── test_data_loader.py     # Tests unitarios
└── data/                       # Directorio de salida (se crea automáticamente)
    ├── siniestros_1_limpio.csv
    ├── siniestros_2_limpio.csv
    ├── siniestros_procesados.parquet
    ├── reporte_limpieza.txt
    └── contexto_rag.json
```

## 🚀 Quick Start

### 1. Instalar dependencias

```powershell
# En Windows (PowerShell)
.\setup.bat

# O manualmente
pip install -r requirements.txt
pip install pytest scikit-learn
```

### 2. Descargar y limpiar datos

```powershell
# Opción A: Usar el script
python scripts/descargar_datos.py

# Opción B: Desde el notebook
# Abrir notebooks/02_analisis_siniestros.ipynb y ejecutar
```

### 3. Analizar datos

```powershell
# Iniciar Jupyter
jupyter notebook notebooks/02_analisis_siniestros.ipynb
```

### 4. Entrenar modelo predictivo

```powershell
# Abrir el notebook de LangChain
jupyter notebook notebooks/03_multiagente_langchain.ipynb
```

## 📋 Fuentes de Datos

### Dataset 1: Siniestros Viales Básicos
- **URL**: `https://www.datos.gov.co/resource/sjpx-eqfp.json`
- **Columnas**: a_o, ipat, clase_siniestro, fecha, hora, jornada, dia_semana, barrios, dirección, zona, autoridad, lat, long, hipotesis, código, condición_víctima, edad, género, lesionados_muertos
- **Registros**: ~50,000+

### Dataset 2: Siniestros - Gravedad y Víctimas
- **URL**: `https://www.datos.gov.co/resource/xx6f-f84h.json`
- **Columnas**: gravedad, fecha, a_o, hora, jornada, dia_semana, barrios, dirección, zona, autoridad, lat, long, condición_víctima, clase_siniestro, género, lesionado, homicidios, clínica, clase_vehículo, marca, tipo_servicio, empresa
- **Registros**: ~2000+

## 🧹 Proceso de Limpieza de Datos

El módulo `data_loader.py` ejecuta los siguientes pasos:

### 1. **Eliminación de Duplicados**
   - Elimina filas completamente duplicadas

### 2. **Validación de Tipos de Datos**
   - Convierte `fecha` a datetime
   - Convierte columnas numéricas a float

### 3. **Limpieza de Valores Nulos**
   - Elimina filas sin información crítica (fecha, año, localización)
   - Reporta % de nulos por columna

### 4. **Eliminación de Outliers**
   - Edades válidas: 0-120
   - Coordenadas geográficas: dentro de Palmira (~-3.5° a -4.0° S, -76.2° a -76.5° O)

### 5. **Estandarización de Texto**
   - Elimina espacios extras
   - Convierte categorías a mayúsculas (GÉNERO, JORNADA, ZONA, etc.)

### 6. **Validación de Rangos**
   - Horas: 0-23
   - Validación lógica de campos

## 🔬 Feature Engineering

El módulo `feature_engineering.py` crea características avanzadas:

### 1. **Features Temporales**
   - `mes`: Mes del año (1-12)
   - `trimestre`: Trimestre (1-4)
   - `semana_ano`: Semana del año
   - `periodo_dia`: MANANA, TARDE, NOCHE, MADRUGADA

### 2. **Features Geográficos**
   - `distancia_centro`: Distancia euclidiana al centro de Palmira
   - `en_centro`: Binario (1=dentro del centro, 0=fuera)

### 3. **Features Categóricos Codificados**
   - Label encoding de: jornada, día_semana, género, zona, clase_siniestro, gravedad
   - Columnas: `{nombre}_encoded`

### 4. **Features de Interacción**
   - `hora_jornada_interaction`: Interacción hora × período del día
   - `genero_edad_interaction`: Interacción género × edad

### 5. **Normalización**
   - StandardScaler para features numéricos
   - Columnas: `{nombre}_normalized`

## 🤖 Integración con LangChain

### Contexto para RAG (Retrieval-Augmented Generation)

El sistema genera contextos estructurados para usar con LangChain:

```python
contexto_rag = {
    "general": "Dataset info (registros, período, columnas)",
    "jornada": "Distribución por jornada",
    "dia_semana": "Distribución por día de semana",
    "genero": "Distribución por género",
    "gravedad": "Distribución por gravedad",
    "edad": "Estadísticas de edad",
}
```

### Ejemplo de Uso

```python
from langchain.llms import OpenAI
from mintic_project.feature_engineering import DatasetPredictor

# Cargar datos procesados
predictor = DatasetPredictor()
df_proc = predictor.preparar_dataset_completo(df1, df2)
contexto = predictor.generar_contexto_rag(df_proc)

# Usar con LangChain
llm = OpenAI(api_key="tu-clave")
prompt = f"""
Contexto: {contexto['general']}
Distribución por jornada: {contexto['jornada']}

Pregunta: ¿Cuándo ocurren más siniestros en Palmira?
"""
respuesta = llm(prompt)
print(respuesta)
```

## 📊 Modelos de ML

El notebook `03_multiagente_langchain.ipynb` entrena:

### Random Forest Classifier
- **Target**: Gravedad del siniestro (leve, moderado, grave, etc.)
- **Features**: Características temporales, geográficas, categóricas
- **Métrica**: Accuracy

### Importancia de Features
El modelo identifica qué variables son más predictivas:
- Jornada y hora
- Día de la semana
- Zona geográfica
- Género y edad

## 🧪 Tests

Ejecutar tests unitarios:

```powershell
# Tests básicos
pytest tests/test_data_loader.py -v

# Con cobertura
pytest tests/test_data_loader.py --cov=src --cov-report=html
```

## 📈 Ejemplo de Output

```
Reporte de Limpieza de Datos
==============================================================

Dataset 1 (Siniestros Viales):
  Registros iniciales: 50000
  Registros finales:   45200
  Registros eliminados: 4800
  % Retenido: 90.4%

Dataset 2 (Gravedad/Víctimas):
  Registros iniciales: 50000
  Registros finales:   46100
  Registros eliminados: 3900
  % Retenido: 92.2%

==============================================================
```

## 🔐 Variables de Entorno

Crear archivo `.env`:

```env
# Para LangChain + OpenAI (opcional)
OPENAI_API_KEY="sk-..."

# URLs de APIs (ya configuradas por defecto)
DATA_SOURCE_URL="https://www.datos.gov.co/resource/sjpx-eqfp.json"
```

## 🚦 Próximos Pasos

1. **Sistema Multiagente Completo**
   - Crear agentes especializados (análisis temporal, geográfico, predicción)
   - Integrar con LangChain Agents

2. **RAG con Normas de Tránsito**
   - Incorporar documento con Código Nacional de Tránsito
   - Vectorizar con embeddings
   - Responder preguntas normativas sobre siniestros

3. **API REST**
   - Crear endpoints con Flask para consultas de predicción
   - Integrar con Streamlit Cloud

4. **Dashboard Interactivo**
   - Visualizaciones en tiempo real
   - Filtros por zona, jornada, período

## 🤝 Contribuir

Ver [`CONTRIBUTING.md`](../CONTRIBUTING.md) para detalles.

## 📚 Referencias

- [datos.gov.co](https://www.datos.gov.co/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [LangChain Docs](https://python.langchain.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://docs.streamlit.io/)

## 📝 Licencia

MIT License
