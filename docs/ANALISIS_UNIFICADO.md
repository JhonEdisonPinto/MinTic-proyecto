# 📊 Análisis Unificado: OCR + CSV + Gemini

## ¿Qué es esto?

Sistema integrado para analizar datos de siniestros viales combinando:
- **OCR**: Extrae texto de documentos legales (PDF)
- **CSV Analytics**: Analiza metadatos y estadísticas de datos
- **Gemini API**: Responde preguntas inteligentes combinando ambas fuentes

---

## 🗂️ Estructura de módulos

### `src/mintic_project/langchain_integration.py`
- `extract_text_from_pdf_ocr()` — Extrae texto de PDFs con OCR
- `answer_with_ocr()` — Responde preguntas sobre PDFs
- `OCRAnalyzer` — Clase para análisis interactivo de PDFs
- `LangChainConfig` — Configuración de Gemini

### `src/mintic_project/db_analysis.py`
- `load_csv_dataset()` — Carga CSV
- `extract_dataset_metadata()` — Extrae columnas, tipos, estadísticas
- `generate_dataset_report()` — Genera reporte textual para Gemini
- `query_dataset_with_gemini()` — Responde preguntas sobre CSV
- `analyze_csv_file()` — Función principal para análisis de CSV

### `src/mintic_project/unified_analyzer.py`
- `UnifiedAnalyzer` — **CLASE PRINCIPAL** que combina OCR + CSV + Gemini
  - `responder_pregunta()` — Responde usando PDF + CSV
  - `generar_resumen_general()` — Resumen ejecutivo
  - `responder_multiples_preguntas()` — Batch de preguntas

---

## 🚀 Cómo usar

### Opción 1: Script Demo (recomendado para empezar)

```powershell
python demo_analysis.py
```

Ejecuta un análisis completo con ejemplos:
1. Análisis del CSV
2. Extracción OCR del PDF
3. Análisis unificado

### Opción 2: Desde Python

```python
from src.mintic_project.unified_analyzer import UnifiedAnalyzer

# Crear analizador (carga PDF + CSV automáticamente)
analyzer = UnifiedAnalyzer(
    pdf_path="data/Ley_769_de_2002.pdf",
    csv_path="data/siniestros_1_limpio.csv"
)

# Preguntar algo
respuesta = analyzer.responder_pregunta(
    "¿Cuál es el tipo de siniestro más frecuente?"
)
print(respuesta)

# Múltiples preguntas
preguntas = [
    "¿En qué jornada ocurren más siniestros?",
    "¿Cuál es la edad promedio de las víctimas?",
]
respuestas = analyzer.responder_multiples_preguntas(preguntas)
for q, r in respuestas.items():
    print(f"P: {q}\nR: {r}\n")
```

### Opción 3: Análisis separado del CSV

```python
from src.mintic_project.db_analysis import analyze_csv_file

result = analyze_csv_file(
    csv_path="data/siniestros_1_limpio.csv",
    question="¿Cuáles son las causas más comunes?"
)

print(result["report"])  # Metadata y estadísticas
print(result["answer"])   # Respuesta de Gemini
```

### Opción 4: Análisis separado del PDF

```python
from src.mintic_project.langchain_integration import OCRAnalyzer

analyzer = OCRAnalyzer("data/Ley_769_de_2002.pdf")
respuesta = analyzer.responder_pregunta(
    "¿Qué dice la ley sobre sanciones por exceso de velocidad?"
)
print(respuesta)
```

---

## 📋 Ejemplo: Preguntas típicas

```python
analyzer = UnifiedAnalyzer()

# Preguntas sobre datos
"¿Cuál es el horario de mayor riesgo de siniestros?"
"¿Qué género es más afectado?"
"¿Dónde ocurren más siniestros (zona urbana o rural)?"

# Preguntas combinadas (PDF + CSV)
"¿Qué dice la ley sobre CHOQUE y cuántos hay en los datos?"
"¿Cuáles son las hipótesis más frecuentes y qué contempla la ley?"
"¿Qué grupos etarios son los más afectados según la ley?"
```

---

## 🔧 Configuración

### `.env` requeridas

```env
GEMINI_API_KEY=tu_clave_gemini_aqui
POPPLER_PATH=C:\Users\Jhon\Documents\GitHub\MinTic-proyecto\tools\poppler\poppler-25.11.0\Library\bin
```

### Archivos necesarios

- `data/Ley_769_de_2002.pdf` — PDF legal (se extrae con OCR)
- `data/siniestros_1_limpio.csv` — Datos de siniestros
- `.venv/` — Python virtual environment (con paquetes instalados)

---

## 📊 Archivos CSV disponibles

- `data/siniestros_1_limpio.csv` — 2,834 registros (2022-2024)
- `data/siniestros_2_limpio.csv` — Registros adicionales

### Columnas principales
```
a_o, ipat, clase_siniestro, fecha, hora, jornada, dia_semana,
barrios_corregimiento_via, direccion, zona, autoridad, lat, long,
hipotesis, codigo, condicion_de_la_victima, edad, genero,
lesionados_y_muertos
```

---

## 🎯 Flujo de funcionamiento

```
┌─────────────────┐
│  Pregunta del   │
│    usuario      │
└────────┬────────┘
         │
         ├──────────────────────┬──────────────────────┐
         │                      │                      │
    ┌────▼─────┐         ┌──────▼──────┐        ┌─────▼────┐
    │   PDF    │         │     CSV     │        │  Gemini  │
    │  (OCR)   │         │  (análisis) │        │  (LLM)   │
    └────┬─────┘         └──────┬──────┘        └─────────┘
         │                      │                      
         └──────────────────────┼──────────────────────┘
                                │
                        ┌───────▼────────┐
                        │  Prompt mixto  │
                        └───────┬────────┘
                                │
                        ┌───────▼────────┐
                        │   Respuesta    │
                        │   unificada    │
                        └────────────────┘
```

---

## ✨ Características

✅ **Extracción OCR automática** — Lee PDFs escaneados
✅ **Análisis de metadatos** — Columnas, tipos, valores únicos
✅ **Estadísticas automáticas** — Min, max, media, frecuencias
✅ **Caché inteligente** — No reextrae OCR si ya existe
✅ **Gemini integrado** — Responde preguntas complejas
✅ **Manejo de Poppler** — Descargado automáticamente
✅ **Fallback robusto** — pypdf → OCR → Gemini

---

## 🐛 Troubleshooting

### Error: "PDF no encontrado"
```
Verifica que exista: data/Ley_769_de_2002.pdf
```

### Error: "CSV no encontrado"
```
Usa:
- data/siniestros_1_limpio.csv
- data/siniestros_2_limpio.csv
```

### Error: "Unable to get page count"
```
Poppler no disponible. Ejecuta:
  (ya está en tools/poppler automáticamente desde .env)
```

### Error: "GEMINI_API_KEY no encontrada"
```
Asegúrate de tener en .env:
  GEMINI_API_KEY=tu_clave_aqui
```

---

## 📝 Ejemplo completo

```python
#!/usr/bin/env python3
from dotenv import load_dotenv
from src.mintic_project.unified_analyzer import UnifiedAnalyzer

load_dotenv()

# Crear analizador
analyzer = UnifiedAnalyzer()

# Preguntas de ejemplo
preguntas = [
    "¿Cuál es el tipo de siniestro más común?",
    "¿En qué jornada (mañana, tarde, noche) ocurren más accidentes?",
    "¿Cuáles son las causas principales según los datos?",
]

print("="*80)
print("ANÁLISIS DE SINIESTROS VIALES")
print("="*80 + "\n")

for pregunta in preguntas:
    print(f"❓ {pregunta}")
    respuesta = analyzer.responder_pregunta(pregunta)
    print(f"✅ {respuesta}\n")
    print("-"*80 + "\n")
```

---

## 🔄 Integración con otros módulos

El sistema se integra con:
- `main.py` — CLI para OCR, queries, modo interactivo
- `test_ocr.py` — Tests de OCR y Gemini
- `tests/test_rag_load_pdf.py` — Tests de pytest

---

## 📚 Documentación relacionada

- `CAMBIOS_OCR.md` — Cambios del sistema RAG → OCR
- `README.md` — Documentación general del proyecto
- `.env.example` — Variables de entorno

---

**Última actualización:** 27 de noviembre, 2025
