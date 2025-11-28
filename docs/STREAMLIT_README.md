# 🚗 Aplicación Streamlit - Análisis de siniestros viales en Palmira

## 📋 Descripción General

Esta es una **aplicación web interactiva** que integra:
- 📄 **OCR**: Extracción de texto de documentos legales (Ley 769 de 2002)
- 📊 **Análisis de Datos**: Exploración interactiva de CSV con siniestros viales en Palmira
- 🔗 **Análisis Unificado**: Combinación de PDF + datos + Gemini API
- 📈 **Reportes**: Visualizaciones y gráficos de estadísticas

## 🚀 Inicio Rápido

### 1. Requisitos previos

```bash
# Python 3.13+
# Entorno virtual activado
# Gemini API Key configurada en .env
```

### 2. Ejecutar la aplicación

#### En Windows PowerShell:
```powershell
.\RUN_STREAMLIT.ps1
```

#### En terminal (cualquier SO):
```bash
streamlit run app/streamlit_app.py
```

La aplicación se abrirá en `http://localhost:8501`

### 3. Estructura de la aplicación

```
🏠 Inicio
├── Tutorial y guía rápida
├── Archivos disponibles
└── Estadísticas principales

📄 Análisis de PDF (OCR)
├── Información del documento
├── Hacer preguntas
└── Vista previa

📈 Exploración de Datos (CSV)
├── Resumen estadístico
├── Exploración de columnas
├── Preguntas con Gemini
└── Datos crudos

🔗 Análisis Unificado
├── Preguntas cruzadas
├── Resumen ejecutivo
└── Información técnica

📋 Reportes y Estadísticas
├── Gráficos principales
├── Series temporales
└── Distribución geográfica

ℹ️ Información
├── Acerca de
├── Archivos
├── Tecnología
└── Contacto
```

## 📊 Funcionalidades Principales

### 1. Análisis de PDF (OCR)

**Propósito**: Extraer y analizar documentos legales

**Características**:
- ✅ Extracción OCR automática con pytesseract
- ✅ Fallback a pypdf si Poppler no está disponible
- ✅ Caché de resultados para rendimiento
- ✅ Preguntas sobre el contenido con Gemini

**Ejemplo de preguntas**:
- "¿Cuál es el objetivo principal de esta ley?"
- "¿Qué sanciones establece para conducir embriagado?"
- "¿Qué dice la ley sobre CHOQUES?"

### 2. Exploración de Datos (CSV)

**Propósito**: Analizar estadísticas de siniestros viales en Palmira

**Características**:
- ✅ Resumen automático (filas, columnas, memoria)
- ✅ Análisis por columna (numérica o categórica)
- ✅ Gráficos interactivos
- ✅ Tabla completa de datos
- ✅ Preguntas sobre los datos

**Ejemplo de preguntas**:
- "¿Cuál es el tipo de siniestro más frecuente?"
- "¿En qué jornada ocurren más accidentes?"
- "¿Qué género es más afectado?"

**Datos disponibles**:
- `data/siniestros_1_limpio.csv` (Palmira, 2,834 registros, 19 columnas)
- `data/siniestros_2_limpio.csv` (datos adicionales Palmira)

### 3. Análisis Unificado

**Propósito**: Combinar contexto legal + datos estadísticos

**Características**:
- ✅ Preguntas que combinan PDF + CSV
- ✅ Resumen ejecutivo automático
- ✅ Información técnica de fuentes
- ✅ Respuestas con doble contexto

**Ejemplo de preguntas**:
- "¿Cuál es el tipo de siniestro más frecuente y qué dice la ley?"
- "¿En qué jornada ocurren más siniestros viales en Palmira y por qué?"

### 4. Reportes y Visualizaciones

**Gráficos disponibles**:
- Tipo de siniestro (Top 10)
- Distribución por jornada
- Zona de ocurrencia (Urbana/Rural)
- Género de víctimas
- Series temporales (por año/mes)
- Distribución geográfica (Top barrios)

## 📁 Archivos y Estructura

```
app/
├── streamlit_app.py ..................... Aplicación Streamlit (este archivo)
└── .streamlit/
    └── config.toml ...................... Configuración de Streamlit

data/
├── Ley_769_de_2002.pdf .................. Código Nacional de Tránsito
├── siniestros_1_limpio.csv .............. Datos principales Palmira (2,834 registros)
├── siniestros_2_limpio.csv .............. Datos adicionales Palmira
└── ocr_cache/
    └── Ley_769_de_2002.txt .............. Texto extraído (caché)

src/mintic_project/
├── langchain_integration.py ............. OCR + Gemini (OCRAnalyzer)
├── db_analysis.py ....................... Análisis CSV (load_csv_dataset)
├── unified_analyzer.py .................. Análisis combinado (UnifiedAnalyzer)
└── ... (otros módulos)
```

## ⚙️ Configuración

### Variables de entorno (.env)

```env
GEMINI_API_KEY=sk-... # Tu clave de API de Gemini
POPPLER_PATH=C:\Users\Jhon\...\tools\poppler\... # Ruta a Poppler (Windows)
```

### Configuración de Streamlit

Editar `app/.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#000000"

[client]
showErrorDetails = true

[logger]
level = "info"
```

## 🔧 Solución de Problemas

### Error: "GEMINI_API_KEY no configurada"

**Solución**:
```bash
# Editar .env
echo "GEMINI_API_KEY=tu-clave-aqui" >> .env
```

### Error: "Poppler not found"

**Solución**:
```bash
# El sistema intenta usar pypdf automáticamente
# Si deseas OCR con imagenes, instala Poppler:
# En Windows: Ya está instalado en tools/poppler/
# Verifica POPPLER_PATH en .env
```

### Error: "CSV no encontrado"

**Solución**:
```bash
# Verifica que los archivos existan:
# data/siniestros_1_limpio.csv (Palmira)
# data/siniestros_2_limpio.csv (Palmira)
```

### La aplicación es lenta

**Soluciones**:
- Cierra otras aplicaciones
- Los datos se cachean automáticamente (OCR, CSV)
- Limpia caché: `rm data/ocr_cache/*.txt`

## 📈 Ejemplos de Uso

### Caso 1: Analizar tipo de siniestro más frecuente

1. Ir a "📈 Exploración de Datos (CSV)"
2. En tab "❓ Preguntas", escribir:
   ```
   ¿Cuál es el tipo de siniestro más frecuente?
   ```
3. Hacer click en "🔍 Analizar datos"

**Resultado esperado**: Gemini responde que CHOQUE es el más frecuente (1,970 casos)

### Caso 2: Pregunta unificada

1. Ir a "🔗 Análisis Unificado"
2. En tab "❓ Preguntas", escribir:
   ```
   ¿Cuál es el tipo de siniestro más frecuente y qué dice la ley?
   ```
3. Hacer click en "🔍 Obtener respuesta unificada"

**Resultado esperado**: Respuesta que combina:
- Datos: CHOQUE es el 60% de los siniestros viales en Palmira
- Ley: Artículos sobre definiciones y sanciones

### Caso 3: Generar reporte

1. Ir a "📋 Reportes y Estadísticas"
2. Ver gráficos automáticos de:
   - Tipos de siniestro
   - Distribución temporal
   - Zonas geográficas

## 🎯 Mejores Prácticas

1. **Preguntas claras**: Sé específico en tus preguntas
   - ❌ "Tell me about this"
   - ✅ "¿Cuál es la causa más común en zona URBANA?"

2. **Formato de preguntas**: Usa puntuación correcta
   - ✅ "¿En qué jornada ocurren más siniestros viales en Palmira?"
   - ✅ "¿Qué dice la ley sobre el CHOQUE?"

3. **Monitorear resultados**: Verifica que las respuestas tengan sentido
   - Si algo no coincide, reformula la pregunta

4. **Usar ejemplos**: Los ejemplos sugeridos en cada sección funcionan bien

## 📞 Soporte

- **Documentación**: Ver `ANALISIS_UNIFICADO.md`
- **Issues técnicos**: Revisar `.env` y rutas de archivos
- **API limits**: Comprobar límites de Gemini API

## 🎓 Recursos Educativos

Esta aplicación demuestra:
- 📚 **Integración de LLMs**: Gemini + LangChain
- 🔍 **OCR y extracción de texto**: pytesseract + pdf2image
- 📊 **Análisis de datos**: pandas + visualizaciones
- 🎨 **Interfaz web**: Streamlit
- 🔗 **Arquitectura modular**: Componentes reutilizables

## ✅ Checklist antes de presentar

- [ ] `.env` tiene GEMINI_API_KEY
- [ ] Archivos CSV existen en `data/`
- [ ] PDF existe en `data/Ley_769_de_2002.pdf`
- [ ] `python -m streamlit run app/streamlit_app.py` funciona
- [ ] Todas las tabs cargan sin errores
- [ ] Las gráficas se muestran correctamente
- [ ] Las preguntas reciben respuestas de Gemini

## 📝 Notas de Desarrollo

- **Cache**: Streamlit cachea módulos con `@st.cache_resource`
- **Estado**: El estado de sesión se mantiene durante la sesión
- **Rendimiento**: OCR se cachea en `data/ocr_cache/`
- **Modularidad**: Cada tab es una función independiente

---

**Última actualización**: Noviembre 2025
**Versión**: 1.0
**Python**: 3.13+
**Status**: ✅ Producción
