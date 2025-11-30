# 🚗 Análisis de siniestros viales en Palmira - MinTIC

**Aplicación web profesional para analizar siniestros viales en Palmira, Colombia**

Combina extracción OCR de documentos legales, análisis de datos estadísticos y respuestas inteligentes con Gemini AI.

---

## 📋 Características Principales

✅ **OCR de Documentos** - Extrae automáticamente texto de la Ley 769 de 2002  
✅ **Análisis de Datos** - Explora 2,834+ registros de siniestros viales en Palmira  
✅ **Inteligencia Artificial** - Respuestas contextualizadas con Gemini API  
✅ **Análisis Unificado** - Combina contexto legal + estadísticas + IA  
✅ **Visualizaciones** - Gráficos interactivos automáticos  
✅ **Interfaz Web** - Aplicación Streamlit profesional  

---

## 🚀 Inicio Rápido

### 1. Instalación

```powershell
# Clonar repositorio
git clone https://github.com/tu-usuario/MinTic-proyecto.git
cd MinTic-proyecto

# Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Configurar API key
cp .env.example .env
# Editar .env y agregar GEMINI_API_KEY
```

### 2. Ejecutar Aplicación

```powershell
# Script automatizado (recomendado)
.\RUN_STREAMLIT.ps1

# O comando directo
streamlit run app/streamlit_app.py
```

Se abrirá en `http://localhost:8501`

---

## 📊 Estructura

```
MinTic-proyecto/
├── app/
│   └── streamlit_app.py          # Aplicación web
├── src/mintic_project/
│   ├── langchain_integration.py   # OCR + Gemini
│   ├── db_analysis.py             # Análisis CSV
│   ├── unified_analyzer.py        # Análisis combinado
│   └── main.py                    # CLI
├── data/
│   ├── Ley_769_de_2002.pdf       # Documento legal
│   └── siniestros_1_limpio.csv   # Datos Palmira (2,834 registros)
├── tests/                         # Tests
├── docs/                          # Documentación
└── RUN_STREAMLIT.ps1             # Script de inicio
```

---

## 🎯 Uso

### Aplicación Web (Streamlit)

**6 secciones principales:**

1. **🏠 Inicio** - Dashboard y tutorial
2. **📄 PDF** - Análisis de la Ley 769 con OCR
3. **📈 Datos** - Exploración de CSV interactiva
4. **🔗 Unificado** - Análisis combinado (PDF + CSV + Gemini)
5. **📋 Reportes** - Gráficos y visualizaciones
6. **ℹ️ Información** - Documentación y soporte

**Ejemplos de preguntas:**

```
# Sobre PDF
"¿Qué sanciones establece para conducir embriagado?"

# Sobre datos
"¿Cuál es el tipo de siniestro más frecuente?"

# Combinadas
"¿El CHOQUE es frecuente y qué dice la ley?"
```

### CLI (Opcional)

```bash
# Extraer texto
python -m src.mintic_project.main extract --pdf data/Ley_769_de_2002.pdf

# Hacer preguntas
python -m src.mintic_project.main query --pdf data/Ley_769_de_2002.pdf --question "..."

# Modo interactivo
python -m src.mintic_project.main interactive --pdf data/Ley_769_de_2002.pdf
```

---

## 🛠️ Stack Técnico

**Backend**: Python 3.13, Pandas, LangChain, Pytesseract  
**Frontend**: Streamlit  
**IA**: Gemini API  
**OCR**: pytesseract + pdf2image + pypdf (fallback)  

---

## 🔧 Configuración

### Variables de Entorno (.env)

```env
# Obligatorio
GEMINI_API_KEY=tu-clave-aqui

# Opcional
POPPLER_PATH=C:\...\tools\poppler\...\bin
GEMINI_MODEL=gemini-2.0-flash-exp
```

Obtén tu API key en [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## 🔍 Solución de Problemas

**"GEMINI_API_KEY no configurada"**
```bash
echo "GEMINI_API_KEY=tu-clave" >> .env
```

**"Streamlit not found"**
```powershell
.\INSTALL_STREAMLIT.ps1
```

**"Poppler not found"**
- Sistema usa `pypdf` como fallback (no requiere Poppler)
- Para OCR con imágenes: descarga [Poppler](https://github.com/oschwartz10612/poppler-windows/releases)

---

## 📚 Documentación

- **[INICIO_RAPIDO.md](docs/INICIO_RAPIDO.md)** - Guía de 3 pasos
- **[STREAMLIT_README.md](docs/STREAMLIT_README.md)** - Documentación completa (500+ líneas)
- **[ANALISIS_UNIFICADO.md](docs/ANALISIS_UNIFICADO.md)** - Guía de análisis
- **[PROYECTO_COMPLETADO.md](docs/PROYECTO_COMPLETADO.md)** - Resumen ejecutivo

---

## 🧪 Testing

```bash
pytest tests/                    # Todos los tests
pytest tests/test_ocr.py -v     # Tests específicos
pytest --cov=src tests/         # Con coverage
```

---

## 📊 Datos

**Fuentes:**
- Ley 769 de 2002 (PDF) - Código Nacional de Tránsito
- Siniestros viales en Palmira (CSV) - 2,834 registros de [datos.gov.co](https://datos.gov.co)

**Actualizar datos:**
```bash
python scripts/descargar_datos.py
```

---

## 🤝 Contribuir

1. Fork el repositorio
2. Crea branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commit (`git commit -am 'Agregar funcionalidad'`)
4. Push (`git push origin feature/nueva-funcionalidad`)
5. Abre Pull Request

---

## 📝 Licencia

MIT License - Ver archivo `LICENSE`

---

## 📞 Contacto

**Equipo**:
Jhon Edison Pinto Hincapie 
Julián Bedoya Palacio 
Daniel Quintero Castaño
Paulina Gómez Hincapie
**Repo**: GitHub MinTic-proyecto  
**Issues**: GitHub Issues  

---

## 🎓 Casos de Uso

1. **Análisis Exploratorio** - Analistas de datos
2. **Investigación Legal** - Estudiantes de derecho
3. **Análisis Ejecutivo** - Gerentes de seguridad vial
4. **Reportes** - Consultores y presentaciones

---

## ✨ Características Destacadas

✅ Caché inteligente  
✅ Fallback automático (pypdf → OCR)  
✅ UX profesional  
✅ Multi-tab (6 secciones)  
✅ Rendimiento optimizado  
✅ Error handling robusto  
✅ Componentes modulares  

---

## 🎯 Roadmap

- [ ] Exportar reportes PDF
- [ ] Gráficos Plotly avanzados
- [ ] Mapa interactivo
- [ ] ML predictivo
- [ ] Multi-idioma
- [ ] Autenticación usuarios

---

**Versión**: 1.0  
**Status**: ✅ Producción  
**Última actualización**: Noviembre 2025  

---

## 🚀 ¡Comienza Ahora!

```powershell
.\RUN_STREAMLIT.ps1
```

¡La aplicación se abrirá automáticamente! 🎊

