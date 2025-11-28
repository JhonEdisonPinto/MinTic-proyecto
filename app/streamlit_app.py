"""Streamlit app: OCR + CSV Analysis + Gemini.

Una aplicación interactiva y completa para:
1. 📄 Análisis de PDFs (OCR)
2. 📊 Exploración de datos (CSV)
3. 🔗 Análisis unificado (PDF + CSV + Gemini)
4. 📈 Reportes y visualizaciones
"""

import streamlit as st
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar página de Streamlit
st.set_page_config(
    page_title="🚗 Análisis de Siniestros Viales",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos personalizados
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .highlight {
        background-color: #ffe6e6;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff4444;
    }
    .success {
        background-color: #e6ffe6;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #44ff44;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# CACHÉ Y ESTADO DE SESIÓN
# ============================================================================

@st.cache_resource
def load_modules():
    """Cargar módulos una sola vez."""
    # Asegurar que el root del proyecto esté en sys.path para poder importar `src.*`
    import sys
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.mintic_project.langchain_integration import (
        OCRAnalyzer,
        LangChainConfig,
        extract_text_from_pdf_ocr,
    )
    from src.mintic_project.db_analysis import (
        load_csv_dataset,
        extract_dataset_metadata,
        generate_dataset_report,
    )
    from src.mintic_project.unified_analyzer import UnifiedAnalyzer

    return {
        "OCRAnalyzer": OCRAnalyzer,
        "LangChainConfig": LangChainConfig,
        "extract_text_from_pdf_ocr": extract_text_from_pdf_ocr,
        "load_csv_dataset": load_csv_dataset,
        "extract_dataset_metadata": extract_dataset_metadata,
        "generate_dataset_report": generate_dataset_report,
        "UnifiedAnalyzer": UnifiedAnalyzer,
    }


# ============================================================================
# PÁGINA PRINCIPAL
# ============================================================================

def main():
    # Encabezado
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🚗 Análisis de Siniestros Viales")
        st.markdown("**MinTIC - Proyecto Colaborativo**")
        st.markdown(
            "Análisis integrado: OCR + Datos + IA (Gemini)",
            help="Combina extracción de documentos legales, análisis de datos y respuestas inteligentes"
        )

    st.divider()

    # Verificar configuración
    from dotenv import load_dotenv
    import os

    load_dotenv()
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))

    if not has_gemini:
        st.warning("⚠️  GEMINI_API_KEY no configurada. Algunas funciones estarán limitadas.")

    # Sidebar para navegación
    st.sidebar.title("🧭 Navegación")
    page = st.sidebar.radio(
        "Selecciona una sección:",
        [
            "📊 Inicio",
            "📄 Análisis de PDF (OCR)",
            "📈 Exploración de Datos (CSV)",
            "🔗 Análisis Unificado",
            "📋 Reportes y Estadísticas",
            "ℹ️ Información",
        ],
    )

    # Módulos cargados
    modules = load_modules()

    # Renderizar página seleccionada
    if page == "📊 Inicio":
        page_home(modules)
    elif page == "📄 Análisis de PDF (OCR)":
        page_ocr_analysis(modules)
    elif page == "📈 Exploración de Datos (CSV)":
        page_csv_analysis(modules)
    elif page == "🔗 Análisis Unificado":
        page_unified_analysis(modules)
    elif page == "📋 Reportes y Estadísticas":
        page_reports(modules)
    elif page == "ℹ️ Información":
        page_info()


# ============================================================================
# PÁGINA: INICIO
# ============================================================================

def page_home(modules):
    """Página principal con resumen y guía rápida."""
    st.header("🏠 Inicio")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 PDF")
        st.markdown("Extrae texto de documentos legales usando OCR")
        st.write("- Ley 769 de 2002")
        st.write("- Extracción automática")
        st.write("- Análisis con Gemini")

    with col2:
        st.markdown("### 📊 Datos")
        st.markdown("Analiza archivos CSV de siniestros")
        st.write("- 2,834+ registros")
        st.write("- 19 columnas")
        st.write("- Estadísticas automáticas")

    with col3:
        st.markdown("### 🔗 Unificado")
        st.markdown("Combina PDF + datos + IA")
        st.write("- Preguntas cruzadas")
        st.write("- Contexto dual")
        st.write("- Respuestas inteligentes")

    st.divider()

    st.markdown("## 🚀 Guía Rápida")

    tabs = st.tabs(["Tutorial", "Archivos disponibles", "Estadísticas"])

    with tabs[0]:
        st.markdown("""
        ### Pasos para usar la aplicación:

        1. **📄 Sección PDF**: Haz preguntas sobre la Ley 769 de 2002
        2. **📈 Sección CSV**: Explora datos de siniestros viales
        3. **🔗 Sección Unificada**: Combina PDF + CSV para análisis completo
        4. **📋 Reportes**: Genera reportes y visualizaciones

        ### Ejemplos de preguntas:
        - "¿Cuál es el tipo de siniestro más frecuente?"
        - "¿En qué horario ocurren más accidentes?"
        - "¿Qué dice la ley sobre CHOQUE?"
        - "¿Cuáles son las causas principales?"
        """)

    with tabs[1]:
        st.markdown("""
        ### Archivos disponibles:

        **PDFs:**
        - `data/Ley_769_de_2002.pdf` — Código Nacional de Tránsito

        **CSVs:**
        - `data/siniestros_1_limpio.csv` — Datos 2022-2024 (2,834 registros)
        - `data/siniestros_2_limpio.csv` — Datos adicionales

        **Caché:**
        - `data/ocr_cache/Ley_769_de_2002.txt` — Texto OCR cacheado
        """)

    with tabs[2]:
        try:
            df = modules["load_csv_dataset"]("data/siniestros_1_limpio.csv")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de registros", f"{len(df):,}")
            
            with col2:
                st.metric("Columnas", len(df.columns))
            
            with col3:
                choques = (df["clase_siniestro"] == "CHOQUE").sum()
                st.metric("Choques", f"{choques:,}")
            
            with col4:
                zona_urbana = (df["zona"] == "URBANA").sum()
                st.metric("Zona urbana", f"{zona_urbana:,}")

        except Exception as e:
            st.error(f"Error cargando datos: {e}")


# ============================================================================
# PÁGINA: ANÁLISIS DE PDF (OCR)
# ============================================================================

def page_ocr_analysis(modules):
    """Análisis de PDF usando OCR."""
    st.header("📄 Análisis de PDF (OCR)")
    st.markdown("Extrae texto de documentos legales y responde preguntas.")

    pdf_path = "data/Ley_769_de_2002.pdf"

    # Verificar que el PDF existe
    if not Path(pdf_path).exists():
        st.error(f"❌ PDF no encontrado: {pdf_path}")
        return

    # Cargar analizador
    try:
        ocr_analyzer = modules["OCRAnalyzer"](pdf_path)
    except Exception as e:
        st.error(f"Error cargando PDF: {e}")
        return

    # Tabs
    tabs = st.tabs(["📋 Información del PDF", "❓ Hacer Preguntas", "📊 Vista previa"])

    with tabs[0]:
        st.subheader("Información del documento")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Archivo**: `Ley_769_de_2002.pdf`")
            st.markdown("**Tipo**: Código Nacional de Tránsito de Colombia")
            st.markdown("**Año**: 2002")

        with col2:
            try:
                texto = ocr_analyzer.extraer_texto()
                st.metric("Caracteres extraídos", f"{len(texto):,}")
                st.metric("Palabras aproximadas", f"{len(texto.split()):,}")
            except Exception as e:
                st.warning(f"Error extrayendo texto: {e}")

    with tabs[1]:
        st.subheader("❓ Haz una pregunta sobre el PDF")

        # Ejemplos de preguntas
        ejemplos = [
            "¿Cuál es el objetivo principal de esta ley?",
            "¿Qué sanciones establece para conducir en estado de embriaguez?",
            "¿Cuáles son los requisitos para obtener licencia de conducción?",
            "¿Qué dice la ley sobre los CHOQUES?",
        ]

        col_input, col_examples = st.columns([2, 1])

        with col_input:
            pregunta = st.text_area(
                "Escribe tu pregunta:",
                height=100,
                placeholder="¿Qué es...? ¿Cuáles son...? ¿Qué dice la ley sobre...?",
            )

        with col_examples:
            st.markdown("**Ejemplos:**")
            for i, ejemplo in enumerate(ejemplos, 1):
                st.caption(f"{i}. {ejemplo}")

        if st.button("🔍 Buscar respuesta", type="primary"):
            if not pregunta.strip():
                st.warning("Por favor, escribe una pregunta.")
            else:
                with st.spinner("⏳ Procesando con Gemini..."):
                    try:
                        respuesta = ocr_analyzer.responder_pregunta(pregunta)
                        st.markdown("### 📝 Respuesta")
                        st.success(respuesta)
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tabs[2]:
        st.subheader("📖 Vista previa del documento")
        try:
            texto = ocr_analyzer.extraer_texto()
            # Mostrar primeros 2000 caracteres
            st.text_area(
                "Primeros 2000 caracteres del PDF:",
                value=texto[:2000],
                height=300,
                disabled=True,
            )
            st.caption(f"Total: {len(texto):,} caracteres")
        except Exception as e:
            st.error(f"Error: {e}")


# ============================================================================
# PÁGINA: EXPLORACIÓN DE DATOS (CSV)
# ============================================================================

def page_csv_analysis(modules):
    """Análisis y exploración de CSV."""
    st.header("📈 Exploración de Datos (CSV)")
    st.markdown("Analiza datos de siniestros viales y responde preguntas.")

    # Seleccionar archivo CSV
    csv_options = {
        "siniestros_1_limpio.csv": "data/siniestros_1_limpio.csv",
        "siniestros_2_limpio.csv": "data/siniestros_2_limpio.csv",
    }

    selected_csv = st.selectbox("Selecciona un archivo CSV:", list(csv_options.keys()))
    csv_path = csv_options[selected_csv]

    if not Path(csv_path).exists():
        st.error(f"❌ CSV no encontrado: {csv_path}")
        return

    # Cargar datos
    try:
        df = modules["load_csv_dataset"](csv_path)
    except Exception as e:
        st.error(f"Error cargando CSV: {e}")
        return

    # Tabs
    tabs = st.tabs(["📊 Resumen", "🔎 Exploración", "❓ Preguntas", "📋 Datos"])

    with tabs[0]:
        st.subheader("Resumen del dataset")

        # Métrica principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Filas", f"{len(df):,}")
        with col2:
            st.metric("Columnas", len(df.columns))
        with col3:
            nulos = df.isna().sum().sum()
            st.metric("Valores nulos", f"{nulos:,}")
        with col4:
            st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Reporte
        try:
            metadata = modules["extract_dataset_metadata"](df)
            report = modules["generate_dataset_report"](df, metadata)
            st.markdown("### 📋 Reporte Detallado")
            st.text(report)
        except Exception as e:
            st.error(f"Error generando reporte: {e}")

    with tabs[1]:
        st.subheader("🔎 Exploración de columnas")

        col1, col2 = st.columns(2)

        with col1:
            # Columnas numéricas
            st.markdown("**Columnas numéricas:**")
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    st.write(f"- {col}: {df[col].dtype}")
            else:
                st.write("No hay columnas numéricas")

        with col2:
            # Columnas categóricas
            st.markdown("**Columnas categóricas:**")
            cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if cat_cols:
                for col in cat_cols:
                    st.write(f"- {col} ({df[col].nunique()} únicos)")
            else:
                st.write("No hay columnas categóricas")

        st.divider()

        # Análisis por columna seleccionada
        selected_col = st.selectbox("Analiza una columna:", df.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {selected_col}")
            st.write(f"Tipo: {df[selected_col].dtype}")
            st.write(f"Valores únicos: {df[selected_col].nunique()}")
            st.write(f"Valores nulos: {df[selected_col].isna().sum()}")

        with col2:
            if df[selected_col].dtype == "object":
                # Top valores para categóricas
                top = df[selected_col].value_counts().head(10)
                st.bar_chart(top)
            else:
                # Histograma para numéricas
                st.write(df[selected_col].describe())

    with tabs[2]:
        st.subheader("❓ Haz preguntas sobre los datos")

        ejemplos_csv = [
            "¿Cuál es el tipo de siniestro más frecuente?",
            "¿En qué jornada ocurren más siniestros?",
            "¿Cuáles son las causas principales?",
            "¿Qué género es más afectado?",
            "¿Dónde ocurren más siniestros (urbana o rural)?",
        ]

        col_input, col_examples = st.columns([2, 1])

        with col_input:
            pregunta_csv = st.text_area(
                "Escribe tu pregunta sobre los datos:",
                height=100,
                placeholder="¿Cuál es...? ¿Qué tipo...? ¿Dónde...?",
                key="csv_question",
            )

        with col_examples:
            st.markdown("**Ejemplos:**")
            for i, ejemplo in enumerate(ejemplos_csv, 1):
                st.caption(f"{i}. {ejemplo}")

        if st.button("🔍 Analizar datos", type="primary", key="csv_analyze"):
            if not pregunta_csv.strip():
                st.warning("Por favor, escribe una pregunta.")
            else:
                with st.spinner("⏳ Analizando con Gemini..."):
                    try:
                        from src.mintic_project.db_analysis import query_dataset_with_gemini
                        respuesta = query_dataset_with_gemini(pregunta_csv, df)
                        st.markdown("### 📊 Análisis")
                        st.success(respuesta)
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tabs[3]:
        st.subheader("📋 Datos crudos")
        st.dataframe(df, use_container_width=True)


# ============================================================================
# PÁGINA: ANÁLISIS UNIFICADO
# ============================================================================

def page_unified_analysis(modules):
    """Análisis que combina PDF + CSV + Gemini."""
    st.header("🔗 Análisis Unificado")
    st.markdown("Combina información legal (PDF) + datos reales (CSV) + IA (Gemini)")

    # Cargar UnifiedAnalyzer
    try:
        analyzer = modules["UnifiedAnalyzer"]()
    except Exception as e:
        st.error(f"Error inicializando analizador: {e}")
        return

    # Tabs
    tabs = st.tabs(["❓ Preguntas", "📊 Resumen ejecutivo", "⚙️ Detalles"])

    with tabs[0]:
        st.subheader("❓ Haz una pregunta")
        st.markdown(
            "La pregunta se responde combinando contexto legal + datos estadísticos + IA"
        )

        ejemplos_unificado = [
            "¿Cuál es el tipo de siniestro más frecuente y qué dice la ley al respecto?",
            "¿En qué jornada ocurren más siniestros?",
            "¿Cuáles son las hipótesis (causas) más comunes en los datos?",
            "¿Qué género es más afectado según los datos?",
        ]

        col_input, col_examples = st.columns([2, 1])

        with col_input:
            pregunta_unificada = st.text_area(
                "Escribe tu pregunta:",
                height=120,
                placeholder="Combina información legal con datos...",
                key="unified_question",
            )

        with col_examples:
            st.markdown("**Sugerencias:**")
            for i, ejemplo in enumerate(ejemplos_unificado, 1):
                st.caption(f"{i}. {ejemplo}")

        if st.button("🔍 Obtener respuesta unificada", type="primary", key="unified_search"):
            if not pregunta_unificada.strip():
                st.warning("Por favor, escribe una pregunta.")
            else:
                with st.spinner("⏳ Procesando con contexto dual..."):
                    try:
                        respuesta = analyzer.responder_pregunta(pregunta_unificada)
                        st.markdown("### 🎯 Respuesta Unificada")
                        st.success(respuesta)
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tabs[1]:
        st.subheader("📊 Resumen Ejecutivo")
        st.markdown("Resumen que combina contexto legal + estadísticas de datos")

        if st.button("📋 Generar resumen", type="primary"):
            with st.spinner("⏳ Generando resumen..."):
                try:
                    resumen = analyzer.generar_resumen_general()
                    st.success(resumen)
                except Exception as e:
                    st.error(f"Error: {e}")

    with tabs[2]:
        st.subheader("⚙️ Información técnica")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**PDF:**")
            st.caption("Ley 769 de 2002 - Código Nacional de Tránsito")
            if analyzer.pdf_text:
                st.metric("Caracteres", f"{len(analyzer.pdf_text):,}")
            
        with col2:
            st.markdown("**CSV:**")
            st.caption("siniestros_1_limpio.csv")
            if analyzer.df is not None:
                st.metric("Registros", f"{len(analyzer.df):,}")


# ============================================================================
# PÁGINA: REPORTES
# ============================================================================

def page_reports(modules):
    """Reportes y visualizaciones."""
    st.header("📋 Reportes y Estadísticas")

    # Cargar datos
    try:
        df = modules["load_csv_dataset"]("data/siniestros_1_limpio.csv")
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return

    tabs = st.tabs(["📊 Gráficos", "📈 Series temporales", "🗺️ Geográfico"])

    with tabs[0]:
        st.subheader("Visualizaciones principales")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Tipo de siniestro más frecuente")
            clase_siniestro = df["clase_siniestro"].value_counts().head(10)
            st.bar_chart(clase_siniestro)

        with col2:
            st.markdown("### Distribución por jornada")
            jornada = df["jornada"].value_counts()
            st.pie_chart(jornada)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Zona de ocurrencia")
            zona = df["zona"].value_counts()
            st.bar_chart(zona)

        with col2:
            st.markdown("### Género de víctimas")
            genero = df["genero"].value_counts()
            st.bar_chart(genero)

    with tabs[1]:
        st.subheader("Tendencias en el tiempo")

        # Convertir fecha a datetime
        df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")

        # Por año
        siniestros_por_año = df.groupby(df["fecha_dt"].dt.year).size()
        st.line_chart(siniestros_por_año, title="Siniestros por año")

        # Por mes
        siniestros_por_mes = df.groupby(df["fecha_dt"].dt.to_period("M")).size()
        st.line_chart(siniestros_por_mes, title="Siniestros por mes")

    with tabs[2]:
        st.subheader("Distribución geográfica")

        # Top barrios
        st.markdown("### Barrios/vías con más siniestros")
        top_barrios = df["barrios_corregimiento_via"].value_counts().head(15)
        st.bar_chart(top_barrios)

        # Top direcciones
        st.markdown("### Direcciones más críticas")
        top_direcciones = df["direccion"].value_counts().head(10)
        st.dataframe(top_direcciones.reset_index(), use_container_width=True)


# ============================================================================
# PÁGINA: INFORMACIÓN
# ============================================================================

def page_info():
    """Página de información y ayuda."""
    st.header("ℹ️ Información")

    tabs = st.tabs(["Acerca de", "Archivos", "Tecnología", "Contacto"])

    with tabs[0]:
        st.markdown("""
        ## 🚗 Análisis de Siniestros Viales

        Esta es una aplicación desarrollada como parte del **proyecto MinTIC** 
        para analizar datos de siniestros viales de Colombia.

        ### Características principales:
        - ✅ Extracción OCR de documentos legales (Ley 769 de 2002)
        - ✅ Análisis automático de datos CSV
        - ✅ Integración con Gemini API para respuestas inteligentes
        - ✅ Reportes y visualizaciones interactivas
        - ✅ Análisis combinado (PDF + datos + IA)

        ### Objetivos:
        1. Procesar y analizar datos de siniestros viales
        2. Combinar información legal con datos estadísticos
        3. Proporcionar respuestas inteligentes y basadas en datos
        4. Facilitar la toma de decisiones en seguridad vial

        **Desarrollo:** Equipo MinTIC
        **Fecha:** Noviembre 2025
        """)

    with tabs[1]:
        st.markdown("""
        ## 📁 Archivos disponibles

        ### PDFs:
        - `data/Ley_769_de_2002.pdf` - Código Nacional de Tránsito de Colombia

        ### CSVs de siniestros:
        - `data/siniestros_1_limpio.csv` - 2,834 registros (2022-2024)
        - `data/siniestros_2_limpio.csv` - Datos adicionales

        ### Caché:
        - `data/ocr_cache/` - Texto OCR cacheado para rendimiento

        ### Columnas del CSV:
        - a_o, ipat, clase_siniestro, fecha, hora
        - jornada, dia_semana, barrios_corregimiento_via
        - direccion, zona, autoridad, lat, long
        - hipotesis, codigo, condicion_de_la_victima
        - edad, genero, lesionados_y_muertos
        """)

    with tabs[2]:
        st.markdown("""
        ## 🛠️ Tecnología utilizada

        ### Backend:
        - **Python 3.13** - Lenguaje principal
        - **Pandas** - Análisis de datos
        - **LangChain** - Integración con LLMs
        - **Gemini API** - Modelo de lenguaje
        - **Pytesseract** - Extracción OCR
        - **pdf2image** - Conversión PDF a imagen

        ### Frontend:
        - **Streamlit** - Framework de aplicación
        - **Plotly** - Visualizaciones (opcional)

        ### Infraestructura:
        - **Git** - Control de versiones
        - **Python venv** - Entorno virtual
        - **FAISS** - Búsqueda vectorial (disponible)

        ### Configuración:
        - Variables de entorno en `.env`
        - GEMINI_API_KEY para IA
        - POPPLER_PATH para OCR
        """)

    with tabs[3]:
        st.markdown("""
        ## 📧 Contacto e información

        ### Equipo del proyecto:
        - Desarrollo: Equipo MinTIC
        - Coordinación: Jhon Doe
        
        ### Repositorio:
        - GitHub: MinTic-proyecto
        
        ### Documentación:
        - README.md - Guía general
        - ANALISIS_UNIFICADO.md - Guía de análisis
        - CAMBIOS_OCR.md - Cambios técnicos
        
        ### Soporte:
        - Para issues o preguntas, consulta la documentación
        - Verifica que GEMINI_API_KEY esté configurada
        - Asegúrate de tener Poppler instalado
        """)


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
