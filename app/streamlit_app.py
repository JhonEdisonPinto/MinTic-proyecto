"""Streamlit app: Análisis Inteligente de Siniestros Viales Colombia.

Plataforma completa para análisis de siniestros viales con:
1. 📂 Gestión dinámica de datasets (múltiples ciudades)
2. 📄 Análisis de PDFs legales con OCR
3. 📊 Exploración adaptable de datos
4. 🤖 Análisis unificado con IA (Gemini)
5. 📊 Reportes detallados y visualizaciones interactivas
6. 🗺️ Mapas geográficos de siniestros
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Siniestros Viales Colombia",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos personalizados
st.markdown("""
<style>
.metric-card { background-color: #f0f2f6; padding: 16px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def normalize_text(text):
    """Normaliza texto removiendo tildes, caracteres especiales y convirtiendo a minúsculas.
    
    Args:
        text: Texto a normalizar
        
    Returns:
        str: Texto normalizado
    """
    import unicodedata
    import re
    # Remover tildes
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Convertir a minúsculas
    text = text.lower()
    # Remover caracteres especiales (mantener solo letras y números)
    text = re.sub(r'[^a-z0-9]', '', text)
    return text.strip()


def find_column(df, possible_names):
    """Encuentra la primera columna que existe en el DataFrame con búsqueda flexible.
    
    Busca coincidencias exactas primero, luego usa normalización para manejar:
    - Tildes (dirección vs direccion)
    - Caracteres especiales (direcci_n vs direccion)
    - Mayúsculas/minúsculas
    - Similitud de prefijos (direccin vs direccion)
    
    Args:
        df: DataFrame de pandas
        possible_names: Lista de nombres posibles de columnas
        
    Returns:
        str: Nombre de la columna encontrada, o None si no existe ninguna
    """
    # Primero: coincidencia exacta
    for name in possible_names:
        if name in df.columns:
            return name
    
    # Segundo: coincidencia normalizada
    normalized_possible = [normalize_text(name) for name in possible_names]
    
    for col in df.columns:
        normalized_col = normalize_text(col)
        
        # Coincidencia exacta normalizada
        if normalized_col in normalized_possible:
            return col
        
        # Coincidencia por substring bidireccional
        for possible in normalized_possible:
            # La columna contiene el término buscado
            if possible in normalized_col:
                return col
            # El término buscado contiene la columna
            if normalized_col in possible and len(normalized_col) >= 5:
                return col
            
            # Similitud por prefijo común (al menos 70% del más corto)
            min_len = min(len(normalized_col), len(possible))
            if min_len >= 5:  # Solo para palabras de al menos 5 caracteres
                # Contar caracteres coincidentes al inicio
                common_prefix = 0
                for i in range(min_len):
                    if normalized_col[i] == possible[i]:
                        common_prefix += 1
                    else:
                        break
                
                # Si al menos 70% del más corto coincide al inicio
                if common_prefix / min_len >= 0.7:
                    return col
    
    return None


@st.cache_resource
def load_modules():
    """Carga y expone módulos del paquete `src.mintic_project` una sola vez."""
    import sys
    from pathlib import Path as _Path

    project_root = _Path(__file__).resolve().parent.parent
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
        check_data_quality,
    )
    from src.mintic_project.unified_analyzer import UnifiedAnalyzer

    return {
        "OCRAnalyzer": OCRAnalyzer,
        "LangChainConfig": LangChainConfig,
        "extract_text_from_pdf_ocr": extract_text_from_pdf_ocr,
        "load_csv_dataset": load_csv_dataset,
        "extract_dataset_metadata": extract_dataset_metadata,
        "generate_dataset_report": generate_dataset_report,
        "check_data_quality": check_data_quality,
        "UnifiedAnalyzer": UnifiedAnalyzer,
    }


# ============================================================================
# PÁGINA PRINCIPAL
# ============================================================================

def main():
    # Asegurar que el proyecto esté en sys.path
    import sys
    from pathlib import Path as _Path
    project_root = _Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Encabezado
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🚗 Análisis de Siniestros Viales Colombia")
        st.markdown("**MinTIC - Proyecto Colaborativo 2025**")
        st.markdown(
            "🌐 Plataforma nacional multi-ciudad con IA",
            help="Gestión dinámica de datasets + OCR legal + Análisis adaptable + Reportes robustos"
        )

    st.divider()

    # Verificar configuración
    from dotenv import load_dotenv
    import os

    load_dotenv()
    
    # Intentar cargar desde st.secrets (Streamlit Cloud) o desde .env (local)
    try:
        # Acceder a `st.secrets` puede lanzar StreamlitSecretNotFoundError
        # si no existe un secrets.toml en las ubicaciones esperadas.
        gemini_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
    except Exception:
        # Fallback: usar variable de entorno si no hay archivo de secrets
        gemini_key = os.getenv("GEMINI_API_KEY")

    has_gemini = bool(gemini_key)
    
    # Asegurar que la variable esté en el entorno para los módulos
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

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
            "📋 Reportes y Power BI",
            "ℹ️ Información",
        ],
    )

    # ------------------------------------------------------------------
    # GESTIÓN DE DATASETS DINÁMICOS
    # ------------------------------------------------------------------
    dataset_expander = st.sidebar.expander("📂 Gestión de Datasets")
    
    # Cargar gestor
    from src.mintic_project.data_loader import DatasetManager, descargar_y_limpiar_dataset
    
    if "dataset_manager" not in st.session_state:
        st.session_state.dataset_manager = DatasetManager()
    
    manager = st.session_state.dataset_manager
    
    with dataset_expander:
        st.markdown("### Datasets disponibles")
        
        # Selector de dataset activo
        datasets_list = list(manager.list_datasets().keys())
        if datasets_list:
            current_idx = datasets_list.index(manager.active_dataset) if manager.active_dataset in datasets_list else 0
            selected = st.selectbox(
                "Dataset activo:",
                datasets_list,
                index=current_idx,
                key="dataset_selector"
            )
            
            if selected != manager.active_dataset:
                manager.set_active(selected)
                st.success(f"✓ Dataset activo: {selected}")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### Agregar nuevo dataset")
        
        with st.form("add_dataset_form"):
            new_name = st.text_input("Nombre del dataset", placeholder="Ej: Yopal - Siniestros")
            new_url = st.text_input(
                "URL de datos.gov.co", 
                placeholder="https://www.datos.gov.co/resource/xxxx-xxxx.json",
                help="Copia la URL base del dataset (sin parámetros de query)"
            )
            submit = st.form_submit_button("➕ Agregar")
            
            if submit:
                if new_name and new_url:
                    # Limpiar URL si tiene query params
                    from urllib.parse import urlparse
                    parsed = urlparse(new_url)
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    if not clean_url.endswith('.json'):
                        clean_url = clean_url.rstrip('/') + '.json'
                    
                    if manager.add_dataset(new_name, clean_url):
                        st.success(f"✓ Dataset '{new_name}' agregado")
                        st.info("💡 Ahora haz clic en '⬇️ Descargar dataset activo' para obtener los datos")
                        st.rerun()
                    else:
                        st.error(f"❌ Dataset '{new_name}' ya existe")
                else:
                    st.warning("Por favor completa ambos campos")
        
        st.markdown("---")
        st.markdown("### Eliminar dataset")
        
        # Obtener lista de datasets eliminables (excluir predeterminados)
        datasets_dict = manager.list_datasets()
        defaults = list(DatasetManager.DEFAULTS.keys())
        eliminables = {k: v for k, v in datasets_dict.items() if k not in defaults}
        
        if eliminables:
            dataset_to_delete = st.selectbox(
                "Selecciona dataset a eliminar:",
                list(eliminables.keys()),
                key="delete_selector"
            )
            
            if st.button("🗑️ Eliminar dataset", type="secondary"):
                if manager.remove_dataset(dataset_to_delete):
                    st.success(f"✓ Dataset '{dataset_to_delete}' eliminado")
                    # Si era el activo, cambiar a uno predeterminado
                    if manager.active_dataset == dataset_to_delete:
                        manager.set_active(defaults[0] if defaults else list(datasets_dict.keys())[0])
                    st.rerun()
                else:
                    st.error("❌ No se puede eliminar dataset predeterminado")
        else:
            st.caption("No hay datasets personalizados para eliminar")
        
        st.markdown("---")
        
        # Botón para descargar dataset activo
        if st.button("⬇️ Descargar dataset activo", type="primary"):
            with st.spinner(f"Descargando '{manager.active_dataset}'..."):
                url = manager.get_active_url()
                if url:
                    df = descargar_y_limpiar_dataset(url, manager.active_dataset, "data", 50000)
                    if not df.empty:
                        st.success(f"✓ Descargado: {len(df)} registros")
                        st.info("✓ Archivo guardado. La app se recargará automáticamente.")
                        # Recargar app
                        try:
                            st.rerun()
                        except:
                            st.info("Recarga la página para ver los cambios")
                    else:
                        st.error("❌ No se pudieron descargar datos. Verifica la URL.")

    # ------------------------------------------------------------------
    # CONTROL: Actualizar datasets desde la API (LEGACY - mantener compatibilidad)
    # ------------------------------------------------------------------
    data_expander = st.sidebar.expander("🔁 Actualización Rápida")
    data_expander.markdown("Actualizar todos los datasets predeterminados")
    try:
        if data_expander.button("🔁 Actualizar todos"):
            with st.spinner("Descargando y procesando datos desde la API..."):
                from src.mintic_project.data_loader import procesar_siniestros

                df1, df2 = procesar_siniestros(directorio_salida="data", limite_registros=50000)

                if (not df1.empty) or (not df2.empty):
                    st.success(f"Datos actualizados: siniestros_1={len(df1)} filas, siniestros_2={len(df2)} filas")
                else:
                    st.error("No se pudieron descargar/actualizar los datos. Revisa los logs.")

                # Recargar la app para que lea los CSV nuevos (seguro)
                try:
                    if hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()
                    else:
                        st.info("Datos actualizados. Recarga la página manualmente para ver los cambios.")
                except Exception:
                    # En algunos entornos la función puede no estar disponible
                    st.info("Datos actualizados. Recarga la página manualmente para ver los cambios.")
    except Exception as e:
        data_expander.warning(f"No se pudo iniciar el actualizador de datos: {e}")

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
    elif page == "📋 Reportes y Power BI":
        page_reports(modules)
    elif page == "ℹ️ Información":
        page_info()


# ============================================================================
# PÁGINA: INICIO
# ============================================================================

def page_home(modules):
    """Página principal con resumen y guía rápida."""
    st.header("🏠 Inicio - siniestros viales en Palmira")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 PDF")
        st.markdown("Extrae texto de documentos legales usando OCR")
        st.write("- Ley 769 de 2002")
        st.write("- Extracción automática")
        st.write("- Análisis con Gemini")

    with col2:
        st.markdown("### 📊 Datos")
        st.markdown("Analiza archivos CSV de siniestros viales en Palmira")
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

    tabs = st.tabs(["Tutorial", "Archivos disponibles"])

    with tabs[0]:
        st.markdown("""
        ### Pasos para usar la aplicación:

        1. **📄 Sección PDF**: Haz preguntas sobre la Ley 769 de 2002
        2. **📈 Sección CSV**: Explora datos de siniestros viales en Palmira
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
        - `data/siniestros_1_limpio.csv` — Datos de siniestros viales en Palmira 2022-2024 (2,834 registros)
        - `data/siniestros_2_limpio.csv` — Datos adicionales de Palmira

        **Caché:**
        - `data/ocr_cache/Ley_769_de_2002.txt` — Texto OCR cacheado
        """)


 


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
    """Análisis y exploración de CSV - usa dataset activo del gestor."""
    st.header("📈 Exploración de Datos (CSV)")
    
    # Obtener dataset activo del gestor
    if "dataset_manager" in st.session_state:
        manager = st.session_state.dataset_manager
        active_name = manager.active_dataset
        st.markdown(f"**Dataset activo:** `{active_name}`")
        st.caption("Cambia el dataset desde el panel lateral → 📂 Gestión de Datasets")
    else:
        st.warning("⚠️ No se pudo cargar el gestor de datasets")
        active_name = None
    
    # Determinar ruta del CSV basado en el dataset activo
    if active_name:
        # Convertir nombre a nombre de archivo
        csv_filename = active_name.lower().replace(" ", "_").replace("-", "_") + ".csv"
        csv_path = f"data/{csv_filename}"
    else:
        # Fallback: usar selector legacy
        csv_options = {
            "siniestros_1_limpio.csv": "data/siniestros_1_limpio.csv",
            "siniestros_2_limpio.csv": "data/siniestros_2_limpio.csv",
        }
        selected_csv = st.selectbox("Selecciona un archivo CSV:", list(csv_options.keys()))
        csv_path = csv_options[selected_csv]

    if not Path(csv_path).exists():
        st.error(f"❌ CSV no encontrado: {csv_path}")
        st.info("💡 Descarga el dataset desde el panel lateral → 📂 Gestión de Datasets → ⬇️ Descargar dataset activo")
        return

    # Cargar datos
    try:
        df = modules["load_csv_dataset"](csv_path)
    except Exception as e:
        st.error(f"Error cargando CSV: {e}")
        return
    
    # Evaluar calidad de datos
    quality = modules["check_data_quality"](df)

    # Tabs
    tabs = st.tabs(["📊 Resumen", "🔎 Exploración", "❓ Preguntas", "📋 Datos"])

    with tabs[0]:
        st.subheader("Resumen del dataset")
        
        # Card de calidad de datos
        st.markdown("### 🎯 Calidad de Datos")
        col_q1, col_q2, col_q3 = st.columns(3)
        
        with col_q1:
            score_color = "🟢" if quality["score"] >= 80 else "🟡" if quality["score"] >= 60 else "🔴"
            st.metric("Calidad General", f"{score_color} {quality['score']}%")
        
        with col_q2:
            st.metric("Completitud", f"{quality['completeness']}%")
        
        with col_q3:
            st.metric("Filas duplicadas", quality["duplicates"])
        
        # Issues detectados
        if quality["issues"]:
            with st.expander("⚠️ Problemas detectados"):
                for issue in quality["issues"]:
                    st.warning(issue)
        
        # Outliers
        if quality["outliers"]:
            with st.expander("📊 Outliers por columna"):
                for col, count in quality["outliers"].items():
                    st.caption(f"• {col}: {count} valores atípicos")
        
        st.divider()

        # Resumen ejecutivo visual
        st.markdown("### 📊 Resumen Ejecutivo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📐 Estructura**")
            st.metric("Filas", f"{len(df):,}")
            st.metric("Columnas", len(df.columns))
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            st.metric("Memoria", f"{memory_mb:.2f} MB")
        
        with col2:
            st.markdown("**🔢 Tipos de Datos**")
            numeric_count = len(df.select_dtypes(include=["number"]).columns)
            categorical_count = len(df.select_dtypes(include=["object"]).columns)
            datetime_count = len(df.select_dtypes(include=["datetime"]).columns)
            st.metric("Numéricas", numeric_count)
            st.metric("Categóricas", categorical_count)
            st.metric("Fecha/Hora", datetime_count)
        
        with col3:
            st.markdown("**🎯 Integridad**")
            nulos = df.isna().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            completitud = ((total_cells - nulos) / total_cells * 100) if total_cells > 0 else 0
            st.metric("Valores nulos", f"{nulos:,}")
            st.metric("Completitud", f"{completitud:.1f}%")
            st.metric("Duplicados", quality["duplicates"])

        # Reporte detallado
        try:
            metadata = modules["extract_dataset_metadata"](df)
            report = modules["generate_dataset_report"](df, metadata)
            
            with st.expander("📋 Ver Reporte Detallado Completo", expanded=False):
                st.code(report, language=None)
                
                # Botón de descarga
                st.download_button(
                    label="💾 Descargar Reporte",
                    data=report,
                    file_name=f"reporte_{active_name.lower().replace(' ', '_')}.txt",
                    mime="text/plain"
                )
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
        st.caption("El asistente puede analizar el DataFrame directamente usando operaciones de Pandas.")

        ejemplos_csv = [
            "¿Cuál es el tipo de siniestro más frecuente?",
            "¿En qué jornada ocurren más siniestros?",
            "¿Cuáles son las causas principales?",
            "¿Qué género es más afectado?",
            "¿Dónde ocurren más siniestros (urbana o rural)?",
            "¿Cuántos choques hubo en 2023?",
        ]

        col_input, col_examples = st.columns([2, 1])

        with col_input:
            pregunta_csv = st.text_area(
                "Escribe tu pregunta sobre los datos:",
                height=100,
                placeholder="¿Cuál es...? ¿Qué tipo...? ¿Dónde...?",
                key="csv_question",
            )
            advanced = st.checkbox(
                "🔧 Modo avanzado (permitir ejecución de código Pandas)",
                value=False,
                help="Habilita ejecución directa de código. Usa solo en entornos confiables."
            )

        with col_examples:
            st.markdown("**Ejemplos:**")
            for i, ejemplo in enumerate(ejemplos_csv, 1):
                st.caption(f"{i}. {ejemplo}")

        if st.button("🔍 Analizar datos", type="primary", key="csv_analyze"):
            if not pregunta_csv.strip():
                st.warning("Por favor, escribe una pregunta.")
            else:
                with st.spinner("⏳ Consultando agente de Pandas..."):
                    try:
                        from src.mintic_project.db_analysis import query_with_pandas_agent
                        respuesta = query_with_pandas_agent(pregunta_csv, df, dangerous=advanced)
                        st.markdown("### 📊 Respuesta")
                        st.markdown(respuesta)
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tabs[3]:
        st.subheader("📋 Datos crudos")
        # Usar width='stretch' para ocupar el ancho disponible (evita error width=None)
        st.dataframe(df, width='stretch')


# ============================================================================
# PÁGINA: ANÁLISIS UNIFICADO
# ============================================================================

def page_unified_analysis(modules):
    """Análisis que combina PDF + CSV + Gemini."""
    st.header("🔗 Análisis Unificado")
    st.markdown("Combina información legal (PDF) + datos reales (CSV) + IA (Gemini)")

    # Obtener dataset activo
    if "dataset_manager" in st.session_state:
        manager = st.session_state.dataset_manager
        active_name = manager.active_dataset
        csv_filename = active_name.lower().replace(" ", "_").replace("-", "_") + ".csv"
        csv_path = f"data/{csv_filename}"
        st.caption(f"📊 Dataset activo: **{active_name}**")
    else:
        csv_path = "data/siniestros_1_limpio.csv"
        st.caption("📊 Dataset: siniestros_1_limpio.csv (predeterminado)")
    
    if not Path(csv_path).exists():
        st.error(f"❌ CSV no encontrado: {csv_path}")
        st.info("💡 Descarga el dataset desde el panel lateral → 📂 Gestión de Datasets → ⬇️ Descargar dataset activo")
        return

    # Cargar UnifiedAnalyzer con el dataset activo
    try:
        analyzer = modules["UnifiedAnalyzer"](csv_path=csv_path)
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
            st.caption("siniestros_1_limpio.csv (Palmira)")
            if analyzer.df is not None:
                st.metric("Registros", f"{len(analyzer.df):,}")


# ============================================================================
# PÁGINA: REPORTES
# ============================================================================

def page_reports(modules):
    """Reportes y visualizaciones - usa dataset activo del gestor."""
    st.header("📋 Reportes y Power BI")
    
    # Obtener dataset activo
    if "dataset_manager" in st.session_state:
        manager = st.session_state.dataset_manager
        active_name = manager.active_dataset
        csv_filename = active_name.lower().replace(" ", "_").replace("-", "_") + ".csv"
        csv_path = f"data/{csv_filename}"
        st.caption(f"Dataset: `{active_name}`")
    else:
        csv_path = "data/siniestros_1_limpio.csv"

    # Cargar datos
    try:
        df = modules["load_csv_dataset"](csv_path)
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.info("💡 Descarga el dataset desde el panel lateral")
        return

    # Panel de información de columnas detectadas
    with st.expander("🔍 Columnas detectadas en este dataset"):
        col1, col2, col3 = st.columns(3)
        
        column_mapping = {
            'tipo_siniestro': ['clase_siniestro', 'tipo_siniestro', 'tipo', 'clase'],
            'tiempo': ['jornada', 'periodo', 'hora', 'turno'],
            'zona': ['zona', 'area', 'sector', 'localidad'],
            'genero': ['genero', 'sexo', 'genero_victima', 'sexo_victima'],
            'gravedad': ['gravedad', 'severidad', 'tipo_herida', 'estado'],
            'fecha': ['fecha', 'fecha_ocurrencia', 'fecha_siniestro', 'date', 'timestamp'],
            'barrio': ['barrios_corregimiento_via', 'barrio', 'comuna', 'localidad', 'sector'],
            'direccion': ['direccion', 'direccion_ocurrencia', 'address', 'via', 'calle'],
            'coordenadas': ['latitud', 'longitud', 'lat', 'long', 'lon', 'latitude', 'longitude', 'y', 'x']
        }
        
        with col1:
            st.markdown("**📊 Categóricas:**")
            for key in ['tipo_siniestro', 'tiempo', 'zona']:
                col = find_column(df, column_mapping[key])
                if col:
                    st.success(f"✓ {col}")
                else:
                    st.caption(f"✗ {key.replace('_', ' ')}")
        
        with col2:
            st.markdown("**👥 Demográficas:**")
            for key in ['genero', 'gravedad']:
                col = find_column(df, column_mapping[key])
                if col:
                    st.success(f"✓ {col}")
                else:
                    st.caption(f"✗ {key.replace('_', ' ')}")
        
        with col3:
            st.markdown("**📍 Geográficas:**")
            for key in ['fecha', 'barrio', 'direccion']:
                col = find_column(df, column_mapping[key])
                if col:
                    st.success(f"✓ {col}")
                else:
                    st.caption(f"✗ {key.replace('_', ' ')}")
            
            # Coordenadas - mostrar lat y lon por separado
            lat_col = find_column(df, ['latitud', 'lat', 'latitude', 'y'])
            lon_col = find_column(df, ['longitud', 'long', 'lon', 'longitude', 'x'])
            if lat_col and lon_col:
                st.success(f"✓ {lat_col} / {lon_col}")
            else:
                st.caption(f"✗ coordenadas")
        
        st.caption(f"Total de columnas en el dataset: **{len(df.columns)}**")

    # Pestañas principales: Reportes locales y Power BI embebido
    main_tabs = st.tabs(["📋 Reportes y Estadísticas", "📊 Power BI"])

    # --- Pestaña 1: reportes locales (mantener sub-pestañas existentes)
    with main_tabs[0]:
        sub_tabs = st.tabs(["📊 Gráficos", "📈 Series temporales", "🗺️ Geográfico"])

        with sub_tabs[0]:
            st.subheader("Visualizaciones principales")
            
            # Mapeo de columnas comunes (flexibilidad para diferentes datasets)
            column_mapping = {
                'tipo_siniestro': ['clase_siniestro', 'tipo_siniestro', 'tipo', 'clase'],
                'tiempo': ['jornada', 'periodo', 'hora', 'turno'],
                'zona': ['zona', 'area', 'sector', 'localidad'],
                'genero': ['genero', 'sexo', 'genero_victima', 'sexo_victima'],
                'gravedad': ['gravedad', 'severidad', 'tipo_herida', 'estado']
            }

            col1, col2 = st.columns(2)

            with col1:
                tipo_col = find_column(df, column_mapping['tipo_siniestro'])
                if tipo_col:
                    st.markdown("### Tipo de siniestro más frecuente")
                    clase_siniestro = df[tipo_col].value_counts().head(10)
                    st.bar_chart(clase_siniestro)
                else:
                    st.info("📊 No se encontró columna de tipo de siniestro")

            with col2:
                tiempo_col = find_column(df, column_mapping['tiempo'])
                if tiempo_col:
                    st.markdown(f"### Distribución por {tiempo_col}")
                    tiempo_data = df[tiempo_col].value_counts()
                    fig = px.pie(values=tiempo_data.values, names=tiempo_data.index, title="")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("⏰ No se encontró columna de tiempo/jornada")

            col1, col2 = st.columns(2)

            with col1:
                zona_col = find_column(df, column_mapping['zona'])
                if zona_col:
                    st.markdown(f"### {zona_col.title()}")
                    zona = df[zona_col].value_counts()
                    st.bar_chart(zona)
                else:
                    st.info("🗺️ No se encontró columna de zona/área")

            with col2:
                genero_col = find_column(df, column_mapping['genero'])
                if genero_col:
                    st.markdown(f"### {genero_col.title()}")
                    genero = df[genero_col].value_counts()
                    st.bar_chart(genero)
                else:
                    gravedad_col = find_column(df, column_mapping['gravedad'])
                    if gravedad_col:
                        st.markdown(f"### {gravedad_col.title()}")
                        gravedad = df[gravedad_col].value_counts()
                        st.bar_chart(gravedad)
                    else:
                        st.info("👤 No se encontró columna de género o gravedad")

        with sub_tabs[1]:
            st.subheader("Tendencias en el tiempo")

            # Buscar columna de fecha
            fecha_col = find_column(df, ['fecha', 'fecha_ocurrencia', 'fecha_siniestro', 'date', 'timestamp'])
            
            if fecha_col:
                # Convertir fecha a datetime
                df["fecha_dt"] = pd.to_datetime(df[fecha_col], errors="coerce")
                
                if df["fecha_dt"].notna().sum() > 0:
                    # Por año
                    st.markdown("### Siniestros por año")
                    siniestros_por_año = df.groupby(df["fecha_dt"].dt.year).size()
                    st.line_chart(siniestros_por_año)

                    # Por mes
                    st.markdown("### Siniestros por mes")
                    siniestros_por_mes = df.groupby(df["fecha_dt"].dt.to_period("M")).size()
                    # Convertir Period a string para visualización
                    siniestros_por_mes_df = pd.DataFrame({
                        'Fecha': siniestros_por_mes.index.astype(str),
                        'Cantidad': siniestros_por_mes.values
                    })
                    st.line_chart(siniestros_por_mes_df.set_index('Fecha'))
                else:
                    st.warning("⚠️ No se pudieron convertir las fechas a formato válido")
            else:
                st.info("📅 No se encontró columna de fecha en este dataset")

        with sub_tabs[2]:
            st.subheader("Distribución geográfica")

            # Buscar columnas geográficas
            barrio_col = find_column(df, ['barrios_corregimiento_via', 'barrio', 'comuna', 'localidad', 'sector'])
            direccion_col = find_column(df, ['direccion', 'direccion_ocurrencia', 'address', 'via', 'calle'])
            
            if barrio_col:
                st.markdown(f"### Top 15 {barrio_col.replace('_', ' ').title()}")
                top_barrios = df[barrio_col].value_counts().head(15)
                st.bar_chart(top_barrios)
            else:
                st.info("🏘️ No se encontró columna de barrio/sector")

            if direccion_col:
                st.markdown(f"### Top 10 {direccion_col.replace('_', ' ').title()}")
                top_direcciones = df[direccion_col].value_counts().head(10)
                st.dataframe(top_direcciones.reset_index(), width='stretch')
            else:
                st.info("📍 No se encontró columna de dirección")
            
            # Coordenadas si existen
            lat_col = find_column(df, ['latitud', 'lat', 'latitude', 'y'])
            lon_col = find_column(df, ['longitud', 'long', 'lon', 'longitude', 'x'])
            
            if lat_col and lon_col:
                st.markdown("### 🗺️ Mapa de siniestros")
                try:
                    # Convertir a numérico y filtrar coordenadas válidas
                    map_data = df[[lat_col, lon_col]].copy()
                    
                    # Reemplazar coma por punto en caso de formato europeo
                    if map_data[lat_col].dtype == 'object':
                        map_data[lat_col] = map_data[lat_col].astype(str).str.replace(',', '.')
                    if map_data[lon_col].dtype == 'object':
                        map_data[lon_col] = map_data[lon_col].astype(str).str.replace(',', '.')
                    
                    map_data[lat_col] = pd.to_numeric(map_data[lat_col], errors='coerce')
                    map_data[lon_col] = pd.to_numeric(map_data[lon_col], errors='coerce')
                    map_data = map_data.dropna()
                    
                    # Filtrar coordenadas en rango válido
                    map_data = map_data[
                        (map_data[lat_col] >= -90) & (map_data[lat_col] <= 90) &
                        (map_data[lon_col] >= -180) & (map_data[lon_col] <= 180)
                    ]
                    
                    if len(map_data) > 0:
                        st.map(map_data.rename(columns={lat_col: 'lat', lon_col: 'lon'}))
                        st.caption(f"Mostrando {len(map_data):,} registros con coordenadas válidas de {len(df):,} totales")
                    else:
                        st.info("No hay coordenadas válidas para mostrar en el mapa")
                except Exception as e:
                    st.warning(f"⚠️ Error al procesar coordenadas: {str(e)}")
            else:
                st.info("🌍 No se encontraron columnas de coordenadas (lat/lon)")

    # --- Pestaña 2: Power BI embebido
    with main_tabs[1]:
        st.subheader("Power BI — Informe embebido")
        st.markdown("Este informe se usa mediante 'Publish to web' (público).")

        # URL por defecto (el usuario puede cambiarla luego si lo desea)
        embed_url = "https://app.powerbi.com/view?r=eyJrIjoiNWI0N2ZjYzEtNDg3Yy00MWJkLWExNDMtYzQ5MWJjZjFmNWJjIiwidCI6IjU3N2ZjMWQ4LTA5MjItNDU4ZS04N2JmLWVjNGY0NTVlYjYwMCIsImMiOjR9"

        import streamlit.components.v1 as components

        html = f"""
        <iframe width="100%" height="720" src="{embed_url}" frameborder="0" allowFullScreen="true"></iframe>
        """
        components.html(html, height=760)

        # Mostrar métricas rápidas del dataset (si está disponible)
        try:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total de registros", f"{len(df):,}")

            with col2:
                st.metric("Columnas", len(df.columns))

            with col3:
                # Buscar columna de tipo para métrica dinámica
                tipo_col = find_column(df, ['clase_siniestro', 'tipo_siniestro', 'tipo'])
                if tipo_col and len(df) > 0:
                    most_common = df[tipo_col].value_counts().index[0]
                    count = (df[tipo_col] == most_common).sum()
                    st.metric(f"{most_common.title()}", f"{count:,}")
                else:
                    st.metric("Tipo más común", "—")

            with col4:
                # Buscar columna de zona para métrica dinámica
                zona_col = find_column(df, ['zona', 'area', 'sector'])
                if zona_col and len(df) > 0:
                    most_common_zona = df[zona_col].value_counts().index[0]
                    count_zona = (df[zona_col] == most_common_zona).sum()
                    st.metric(f"{most_common_zona.title()}", f"{count_zona:,}")
                else:
                    st.metric("Zona principal", "—")

        except Exception as e:
            st.warning(f"No se pudieron calcular métricas rápidas: {str(e)}")


# ============================================================================
# PÁGINA: INFORMACIÓN
# ============================================================================

def page_info():
    """Página de información y ayuda."""
    st.header("ℹ️ Información")

    tabs = st.tabs(["Acerca de", "Archivos", "Tecnología", "Contacto"])

    with tabs[0]:
        st.markdown("""
        ## 🚗 Análisis Inteligente de Siniestros Viales Colombia

        Plataforma desarrollada como parte del **proyecto MinTIC** para analizar
        datos de siniestros viales a nivel nacional de Colombia.

        ### ✨ Características principales:
        
        **📂 Gestión de Datasets:**
        - ✅ Soporte multi-ciudad (Palmira, Yopal, Cali, Bogotá, etc.)
        - ✅ Carga dinámica desde datos.gov.co
        - ✅ Agregar, eliminar y cambiar entre datasets
        - ✅ Descarga y limpieza automática
        
        **📊 Análisis Adaptable:**
        - ✅ Detección automática de columnas (tildes, caracteres especiales)
        - ✅ Reportes detallados con calidad de datos
        - ✅ Visualizaciones que se adaptan al schema de cada dataset
        - ✅ Mapas interactivos con coordenadas geográficas
        
        **🤖 Inteligencia Artificial:**
        - ✅ Extracción OCR de documentos legales (Ley 769 de 2002)
        - ✅ Agente Pandas para consultas en lenguaje natural
        - ✅ Integración con Gemini API
        - ✅ Análisis unificado (PDF + datos + IA)
        
        **📊 Reportes Profesionales:**
        - ✅ Score de calidad de datos (0-100)
        - ✅ Detección de outliers y duplicados
        - ✅ Análisis de completitud y recomendaciones
        - ✅ Exportación de reportes en formato texto

        ### 🎯 Objetivos:
        1. Centralizar datos de siniestros viales de Colombia
        2. Proporcionar análisis robusto y escalable
        3. Facilitar la toma de decisiones basadas en datos
        4. Democratizar el acceso a información de seguridad vial

        **👥 Desarrollo:** Equipo MinTIC  
        **📅 Última actualización:** Diciembre 2025  
        **🌐 Alcance:** Nacional (Colombia)
        """)

    with tabs[1]:
        st.markdown("""
        ## 📁 Estructura de Archivos

        ### 📊 Datasets:
        La aplicación maneja múltiples datasets dinámicamente:
        
        **Predeterminados:**
        - `data/siniestros_siniestros_palmira_2022-2024.csv` - Palmira 2022-2024
        - `data/siniestros_siniestros_palmira_2021.csv` - Palmira 2021
        
        **Dinámicos:** (se descargan desde datos.gov.co)
        - `data/siniestros_<nombre_ciudad>.csv` - Datasets agregados por el usuario
        - Configuración en: `data/datasets_config.json`

        ### 📝 Documentos Legales:
        - `data/Ley_769_de_2002.pdf` - Código Nacional de Tránsito de Colombia
        - `data/ocr_cache/` - Caché de texto OCR procesado

        ### 📊 Reportes:
        - `data/reporte_limpieza.txt` - Log de limpieza de datos
        - Reportes descargables desde la interfaz

        ### 📊 Columnas detectadas automáticamente:
        
        **Categorías principales:**
        - Tipo de siniestro: `clase_siniestro`, `tipo_siniestro`, `tipo`
        - Temporal: `jornada`, `fecha`, `hora`, `dia_semana`
        - Geográfica: `zona`, `barrio`, `direccion`, `latitud`, `longitud`
        - Demográfica: `genero`, `edad`, `gravedad`
        - Contexto: `hipotesis`, `autoridad`, `condicion_victima`
        
        🔍 La aplicación detecta automáticamente variaciones:
        - Con/sin tildes: `dirección` vs `direccion`
        - Caracteres especiales: `direcci_n` vs `direccion`
        - Mayúsculas/minúsculas: `DIRECCION` vs `direccion`
        """)

    with tabs[2]:
        st.markdown("""
        ## 🛠️ Stack Tecnológico

        ### 🐍 Backend:
        - **Python 3.13** - Lenguaje principal
        - **Pandas** - Análisis y manipulación de datos
        - **NumPy** - Cálculos numéricos
        - **Requests** - Descarga de datasets desde APIs
        
        ### 🤖 Inteligencia Artificial:
        - **LangChain** - Framework de LLMs
        - **Gemini 2.0 Flash** - Modelo de lenguaje (Google)
        - **Pandas Agent** - Consultas en lenguaje natural
        - **langchain-google-genai** - Integración con Gemini
        
        ### 📄 Procesamiento de Documentos:
        - **Pytesseract** - Motor OCR
        - **pdf2image** - Conversión PDF a imágenes
        - **Pillow (PIL)** - Procesamiento de imágenes
        - **Poppler** - Renderizado de PDFs
        
        ### 🎨 Frontend:
        - **Streamlit** - Framework de aplicación web
        - **Plotly Express** - Gráficos interactivos
        - **st.map()** - Mapas geográficos
        
        ### 📊 Análisis de Datos:
        - **IQR Method** - Detección de outliers
        - **Quality Scoring** - Sistema de puntuación 0-100
        - **Metadata Extraction** - Análisis automático de estructura
        - **Normalización de texto** - Coincidencia flexible de columnas

        ### 💾 Infraestructura:
        - **Git** - Control de versiones
        - **Python venv** - Entorno virtual aislado
        - **JSON** - Persistencia de configuración
        - **datos.gov.co API** - Fuente de datos abiertos

        ### ⚙️ Configuración:
        ```env
        GEMINI_API_KEY=tu_clave_api
        POPPLER_PATH=C:/Program Files/poppler/Library/bin
        ```
        
        ### 📦 Dependencias principales:
        ```
        streamlit>=1.40.0
        pandas>=2.2.0
        langchain>=0.3.12
        langchain-google-genai>=2.0.6
        pytesseract>=0.3.13
        pdf2image>=1.17.0
        plotly>=5.24.1
        requests>=2.32.3
        python-dotenv>=1.0.1
        ```
        """)

    with tabs[3]:
        st.markdown("""
        ## 📧 Información de Contacto

        ### 👥 Equipo del Proyecto:
        - **Desarrollo:** Equipo MinTIC Colombia
        - **Tipo:** Proyecto Académico
        - **Año:** 2025
        
        ### 📚 Documentación Disponible:
        
        **Archivos principales:**
        - `README.md` - Guía general del proyecto
        - `ANALISIS_UNIFICADO.md` - Documentación de análisis
        - `CAMBIOS_OCR.md` - Registro de cambios técnicos
        - `requirements.txt` - Dependencias Python
        
        **Código fuente:**
        - `app/streamlit_app.py` - Aplicación principal
        - `src/mintic_project/` - Módulos backend
          - `data_loader.py` - Gestión de datasets
          - `db_analysis.py` - Análisis y reportes
          - `langchain_integration.py` - OCR y LLM
          - `unified_analyzer.py` - Análisis unificado

        ### 🚀 Cómo Empezar:
        
        **1. Configuración inicial:**
        ```bash
        # Clonar repositorio
        git clone <repo-url>
        cd MinTic-proyecto
        
        # Crear entorno virtual
        python -m venv .venv
        .venv\\Scripts\\activate
        
        # Instalar dependencias
        pip install -r requirements.txt
        ```
        
        **2. Configurar variables:**
        ```bash
        # Crear archivo .env
        GEMINI_API_KEY=tu_clave_aqui
        POPPLER_PATH=C:/Program Files/poppler/Library/bin
        ```
        
        **3. Ejecutar aplicación:**
        ```bash
        streamlit run app/streamlit_app.py
        ```

        ### ❓ Soporte:
        
        **Problemas comunes:**
        - ❌ `GEMINI_API_KEY not found` → Configura `.env` o Streamlit secrets
        - ❌ `Poppler not found` → Instala Poppler y configura PATH
        - ❌ `CSV not found` → Descarga dataset desde panel lateral
        - ❌ `Import errors` → Limpia cache con `Get-ChildItem __pycache__ | Remove-Item`
        
        **Recursos:**
        - 🌐 [datos.gov.co](https://datos.gov.co) - Fuente de datos abiertos
        - 📚 [Streamlit Docs](https://docs.streamlit.io)
        - 🤖 [LangChain Docs](https://python.langchain.com)
        - ✨ [Gemini API](https://ai.google.dev/gemini-api)
        
        ### 🌟 Contribuciones:
        Este es un proyecto académico open-source. Las mejoras y sugerencias
        son bienvenidas a través de pull requests o issues en el repositorio.
        """)


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
