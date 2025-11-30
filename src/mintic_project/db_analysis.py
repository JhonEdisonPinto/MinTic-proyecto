"""Análisis de CSV de siniestros y respuesta de preguntas con Gemini.

Este módulo proporciona funciones para:
- Cargar datos de CSV
- Extraer metadatos y estadísticas
- Responder preguntas sobre los datos usando Gemini

Casos de uso:
- Analizar datos de siniestros desde CSV
- Generar reportes automáticos
- Responder preguntas sobre patrones en los datos
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv_dataset(csv_path: str) -> pd.DataFrame:
    """Cargar CSV y devolver DataFrame con información sobre carga."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    
    df = pd.read_csv(path)
    logger.info(f"✓ CSV cargado: {csv_path} ({len(df)} filas, {len(df.columns)} columnas)")
    return df


def extract_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Extrae metadatos y estadísticas del DataFrame.
    
    Retorna un diccionario con:
    - shape: (filas, columnas)
    - columns: nombres y tipos de datos
    - missing: porcentaje de valores nulos por columna
    - numeric_stats: min, max, mean para columnas numéricas
    - unique_values: count de valores únicos por columna
    """
    metadata = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_percent": {col: (df[col].isna().sum() / len(df) * 100) for col in df.columns},
        "unique_counts": {col: df[col].nunique() for col in df.columns},
        "numeric_stats": {},
        "categorical_samples": {},
    }
    
    # Estadísticas numéricas
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        metadata["numeric_stats"][col] = {
            "min": float(df[col].min()) if not df[col].isna().all() else None,
            "max": float(df[col].max()) if not df[col].isna().all() else None,
            "mean": float(df[col].mean()) if not df[col].isna().all() else None,
            "median": float(df[col].median()) if not df[col].isna().all() else None,
        }
    
    # Muestras de valores categóricos
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        top_values = df[col].value_counts().head(5).to_dict()
        metadata["categorical_samples"][col] = top_values
    
    return metadata


def generate_dataset_report(df: pd.DataFrame, metadata: Optional[Dict] = None) -> str:
    """Genera un reporte textual del dataset para pasarlo a Gemini."""
    if metadata is None:
        metadata = extract_dataset_metadata(df)
    
    rows, cols = metadata["shape"]
    report = f"""
=== REPORTE DEL DATASET ===

📊 DIMENSIONES:
- Filas: {rows:,}
- Columnas: {cols}

📋 COLUMNAS Y TIPOS:
"""
    
    for col, dtype in metadata["dtypes"].items():
        missing = metadata["missing_percent"].get(col, 0)
        unique = metadata["unique_counts"].get(col, 0)
        report += f"  • {col:<35} | Tipo: {dtype:<10} | Nulos: {missing:.1f}% | Únicos: {unique}\n"
    
    # Estadísticas numéricas
    if metadata["numeric_stats"]:
        report += "\n📈 ESTADÍSTICAS NUMÉRICAS:\n"
        for col, stats in metadata["numeric_stats"].items():
            if stats["min"] is not None:
                report += f"  • {col:<35} | Min: {stats['min']:<10.2f} | Max: {stats['max']:<10.2f} | Promedio: {stats['mean']:<10.2f}\n"
    
    # Muestras categóricas
    if metadata["categorical_samples"]:
        report += "\n🏷️  VALORES FRECUENTES (CATEGÓRICOS):\n"
        for col, top_values in metadata["categorical_samples"].items():
            report += f"  • {col}:\n"
            for val, count in list(top_values.items())[:3]:
                report += f"      - {val}: {count} registros\n"
    
    return report


def query_dataset_with_gemini(question: str, df: pd.DataFrame, llm=None) -> str:
    """Responde una pregunta sobre un dataset usando Gemini."""
    from src.mintic_project.langchain_integration import LangChainConfig
    
    if llm is None:
        config = LangChainConfig()
        llm = config.crear_llm()
        if llm is None:
            return "⚠️  No hay LLM disponible. Configura GEMINI_API_KEY."
    
    logger.info(f"❓ Pregunta sobre datos: {question}")
    
    # Generar reporte del dataset
    metadata = extract_dataset_metadata(df)
    report = generate_dataset_report(df, metadata)
    
    # Crear prompt
    prompt = f"""Eres un experto en análisis de datos. Se te proporciona un reporte detallado de un dataset con información sobre siniestros viales.

REPORTE DEL DATASET:
{report}

PREGUNTA: {question}

INSTRUCCIONES:
- Responde basándote en el reporte del dataset
- Si necesitas información más detallada, puedes hacer suposiciones razonables basadas en los datos
- Sé preciso y proporciona números cuando sea posible
- Si la pregunta no puede responderse con la información disponible, indícalo claramente
"""
    
    try:
        logger.info("⏳ Generando respuesta con Gemini...")
        response = llm.invoke(prompt)
        
        if hasattr(response, "content"):
            return response.content
        return str(response)
    except Exception as e:
        logger.error(f"❌ Error generando respuesta: {e}")
        return "⚠️  Error al generar la respuesta."


def analyze_csv_file(csv_path: str, question: str = None, llm=None) -> Dict[str, Any]:
    """Función principal: carga CSV, extrae metadata, y opcionalmente responde preguntas.
    
    Args:
        csv_path: Ruta al archivo CSV
        question: Pregunta opcional sobre los datos
        llm: Instancia de LLM (si None, se crea una)
    
    Returns:
        Dict con metadata, report, y respuesta (si pregunta se proporcionó)
    """
    df = load_csv_dataset(csv_path)
    metadata = extract_dataset_metadata(df)
    report = generate_dataset_report(df, metadata)
    
    result = {
        "file": csv_path,
        "shape": metadata["shape"],
        "columns": metadata["columns"],
        "metadata": metadata,
        "report": report,
    }
    
    if question:
        result["question"] = question
        result["answer"] = query_dataset_with_gemini(question, df, llm)
    
    return result


# =============================================================================
# LLM + Pandas Agent: consultas directamente sobre el DataFrame
# =============================================================================
def create_pandas_agent(df: pd.DataFrame, model: str = "gemini-2.5-flash", dangerous: bool = False):
    """Crear un agente de LangChain para consultar el DataFrame con Gemini.

    Parámetros:
        df: DataFrame objetivo
        model: nombre del modelo Gemini
        dangerous: si True habilita `allow_dangerous_code` (ejecución arbitraria). Úsalo solo si confías en el entorno.

    Si `dangerous=False` y el agente requiere ejecución insegura, se devuelve None y se usará un fallback seguro.
    """
    # Define el system prompt del agente para enriquecer las respuestas
    agent_prefix = (
        "Eres un analista de datos experto especializado en siniestros viales. "
        "Tu objetivo es analizar el DataFrame y proporcionar respuestas completas y estructuradas que incluyan:\n"
        "1. Respuesta directa a la pregunta\n"
        "2. Contexto relevante (tendencias, distribuciones, comparaciones)\n"
        "3. Insights y observaciones clave\n"
        "4. Recomendaciones prácticas cuando sea aplicable\n\n"
        "Usa operaciones de Pandas para extraer información precisa y fundamenta tus respuestas con datos concretos."
    )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        logger.error(f"❌ Falta langchain-google-genai: {e}")
        return None

    # Verificar dependencia tabulate (log informativo, no bloquear si falta)
    try:
        import tabulate  # noqa: F401
        logger.info(f"tabulate detectado: versión={getattr(tabulate, '__version__', 'desconocida')}")
    except ImportError:
        logger.warning("⚠️ 'tabulate' no detectado. Se usará fallback si el agente falla. Ejecuta 'pip install tabulate' para capacidades completas.")

    # Intentar distintas ubicaciones de create_pandas_dataframe_agent
    create_agent_fn = None
    for path in [
        "langchain_experimental.agents",
        "langchain_experimental.agents.agent_toolkits",
        "langchain_experimental.openai_minutes.agents",  # fallback hipotético
    ]:
        try:
            module = __import__(path, fromlist=["create_pandas_dataframe_agent"])
            create_agent_fn = getattr(module, "create_pandas_dataframe_agent")
            break
        except Exception:
            continue

    if not create_agent_fn:
        logger.error("❌ No se encontró create_pandas_dataframe_agent. Instala/actualiza 'langchain-experimental'.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(model=model, convert_system_message_to_human=True)
        agent = create_agent_fn(
            llm,
            df,
            verbose=False,
            allow_dangerous_code=dangerous,
            prefix=agent_prefix,
            include_df_in_prompt=False,
        )
        return agent
    except Exception as e:
        logger.error(f"❌ Error creando agente Pandas: {e}")
        return None


def safe_dataframe_summary(df: pd.DataFrame) -> str:
    """Generar un resumen seguro del DataFrame para que el LLM pueda razonar sin ejecutar código.

    Incluye: shape, columnas, top 5 de categóricas, describe de numéricas.
    """
    lines = []
    lines.append(f"Shape: {df.shape}")
    lines.append("Columnas:")
    for c in df.columns:
        lines.append(f"  - {c} ({df[c].dtype})")
    # Numéricas
    num_cols = df.select_dtypes(include=["number"]).columns
    if num_cols.any():
        lines.append("\nEstadísticas numéricas (describe):")
        desc = df[num_cols].describe().to_string()
        lines.append(desc)
    # Categóricas
    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        vc = df[c].value_counts().head(5)
        lines.append(f"\nTop valores {c}:")
        for val, cnt in vc.items():
            lines.append(f"  {val}: {cnt}")
    return "\n".join(lines)[:8000]


def _format_structured_answer(title: str, answer: str, notes: Optional[str] = None, preview_df: Optional[pd.DataFrame] = None) -> str:
    """Formatea una respuesta en secciones cortas y escaneables."""
    sections = []
    sections.append(f"### {title}")
    sections.append(f"**Respuesta:** {answer.strip()}")
    if preview_df is not None and not preview_df.empty:
        try:
            from tabulate import tabulate
            table = tabulate(preview_df.head(10), headers=preview_df.columns, tablefmt="github")
            sections.append("**Vista previa:**\n" + table)
        except Exception:
            sections.append("**Vista previa:** (instala 'tabulate' para tabla)\n" + preview_df.head(10).to_string())
    if notes:
        sections.append(f"**Notas:** {notes.strip()}")
    return "\n\n".join(sections)


def query_with_pandas_agent(question: str, df: pd.DataFrame, dangerous: bool = False) -> str:
    """Intentar responder usando el agente Pandas; si no se puede, usar fallback seguro.

    Fallback: se construye un contexto con resumen del DF y se pasa al LLM como texto.
    """
    agent = create_pandas_agent(df, dangerous=dangerous)
    if agent is None:
        # Fallback seguro: usar reporte textual y Gemini normal
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", convert_system_message_to_human=True)
            context = safe_dataframe_summary(df)
            prompt = (
                "Eres un analista de datos experto especializado en siniestros viales. Tu tarea es proporcionar un análisis completo y profesional.\n\n"
                "INSTRUCCIONES:\n"
                "1. Responde la pregunta de forma directa y precisa\n"
                "2. Proporciona contexto relevante: tendencias, patrones, distribuciones\n"
                "3. Incluye insights clave y observaciones importantes\n"
                "4. Cuando sea aplicable, ofrece recomendaciones prácticas basadas en los datos\n"
                "5. Estructura tu respuesta con títulos y secciones claras\n"
                "6. Usa números concretos y porcentajes cuando estén disponibles\n\n"
                f"DATOS DEL DATASET:\n{context}\n\n"
                f"PREGUNTA DEL USUARIO: {question}\n\n"
                "ANÁLISIS COMPLETO:"
            )
            resp = llm.invoke(prompt)
            raw = getattr(resp, "content", str(resp))[:4000]
            return _format_structured_answer(
                title="Análisis del DataFrame (Fallback)",
                answer=raw,
                notes=("Consulta ejecutada en modo seguro sin código Pandas. "
                       "Activa 'modo avanzado' para operaciones directas si es necesario."),
            )
        except Exception as e:
            logger.error(f"❌ Fallback también falló: {e}")
            return "⚠️  No se pudo responder la pregunta (falta agente y fallback falló)."
    try:
        respuesta = agent.invoke({"input": question})
        # Algunos agentes devuelven dict con 'output' y 'intermediate_steps'
        out_text = None
        preview = None
        if isinstance(respuesta, dict):
            out_text = str(respuesta.get("output", "")).strip()
            # Intentar extraer algún dataframe utilizado en pasos intermedios (si disponible)
            steps = respuesta.get("intermediate_steps") or []
            for step in steps:
                try:
                    # step puede ser tupla (AgentAction, str) o similar
                    if isinstance(step, tuple) and hasattr(step[0], "tool_input"):
                        # No confiable; omitir
                        pass
                except Exception:
                    pass
        else:
            out_text = str(respuesta).strip()

        if not out_text:
            out_text = "(Sin salida textual del agente)"
        return _format_structured_answer(
            title="Resultado del Agente de Pandas",
            answer=out_text[:4000],
            preview_df=preview,
            notes=("El agente puede ejecutar operaciones de Pandas. "
                   "Si la respuesta parece ambigua, intenta ser más específico."),
        )
    except Exception as e:
        logger.error(f"❌ Error ejecutando agente Pandas: {e}")
        return "⚠️  Error ejecutando el agente Pandas. Prueba con otra pregunta o activa modo avanzado."


if __name__ == "__main__":
    # Prueba: analizar los CSVs disponibles
    csv_files = [
        "data/siniestros_1_limpio.csv",
        "data/siniestros_2_limpio.csv",
    ]
    
    for csv_path in csv_files:
        if Path(csv_path).exists():
            print(f"\n{'='*80}")
            print(f"ANALIZANDO: {csv_path}")
            print('='*80)
            
            result = analyze_csv_file(
                csv_path,
                question="¿Cuál es el tipo de siniestro más frecuente?",
            )
            
            print(result["report"])
            if "answer" in result:
                print(f"\n💬 RESPUESTA A LA PREGUNTA:\n{result['answer']}\n")
