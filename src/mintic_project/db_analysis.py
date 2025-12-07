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


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluar calidad del dataset con métricas detalladas.
    
    Retorna:
        - score: porcentaje de calidad general (0-100)
        - completeness: % de celdas no nulas
        - duplicates: cantidad de filas duplicadas
        - outliers: columnas con posibles outliers
        - issues: lista de problemas detectados
    """
    total_cells = df.shape[0] * df.shape[1]
    non_null_cells = df.count().sum()
    completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
    
    duplicates = df.duplicated().sum()
    
    issues = []
    outliers = {}
    
    # Detectar outliers en columnas numéricas (IQR)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outlier_count > 0:
            outliers[col] = int(outlier_count)
    
    # Issues detectados
    if completeness < 90:
        issues.append(f"Completitud baja: {completeness:.1f}%")
    if duplicates > 0:
        issues.append(f"{duplicates} filas duplicadas")
    if outliers:
        issues.append(f"{len(outliers)} columnas con outliers")
    
    # Columnas con muchos nulos
    null_pct = (df.isnull().sum() / len(df) * 100)
    high_null_cols = null_pct[null_pct > 20].to_dict()
    if high_null_cols:
        issues.append(f"{len(high_null_cols)} columnas con >20% nulos")
    
    # Score ponderado
    score = (
        completeness * 0.5 +  # 50% peso
        (100 - (duplicates / len(df) * 100)) * 0.3 +  # 30% peso
        (100 - min(len(outliers) / max(len(numeric_cols), 1) * 100, 100)) * 0.2  # 20% peso
    )
    
    return {
        "score": round(score, 1),
        "completeness": round(completeness, 1),
        "duplicates": duplicates,
        "outliers": outliers,
        "high_null_columns": {k: round(v, 1) for k, v in high_null_cols.items()},
        "issues": issues,
        "total_rows": len(df),
        "total_columns": len(df.columns)
    }


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
    """Genera un reporte textual robusto y detallado del dataset."""
    if metadata is None:
        metadata = extract_dataset_metadata(df)
    
    rows, cols = metadata["shape"]
    
    # Calcular métricas de calidad
    quality = check_data_quality(df)
    
    # Detectar tipos de columnas
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    
    # Memoria utilizada
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    report = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                   📊 REPORTE DETALLADO DEL DATASET                    ║
╚═══════════════════════════════════════════════════════════════════════╝

🎯 CALIDAD DE DATOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Score de Calidad:        {quality['score']}% {'🟢' if quality['score'] >= 80 else '🟡' if quality['score'] >= 60 else '🔴'}
  • Completitud:             {quality['completeness']}%
  • Filas Duplicadas:        {quality['duplicates']:,}
  • Columnas con Outliers:   {len(quality['outliers'])}

📐 DIMENSIONES Y ESTRUCTURA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Total de Filas:          {rows:,}
  • Total de Columnas:       {cols}
  • Celdas Totales:          {rows * cols:,}
  • Memoria Utilizada:       {memory_mb:.2f} MB
  • Columnas Numéricas:      {len(numeric_cols)}
  • Columnas Categóricas:    {len(categorical_cols)}
  • Columnas Fecha/Hora:     {len(datetime_cols)}

📋 DETALLE DE COLUMNAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    # Tabla de columnas con más información
    report += "\n\n  {:<30} | {:<12} | {:>8} | {:>10} | {:>10}".format(
        "Columna", "Tipo", "Nulos%", "Únicos", "Cardinalidad"
    )
    report += "\n  " + "─" * 88
    
    for col in df.columns:
        dtype = str(metadata["dtypes"][col])
        missing = metadata["missing_percent"].get(col, 0)
        unique = metadata["unique_counts"].get(col, 0)
        cardinality = (unique / rows * 100) if rows > 0 else 0
        
        # Indicador de tipo
        if 'int' in dtype or 'float' in dtype:
            tipo_icon = "🔢"
        elif 'object' in dtype:
            tipo_icon = "📝"
        elif 'datetime' in dtype:
            tipo_icon = "📅"
        else:
            tipo_icon = "❓"
        
        report += f"\n  {col:<30} | {tipo_icon} {dtype:<9} | {missing:>7.1f}% | {unique:>10,} | {cardinality:>9.1f}%"
    
    # Estadísticas numéricas detalladas
    if metadata["numeric_stats"]:
        report += "\n\n📈 ESTADÍSTICAS NUMÉRICAS DETALLADAS:\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        for col, stats in metadata["numeric_stats"].items():
            if stats["min"] is not None:
                rango = stats['max'] - stats['min']
                outlier_count = quality['outliers'].get(col, 0)
                
                report += f"\n  🔢 {col}:\n"
                report += f"     • Mínimo:      {stats['min']:>15,.2f}\n"
                report += f"     • Máximo:      {stats['max']:>15,.2f}\n"
                report += f"     • Promedio:    {stats['mean']:>15,.2f}\n"
                report += f"     • Mediana:     {stats['median']:>15,.2f}\n"
                report += f"     • Rango:       {rango:>15,.2f}\n"
                if outlier_count > 0:
                    report += f"     • ⚠️ Outliers:  {outlier_count:>15,} valores atípicos\n"
    
    # Análisis categórico mejorado
    if metadata["categorical_samples"]:
        report += "\n\n🏷️  ANÁLISIS DE VARIABLES CATEGÓRICAS:\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        for col, top_values in metadata["categorical_samples"].items():
            total_in_col = df[col].notna().sum()
            report += f"\n  📝 {col} ({metadata['unique_counts'][col]} valores únicos):\n"
            
            for i, (val, count) in enumerate(list(top_values.items())[:5], 1):
                pct = (count / total_in_col * 100) if total_in_col > 0 else 0
                bar_length = int(pct / 2)  # Barra visual
                bar = "█" * bar_length
                report += f"     {i}. {str(val)[:40]:<40} │ {bar:<50} {count:>7,} ({pct:>5.1f}%)\n"
    
    # Problemas detectados
    if quality['issues']:
        report += "\n\n⚠️  PROBLEMAS DETECTADOS:\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for issue in quality['issues']:
            report += f"  ❌ {issue}\n"
    
    # Columnas con alta proporción de nulos
    if quality['high_null_columns']:
        report += "\n\n🕳️  COLUMNAS CON ALTA PROPORCIÓN DE NULOS (>20%):\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for col, pct in sorted(quality['high_null_columns'].items(), key=lambda x: x[1], reverse=True):
            report += f"  • {col:<35} {pct:>6.1f}% nulos\n"
    
    # Recomendaciones
    report += "\n\n💡 RECOMENDACIONES:\n"
    report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    recommendations = []
    
    if quality['duplicates'] > 0:
        recommendations.append(f"  ✓ Eliminar {quality['duplicates']} filas duplicadas")
    
    if quality['high_null_columns']:
        recommendations.append(f"  ✓ Analizar {len(quality['high_null_columns'])} columnas con >20% nulos (imputar o eliminar)")
    
    if quality['outliers']:
        recommendations.append(f"  ✓ Revisar outliers en {len(quality['outliers'])} columnas numéricas")
    
    if quality['score'] < 80:
        recommendations.append("  ✓ Realizar limpieza profunda para mejorar calidad general")
    
    # Recomendaciones por tipo de columna
    low_cardinality = [col for col, cnt in metadata['unique_counts'].items() 
                       if cnt < 10 and col in categorical_cols]
    if low_cardinality:
        recommendations.append(f"  ✓ Considerar codificar {len(low_cardinality)} columnas de baja cardinalidad")
    
    high_cardinality = [col for col, cnt in metadata['unique_counts'].items() 
                        if cnt > rows * 0.8 and col in categorical_cols]
    if high_cardinality:
        recommendations.append(f"  ✓ Evaluar {len(high_cardinality)} columnas de alta cardinalidad (posibles IDs)")
    
    if recommendations:
        report += "\n".join(recommendations)
    else:
        report += "  ✅ Dataset en excelente estado, no se detectaron problemas críticos\n"
    
    report += "\n\n" + "═" * 75 + "\n"
    
    return report


# query_dataset_with_gemini reemplazado por query_with_pandas_agent (línea ~284)

# analyze_csv_file reemplazado por funciones modulares (load_csv_dataset, query_with_pandas_agent)

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
    # Script de prueba: cargar CSV y mostrar estadísticas
    import sys
    
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/siniestros_1_limpio.csv"
    
    try:
        df = load_csv_dataset(csv_file)
        print(f"\n✓ CSV cargado: {df.shape}")
        
        metadata = extract_dataset_metadata(df)
        report = generate_dataset_report(df, metadata)
        
        print("\n" + "="*60)
        print(report)
        print("="*60)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
