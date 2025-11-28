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
