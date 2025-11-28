"""Integración: Análisis de PDF (OCR) + CSV (metadata) + Gemini.

Este módulo combina:
1. Extracción OCR del PDF legal
2. Análisis de metadata del CSV de siniestros
3. Generación de respuestas unificadas con Gemini

Permite responder preguntas que cruzan información legal con datos de siniestros.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class UnifiedAnalyzer:
    """Analizador que combina OCR (legal) + CSV (datos) + Gemini (respuestas)."""
    
    def __init__(
        self,
        pdf_path: str = "data/Ley_769_de_2002.pdf",
        csv_path: str = "data/siniestros_1_limpio.csv",
        config=None
    ):
        """Inicializar el analizador.
        
        Args:
            pdf_path: Ruta del PDF legal (Ley 769)
            csv_path: Ruta del CSV de siniestros
            config: Configuración de LangChain (si None, se crea una)
        """
        from src.mintic_project.langchain_integration import (
            OCRAnalyzer,
            LangChainConfig,
        )
        from src.mintic_project.db_analysis import (
            load_csv_dataset,
            extract_dataset_metadata,
            generate_dataset_report,
        )
        
        self.pdf_path = pdf_path
        self.csv_path = csv_path
        self.config = config or LangChainConfig()
        self.llm = self.config.crear_llm()
        
        # Cargar análisis OCR
        if Path(pdf_path).exists():
            self.ocr_analyzer = OCRAnalyzer(pdf_path, config=self.config)
            self.pdf_text = self.ocr_analyzer.extraer_texto()
            logger.info(f"✓ PDF cargado: {len(self.pdf_text)} caracteres")
        else:
            logger.warning(f"⚠️  PDF no encontrado: {pdf_path}")
            self.ocr_analyzer = None
            self.pdf_text = None
        
        # Cargar análisis CSV
        if Path(csv_path).exists():
            self.df = load_csv_dataset(csv_path)
            self.csv_metadata = extract_dataset_metadata(self.df)
            self.csv_report = generate_dataset_report(self.df, self.csv_metadata)
            logger.info(f"✓ CSV cargado: {self.df.shape[0]} filas")
        else:
            logger.warning(f"⚠️  CSV no encontrado: {csv_path}")
            self.df = None
            self.csv_metadata = None
            self.csv_report = None
    
    def responder_pregunta(self, question: str) -> str:
        """Responder pregunta usando PDF + CSV + Gemini.
        
        Combina:
        1. Contexto legal del PDF
        2. Metadatos/estadísticas del CSV
        3. Capacidad de análisis de Gemini
        """
        if not self.llm:
            return "⚠️  No hay LLM disponible. Configura GEMINI_API_KEY."
        
        logger.info(f"❓ Pregunta unificada: {question}")
        
        # Construir contexto combinado
        contexto_pdf = ""
        if self.pdf_text:
            # Usar solo los primeros 5000 caracteres del PDF para no saturar el prompt
            contexto_pdf = self.pdf_text[:5000]
        
        contexto_csv = ""
        if self.csv_report:
            contexto_csv = self.csv_report
        
        # Crear prompt unificado
        prompt = f"""Eres un experto en seguridad vial y análisis de siniestros. 
Se te proporciona:
1. Información legal del código de tránsito (PDF)
2. Estadísticas reales de siniestros viales (datos CSV)

{"=== CONTEXTO LEGAL (Extracto de Ley 769 de 2002) ===" if contexto_pdf else ""}
{contexto_pdf}

{"=== ESTADÍSTICAS DE SINIESTROS ===" if contexto_csv else ""}
{contexto_csv}

PREGUNTA: {question}

INSTRUCCIONES:
- Responde combinando la información legal con los datos reales
- Referencia artículos específicos cuando sea relevante
- Usa números y estadísticas del dataset para sustentar tu respuesta
- Sé conciso pero informativo
- Si alguna información no está disponible, indícalo claramente
"""
        
        try:
            logger.info("⏳ Generando respuesta unificada con Gemini...")
            response = self.llm.invoke(prompt)
            
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"❌ Error generando respuesta: {e}")
            return "⚠️  Error al generar la respuesta."
    
    def generar_resumen_general(self) -> str:
        """Generar un resumen general del estado de los datos y contexto legal."""
        logger.info("Generando resumen general...")
        
        prompts_summary = f"""Eres un experto en seguridad vial. Resume la siguiente información en 3-4 párrafos:

1. Marco legal: Ley 769 de 2002 (primeras 3000 caracteres)
{self.pdf_text[:3000] if self.pdf_text else "No disponible"}

2. Datos de siniestros: estadísticas generales
{self.csv_report if self.csv_report else "No disponible"}

Genera un resumen ejecutivo que combine ambas perspectivas.
"""
        
        try:
            response = self.llm.invoke(prompts_summary)
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return "⚠️  Error generando resumen."
    
    def responder_multiples_preguntas(self, preguntas: list) -> Dict[str, str]:
        """Responder múltiples preguntas sobre los datos."""
        resultados = {}
        for i, pregunta in enumerate(preguntas, 1):
            logger.info(f"Procesando pregunta {i}/{len(preguntas)}: {pregunta}")
            resultados[pregunta] = self.responder_pregunta(pregunta)
        return resultados


if __name__ == "__main__":
    print("="*80)
    print("ANALIZADOR UNIFICADO: OCR + CSV + Gemini")
    print("="*80)
    
    # Crear analizador
    analyzer = UnifiedAnalyzer()
    
    # Ejemplo: preguntas
    preguntas_ejemplo = [
        "¿Cuál es el tipo de siniestro más frecuente en los datos y qué dice la ley al respecto?",
        "¿En qué jornada (mañana, tarde, noche) ocurren más siniestros?",
        "¿Cuál es el género más afectado en los siniestros y cuáles son las implicaciones legales?",
    ]
    
    for pregunta in preguntas_ejemplo:
        print(f"\n{'='*80}")
        print(f"P: {pregunta}")
        print('='*80)
        respuesta = analyzer.responder_pregunta(pregunta)
        print(f"R: {respuesta}\n")
