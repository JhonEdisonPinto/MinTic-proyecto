"""Configuración y utilidades para OCR usando pytesseract

Este módulo proporciona funciones para extraer texto de PDFs usando OCR
y responder preguntas sobre el contenido usando Gemini API.
"""
import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# FUNCIONES OCR BASADAS EN PYTESSERACT
# ============================================================================

def extract_text_from_pdf_ocr(
    pdf_path: str,
    cache_txt: bool = True,
    cache_dir: str = "data/ocr_cache"
) -> str:
    """Extrae texto de un PDF usando OCR (pytesseract).
    
    CARACTERÍSTICAS:
    - Usa pytesseract para extraer texto de imágenes
    - Cachea el resultado en un archivo .txt para no re-procesar
    - Intenta múltiples métodos de conversión PDF->imagen
    
    Args:
        pdf_path: Ruta del PDF
        cache_txt: Si True, guarda resultado en .txt y lo reutiliza
        cache_dir: Directorio para guardar archivos .txt cacheados
    
    Returns:
        Texto extraído del PDF
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")
    
    logger.info(f"📄 Extrayendo texto OCR de: {pdf_path}")

    # 1. Verificar si hay cache
    if cache_txt:
        cache_path = Path(cache_dir) / f"{pdf_path_obj.stem}.txt"
        if cache_path.exists():
            logger.info(f"✓ Usando cache: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

    # 2. Intentar extracción de texto sin OCR usando pypdf (más rápido, sin dependencias externas)
    try:
        from pypdf import PdfReader

        logger.info("⏳ Intentando extracción con pypdf (sin OCR)...")
        reader = PdfReader(str(pdf_path_obj))
        pages_text = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            pages_text.append(f"\n--- PÁGINA {i+1} ---\n{txt}\n")

        full_text = "".join(pages_text).strip()
        if full_text and len(full_text) > 100:
            logger.info(f"✓ Extracción con pypdf exitosa: {len(full_text)} caracteres")
            if cache_txt:
                cache_dir_obj = Path(cache_dir)
                cache_dir_obj.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir_obj / f"{pdf_path_obj.stem}.txt"
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                logger.info(f"✓ Texto guardado en cache: {cache_path}")
            return full_text
        else:
            logger.info("⚠️  Extracción con pypdf devolvió texto vacío o corto, proceder a OCR")
    except Exception as e:
        logger.info(f"⚠️  pypdf no disponible o falló: {e}")

    # 3. Si pypdf no pudo, usar OCR: convertir a imágenes y aplicar pytesseract
    try:
        import pytesseract
        from pdf2image import convert_from_path

        poppler_path = os.getenv("POPPLER_PATH") or os.getenv("POPPLER_BIN")
        logger.info("⏳ Convirtiendo PDF a imágenes para OCR...")
        if poppler_path:
            images = convert_from_path(str(pdf_path_obj), poppler_path=poppler_path)
        else:
            images = convert_from_path(str(pdf_path_obj))

        logger.info(f"✓ {len(images)} páginas convertidas")

        logger.info("⏳ Ejecutando OCR (esto puede tardar)...")
        all_text = ""
        for i, img in enumerate(images):
            logger.info(f"   Procesando página {i+1}/{len(images)}...")
            text = pytesseract.image_to_string(img, lang="spa+eng")
            all_text += f"\n--- PÁGINA {i+1} ---\n{text}\n"

        logger.info(f"✓ OCR completado: {len(all_text)} caracteres extraídos")

        # 4. Guardar en cache
        if cache_txt:
            cache_dir_obj = Path(cache_dir)
            cache_dir_obj.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir_obj / f"{pdf_path_obj.stem}.txt"
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(all_text)
            logger.info(f"✓ Texto guardado en cache: {cache_path}")

        return all_text
    except ImportError as e:
        logger.error(f"❌ Falta dependencia Python para OCR: {e}")
        logger.error("   Instala: pip install pytesseract pdf2image pillow pypdf")
        raise
    except Exception as e:
        # Detectar error común de pdf2image cuando poppler no está disponible
        msg = str(e)
        if "Unable to get page count" in msg or "page count" in msg or "Poppler" in msg:
            logger.error("❌ Error en OCR: %s", e)
            logger.error("   Parece que Poppler no está instalado o no está en PATH.")
            logger.error("   En Windows descarga Poppler: https://github.com/oschwartz10612/poppler-windows/releases")
            logger.error("   O configura la variable POPPLER_PATH en tu .env con la ruta al binario 'poppler-xx/bin'")
        else:
            logger.error(f"❌ Error en OCR: {e}")
        raise


def _create_gemini_llm(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 2048
):
    """Crear y devolver una instancia de ChatGoogleGenerativeAI (Gemini).
    
    CARACTERÍSTICAS:
    - Lee GEMINI_API_KEY del .env
    - Manejo de errores mejorado
    - Verificación de API key
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("❌ GEMINI_API_KEY no encontrada en variables de entorno")
        logger.error("   Configúrala en tu archivo .env")
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        logger.info(f"✓ LLM Gemini creado: {model}")
        return llm
    except Exception as e:
        logger.error(f"❌ Error creando LLM Gemini: {e}")
        return None


def answer_with_ocr(
    question: str,
    pdf_path: str = "data/Ley_769_de_2002.pdf",
    llm=None,
    max_context_length: int = 100000
) -> str:
    """Responde preguntas sobre un PDF usando OCR + Gemini.
    
    CARACTERÍSTICAS:
    - Extrae texto con OCR
    - Usa Gemini para responder preguntas
    - Limita contexto para no exceder límites de token
    
    Args:
        question: La pregunta a responder
        pdf_path: Ruta del PDF
        llm: Instancia de LLM (si None, se crea una)
        max_context_length: Máximo de caracteres de contexto
    
    Returns:
        Respuesta de Gemini
    """
    
    logger.info(f"❓ Pregunta: {question}")
    
    # 1. Extraer texto del PDF con OCR
    try:
        full_text = extract_text_from_pdf_ocr(pdf_path)
    except Exception as e:
        logger.error(f"❌ Error extrayendo texto: {e}")
        return "⚠️  No pude extraer texto del PDF con OCR."
    
    # 2. Limitar contexto si es muy largo
    if len(full_text) > max_context_length:
        logger.warning(f"⚠️  Contexto muy largo ({len(full_text)} chars), limitando a {max_context_length}")
        # Intentar usar solo las primeras secciones relevantes
        full_text = full_text[:max_context_length]
    
    logger.info(f"✓ Contexto preparado: {len(full_text)} caracteres")
    
    # 3. Preparar LLM
    if llm is None:
        llm = _create_gemini_llm()
        if llm is None:
            return "⚠️  No hay LLM disponible. Configura GEMINI_API_KEY."

    # 4. Crear prompt
    prompt = f"""Eres un experto en la Ley 769 de 2002 (Código Nacional de Tránsito de Colombia).

CONTENIDO del PDF (extraído con OCR):
{full_text}

PREGUNTA: {question}

INSTRUCCIONES:
- Responde SOLO basándote en el contenido anterior
- Cita artículos específicos cuando sea posible
- Si la información no está en el contenido, di claramente "No encuentro esta información en el PDF"
- Sé preciso y conciso
- Entiende que el OCR puede tener errores menores de ortografía
"""

    # 5. Invocar LLM
    try:
        logger.info("⏳ Generando respuesta con Gemini...")
        response = llm.invoke(prompt)
        
        if hasattr(response, "content"):
            result = response.content
        else:
            result = str(response)
        
        logger.info("✓ Respuesta generada")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error generando respuesta: {e}")
        return "⚠️  Error al generar la respuesta."


# ============================================================================
# CLASES PRINCIPALES
# ============================================================================

class LangChainConfig:
    """Configuración para integrar LangChain con Gemini API Key."""

    def __init__(
        self,
        provider: str = "google",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 25000,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Lee la API KEY desde el .env
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")

        if provider == "google" and not self.gemini_api_key:
            logger.warning("⚠️  GEMINI_API_KEY no encontrada en el entorno")

    def crear_llm(self):
        """Crear LLM usando Gemini API Key."""
        if not self.gemini_api_key:
            logger.error("❌ GEMINI_API_KEY no configurada")
            return None

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.gemini_api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                convert_system_message_to_human=True,
            )
        except Exception as e:
            logger.error(f"❌ Error creando LLM: {e}")
            return None


class OCRAnalyzer:
    """Analizador de PDFs usando OCR + Gemini."""

    def __init__(self, pdf_path: str = "data/Ley_769_de_2002.pdf", config: Optional[LangChainConfig] = None):
        """Inicializar analizador OCR.
        
        Args:
            pdf_path: Ruta del PDF a analizar
            config: Configuración de LangChain (si None, se crea una)
        """
        self.pdf_path = pdf_path
        self.config = config or LangChainConfig()
        self.llm = self.config.crear_llm()
        self.extracted_text = None

    def extraer_texto(self) -> str:
        """Extraer texto del PDF usando OCR."""
        if self.extracted_text is not None:
            logger.info("✓ Usando texto en caché")
            return self.extracted_text
        
        try:
            self.extracted_text = extract_text_from_pdf_ocr(self.pdf_path)
            return self.extracted_text
        except Exception as e:
            logger.error(f"❌ Error extrayendo texto: {e}")
            return ""

    def responder_pregunta(self, pregunta: str) -> str:
        """Responder una pregunta sobre el PDF usando OCR.
        
        Args:
            pregunta: La pregunta a responder
        
        Returns:
            Respuesta de Gemini
        """
        if self.llm is None:
            logger.error("❌ LLM no disponible")
            return "⚠️  Configura GEMINI_API_KEY"
        
        return answer_with_ocr(pregunta, self.pdf_path, llm=self.llm)

    def preguntas_multiples(self, preguntas: List[str]) -> Dict[str, str]:
        """Responder múltiples preguntas.
        
        Args:
            preguntas: Lista de preguntas
        
        Returns:
            Diccionario {pregunta: respuesta}
        """
        resultados = {}
        for pregunta in preguntas:
            logger.info(f"Procesando: {pregunta}")
            resultados[pregunta] = self.responder_pregunta(pregunta)
        return resultados


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SISTEMA OCR - ANÁLISIS CON PYTESSERACT")
    print("=" * 60)

    # Verificar configuración
    config = LangChainConfig()
    print(f"\n✓ Configuración:")
    print(f"  - Modelo: {config.model}")
    print(f"  - API Key: {'Configurada' if config.gemini_api_key else 'NO configurada'}")

    # Probar OCR
    if config.gemini_api_key:
        print("\n🧪 Probando consulta OCR...")
        analyzer = OCRAnalyzer()
        respuesta = analyzer.responder_pregunta("Me choqué con otro vehículo, ¿qué debo hacer?")
        print(f"\n📝 Respuesta:\n{respuesta}")
    else:
        print("\n⚠️  Configura GEMINI_API_KEY para probar el sistema")