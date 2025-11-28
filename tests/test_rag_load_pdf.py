"""Prueba pytest para RAG: indexa y consulta el PDF Ley_769_de_2002.pdf

Requisitos:
"""
"""Prueba pytest para OCR: extrae y analiza el PDF Ley_769_de_2002.pdf

Requisitos:
- Coloca `Ley_769_de_2002.pdf` en la carpeta `data/`.
- Tesseract-OCR debe estar instalado en el sistema.
- Si `GEMINI_API_KEY` está definido en el entorno, el test intentará hacer preguntas.
"""
import os
from pathlib import Path
import pytest

from dotenv import load_dotenv
load_dotenv()

from src.mintic_project.langchain_integration import (
    extract_text_from_pdf_ocr,
    OCRAnalyzer,
    LangChainConfig,
)


def test_ocr_extraction():
    """Prueba: Extracción de texto OCR del PDF.
    
    Verifica que:
    1. El archivo PDF existe
    2. Se puede extraer texto usando OCR
    3. El texto es no-vacío
    """
    data_dir = Path("data")
    pdf_path = data_dir / "Ley_769_de_2002.pdf"

    if not pdf_path.exists():
        pytest.skip("PDF no encontrado")

    # Extraer texto
    text = extract_text_from_pdf_ocr(str(pdf_path), cache_txt=True)
    
    # Verificaciones
    assert text is not None, "OCR debería retornar texto"
    assert len(text) > 100, "OCR debería extraer más de 100 caracteres"
    assert "Ley" in text or "ley" in text or "769" in text, "El texto debería contener referencias a la ley"


def test_ocr_analyzer_creation():
    """Prueba: Creación de OCRAnalyzer.
    
    Verifica que:
    1. Se puede instanciar OCRAnalyzer
    2. Se configura correctamente
    """
    data_dir = Path("data")
    pdf_path = data_dir / "Ley_769_de_2002.pdf"

    if not pdf_path.exists():
        pytest.skip("PDF no encontrado")

    analyzer = OCRAnalyzer(str(pdf_path))
    assert analyzer is not None, "OCRAnalyzer debería instanciarse"
    assert analyzer.pdf_path == str(pdf_path), "PDF path debería guardarse"


def test_ocr_with_gemini():
    """Prueba: Responder preguntas con Gemini (solo si API key existe).
    
    Verifica que:
    1. GEMINI_API_KEY está configurada
    2. Se puede hacer una pregunta al PDF
    3. Se recibe una respuesta no-vacía
    """
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY no configurada")

    data_dir = Path("data")
    pdf_path = data_dir / "Ley_769_de_2002.pdf"

    if not pdf_path.exists():
        pytest.skip("PDF no encontrado")

    analyzer = OCRAnalyzer(str(pdf_path))
    
    # Hacer una pregunta simple
    response = analyzer.responder_pregunta("¿Cuál es el tema principal?")
    
    assert response is not None, "Debería recibir una respuesta"
    assert isinstance(response, str), "La respuesta debería ser string"
    assert len(response) > 10, "La respuesta debería tener contenido"
