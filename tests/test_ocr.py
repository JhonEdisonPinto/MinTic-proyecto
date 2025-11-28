#!/usr/bin/env python3
"""Script de prueba para el sistema OCR."""

import os
import sys
from pathlib import Path

# Asegurar que podemos importar desde src
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.mintic_project.langchain_integration import (
    extract_text_from_pdf_ocr,
    OCRAnalyzer,
    LangChainConfig
)

load_dotenv()

def test_ocr_extraction():
    """Prueba: Extracción de texto con OCR."""
    print("\n" + "="*60)
    print("TEST 1: Extracción OCR de PDF")
    print("="*60)
    
    pdf_path = "data/Ley_769_de_2002.pdf"
    if not Path(pdf_path).exists():
        print(f"⚠️  PDF no encontrado: {pdf_path}")
        return False
    
    try:
        print(f"📄 Extrayendo texto de: {pdf_path}")
        texto = extract_text_from_pdf_ocr(pdf_path, cache_txt=True)
        
        if texto and len(texto) > 100:
            print(f"✅ Extracción exitosa: {len(texto)} caracteres")
            print(f"   Primeros 200 caracteres: {texto[:200]}...")
            return True
        else:
            print("❌ Texto extraído muy corto o vacío")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_gemini_config():
    """Prueba: Configuración de Gemini."""
    print("\n" + "="*60)
    print("TEST 2: Configuración Gemini")
    print("="*60)
    
    config = LangChainConfig()
    print(f"  Modelo: {config.model}")
    print(f"  API Key: {'✅ Configurada' if config.gemini_api_key else '❌ NO configurada'}")
    
    if not config.gemini_api_key:
        print("⚠️  GEMINI_API_KEY no encontrada en .env")
        return False
    
    llm = config.crear_llm()
    if llm:
        print("✅ LLM creado exitosamente")
        return True
    else:
        print("❌ Error al crear LLM")
        return False


def test_analyzer():
    """Prueba: OCRAnalyzer."""
    print("\n" + "="*60)
    print("TEST 3: OCRAnalyzer")
    print("="*60)
    
    pdf_path = "data/Ley_769_de_2002.pdf"
    if not Path(pdf_path).exists():
        print(f"⚠️  PDF no encontrado: {pdf_path}")
        return False
    
    try:
        analyzer = OCRAnalyzer(pdf_path)
        print(f"✅ Analizador creado")
        
        if not analyzer.llm:
            print("⚠️  LLM no disponible (configura GEMINI_API_KEY)")
            return False
        
        print("✅ LLM disponible")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_sample_question():
    """Prueba: Hacer una pregunta simple."""
    print("\n" + "="*60)
    print("TEST 4: Pregunta de prueba")
    print("="*60)
    
    pdf_path = "data/Ley_769_de_2002.pdf"
    if not Path(pdf_path).exists():
        print(f"⚠️  PDF no encontrado: {pdf_path}")
        return False
    
    config = LangChainConfig()
    if not config.gemini_api_key:
        print("⚠️  GEMINI_API_KEY no configurada")
        return False
    
    try:
        analyzer = OCRAnalyzer(pdf_path, config=config)
        print("⏳ Haciendo pregunta al PDF...")
        
        respuesta = analyzer.responder_pregunta("¿Cuál es el objetivo principal de este documento?")
        
        if respuesta and len(respuesta) > 10:
            print(f"✅ Respuesta recibida ({len(respuesta)} caracteres)")
            print(f"\n{respuesta[:500]}...\n")
            return True
        else:
            print("❌ Respuesta vacía o muy corta")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecutar todas las pruebas."""
    print("\n" + "🧪 PRUEBAS DEL SISTEMA OCR " + "="*40)
    
    resultados = {
        "Extracción OCR": test_ocr_extraction(),
        "Config Gemini": test_gemini_config(),
        "OCRAnalyzer": test_analyzer(),
        "Pregunta de prueba": test_sample_question(),
    }
    
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBAS")
    print("="*60)
    
    for test_name, resultado in resultados.items():
        estado = "✅ PASÓ" if resultado else "❌ FALLÓ"
        print(f"{test_name}: {estado}")
    
    total = len(resultados)
    pasadas = sum(1 for v in resultados.values() if v)
    print(f"\nTotal: {pasadas}/{total} pruebas pasadas")
    
    if pasadas == total:
        print("\n✨ ¡Todas las pruebas pasaron!")
        return 0
    else:
        print(f"\n⚠️  {total - pasadas} pruebas fallaron")
        return 1


if __name__ == "__main__":
    sys.exit(main())
