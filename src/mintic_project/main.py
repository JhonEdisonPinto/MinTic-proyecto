#!/usr/bin/env python3
"""Runner principal para consultar PDFs usando OCR + Gemini.

Uso:
  - Extraer texto de un PDF con OCR:
      python -m src.mintic_project.main extract --pdf data/Ley_769_de_2002.pdf

  - Hacer una pregunta sobre un PDF:
      python -m src.mintic_project.main query --pdf data/Ley_769_de_2002.pdf --question "¿Qué es...?"

  - Modo interactivo:
      python -m src.mintic_project.main interactive --pdf data/Ley_769_de_2002.pdf

Este script asume que tienes un entorno Python activado y que `GEMINI_API_KEY`
está definido en `.env` o en las variables de entorno.
Requiere: pytesseract, pdf2image, pillow
"""
import argparse
import logging
import sys
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_extract(args):
    """Extraer texto de un PDF usando OCR."""
    from src.mintic_project.langchain_integration import extract_text_from_pdf_ocr

    pdf_path = args.pdf or "data/Ley_769_de_2002.pdf"

    if not Path(pdf_path).exists():
        logger.error("PDF no encontrado: %s", pdf_path)
        sys.exit(1)

    logger.info("Extrayendo texto con OCR de '%s'", pdf_path)
    try:
        text = extract_text_from_pdf_ocr(pdf_path, cache_txt=True)
        logger.info(f"✓ Extracción completada: {len(text)} caracteres")
        
        # Guardar en archivo
        output_file = f"{Path(pdf_path).stem}_ocr.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"✓ Texto guardado en: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


def cmd_query(args):
    """Hacer una pregunta sobre un PDF usando OCR."""
    from src.mintic_project.langchain_integration import answer_with_ocr, LangChainConfig

    pdf_path = args.pdf or "data/Ley_769_de_2002.pdf"
    question = args.question
    
    if not question:
        logger.error("Proporciona una pregunta con --question")
        sys.exit(1)

    if not Path(pdf_path).exists():
        logger.error("PDF no encontrado: %s", pdf_path)
        sys.exit(1)

    cfg = LangChainConfig()
    llm = cfg.crear_llm()
    
    if not llm:
        logger.error("❌ No se pudo crear el LLM. Verifica GEMINI_API_KEY")
        sys.exit(1)

    logger.info("Analizando PDF con OCR...")
    try:
        resp = answer_with_ocr(question, pdf_path=pdf_path, llm=llm)
        print("\n" + "="*60)
        print("RESPUESTA:")
        print("="*60 + "\n")
        print(resp)
        print("\n" + "="*60 + "\n")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


def cmd_interactive(args):
    """Modo interactivo para hacer preguntas sobre un PDF."""
    from src.mintic_project.langchain_integration import OCRAnalyzer, LangChainConfig

    pdf_path = args.pdf or "data/Ley_769_de_2002.pdf"
    
    if not Path(pdf_path).exists():
        logger.error("PDF no encontrado: %s", pdf_path)
        sys.exit(1)

    cfg = LangChainConfig()
    analyzer = OCRAnalyzer(pdf_path, config=cfg)
    
    logger.info("Extrayendo texto del PDF (primera vez puede tardar)...")
    text = analyzer.extraer_texto()
    
    if not text:
        logger.error("❌ No se pudo extraer texto del PDF")
        sys.exit(1)
    
    logger.info(f"✓ Texto extraído: {len(text)} caracteres")
    print("\n" + "="*60)
    print("Modo interactivo OCR (escribe 'salir' para terminar)")
    print("="*60)
    
    while True:
        try:
            q = input('\n❓ Pregunta: ').strip()
        except (KeyboardInterrupt, EOFError):
            print('\n👋 Saliendo...')
            break
        
        if not q:
            continue
        
        if q.lower() in ("salir", "exit", "quit"):
            print('👋 Saliendo...')
            break
        
        print('\n⏳ Generando respuesta...')
        resp = analyzer.responder_pregunta(q)
        print('\n' + "="*60)
        print('RESPUESTA:')
        print("="*60 + '\n')
        print(resp)
        print('\n' + "="*60)


def build_parser():
    p = argparse.ArgumentParser(description="Runner para OCR + Gemini (análisis de PDFs)")
    sub = p.add_subparsers(dest="cmd")

    p_extract = sub.add_parser("extract", help="Extraer texto de un PDF con OCR")
    p_extract.add_argument("--pdf", help="Ruta al PDF (default: data/Ley_769_de_2002.pdf)")

    p_query = sub.add_parser("query", help="Hacer una pregunta sobre un PDF")
    p_query.add_argument("--pdf", help="Ruta al PDF (default: data/Ley_769_de_2002.pdf)")
    p_query.add_argument("--question", required=True, help="Pregunta a responder")

    p_inter = sub.add_parser("interactive", help="Modo interactivo de preguntas")
    p_inter.add_argument("--pdf", help="Ruta al PDF (default: data/Ley_769_de_2002.pdf)")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "query":
        cmd_query(args)
    elif args.cmd == "interactive":
        cmd_interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
