"""Test completo del sistema RAG - TODO EN UN ARCHIVO"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Any

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES RAG (incluidas en este archivo)
# ============================================================================

def _leer_documento(path: str) -> str:
    """Leer documento: soporta .txt, .md y .pdf"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Documento no encontrado: {path}")

    suffix = p.suffix.lower()
    
    if suffix in {".txt", ".md"}:
        return p.read_text(encoding="utf-8")

    if suffix == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(p))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
            resultado = "\n\n".join(pages)
            logger.info(f"PDF leído: {len(pages)} páginas, {len(resultado)} caracteres")
            return resultado
        except Exception as e:
            logger.error(f"Error leyendo PDF: {e}")
            raise

    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        raise ValueError(f"Formato no soportado: {suffix}")


def index_document_to_faiss(
    doc_path: str,
    index_dir: str = "data/faiss_ley769",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Any:
    """Indexa un documento en FAISS."""
    logger.info(f"=== INICIANDO INDEXACIÓN ===")
    logger.info(f"Documento: {doc_path}")
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        logger.error("Faltan dependencias. Instala: pip install langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers")
        raise

    # 1. Leer documento
    logger.info("Paso 1: Leyendo documento...")
    text = _leer_documento(doc_path)
    logger.info(f"✓ Texto leído: {len(text)} caracteres")

    # 2. Dividir en chunks
    logger.info("Paso 2: Dividiendo en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    logger.info(f"✓ Chunks creados: {len(chunks)}")
    
    if len(chunks) == 0:
        raise ValueError("No se generaron chunks")
    
    logger.info(f"Muestra chunk 1:\n{chunks[0][:200]}...\n")

    # 3. Crear documentos
    logger.info("Paso 3: Creando documentos...")
    docs = [
        Document(
            page_content=chunk,
            metadata={"source": doc_path, "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    logger.info(f"✓ {len(docs)} documentos creados")

    # 4. Crear embeddings
    logger.info("Paso 4: Cargando embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("✓ Embeddings cargados")

    # 5. Crear índice FAISS
    logger.info("Paso 5: Creando índice FAISS...")
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    logger.info("✓ Índice creado")
    
    # 6. Guardar
    logger.info("Paso 6: Guardando índice...")
    vectorstore.save_local(str(index_path))
    logger.info(f"✓ Guardado en: {index_path.absolute()}")
    
    logger.info("=== INDEXACIÓN COMPLETADA ===\n")
    return vectorstore


def load_faiss_index(index_dir: str = "data/faiss_ley769") -> Any:
    """Carga un índice FAISS."""
    logger.info(f"Cargando índice desde: {index_dir}")
    
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError("Instala: pip install langchain-community faiss-cpu sentence-transformers")

    p = Path(index_dir)
    if not p.exists():
        raise FileNotFoundError(f"Índice no encontrado: {index_dir}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.load_local(
        str(p),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    logger.info(f"✓ Índice cargado")
    return vectorstore


def answer_with_faiss(
    question: str,
    index_dir: str = "data/faiss_ley769",
    k: int = 4,
    verbose: bool = True
) -> Optional[str]:
    """Responde una pregunta usando RAG."""
    if verbose:
        logger.info("=" * 70)
        logger.info("=== CONSULTA RAG ===")
        logger.info(f"Pregunta: {question}")
        logger.info("=" * 70)
    
    # 1. Cargar índice
    if verbose:
        logger.info("\n[1/5] Cargando índice...")
    try:
        vectorstore = load_faiss_index(index_dir)
        if verbose:
            logger.info("✓ Índice cargado")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None

    # 2. Recuperar documentos
    if verbose:
        logger.info(f"\n[2/5] Recuperando {k} documentos...")
    try:
        docs = vectorstore.similarity_search(question, k=k)
        
        if verbose:
            logger.info(f"✓ {len(docs)} documentos encontrados")
            for i, doc in enumerate(docs, 1):
                logger.info(f"\n--- Doc {i} ---")
                logger.info(f"{doc.page_content[:200]}...")
        
        if not docs:
            return "No encontré información relevante."
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None

    # 3. Crear contexto
    if verbose:
        logger.info("\n[3/5] Creando contexto...")
    
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    if verbose:
        logger.info(f"✓ Contexto: {len(context)} caracteres")

    # 4. Crear LLM
    if verbose:
        logger.info("\n[4/5] Configurando LLM...")
    
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("❌ GEMINI_API_KEY no configurada")
        return None
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=2000,
        )
        if verbose:
            logger.info("✓ LLM creado")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None

    # 5. Generar respuesta
    if verbose:
        logger.info("\n[5/5] Generando respuesta...")
    
    try:
        prompt_text = f"""Eres un experto en leyes de tránsito colombianas. Responde basándote SOLO en el contexto.

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:"""

        response = llm.invoke(prompt_text)
        
        if hasattr(response, 'content'):
            respuesta = response.content
        elif isinstance(response, str):
            respuesta = response
        else:
            respuesta = str(response)
        
        if verbose:
            logger.info("✓ Respuesta generada")
            logger.info("=" * 70)
            logger.info("RESPUESTA:")
            logger.info("=" * 70)
            logger.info(respuesta)
            logger.info("=" * 70 + "\n")
        
        return respuesta.strip()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# FUNCIONES DE PRUEBA
# ============================================================================

def crear_documento_test():
    """Crea documento de prueba"""
    print("\n📝 CREANDO DOCUMENTO DE PRUEBA")
    print("=" * 70)
    
    doc_path = Path("data/test_ley_transito.txt")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    
    contenido = """LEY 769 DE 2002 - CÓDIGO NACIONAL DE TRÁNSITO

ARTÍCULO 106 - LÍMITES DE VELOCIDAD

En Colombia, los límites de velocidad son:

1. VÍAS URBANAS: 60 kilómetros por hora
2. VÍAS RURALES: 80 kilómetros por hora  
3. AUTOPISTAS: 100 kilómetros por hora

ARTÍCULO 131 - SANCIONES

Clasificación de infracciones:

1. INFRACCIONES LEVES: Multa de 15 salarios mínimos
2. INFRACCIONES GRAVES: Multa de 30 salarios mínimos
3. INFRACCIONES MUY GRAVES: Multa de 60 salarios mínimos

ESTADÍSTICAS

- 35% de siniestros por exceso de velocidad
- 25% por conducción en embriaguez
- 45% ocurren en jornada nocturna
- 60% en zonas urbanas

PREVENCIÓN

Medidas principales:
1. Respetar límites de velocidad
2. No conducir bajo alcohol
3. Usar cinturón de seguridad
4. Mantenimiento del vehículo
"""
    
    doc_path.write_text(contenido, encoding='utf-8')
    print(f"✓ Documento creado: {doc_path}")
    print(f"  Ubicación: {doc_path.absolute()}")
    
    return str(doc_path)


def test_completo():
    """Prueba completa del sistema"""
    
    print("\n" + "=" * 70)
    print("🚀 PRUEBA COMPLETA DEL SISTEMA RAG")
    print("=" * 70)
    
    # Cargar .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("\n✓ Variables de entorno cargadas")
    except:
        print("\n⚠️  python-dotenv no disponible")
    
    # Verificar API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    print(f"GEMINI_API_KEY: {'✓ Configurada' if api_key else '❌ NO configurada'}")
    
    if not api_key:
        print("\n❌ Configura GEMINI_API_KEY en .env")
        return False
    
    # Crear documento
    doc_path = crear_documento_test()
    
    # PASO 1: INDEXAR
    print("\n" + "=" * 70)
    print("PASO 1: INDEXACIÓN")
    print("=" * 70)
    
    try:
        vectorstore = index_document_to_faiss(
            doc_path=doc_path,
            index_dir="data/faiss_test_debug",
            chunk_size=500,
            chunk_overlap=100
        )
        print("\n✅ INDEXACIÓN EXITOSA")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # PASO 2: BÚSQUEDA DIRECTA
    print("\n" + "=" * 70)
    print("PASO 2: BÚSQUEDA DIRECTA")
    print("=" * 70)
    
    try:
        query = "límites de velocidad"
        print(f"\nBuscando: '{query}'")
        
        docs = vectorstore.similarity_search(query, k=2)
        print(f"✓ {len(docs)} documentos encontrados\n")
        
        for i, doc in enumerate(docs, 1):
            print(f"--- Doc {i} ---")
            print(doc.page_content)
            print()
        
        print("✅ BÚSQUEDA EXITOSA")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    # PASO 3: RAG COMPLETO
    print("\n" + "=" * 70)
    print("PASO 3: RAG COMPLETO")
    print("=" * 70)
    
    preguntas = [
        "¿Cuál es el límite de velocidad en vías urbanas?",
        "¿Qué porcentaje de siniestros ocurren por exceso de velocidad?",
        "¿Cuál es la multa por infracciones graves?",
    ]
    
    try:
        for i, pregunta in enumerate(preguntas, 1):
            print(f"\n{'='*70}")
            print(f"PREGUNTA {i}: {pregunta}")
            print('='*70)
            
            respuesta = answer_with_faiss(
                question=pregunta,
                index_dir="data/faiss_test_debug",
                k=3,
                verbose=True
            )
            
            if not respuesta:
                print("❌ NO SE OBTUVO RESPUESTA")
                return False
            
            print("\n" + "-"*70)
        
        print("\n" + "="*70)
        print("✅ TODAS LAS PRUEBAS EXITOSAS")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    exito = test_completo()
    sys.exit(0 if exito else 1)