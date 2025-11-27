"""Configuración y utilidades para integración con LangChain.

Este módulo proporciona clases y funciones para integrar el sistema
de análisis de siniestros con LangChain para crear agentes multiagente.
"""
import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import mimetypes
import io

logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES RAG CORREGIDAS
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
    config: Optional[Any] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: Optional[str] = None,
) -> Any:
    """Indexa un documento en FAISS usando LangChain."""
    logger.info(f"Indexando documento: {doc_path}")
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        logger.error("Instala: pip install langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers")
        raise

    text = _leer_documento(doc_path)
    logger.info(f"Texto leído: {len(text)} caracteres")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    logger.info(f"Chunks creados: {len(chunks)}")
    
    if len(chunks) == 0:
        raise ValueError("No se generaron chunks")

    docs = [
        Document(
            page_content=chunk,
            metadata={"source": doc_path, "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("Embeddings creados")

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(index_path))
    logger.info(f"Índice guardado en: {index_path}")
    
    return vectorstore



def load_faiss_index(index_dir: str = "data/faiss_ley769") -> Any:
    """Carga un índice FAISS previamente guardado."""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError("Instala: pip install langchain-community faiss-cpu sentence-transformers")
    # Permitir varios nombres comunes de carpeta de índice
    candidates = [index_dir, "data/faiss_ley769", "data/faiss_ley769_de_2002", "data/faiss_ley769/", "data/faiss_ley769_de_2002/"]
    p = None
    for c in candidates:
        pc = Path(c)
        if pc.exists():
            p = pc
            index_dir = c
            break

    if p is None:
        # intentar listar posibles subdirectorios en data/
        data_dir = Path("data")
        if data_dir.exists():
            for child in data_dir.iterdir():
                if child.is_dir() and "faiss" in child.name.lower():
                    p = child
                    index_dir = str(child)
                    break

    if p is None:
        raise FileNotFoundError(f"Índice FAISS no encontrado en: {index_dir}. Busqué: {candidates}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vs = FAISS.load_local(str(p), embeddings, allow_dangerous_deserialization=True)
    logger.info(f"Índice FAISS cargado desde {index_dir}")
    return vs


def answer_with_faiss(
    question: str,
    index_dir: str = "data/faiss_ley769",
    config: Optional[Any] = None,
    k: int = 4,
) -> Optional[str]:
    """Responder pregunta usando RAG con índice FAISS y LLM configurado.

    Usa `config` (LangChainConfig) si se proporciona; en caso contrario intenta
    crear LLM desde variables de entorno.
    """
    logger.info(f"Respondiendo pregunta: {question}")

    # Cargar índice
    try:
        vs = load_faiss_index(index_dir)
    except Exception as e:
        logger.error("No se pudo cargar índice FAISS: %s", e)
        return None

    # Recuperar documentos (usar método similarity_search si está disponible)
    docs = None
    try:
        if hasattr(vs, "similarity_search"):
            docs = vs.similarity_search(question, k=k)
        elif hasattr(vs, "search"):
            docs = vs.search(question, k=k)
        else:
            # intentar retriever
            retriever = None
            try:
                retriever = vs.as_retriever(search_kwargs={"k": k})
            except Exception:
                try:
                    retriever = vs.as_retriever({"k": k})
                except Exception:
                    retriever = None

            if retriever:
                # algunos retrievers tienen .get_relevant_documents o .get_relevant
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(question)
                elif hasattr(retriever, "retrieve"):
                    docs = retriever.retrieve(question)
                elif hasattr(retriever, "invoke"):
                    try:
                        docs = retriever.invoke(question)
                    except Exception:
                        docs = retriever(question)

        if not docs:
            logger.warning("No se encontraron documentos relevantes")
            return "No encontré información relevante para tu pregunta."

    except Exception as e:
        logger.error(f"Error recuperando documentos: {e}")
        return None

    # Crear contexto
    context = "\n\n---\n\n".join([getattr(doc, "page_content", str(doc)) for doc in docs])
    logger.info(f"Contexto creado: {len(context)} caracteres")

    # Preparar LLM: preferir config.crear_llm()
    if config is None:
        config = LangChainConfig()

    llm = None
    try:
        llm = config.crear_llm()
    except Exception:
        llm = None

    if not llm:
        logger.error("LLM no disponible; verifica GEMINI_API_KEY o la configuración del proveedor")
        return None

    # Generar respuesta usando el LLM (manejo flexible de distintos wrappers)
    prompt_text = (
        "Eres un experto en leyes de tránsito colombianas. Usa ÚNICAMENTE el contexto para responder.\n\n"
        f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}\n\n"
        "INSTRUCCIONES:\n- Responde basándote SOLO en el contexto\n- Si no tienes la información, dilo\n- Cita artículos cuando sea relevante\n\nRESPUESTA:"
    )

    # Intentar distintos métodos comunes en wrappers LLM
    try:
        # 1) si el LLM tiene invoke (langchain_google_genai) utilizarlo
        if hasattr(llm, "invoke"):
            try:
                resp = llm.invoke(prompt_text)
                if hasattr(resp, "content"):
                    return str(resp.content).strip()
                if isinstance(resp, str):
                    return resp.strip()
                # fallback
                return str(resp).strip()
            except Exception as e:
                logger.debug(f"llm.invoke falló: {e}")

        # 2) generate
        if hasattr(llm, "generate"):
            try:
                gen = llm.generate([prompt_text])
                if hasattr(gen, "generations"):
                    try:
                        return gen.generations[0][0].text.strip()
                    except Exception:
                        return str(gen).strip()
                return str(gen).strip()
            except Exception:
                logger.debug("llm.generate falló")

        # 3) __call__
        try:
            call_res = llm(prompt_text)
            if isinstance(call_res, str):
                return call_res.strip()
            if hasattr(call_res, "text"):
                return str(call_res.text).strip()
            return str(call_res).strip()
        except Exception:
            logger.debug("llm(...) falló")

        # 4) predict
        if hasattr(llm, "predict"):
            try:
                p = llm.predict(prompt_text)
                return str(p).strip()
            except Exception:
                logger.debug("llm.predict falló")

        logger.error("No se pudo obtener texto del LLM con los métodos probados.")
        return "No se pudo generar una respuesta."

    except Exception as e:
        logger.error(f"Error generando respuesta: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None



class LangChainConfig:
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()
    """
    Configuración para integrar LangChain con Gemini API Key gratuita.
    """

    def __init__(
        self,
        provider: str = "google",
        model: str = "gemini-2.5-flash",  # Modelo actualizado
        temperature: float = 0.7,
        max_tokens: int = 2500,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Lee la API KEY desde el .env
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")

        if provider == "google" and not self.gemini_api_key:
            logger.warning("⚠️ No se encontró GEMINI_API_KEY en el entorno")

    def crear_llm(self):
        """Crear LLM usando Gemini API Key gratuita."""
        if not self.gemini_api_key:
            logger.error("❌ GEMINI_API_KEY no configurada")
            return None

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            logger.info(f"Creando LLM Gemini modelo={self.model}")

            return ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.gemini_api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,  # ✔️ CORREGIDO
                convert_system_message_to_human=True,
            )
        except Exception as e:
            logger.error(f"Error creando LLM Gemini: {e}")
            return None


class RAGContextManager:
    """Gestor de contextos para RAG (Retrieval-Augmented Generation)."""

    def __init__(self, contexto_path: str = "data/contexto_rag.json"):
        """Inicializar gestor de contextos.

        Args:
            contexto_path: Ruta al archivo de contexto RAG.
        """
        self.contexto_path = Path(contexto_path)
        self.contexto = {}
        self.cargar_contexto()

    def cargar_contexto(self):
        """Cargar contexto desde archivo JSON."""
        if self.contexto_path.exists():
            try:
                with open(self.contexto_path, "r", encoding="utf-8") as f:
                    self.contexto = json.load(f)
                logger.info(f"✓ Contexto RAG cargado: {len(self.contexto)} temas")
            except Exception as e:
                logger.error(f"Error cargando contexto: {e}")
                self.contexto = {}
        else:
            logger.warning(f"Archivo de contexto no encontrado: {self.contexto_path}")

    def obtener_contexto(self, tema: str, default: str = "") -> str:
        """Obtener contexto de un tema específico.

        Args:
            tema: Tema para obtener contexto.
            default: Valor por defecto si no existe.

        Returns:
            Contexto del tema.
        """
        return self.contexto.get(tema, default)

    def crear_prompt_rag(
        self, pregunta: str, temas_relevantes: List[str] = None
    ) -> str:
        """Crear prompt que combina pregunta con contexto RAG.

        Args:
            pregunta: Pregunta del usuario.
            temas_relevantes: Lista de temas RAG a incluir.

        Returns:
            Prompt completo para LLM.
        """
        if temas_relevantes is None:
            temas_relevantes = list(self.contexto.keys())

        prompt = "Contexto de Siniestros Viales en Palmira:\n\n"

        for tema in temas_relevantes:
            if tema in self.contexto:
                prompt += f"--- {tema.upper()} ---\n"
                prompt += self.contexto[tema]
                prompt += "\n\n"

        prompt += f"Pregunta: {pregunta}\n\n"
        prompt += "Basándote en el contexto anterior, proporciona un análisis detallado.\n"

        return prompt


class MultiagenteSiniestros:
    """Gestor de multiagentes para análisis de siniestros."""

    def __init__(self, config: Optional[LangChainConfig] = None):
        """Inicializar sistema multiagente.

        Args:
            config: Configuración de LangChain.
        """
        self.config = config or LangChainConfig()
        self.rag_manager = RAGContextManager()
        self.llm = self.config.crear_llm()

        # Definir tipos de agentes
        self.agentes = {
            "temporal": {
                "descripcion": "Analiza patrones temporales de siniestros",
                "temas": ["jornada", "dia_semana"],
            },
            "geografico": {
                "descripcion": "Analiza distribución geográfica",
                "temas": ["general"],
            },
            "prediccion": {
                "descripcion": "Predice riesgo de siniestros",
                "temas": ["general", "gravedad"],
            },
            "normativo": {
                "descripcion": "Responde sobre normas de tránsito",
                "temas": ["general"],
            },
        }

    def analizar(self, pregunta: str, tipo_agente: str = "temporal") -> Optional[str]:
        if not self.llm:
            logger.error(f"LLM no disponible. Configurar credenciales para el proveedor '{self.config.provider}'")
            return None

        if tipo_agente not in self.agentes:
            logger.error(f"Tipo de agente no válido: {tipo_agente}")
            return None

        agente = self.agentes[tipo_agente]
        temas = agente["temas"]

        logger.info(f"Usando agente: {tipo_agente}")
        logger.info(f"Temas relevantes: {temas}")

        # Crear prompt con contexto RAG
        prompt = self.rag_manager.crear_prompt_rag(pregunta, temas)

        try:
            respuesta = self.llm.invoke(prompt)  # <-- ✔️ CORREGIDO
            return respuesta
        except Exception as e:
            logger.error(f"Error en análisis: {e}")
            return None
    def listar_agentes(self) -> Dict[str, str]:
        """Listar agentes disponibles.

        Returns:
            Diccionario con agentes y sus descripciones.
        """
        return {
            nombre: agente["descripcion"]
            for nombre, agente in self.agentes.items()
        }


# Ejemplos de uso
if __name__ == "__main__":
    print("Configuración de LangChain para MinTIC")
    print("=" * 60)

    # 1. Verificar configuración
    config = LangChainConfig()
    print(f"✓ Configuración LangChain lista")
    print(f"  - Proveedor: {config.provider}")
    print(f"  - Modelo: {config.model}")
    print(f"  - Temperatura: {config.temperature}")
    print(f"  - GEMINI_API_KEY configurada: {'Sí' if config.gemini_api_key else 'No'}")

    # 2. Cargar contextos RAG
    rag = RAGContextManager()
    print(f"\n✓ Contextos RAG cargados: {len(rag.contexto)} temas")

    # 3. Crear multiagente (requiere credenciales según proveedor)
    multiagente = MultiagenteSiniestros(config)
    print(f"\n✓ Agentes disponibles:")
    for nombre, desc in multiagente.listar_agentes().items():
        print(f"  - {nombre}: {desc}")

    # 4. Ejemplo de análisis (requiere GEMINI_API_KEY)
    if config.gemini_api_key:
        print("\n🤖 Ejecutando análisis de ejemplo...")
        respuesta = multiagente.analizar(
            "¿En qué jornada ocurren más siniestros?", tipo_agente="temporal"
        )
        if respuesta:
            print(f"\nRespuesta del agente:\n{respuesta}")
    else:
        print("\n⚠️  Para usar análisis, configurar GEMINI_API_KEY en .env")
