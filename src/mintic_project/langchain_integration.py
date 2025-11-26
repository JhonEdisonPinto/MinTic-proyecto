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


def _leer_documento(path: str) -> str:
    """Leer documento sencillo: soporta .txt y .pdf (si está instalado PyPDF2).

    Devuelve el texto combinado.
    """
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
                pages.append(page.extract_text() or "")
            return "\n\n".join(pages)
        except Exception as e:
            logger.error("No se pudo leer PDF (PyPDF2 requerido): %s", e)
            raise

    # Intentar leer con text-mode por defecto
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        raise ValueError(f"Formato de documento no soportado: {suffix}")


def index_document_to_faiss(
    doc_path: str,
    index_dir: str = "data/faiss_ley769",
    config: Optional[Any] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: Optional[str] = None,
) -> Any:
    """Indexa un documento en FAISS usando LangChain.

    Args:
        doc_path: Ruta al documento (txt o pdf)
        index_dir: Carpeta donde guardar el índice FAISS
        config: LangChainConfig para seleccionar proveedor de embeddings/LLM
        chunk_size/chunk_overlap: parámetros del splitter
        embedding_model: nombre del modelo de embeddings (opcional)

    Retorna el vectorstore creado.
    """
    from pathlib import Path

    config = config or LangChainConfig()

    try:
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
    except Exception as e:
        logger.error("Instala langchain y dependencias (pip install langchain faiss-cpu): %s", e)
        raise

    text = _leer_documento(doc_path)
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    # Crear embeddings: usar HuggingFace local
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Usando HuggingFaceEmbeddings locales")
    except Exception as e:
        logger.error("No se pudo crear embeddings locales HuggingFace: %s", e)
        raise

    # Crear vectorstore FAISS
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        # Guardar localmente
        vectorstore.save_local(str(index_path))
        logger.info(f"Índice FAISS guardado en {index_path}")
        return vectorstore
    except Exception as e:
        logger.error("Error creando/guardando FAISS: %s", e)
        raise


def load_faiss_index(index_dir: str = "data/faiss_ley769") -> Any:
    """Cargar un índice FAISS previamente guardado.

    Retorna el vectorstore si existe.
    """
    try:
        from langchain_community.vectorstores import FAISS
    except Exception as e:
        logger.error("langchain no instalado: %s", e)
        raise

    p = Path(index_dir)
    if not p.exists():
        raise FileNotFoundError(f"Índice FAISS no encontrado en: {index_dir}")

    # Usar HuggingFace embeddings locales para cargar índice
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        logger.error("No se pudo crear embeddings para cargar índice: %s", e)
        raise

    vs = FAISS.load_local(str(p), embeddings, allow_dangerous_deserialization=True)
    logger.info(f"Índice FAISS cargado desde {index_dir}")
    return vs


def answer_with_faiss(question: str, index_dir: str = "data/faiss_ley769", config: Optional[Any] = None, k: int = 4) -> Optional[str]:
    """Responder pregunta usando RAG con índice FAISS y LLM configurado.

    Devuelve la respuesta textual del LLM o None en caso de error.
    """
    config = config or LangChainConfig(max_tokens=2000)  # Aumentar límite de tokens
    try:
        vs = load_faiss_index(index_dir)
    except Exception as e:
        logger.error("No se pudo cargar índice FAISS: %s", e)
        return None

    try:
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    except Exception:
        # Compatibilidad con versiones antiguas
        retriever = vs.as_retriever(search_kwargs={"k": k})

    llm = config.crear_llm()
    if not llm:
        logger.error("LLM no disponible; configura credenciales.")
        return None

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # Obtener documentos relevantes (compatible con distintas versiones)
        docs = None
        for method_name in (
            "get_relevant_documents",
            "get_relevant_documents",
            "retrieve",
            "get_documents",
            "invoke",
        ):
            try:
                if hasattr(retriever, method_name):
                    method = getattr(retriever, method_name)
                    # Algunas APIs esperan (query) otras (query, k)
                    try:
                        docs = method(question)
                    except TypeError:
                        docs = method(question, k=k)
                    break
            except Exception:
                continue

        if not docs:
            logger.warning("No se obtuvieron documentos del retriever; continuando sin contexto.")
            docs = []

        context = "\n".join([getattr(d, "page_content", str(d)) for d in docs if d])

        # Prompt para el LLM
        prompt_template = (
            "Eres un experto en leyes colombianas. Usa el siguiente contexto para responder la pregunta de forma completa y detallada.\n\n"
            "Contexto:\n{context}\n\nPregunta: {question}\n\nRespuesta completa:"
        )

        # First try: runnable chain (prompt -> llm -> parser) if both are compatible
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm | StrOutputParser()
            result = None
            try:
                result = chain.invoke({"context": context, "question": question})
            except Exception:
                # Some runnables may raise; fall back below
                result = None
            if result:
                return result if isinstance(result, str) else str(result)
        except Exception:
            logger.debug("Runnables chain no disponible o falló, intentando llamadas directas al LLM.")

        # Fallback: call the LLM directly with a formatted prompt
        prompt_text = prompt_template.format(context=context, question=question)

        # Try common LLM call patterns
        llm_result_text = None

        # 1) generate (may return object with .generations)
        try:
            if hasattr(llm, "generate"):
                gen = llm.generate([prompt_text])
                # Try common extraction points
                if hasattr(gen, "generations"):
                    try:
                        llm_result_text = gen.generations[0][0].text
                    except Exception:
                        try:
                            llm_result_text = gen.generations[0][0][0].text
                        except Exception:
                            llm_result_text = str(gen)
                else:
                    llm_result_text = str(gen)
        except Exception:
            logger.debug("LLM.generate falló o no aplica.")

        # 2) __call__ or predict
        if not llm_result_text:
            try:
                if hasattr(llm, "__call__"):
                    call_res = llm(prompt_text)
                    if isinstance(call_res, str):
                        llm_result_text = call_res
                    else:
                        # try common attribute names
                        llm_result_text = getattr(call_res, "text", None) or getattr(call_res, "output", None) or str(call_res)
            except Exception:
                logger.debug("LLM.__call__ falló o no aplicable.")

        if not llm_result_text and hasattr(llm, "predict"):
            try:
                p = llm.predict(prompt_text)
                llm_result_text = p if isinstance(p, str) else str(p)
            except Exception:
                logger.debug("LLM.predict falló o no aplicable.")

        if llm_result_text:
            return llm_result_text

        logger.error("No se pudo obtener texto del LLM con los métodos probados.")
        return "No se pudo generar una respuesta."
    except Exception as e:
        logger.error("Error ejecutando cadena RAG: %s", e)
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
