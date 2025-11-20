"""Configuración y utilidades para integración con LangChain.

Este módulo proporciona clases y funciones para integrar el sistema
de análisis de siniestros con LangChain para crear agentes multiagente.
"""
import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LangChainConfig:
    """Configuración para integración con LangChain."""

    def __init__(self, env_path: Optional[str] = None):
        """Inicializar configuración de LangChain.

        Args:
            env_path: Ruta al archivo .env (opcional).
        """
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.temperature = 0.7
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 500

        if not self.openai_api_key:
            logger.warning(
                "⚠️  OPENAI_API_KEY no configurada. "
                "LangChain con OpenAI no funcionará sin esta clave."
            )

    def crear_llm(self):
        """Crear instancia de LLM (requiere langchain y openai).

        Returns:
            Instancia de OpenAI LLM o None si falta API key.
        """
        if not self.openai_api_key:
            logger.error("No se puede crear LLM sin OPENAI_API_KEY")
            return None

        try:
            from langchain.llms import OpenAI

            return OpenAI(
                api_key=self.openai_api_key,
                temperature=self.temperature,
                model_name=self.model,
                max_tokens=self.max_tokens,
            )
        except ImportError:
            logger.error("Instala langchain y openai: pip install langchain openai")
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

    def analizar(
        self, pregunta: str, tipo_agente: str = "temporal"
    ) -> Optional[str]:
        """Analizar pregunta con agente específico.

        Args:
            pregunta: Pregunta a analizar.
            tipo_agente: Tipo de agente a usar.

        Returns:
            Respuesta del LLM o None si no hay LLM disponible.
        """
        if not self.llm:
            logger.error("LLM no disponible. Configurar OPENAI_API_KEY")
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
            respuesta = self.llm(prompt)
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
    print(f"  - Modelo: {config.model}")
    print(f"  - Temperatura: {config.temperature}")
    print(f"  - API Key configurada: {'Sí' if config.openai_api_key else 'No'}")

    # 2. Cargar contextos RAG
    rag = RAGContextManager()
    print(f"\n✓ Contextos RAG cargados: {len(rag.contexto)} temas")

    # 3. Crear multiagente (requiere OPENAI_API_KEY)
    multiagente = MultiagenteSiniestros(config)
    print(f"\n✓ Agentes disponibles:")
    for nombre, desc in multiagente.listar_agentes().items():
        print(f"  - {nombre}: {desc}")

    # 4. Ejemplo de análisis (comentado si no hay API key)
    if config.openai_api_key:
        print("\n🤖 Ejecutando análisis de ejemplo...")
        respuesta = multiagente.analizar(
            "¿En qué jornada ocurren más siniestros?", tipo_agente="temporal"
        )
        if respuesta:
            print(f"\nRespuesta del agente:\n{respuesta}")
    else:
        print("\n⚠️  Para usar análisis, configurar OPENAI_API_KEY en .env")
