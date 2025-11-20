"""Módulo para preparar datos para sistemas multiagente de predicción.

Este módulo transforma los datos limpios en features listos para
modelos de ML y sistemas de predicción con LangChain.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Ingeniería de características para predicción de siniestros."""

    def __init__(self):
        """Inicializar Feature Engineering."""
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def procesar_para_prediccion(
        self, df: pd.DataFrame, columnas_target: List[str] = None
    ) -> pd.DataFrame:
        """Procesar datos para predicción.

        Args:
            df: DataFrame limpio.
            columnas_target: Columnas objetivo para predicción.

        Returns:
            DataFrame con features procesados.
        """
        df_procesado = df.copy()

        # 1. Crear features temporales
        df_procesado = self._crear_features_temporales(df_procesado)

        # 2. Crear features geográficos
        df_procesado = self._crear_features_geograficos(df_procesado)

        # 3. Crear features categóricos
        df_procesado = self._crear_features_categoricos(df_procesado)

        # 4. Crear features de interacción
        df_procesado = self._crear_features_interaccion(df_procesado)

        # 5. Normalizar features numéricos
        df_procesado = self._normalizar_numericos(df_procesado)

        logger.info(f"✓ Features creados. Dimensiones finales: {df_procesado.shape}")

        return df_procesado

    def _crear_features_temporales(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features derivados de fecha/hora."""
        # Convertir fecha si es necesario
        if "fecha" in df.columns and df["fecha"].dtype == "object":
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

        if "fecha" in df.columns:
            df["mes"] = df["fecha"].dt.month
            df["trimestre"] = df["fecha"].dt.quarter
            df["semana_ano"] = df["fecha"].dt.isocalendar().week
            logger.info("  - Features temporales creados (mes, trimestre, semana)")

        # Convertir hora a período del día
        if "hora" in df.columns:
            def categorizar_hora(hora):
                if pd.isna(hora):
                    return np.nan
                if 6 <= hora < 12:
                    return "MANANA"
                elif 12 <= hora < 18:
                    return "TARDE"
                elif 18 <= hora < 24:
                    return "NOCHE"
                else:
                    return "MADRUGADA"

            df["periodo_dia"] = df["hora"].apply(categorizar_hora)
            logger.info("  - Período del día creado (MANANA, TARDE, NOCHE, MADRUGADA)")

        return df

    def _crear_features_geograficos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features derivados de ubicación."""
        # Distancia aproximada del centro de Palmira (4°S, 76°W)
        PALMIRA_LAT, PALMIRA_LON = -3.73, -76.28

        if "lat" in df.columns and "long" in df.columns:
            try:
                df["distancia_centro"] = np.sqrt(
                    (df["lat"] - PALMIRA_LAT) ** 2
                    + (df["long"] - PALMIRA_LON) ** 2
                )
                df["en_centro"] = (
                    (
                        (df["lat"] >= PALMIRA_LAT - 0.1)
                        & (df["lat"] <= PALMIRA_LAT + 0.1)
                        & (df["long"] >= PALMIRA_LON - 0.1)
                        & (df["long"] <= PALMIRA_LON + 0.1)
                    )
                    .astype(int)
                )
                logger.info(
                    "  - Features geográficos creados (distancia_centro, en_centro)"
                )
            except Exception as e:
                logger.warning(f"  - No se pudieron crear features geográficos: {e}")

        return df

    def _crear_features_categoricos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codificar variables categóricas."""
        columnas_categoricas = [
            "jornada",
            "dia_semana",
            "genero",
            "zona",
            "clase_siniestro",
            "gravedad",
            "periodo_dia",
        ]

        for col in df.columns:
            if col in columnas_categoricas and df[col].dtype == "object":
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Manejar valores nulos
                    valores_unicos = df[col].fillna("DESCONOCIDO").unique()
                    self.label_encoders[col].fit(valores_unicos)

                # Aplicar encoding
                valores_con_nulos = df[col].fillna("DESCONOCIDO")
                df[f"{col}_encoded"] = self.label_encoders[col].transform(valores_con_nulos)

        logger.info(f"  - {len(self.label_encoders)} variables categóricas codificadas")

        return df

    def _crear_features_interaccion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features de interacción entre variables."""
        # Interacción: hora x jornada
        if "hora" in df.columns and "periodo_dia" in df.columns:
            df["hora_jornada_interaction"] = (
                df["hora"].fillna(0) * df["periodo_dia"].fillna("DESCONOCIDO").map(
                    {"MANANA": 1, "TARDE": 2, "NOCHE": 3, "MADRUGADA": 4}
                ).fillna(0)
            )
            logger.info("  - Feature de interacción creado (hora x jornada)")

        # Interacción: genero x edad (si existen)
        if "genero_encoded" in df.columns and "edad" in df.columns:
            df["genero_edad_interaction"] = (
                df["genero_encoded"].fillna(0) * df["edad"].fillna(0)
            )
            logger.info("  - Feature de interacción creado (genero x edad)")

        return df

    def _normalizar_numericos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizar features numéricos."""
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        columnas_a_normalizar = [
            col
            for col in columnas_numericas
            if col not in ["mes", "trimestre", "semana_ano", "periodo_dia"]
        ]

        if columnas_a_normalizar:
            # Crear copia de datos para normalización
            datos_a_normalizar = df[columnas_a_normalizar].fillna(df[columnas_a_normalizar].mean())
            df_normalizado = self.scaler.fit_transform(datos_a_normalizar)

            # Reemplazar con valores normalizados
            for i, col in enumerate(columnas_a_normalizar):
                df[f"{col}_normalized"] = df_normalizado[:, i]

            logger.info(f"  - {len(columnas_a_normalizar)} features normalizados")

        return df


class DatasetPredictor:
    """Prepara datasets completos para predicción."""

    def __init__(self):
        """Inicializar preparador de datasets."""
        self.fe = FeatureEngineering()

    def preparar_dataset_completo(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> pd.DataFrame:
        """Combinar y preparar ambos datasets para predicción.

        Args:
            df1: Primer dataset de siniestros.
            df2: Segundo dataset de siniestros.

        Returns:
            DataFrame combinado y procesado para predicción.
        """
        logger.info("\n" + "="*60)
        logger.info("Preparando datasets para predicción multiagente...")
        logger.info("="*60 + "\n")

        # Combinar datasets por fecha (join en 'fecha' o 'a_o')
        columnas_comunes = set(df1.columns) & set(df2.columns)
        logger.info(f"Columnas comunes: {columnas_comunes}")

        # Intentar merge en columnas comunes
        try:
            if "fecha" in columnas_comunes:
                df_combinado = pd.merge(
                    df1, df2, on="fecha", how="outer", suffixes=("_1", "_2")
                )
            elif "a_o" in columnas_comunes:
                df_combinado = pd.merge(
                    df1, df2, on="a_o", how="outer", suffixes=("_1", "_2")
                )
            else:
                logger.warning("No se encontraron columnas comunes para merge")
                df_combinado = df1
        except Exception as e:
            logger.warning(f"Error en merge: {e}, usando df1")
            df_combinado = df1

        logger.info(f"Dataset combinado: {df_combinado.shape}")

        # Procesar para predicción
        df_procesado = self.fe.procesar_para_prediccion(df_combinado)

        return df_procesado

    def generar_contexto_rag(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generar contexto para RAG (Retrieval-Augmented Generation).

        Este contexto puede usarse con LangChain para responder preguntas
        sobre patrones de siniestros.

        Args:
            df: DataFrame con datos procesados.

        Returns:
            Diccionario con contextos por tema.
        """
        contexto = {}

        # Contexto general
        contexto["general"] = f"""
        Dataset de Siniestros Viales en Palmira
        - Total de registros: {len(df):,}
        - Período: {df['a_o'].min() if 'a_o' in df.columns else 'N/A'} - {df['a_o'].max() if 'a_o' in df.columns else 'N/A'}
        - Columnas disponibles: {len(df.columns)}
        """

        # Contexto por jornada
        if "jornada" in df.columns:
            jornada_dist = df["jornada"].value_counts().to_dict()
            contexto["jornada"] = f"""
            Distribución por jornada:
            {jornada_dist}
            """

        # Contexto por día de semana
        if "dia_semana" in df.columns:
            dia_dist = df["dia_semana"].value_counts().to_dict()
            contexto["dia_semana"] = f"""
            Distribución por día de semana:
            {dia_dist}
            """

        # Contexto por género
        if "genero" in df.columns:
            genero_dist = df["genero"].value_counts().to_dict()
            contexto["genero"] = f"""
            Distribución por género:
            {genero_dist}
            """

        # Contexto por gravedad
        if "gravedad" in df.columns:
            gravedad_dist = df["gravedad"].value_counts().to_dict()
            contexto["gravedad"] = f"""
            Distribución por gravedad:
            {gravedad_dist}
            """

        # Contexto estadístico
        if "edad" in df.columns:
            edad_stats = df["edad"].describe().to_dict()
            contexto["edad"] = f"""
            Estadísticas de edad:
            {edad_stats}
            """

        return contexto
