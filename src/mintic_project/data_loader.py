"""Módulo para descargar y procesar datos desde datos.gov.co.

Este módulo maneja la descarga, validación y limpieza de datasets
de siniestros viales en Palmira desde las APIs de datos.gov.co.
"""
import requests
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SiniestrosAPIClient:
    """Cliente para descargar datos de siniestros desde datos.gov.co."""

    # URLs de las APIs (sin parámetros para flexibilidad)
    API_SINIESTROS_1 = "https://www.datos.gov.co/resource/sjpx-eqfp.json"
    API_SINIESTROS_2 = "https://www.datos.gov.co/resource/xx6f-f84h.json"

    def __init__(self, timeout: int = 30):
        """Inicializar cliente de API.

        Args:
            timeout: Tiempo máximo de espera para requests (segundos).
        """
        self.timeout = timeout
        self.session = requests.Session()

    def descargar_dataset_1(self, limit: int = 50000) -> pd.DataFrame:
        """Descargar primer dataset de siniestros.

        Columnas: a_o, ipat, clase_siniestro, fecha, hora, jornada, etc.

        Args:
            limit: Número máximo de registros a descargar.

        Returns:
            DataFrame con los datos descargados.
        """
        params = {
            "$limit": limit,
            "$offset": 0,
            "$order": "a_o ASC",
        }
        logger.info(f"Descargando dataset 1 (límite: {limit} registros)...")
        try:
            response = self.session.get(
                self.API_SINIESTROS_1, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            logger.info(f"✓ Dataset 1 descargado: {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"Error descargando dataset 1: {e}")
            return pd.DataFrame()

    def descargar_dataset_2(self, limit: int = 50000) -> pd.DataFrame:
        """Descargar segundo dataset de siniestros.

        Columnas: gravedad, fecha, a_o, hora, jornada, etc.

        Args:
            limit: Número máximo de registros a descargar.

        Returns:
        
            DataFrame con los datos descargados.
        """
        params = {
            "$limit": limit,
            "$offset": 0,
            "$order": "a_o ASC",
        }
        logger.info(f"Descargando dataset 2 (límite: {limit} registros)...")
        try:
            response = self.session.get(
                self.API_SINIESTROS_2, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            logger.info(f"✓ Dataset 2 descargado: {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"Error descargando dataset 2: {e}")
            return pd.DataFrame()

    def descargar_ambos(self, limit: int = 50000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Descargar ambos datasets.

        Args:
            limit: Límite de registros por dataset.

        Returns:
            Tupla (df1, df2) con ambos datasets.
        """
        df1 = self.descargar_dataset_1(limit)
        df2 = self.descargar_dataset_2(limit)
        return df1, df2


class LimpiadordeDatos:
    """Limpieza y transformación de datos de siniestros."""

    def __init__(self):
        """Inicializar limpiador."""
        self.reporte_limpieza = {}

    def limpiar_dataset(
        self, df: pd.DataFrame, nombre_dataset: str = "dataset"
    ) -> pd.DataFrame:
        """Ejecutar pipeline completo de limpieza.

        Args:
            df: DataFrame a limpiar.
            nombre_dataset: Nombre del dataset para logging.

        Returns:
            DataFrame limpio.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Limpiando {nombre_dataset}...")
        logger.info(f"{'='*60}")

        # Trabajar sobre una copia para evitar SettingWithCopyWarning
        df = df.copy()

        # Registrar estado inicial
        registros_iniciales = len(df)
        logger.info(f"Registros iniciales: {registros_iniciales:,}")

        # Paso 1: Eliminar duplicados
        df = self._eliminar_duplicados(df, nombre_dataset)
        logger.info(f"  → Después de eliminar duplicados: {len(df):,} registros")

        # Paso 2: Validar tipos de datos
        df = self._validar_tipos(df)
        logger.info(f"  → Después de validar tipos: {len(df):,} registros")

        # Paso 3: Limpiar valores nulos
        df = self._limpiar_nulos(df, nombre_dataset)
        logger.info(f"  → Después de limpiar nulos: {len(df):,} registros")

        # Paso 4: Eliminar outliers
        df = self._eliminar_outliers(df)
        logger.info(f"  → Después de eliminar outliers: {len(df):,} registros")

        # Paso 5: Estandarizar textos
        df = self._estandarizar_texto(df)
        logger.info(f"  → Después de estandarizar texto: {len(df):,} registros")

        # Paso 6: Validar rangos y valores
        df = self._validar_rangos(df)
        logger.info(f"  → Después de validar rangos: {len(df):,} registros")

        registros_finales = len(df)
        porcentaje_retenido = (registros_finales / registros_iniciales * 100) if registros_iniciales > 0 else 0
        logger.info(
            f"\n✓ {nombre_dataset} limpio: {registros_finales:,} registros "
            f"({porcentaje_retenido:.1f}% retenido)"
        )

        self.reporte_limpieza[nombre_dataset] = {
            "iniciales": registros_iniciales,
            "finales": registros_finales,
            "porcentaje_retenido": porcentaje_retenido,
        }

        return df

    def _eliminar_duplicados(self, df: pd.DataFrame, nombre: str) -> pd.DataFrame:
        """Eliminar filas completamente duplicadas."""
        antes = len(df)
        df = df.drop_duplicates()
        despues = len(df)
        if antes > despues:
            logger.warning(f"  - Duplicados eliminados: {antes - despues}")
        return df

    def _validar_tipos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convertir columnas a tipos apropiados."""

        # Convertir columnas de fecha (solo 'fecha') a datetime
        for col in df.columns:
            try:
                if col.lower() == "fecha":
                    df.loc[:, col] = pd.to_datetime(df[col], errors="coerce")
                    logger.info(f"  - Convertida {col} a datetime")
                # Años (a_o, ano, año) como enteros
                elif col.lower() in {"a_o", "a_o", "ano", "año"}:
                    df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
                    logger.info(f"  - Convertida {col} a numérico (año)")
            except Exception as e:
                logger.warning(f"  - No se pudo convertir {col}: {e}")

        # Convertir columnas numéricas (incluye 'hora')
        columnas_numericas = {
            "edad",
            "lat",
            "long",
            "latitud",
            "longitud",
            "lesionados",
            "muertos",
            "homicidios",
            "hora",
        }
        for col in df.columns:
            if col.lower() in columnas_numericas:
                try:
                    df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
                    logger.info(f"  - Convertida {col} a numérico")
                except Exception as e:
                    logger.warning(f"  - No se pudo convertir {col}: {e}")

        # Inferir año ('a_o', 'ano', 'año') desde 'fecha' cuando falte
        year_cols = [c for c in df.columns if c.lower() in {"a_o", "ano", "año"}]
        if "fecha" in df.columns:
            if year_cols:
                for yc in year_cols:
                    try:
                        mask = df[yc].isna() & df["fecha"].notna()
                        if mask.any():
                            df.loc[mask, yc] = df.loc[mask, "fecha"].dt.year
                            logger.info(f"  - Inferidos {mask.sum()} valores de {yc} desde 'fecha'")
                    except Exception as e:
                        logger.warning(f"  - No se pudo inferir año para {yc}: {e}")
            else:
                # Si no existe columna de año, crear 'a_o' basada en 'fecha'
                try:
                    df.loc[:, "a_o"] = pd.to_numeric(df["fecha"].dt.year, errors="coerce")
                    logger.info("  - Creada columna 'a_o' a partir de 'fecha'")
                except Exception as e:
                    logger.warning(f"  - No se pudo crear 'a_o' desde 'fecha': {e}")

        return df

    def _limpiar_nulos(self, df: pd.DataFrame, nombre: str) -> pd.DataFrame:
        """Manejar valores nulos según columna.
        
        Política conservadora: solo elimina filas si carecen de información crítica real.
        No elimina por simples campos nulos (p. ej. 'hora', 'lat', 'long' pueden estar nulos).
        """
        logger.info(f"  - Analizando valores nulos...")
        nulos_por_col = df.isnull().sum()
        if nulos_por_col.sum() > 0:
            logger.info(f"    Valores nulos por columna:")
            for col, cnt in nulos_por_col[nulos_por_col > 0].items():
                pct = (cnt / len(df)) * 100
                logger.info(f"      {col}: {cnt} ({pct:.1f}%)")

        # Solo eliminar filas completamente vacías o sin fecha (campo absolutamente crítico)
        # Nota: 'a_o' y 'barrios_corregimiento_via' pueden inferirse o no ser críticos
        columnas_criticas = ["fecha"]
        for col in columnas_criticas:
            if col in df.columns:
                antes = len(df)
                # Si la columna está totalmente nula, evitamos eliminar todas las filas
                non_null_count = df[col].notna().sum()
                if non_null_count == 0:
                    logger.warning(
                        f"    Columna crítica '{col}' está totalmente vacía; se omite eliminación por esta columna."
                    )
                    continue

                df = df.dropna(subset=[col])
                despues = len(df)
                if antes > despues:
                    logger.warning(
                        f"    Filas sin {col} eliminadas: {antes - despues}"
                    )

        return df

    def _eliminar_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Eliminar outliers en columnas numéricas."""
        columnas_numericas_existentes = df.select_dtypes(include=[np.number]).columns
        for col in columnas_numericas_existentes:
            cname = col.lower()
            if cname == "edad":
                # Edades válidas: 0-120
                antes = len(df)
                df = df[(df[col].isna()) | ((df[col] >= 0) & (df[col] <= 120))]
                despues = len(df)
                if antes > despues:
                    logger.warning(f"  - Outliers en {col} eliminados: {antes - despues}")

            # Saltamos procesamiento individual de lat/long; lo haremos al final de forma conjunta
            elif cname in {"lat", "latitud", "long", "longitud"}:
                continue

        # Filtrado geográfico ULTRA-CONSERVADOR: solo eliminar si coordenadas son claramente inválidas
        # (ej. valores imposibles como 0, 999, o fuera de Colombia por mucho)
        lat_cols = [c for c in df.columns if c.lower() in {"lat", "latitud"}]
        long_cols = [c for c in df.columns if c.lower() in {"long", "longitud"}]

        # Filtrado geográfico DESACTIVADO: se quitó porque estaba eliminando
        # registros válidos en producción. Si se necesita reactivar, añadir
        # una bandera o parámetro para controlarlo y probar con muestras.
        if lat_cols and long_cols:
            lat_col = lat_cols[0]
            long_col = long_cols[0]
            antes = len(df)
            n_both_null = ((df[lat_col].isna()) & (df[long_col].isna())).sum()
            n_one_null = ((df[lat_col].isna()) ^ (df[long_col].isna())).sum()
            n_both_present = (df[lat_col].notna() & df[long_col].notna()).sum()
            logger.info(
                f"  - Filtrado geográfico DESACTIVADO: {antes} registros. "
                f"(ambos nulos: {n_both_null}, al menos uno nulo: {n_one_null}, "
                f"ambos presentes: {n_both_present})"
            )

        return df

    def _estandarizar_texto(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estandarizar columnas de texto."""
        # Seleccionar columnas que potencialmente contienen texto
        columnas_texto = df.select_dtypes(include=["object", "string"]).columns
        for col in columnas_texto:
            try:
                # Coercionar a la dtype 'string' de pandas para manejar NA correctamente
                s = df[col].astype("string")
                # Eliminar espacios extra de manera segura
                s = s.str.strip()

                # Convertir a mayúsculas para ciertas categorías
                if col.lower() in [
                    "genero",
                    "jornada",
                    "dia_semana",
                    "zona",
                    "autoridad",
                    "clase_siniestro",
                    "clase_de_siniestro",
                    "gravedad",
                ]:
                    s = s.str.upper()

                # Asignar de forma segura usando .loc
                df.loc[:, col] = s
            except Exception as e:
                logger.warning(f"  - No se pudo estandarizar {col}: {e}")

        return df

    def _validar_rangos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validar rangos lógicos de valores."""
        # Horas válidas: 0-23
        if "hora" in df.columns:
            # Asegurar que 'hora' sea numérico. Algunas APIs usan formatos '08:30' o '8 AM'.
            serie_hora = df["hora"]
            non_numeric_before = serie_hora.apply(lambda x: not pd.api.types.is_number(x)) .sum()
            # Intentar conversión directa a numérico (esto convertirá '08' -> 8 y dejará NaN donde no sea posible)
            df.loc[:, "hora"] = pd.to_numeric(df["hora"], errors="coerce")

            # Si aún existen valores no numéricos (por ejemplo '08:30' o '8 AM'), intentar extraer dígitos iniciales
            if df["hora"].dtype == object or df["hora"].isna().any():
                try:
                    # Extraer primeras dos cifras de la cadena que representen la hora
                    posibles = df["hora"].astype(str).str.extract(r"(\d{1,2})", expand=False)
                    df.loc[:, "hora"] = pd.to_numeric(posibles, errors="coerce").combine_first(df["hora"])
                except Exception:
                    pass

            non_numeric_after = df["hora"].apply(lambda x: not pd.api.types.is_number(x)).sum()
            logger.info(f"  - 'hora' no numérica antes: {non_numeric_before}, después: {non_numeric_after}")

            antes = len(df)
            df = df[(df["hora"].isna()) | ((df["hora"] >= 0) & (df["hora"] <= 23))]
            despues = len(df)
            if antes > despues:
                logger.warning(f"  - Horas inválidas eliminadas: {antes - despues}")

        return df

    def generar_reporte(self) -> str:
        """Generar reporte de limpieza en formato texto."""
        reporte = "\n" + "=" * 70 + "\n"
        reporte += "REPORTE DE LIMPIEZA DE DATOS\n"
        reporte += "=" * 70 + "\n\n"

        for dataset, stats in self.reporte_limpieza.items():
            reporte += f"{dataset}:\n"
            reporte += f"  Registros iniciales: {stats['iniciales']:,}\n"
            reporte += f"  Registros finales:   {stats['finales']:,}\n"
            reporte += f"  Registros eliminados: {stats['iniciales'] - stats['finales']:,}\n"
            reporte += f"  % Retenido:          {stats['porcentaje_retenido']:.1f}%\n\n"

        return reporte


def procesar_siniestros(
    directorio_salida: str = "data", limite_registros: int = 50000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Función principal para descargar y procesar todos los datos.

    Args:
        directorio_salida: Directorio donde guardar CSVs limpios.
        limite_registros: Límite de registros a descargar por dataset.

    Returns:
        Tupla con (df1_limpio, df2_limpio).
    """
    # Crear directorio de salida
    Path(directorio_salida).mkdir(exist_ok=True)

    # Descargar datos
    cliente = SiniestrosAPIClient()
    df1_crudo, df2_crudo = cliente.descargar_ambos(limit=limite_registros)

    if df1_crudo.empty or df2_crudo.empty:
        logger.error("Error: No se pudieron descargar los datasets")
        return pd.DataFrame(), pd.DataFrame()

    # Limpiar datos
    limpiador = LimpiadordeDatos()
    df1_limpio = limpiador.limpiar_dataset(df1_crudo, "Dataset 1 (Siniestros Viales)")
    df2_limpio = limpiador.limpiar_dataset(df2_crudo, "Dataset 2 (Gravedad/Víctimas)")

    # Guardar datos limpios
    ruta_1 = Path(directorio_salida) / "siniestros_1_limpio.csv"
    ruta_2 = Path(directorio_salida) / "siniestros_2_limpio.csv"

    df1_limpio.to_csv(ruta_1, index=False, encoding="utf-8")
    df2_limpio.to_csv(ruta_2, index=False, encoding="utf-8")

    logger.info(f"\n✓ Datos guardados:")
    logger.info(f"  - {ruta_1}")
    logger.info(f"  - {ruta_2}")

    # Imprimir reporte
    reporte = limpiador.generar_reporte()
    logger.info(reporte)

    # Guardar reporte
    ruta_reporte = Path(directorio_salida) / "reporte_limpieza.txt"
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write(reporte)
    logger.info(f"  - {ruta_reporte}")

    return df1_limpio, df2_limpio
