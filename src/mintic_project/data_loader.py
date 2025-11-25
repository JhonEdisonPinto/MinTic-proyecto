"""Módulo mínimo para descargar y limpiar datasets de siniestros.

Este loader es intencionalmente conservador: sólo elimina filas vacías
y opcionalmente filas con cualquier NULL si se solicita. Además corrige
comas por punto en columnas de latitud/longitud cuando aparezcan.

El objetivo es mantener el formato lo más cercano posible al original
para consumo por herramientas como Power BI.
"""
from pathlib import Path
from typing import Tuple
import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SiniestrosAPIClient:
    """Cliente sencillo para descargar datasets desde datos.gov.co.

    Mantiene la misma interfaz que la versión anterior para compatibilidad.
    """

    API_SINIESTROS_1 = "https://www.datos.gov.co/resource/sjpx-eqfp.json"
    API_SINIESTROS_2 = "https://www.datos.gov.co/resource/xx6f-f84h.json"

    def __init__(self, session=None, timeout: int = 30):
        self.timeout = timeout
        # importar requests bajo demanda para evitar fallos si no está instalado
        try:
            import requests

            self.session = session or requests.Session()
        except Exception:
            self.session = session

    def descargar_dataset_1(self, limit: int = 50000) -> pd.DataFrame:
        params = {"$limit": limit, "$offset": 0}
        logger.info(f"Descargando dataset 1 (límite={limit})...")
        if not self.session:
            logger.warning("No hay session HTTP disponible; espere un DataFrame vacío.")
            return pd.DataFrame()
        try:
            r = self.session.get(self.API_SINIESTROS_1, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error descargando dataset 1: {e}")
            return pd.DataFrame()

    def descargar_dataset_2(self, limit: int = 50000) -> pd.DataFrame:
        params = {"$limit": limit, "$offset": 0}
        logger.info(f"Descargando dataset 2 (límite={limit})...")
        if not self.session:
            logger.warning("No hay session HTTP disponible; espere un DataFrame vacío.")
            return pd.DataFrame()
        try:
            r = self.session.get(self.API_SINIESTROS_2, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error descargando dataset 2: {e}")
            return pd.DataFrame()


class LimpiadordeDatos:
        """Limpieza mínima: eliminar nulos y corregir separadores decimales en coordenadas.

        Reglas:
        - Eliminar filas completamente vacías (todas las columnas NULL).
        - Opcionalmente, eliminar filas con cualquier NULL si `drop_any=True`.
        - Buscar columnas de lat/long por nombres comunes y reemplazar ',' por '.'
        dentro de esas columnas, intentando convertir a numérico.
        """

        def __init__(self):
            self.reporte = {}

        def limpiar_dataset(self, df: pd.DataFrame, nombre: str = "dataset", drop_any: bool = False) -> pd.DataFrame:
            df = df.copy()
            inicial = len(df)
            logger.info(f"Iniciando limpieza mínima para '{nombre}' ({inicial} registros)")

            # 1) Eliminar filas completamente vacías
            df = df.dropna(how="all")
            logger.info(f"  Filas no totalmente vacías: {len(df)}")

            # 2) Eliminar filas con cualquier NULL si el usuario lo pide
            if drop_any:
                antes = len(df)
                df = df.dropna(how="any")
                logger.info(f"  drop_any=True: eliminadas {antes - len(df)} filas")

            # 3) Detectar columnas relevantes
            lat_names = {"lat", "latitud"}
            lon_names = {"long", "longitud"}
            cols = df.columns.tolist()

            lat_col = next((c for c in cols if c.lower() in lat_names), None)
            lon_col = next((c for c in cols if c.lower() in lon_names), None)
            edad_col = next((c for c in cols if c.lower() in {"edad"}), None)

            # --- DETECTAR COLUMNA DE HORA ---
            hora_col = next((c for c in cols if "hora" in c.lower()), None)

            # Función auxiliar para limpiar columnas numéricas con coma/punto
            def normalizar_decimal(col):
                try:
                    serie = df[col].astype(str).str.replace(",", ".", regex=False)
                    df[col] = pd.to_numeric(serie, errors="coerce")
                    logger.info(f"    Columna '{col}' limpiada (coma→punto, numérico)")
                except Exception as e:
                    logger.warning(f"    No se pudo procesar columna '{col}': {e}")

            # Aplicar limpieza base
            for c in [lat_col, lon_col, edad_col]:
                if c:
                    normalizar_decimal(c)

            # --- NUEVO: Limpieza de hora ---
            if hora_col:
                try:
                    logger.info(f"    Corrigiendo columna de hora '{hora_col}'")

                    def fix_hour(h):
                        if pd.isna(h):
                            return None
                        h = str(h).strip()

                        # Reemplazo de errores comunes
                        h = h.replace(";", ":")
                        h = h.replace("::", ":")
                        h = h.replace(" ", "")

                        # Si no tiene ":", intento inferir (ej '310' -> 3:10)
                        if ":" not in h:
                            if len(h) == 4:
                                h = h[:2] + ":" + h[2:]
                            elif len(h) == 3:
                                h = h[0] + ":" + h[1:]
                            else:
                                return None

                        # Validación final
                        partes = h.split(":")
                        if len(partes) != 2:
                            return None

                        try:
                            hh = int(partes[0])
                            mm = int(partes[1])
                            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                                return None
                        except:
                            return None

                        return f"{hh:02d}:{mm:02d}"

                    df[hora_col] = df[hora_col].apply(fix_hour)
                    logger.info(f"    Columna '{hora_col}' corregida y normalizada")
                except Exception as e:
                    logger.warning(f"    Error limpiando la hora: {e}")

            # --- Conversión final para Power BI (puntos → comas) ---
            for c in [lat_col, lon_col, edad_col]:
                if c and df[c].dtype in ("float64", "float32", "int64"):
                    df[c] = df[c].apply(lambda x: str(x).replace(".", ",") if pd.notna(x) else x)
                    logger.info(f"    Columna '{c}' convertida a formato con coma para Power BI")

            finales = len(df)
            pct = (finales / inicial * 100) if inicial else 0
            self.reporte[nombre] = {"iniciales": inicial, "finales": finales, "porcentaje": pct}
            logger.info(f"Limpieza mínima finalizada para '{nombre}': {finales} registros ({pct:.1f}% retenido)")

            return df



def procesar_siniestros(directorio_salida: str = "data", limite_registros: int = 50000, drop_any: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Descargar (si es posible) y guardar datasets limpios con la limpieza mínima.

    Args:
        directorio_salida: carpeta donde se guardarán los CSV
        limite_registros: límite para la descarga
        drop_any: si True elimina filas que contengan cualquier NULL
    """
    Path(directorio_salida).mkdir(parents=True, exist_ok=True)

    cliente = SiniestrosAPIClient()
    df1 = cliente.descargar_dataset_1(limit=limite_registros)
    df2 = cliente.descargar_dataset_2(limit=limite_registros)

    if df1.empty and df2.empty:
        logger.error("No se pudieron descargar datasets; asegúrate de tener conexión o usa archivos locales.")

    limpiador = LimpiadordeDatos()
    if not df1.empty:
        df1_limpio = limpiador.limpiar_dataset(df1, "Dataset 1 (Siniestros Viales)", drop_any=drop_any)
        ruta1 = Path(directorio_salida) / "siniestros_1_limpio.csv"
        df1_limpio.to_csv(ruta1, index=False, encoding="utf-8")
        logger.info(f"Guardado {ruta1} ({len(df1_limpio)} filas)")
    else:
        df1_limpio = pd.DataFrame()

    if not df2.empty:
        df2_limpio = limpiador.limpiar_dataset(df2, "Dataset 2 (Gravedad/Víctimas)", drop_any=drop_any)
        ruta2 = Path(directorio_salida) / "siniestros_2_limpio.csv"
        df2_limpio.to_csv(ruta2, index=False, encoding="utf-8")
        logger.info(f"Guardado {ruta2} ({len(df2_limpio)} filas)")
    else:
        df2_limpio = pd.DataFrame()

    # Guardar reporte simple
    try:
        reporte_path = Path(directorio_salida) / "reporte_limpieza.txt"
        with open(reporte_path, "w", encoding="utf-8") as f:
            for k, v in limpiador.reporte.items():
                f.write(f"{k}: iniciales={v['iniciales']}, finales={v['finales']}, %={v['porcentaje']:.1f}\n")
        logger.info(f"Reporte guardado en {reporte_path}")
    except Exception as e:
        logger.warning(f"No se pudo guardar reporte: {e}")

    return df1_limpio, df2_limpio