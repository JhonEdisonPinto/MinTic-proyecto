"""Módulo para descargar, limpiar y gestionar múltiples datasets de siniestros.

Funcionalidades:
- Descargar datasets predeterminados de datos.gov.co
- Agregar nuevos datasets dinámicamente (URL + nombre personalizado)
- Limpieza automática de todos los datasets
- Persistencia de configuración en JSON
- Selección de dataset activo

El objetivo es mantener el formato lo más cercano posible al original
para consumo por herramientas como Power BI.
"""
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import json
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatasetManager:
    """Gestiona múltiples datasets: predeterminados y personalizados.
    
    Funcionalidades:
    - Lista de datasets predeterminados (Palmira 1 y 2)
    - Agregar datasets personalizados con nombre y URL
    - Persistencia en archivo JSON
    - Marcar dataset activo
    """
    
    # Datasets predeterminados
    DEFAULTS = {
        "siniestros_palmira_2022-2024": "https://www.datos.gov.co/resource/sjpx-eqfp.json",
        "siniestros_palmira_2021": "https://www.datos.gov.co/resource/xx6f-f84h.json",
    }
    
    def __init__(self, config_file: str = "data/datasets_config.json"):
        self.config_file = Path(config_file)
        self.datasets = {}
        self.active_dataset = None
        self._load_config()
    
    def _load_config(self):
        """Cargar configuración desde JSON o inicializar con defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.datasets = data.get("datasets", {})
                    self.active_dataset = data.get("active_dataset")
                    logger.info(f"✓ Configuración cargada: {len(self.datasets)} datasets")
            except Exception as e:
                logger.warning(f"Error cargando config, usando defaults: {e}")
                self.datasets = self.DEFAULTS.copy()
                self.active_dataset = list(self.DEFAULTS.keys())[0]
        else:
            self.datasets = self.DEFAULTS.copy()
            self.active_dataset = list(self.DEFAULTS.keys())[0]
        
        # ROBUSTEZ: Garantizar que los datasets predeterminados siempre existan
        for name, url in self.DEFAULTS.items():
            if name not in self.datasets:
                logger.warning(f"⚠️ Restaurando dataset predeterminado faltante: {name}")
                self.datasets[name] = url
        
        # Validar que el dataset activo exista
        if self.active_dataset not in self.datasets:
            logger.warning(f"⚠️ Dataset activo inválido, usando predeterminado")
            self.active_dataset = list(self.DEFAULTS.keys())[0]
            self._save_config()
        
        # Guardar si hubo cambios
        if not self.config_file.exists() or any(name not in data.get("datasets", {}) for name in self.DEFAULTS):
            self._save_config()
    
    def _save_config(self):
        """Guardar configuración en JSON con protección de datasets predeterminados."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # ROBUSTEZ: Asegurar que los defaults estén presentes antes de guardar
        for name, url in self.DEFAULTS.items():
            if name not in self.datasets:
                logger.warning(f"⚠️ Agregando dataset predeterminado antes de guardar: {name}")
                self.datasets[name] = url
        
        data = {
            "datasets": self.datasets,
            "active_dataset": self.active_dataset,
            "last_updated": datetime.now().isoformat()
        }
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Configuración guardada en {self.config_file}")
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
    
    def add_dataset(self, name: str, url: str) -> bool:
        """Agregar nuevo dataset personalizado."""
        if name in self.datasets:
            logger.warning(f"Dataset '{name}' ya existe")
            return False
        
        self.datasets[name] = url
        self._save_config()
        logger.info(f"✓ Dataset agregado: {name}")
        return True
    
    def remove_dataset(self, name: str) -> bool:
        """Eliminar dataset (solo personalizados, no defaults).
        
        PROTECCIÓN: Los datasets predeterminados NO pueden ser eliminados.
        
        Args:
            name: Nombre del dataset a eliminar
            
        Returns:
            bool: True si se eliminó correctamente, False si no se pudo eliminar
        """
        # ROBUSTEZ: Prevenir eliminación de datasets predeterminados
        if name in self.DEFAULTS:
            logger.warning(f"🛡️ PROTECCIÓN: No se puede eliminar dataset predeterminado '{name}'")
            return False
        
        if name not in self.datasets:
            logger.warning(f"Dataset '{name}' no existe")
            return False
        
        # Eliminar dataset personalizado
        del self.datasets[name]
        
        # Si era el activo, cambiar al primer predeterminado
        if self.active_dataset == name:
            self.active_dataset = list(self.DEFAULTS.keys())[0]
            logger.info(f"📌 Dataset activo cambiado a: {self.active_dataset}")
        
        self._save_config()
        logger.info(f"✓ Dataset personalizado eliminado: {name}")
        return True
    
    def set_active(self, name: str) -> bool:
        """Establecer dataset activo."""
        if name not in self.datasets:
            logger.warning(f"Dataset no encontrado: {name}")
            return False
        
        self.active_dataset = name
        self._save_config()
        logger.info(f"✓ Dataset activo: {name}")
        return True
    
    def get_active_url(self) -> Optional[str]:
        """Obtener URL del dataset activo."""
        if self.active_dataset and self.active_dataset in self.datasets:
            return self.datasets[self.active_dataset]
        return None
    
    def list_datasets(self) -> Dict[str, str]:
        """Listar todos los datasets disponibles."""
        return self.datasets.copy()
    
    def get_dataset_info(self, name: str) -> Dict:
        """Información del dataset."""
        return {
            "name": name,
            "url": self.datasets.get(name),
            "is_default": name in self.DEFAULTS,
            "is_active": name == self.active_dataset
        }





class SiniestrosAPIClient:
    """Cliente para descargar datasets desde URLs de datos.gov.co.
    
    Ahora soporta URLs dinámicas además de las predeterminadas.
    """

    def __init__(self, session=None, timeout: int = 30):
        self.timeout = timeout
        try:
            import requests
            self.session = session or requests.Session()
        except Exception:
            self.session = session

    def descargar_desde_url(self, url: str, limit: int = 50000) -> pd.DataFrame:
        """Descargar dataset desde cualquier URL de datos.gov.co."""
        # Limpiar URL: si tiene query params, usar solo la base
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url)
        
        # Si la URL ya tiene $query o parámetros, extraer solo la base
        base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if not base_url.endswith('.json'):
            base_url = base_url.rstrip('/') + '.json'
        
        # Parámetros estándar de paginación
        params = {"$limit": limit, "$offset": 0}
        
        logger.info(f"Descargando desde {base_url} (límite={limit})...")
        
        if not self.session:
            logger.warning("No hay session HTTP disponible.")
            return pd.DataFrame()
        
        try:
            r = self.session.get(base_url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data)
            logger.info(f"✓ Descargados {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"Error descargando desde {base_url}: {e}")
            return pd.DataFrame()

    # Mantener métodos legacy para compatibilidad
    def descargar_dataset_1(self, limit: int = 50000) -> pd.DataFrame:
        url = "https://www.datos.gov.co/resource/sjpx-eqfp.json"
        return self.descargar_desde_url(url, limit)

    def descargar_dataset_2(self, limit: int = 50000) -> pd.DataFrame:
        url = "https://www.datos.gov.co/resource/xx6f-f84h.json"
        return self.descargar_desde_url(url, limit)


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



def descargar_y_limpiar_dataset(url: str, nombre: str, directorio_salida: str = "data", limite: int = 50000, drop_any: bool = False) -> pd.DataFrame:
    """Descargar y limpiar un dataset individual desde su URL.
    
    Args:
        url: URL del endpoint de datos.gov.co
        nombre: Nombre identificador del dataset
        directorio_salida: Carpeta donde guardar el CSV limpio
        limite: Límite de registros a descargar
        drop_any: Si True, elimina filas con cualquier NULL
    
    Returns:
        DataFrame limpio
    """
    Path(directorio_salida).mkdir(parents=True, exist_ok=True)
    
    cliente = SiniestrosAPIClient()
    df = cliente.descargar_desde_url(url, limit=limite)
    
    if df.empty:
        logger.warning(f"No se descargaron datos para '{nombre}'")
        return pd.DataFrame()
    
    # Limpiar
    limpiador = LimpiadordeDatos()
    df_limpio = limpiador.limpiar_dataset(df, nombre, drop_any=drop_any)
    
    # Guardar
    nombre_archivo = nombre.lower().replace(" ", "_").replace("-", "_") + ".csv"
    ruta = Path(directorio_salida) / nombre_archivo
    df_limpio.to_csv(ruta, index=False, encoding="utf-8")
    logger.info(f"✓ Dataset guardado: {ruta} ({len(df_limpio)} filas)")
    
    return df_limpio


def procesar_siniestros(directorio_salida: str = "data", limite_registros: int = 50000, drop_any: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Descargar y guardar datasets predeterminados (legacy, mantener compatibilidad).
    
    Args:
        directorio_salida: carpeta donde se guardarán los CSV
        limite_registros: límite para la descarga
        drop_any: si True elimina filas que contengan cualquier NULL
    """
    Path(directorio_salida).mkdir(parents=True, exist_ok=True)

    # Usar el nuevo gestor para obtener URLs
    manager = DatasetManager(config_file=f"{directorio_salida}/datasets_config.json")
    datasets_info = manager.list_datasets()
    
    # Descargar primeros dos datasets (compatibilidad con código existente)
    df_list = []
    for nombre, url in list(datasets_info.items())[:2]:
        df = descargar_y_limpiar_dataset(url, nombre, directorio_salida, limite_registros, drop_any)
        df_list.append(df)
    
    df1 = df_list[0] if len(df_list) > 0 else pd.DataFrame()
    df2 = df_list[1] if len(df_list) > 1 else pd.DataFrame()

    return df1, df2