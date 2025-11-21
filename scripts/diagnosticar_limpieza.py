#!/usr/bin/env python
"""Script de diagnóstico: ejecutar limpieza paso por paso para detectar dónde se pierden registros."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import logging

# Configurar logging en VERBOSE
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from mintic_project.data_loader import SiniestrosAPIClient, LimpiadordeDatos

# Descargar datos reales
print("=" * 70)
print("DESCARGANDO DATOS REALES DE LAS APIs...")
print("=" * 70)

cliente = SiniestrosAPIClient()
df1_crudo, df2_crudo = cliente.descargar_ambos(limit=50000)

print(f"\nDataset 1 crudo: {len(df1_crudo)} registros")
print(f"Dataset 2 crudo: {len(df2_crudo)} registros")

# Analizar Dataset 2 paso a paso
print("\n" + "=" * 70)
print("DIAGNÓSTICO DATASET 2 (Gravedad/Víctimas)")
print("=" * 70)

limpiador = LimpiadordeDatos()
df2 = df2_crudo.copy()

print(f"\n0. Inicial: {len(df2)} registros")
print(f"   Columnas: {list(df2.columns)}")
print(f"   Primeras filas:\n{df2.head(2)}")

# Paso 1
df2 = limpiador._eliminar_duplicados(df2, "Dataset 2")
print(f"\n1. Después duplicados: {len(df2)} registros")

# Paso 2
df2 = limpiador._validar_tipos(df2)
print(f"2. Después validar tipos: {len(df2)} registros")

# Paso 3
df2 = limpiador._limpiar_nulos(df2, "Dataset 2")
print(f"3. Después limpiar nulos: {len(df2)} registros")

# Paso 4
df2_antes_outliers = len(df2)
df2 = limpiador._eliminar_outliers(df2)
print(f"4. Después eliminar outliers: {len(df2)} registros (eliminadas {df2_antes_outliers - len(df2)})")

# Paso 5
df2 = limpiador._estandarizar_texto(df2)
print(f"5. Después estandarizar texto: {len(df2)} registros")

# Paso 6
df2_antes_rangos = len(df2)
df2 = limpiador._validar_rangos(df2)
print(f"6. Después validar rangos: {len(df2)} registros (eliminadas {df2_antes_rangos - len(df2)})")

print(f"\nFINAL: {len(df2)} registros retenidos de {len(df2_crudo)} iniciales")
print(f"% retenido: {len(df2) / len(df2_crudo) * 100:.1f}%")

if len(df2) > 0:
    print(f"\nÚltimas filas del resultado:")
    print(df2.tail(3))
