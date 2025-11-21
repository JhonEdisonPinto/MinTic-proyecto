#!/usr/bin/env python
"""Script rápido para probar el limpiador con datos similares a la API real."""
import sys
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from mintic_project.data_loader import LimpiadordeDatos

# Simular Dataset 1 similar al que mostraste
df1_test = pd.DataFrame({
    'a_o': [2022.0, 2022.0, 2022.0, 2022.0, 2022.0],
    'ipat': [8446.0, 8446.0, 8447.0, 8447.0, 8447.0],
    'clase_siniestro': ['CHOQUE', 'CHOQUE', 'CHOQUE', 'CHOQUE', 'CHOQUE'],
    'fecha': ['2022-01-01', '2022-01-01', '2022-01-01', '2022-01-01', '2022-01-01'],
    'hora': [None, None, None, None, None],  # Todas nulas como en tu reporte
    'jornada': ['MANANA', 'MANANA', 'TARDE', 'NOCHE', 'NOCHE'],
    'edad': [22.0, 37.0, 28.0, 31.0, 30.0],
    'genero': ['MASCULINO', 'MASCULINO', 'FEMENINO', 'MASCULINO', 'MASCULINO'],
    'lat': [-3.73, -3.73, -3.74, -3.73, -3.73],
    'long': [-76.28, -76.28, -76.29, -76.28, -76.28],
})

# Simular Dataset 2
df2_test = pd.DataFrame({
    'gravedad': ['HERIDOS', 'HERIDOS', 'DAÑOS', 'DAÑOS', 'MUERTO'],
    'fecha': ['2021-01-02', '2021-01-02', '2021-01-04', '2021-01-04', '2021-01-01'],
    'a_o': [2021, 2021, 2021, 2021, 2021],
    'hora': [None, None, None, None, None],  # Todas nulas
    'jornada': ['NOCHE', 'NOCHE', 'MAÑANA', 'MAÑANA', 'MADRUGADA'],
    'lat': [-3.73, -3.73, -3.74, -3.74, -3.73],
    'long': [-76.28, -76.28, -76.29, -76.29, -76.28],
})

limpiador = LimpiadordeDatos()

print("=" * 70)
print("Probando Dataset 1 (similar a Siniestros Viales)")
print("=" * 70)
print(f"Registros iniciales: {len(df1_test)}")
df1_limpio = limpiador.limpiar_dataset(df1_test.copy(), "Dataset 1 (Test)")
print(f"Registros finales: {len(df1_limpio)}")
print()

print("=" * 70)
print("Probando Dataset 2 (similar a Gravedad/Víctimas)")
print("=" * 70)
print(f"Registros iniciales: {len(df2_test)}")
df2_limpio = limpiador.limpiar_dataset(df2_test.copy(), "Dataset 2 (Test)")
print(f"Registros finales: {len(df2_limpio)}")
print()

print("=" * 70)
print("REPORTE FINAL")
print("=" * 70)
print(limpiador.generar_reporte())

# Validaciones rápidas
assert len(df1_limpio) > 0, "ERROR: Dataset 1 quedó vacío!"
assert len(df2_limpio) > 0, "ERROR: Dataset 2 quedó vacío!"
print("✓ Todos los datasets tienen registros retenidos")
