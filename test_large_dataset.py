"""
Script para probar la protección de muestreo con datasets grandes.
Genera un CSV de prueba con más de 3000 filas.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Generar dataset de prueba grande (3500 filas)
print("📊 Generando dataset de prueba con 3500 filas...")

np.random.seed(42)

# Crear datos simulados de siniestros
data = {
    'FECHA': pd.date_range('2020-01-01', periods=3500, freq='H'),
    'TIPO_SINIESTRO': np.random.choice(['Choque', 'Atropello', 'Volcamiento', 'Caída'], 3500),
    'JORNADA': np.random.choice(['Mañana', 'Tarde', 'Noche', 'Madrugada'], 3500),
    'GENERO': np.random.choice(['Masculino', 'Femenino', 'No reporta'], 3500, p=[0.6, 0.35, 0.05]),
    'ZONA': np.random.choice(['Urbana', 'Rural'], 3500, p=[0.75, 0.25]),
    'HIPOTESIS': np.random.choice(['Imprudencia conductor', 'Estado embriaguez', 'Exceso velocidad', 'Desacato señales'], 3500),
    'GRAVEDAD': np.random.choice(['Leve', 'Grave', 'Mortal'], 3500, p=[0.6, 0.3, 0.1]),
    'EDAD': np.random.randint(15, 75, 3500),
    'HORA': np.random.randint(0, 24, 3500)
}

df = pd.DataFrame(data)

# Guardar en data/
output_path = Path("data/test_large_dataset.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ Dataset de prueba creado: {output_path}")
print(f"   - Total de filas: {len(df):,}")
print(f"   - Columnas: {', '.join(df.columns)}")
print(f"\n🔍 Primeras filas:")
print(df.head())
print(f"\n📈 Estadísticas:")
print(df.describe())
print(f"\n💡 Ahora ejecuta la app y carga este CSV para probar la protección de muestreo.")
