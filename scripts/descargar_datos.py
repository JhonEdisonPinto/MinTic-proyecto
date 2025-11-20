"""Script para descargar y limpiar datos de siniestros viales.

Ejecutar desde la terminal:
    python scripts/descargar_datos.py

O si el paquete está instalado:
    python -c "from mintic_project.data_loader import procesar_siniestros; procesar_siniestros()"
"""
import sys
from pathlib import Path

# Agregar src al path para importar el paquete
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mintic_project.data_loader import procesar_siniestros


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DESCARGADOR Y LIMPIADOR DE DATOS - SINIESTROS VIALES PALMIRA")
    print("=" * 70 + "\n")

    # Descargar y procesar
    df1, df2 = procesar_siniestros(directorio_salida="data", limite_registros=50000)

    if not df1.empty and not df2.empty:
        print("\n✓ Proceso completado exitosamente")
        print(f"\nDimensions:")
        print(f"  Dataset 1: {df1.shape}")
        print(f"  Dataset 2: {df2.shape}")
        print("\nPrimeras filas Dataset 1:")
        print(df1.head())
        print("\nPrimeras filas Dataset 2:")
        print(df2.head())
    else:
        print("\n✗ Error durante el procesamiento")
        sys.exit(1)
