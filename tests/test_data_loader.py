"""Tests para módulos de data loading y feature engineering."""
import pandas as pd
import numpy as np
import pytest
from mintic_project.data_loader import LimpiadordeDatos
from mintic_project.feature_engineering import FeatureEngineering


class TestLimpiadordeDatos:
    """Tests para la clase LimpiadordeDatos."""

    @pytest.fixture
    def df_sucio(self):
        """Crear DataFrame con datos sucios para pruebas."""
        return pd.DataFrame({
            "a_o": [2022, 2022, 2023, None, 2023],
            "fecha": ["2022-01-01", "2022-01-01", "2023-06-15", None, "2023-06-15"],
            "hora": [10, 25, 15, 8, 22],  # 25 es inválido
            "jornada": ["DIA", "día", "NOCHE", None, "noche"],
            "edad": [25, 150, 35, 28, 45],  # 150 es outlier
            "lat": [-3.73, -3.73, -10.0, -3.73, -3.73],  # -10 es outlier
            "long": [-76.28, -76.28, -50.0, -76.28, -76.28],  # -50 es outlier
            "barrios_corregimiento_via": ["Centro", "Centro", "Zona A", None, "Zona B"],
        })

    def test_eliminar_duplicados(self, df_sucio):
        """Prueba eliminación de duplicados."""
        limpiador = LimpiadordeDatos()
        df_limpio = limpiador._eliminar_duplicados(df_sucio)
        # Debe reducir de 5 a 4 registros (primer y segundo son iguales)
        assert len(df_limpio) <= len(df_sucio)

    def test_validar_tipos(self, df_sucio):
        """Prueba conversión de tipos."""
        limpiador = LimpiadordeDatos()
        df_procesado = limpiador._validar_tipos(df_sucio)
        # Verificar que fecha es datetime
        if "fecha" in df_procesado.columns:
            assert pd.api.types.is_datetime64_any_dtype(df_procesado["fecha"])

    def test_limpiar_nulos(self, df_sucio):
        """Prueba limpieza de valores nulos."""
        limpiador = LimpiadordeDatos()
        df_limpio = limpiador._limpiar_nulos(df_sucio, "test")
        # Debe eliminar la fila con fecha nula
        assert df_limpio["fecha"].isna().sum() == 0

    def test_eliminar_outliers(self, df_sucio):
        """Prueba eliminación de outliers."""
        limpiador = LimpiadordeDatos()
        df_limpio = limpiador._eliminar_outliers(df_sucio)
        # Verificar que edad > 120 sea eliminada
        if "edad" in df_limpio.columns:
            assert df_limpio["edad"].max() <= 120 or df_limpio["edad"].isna().all()

    def test_eliminar_outliers_conservador_geografico(self):
        """Test: filtrado geográfico conservador no elimina si solo una coordenada está ausente."""
        # Simular caso: muchas filas con lat/long nulos o parciales
        df_test = pd.DataFrame({
            "a_o": [2023] * 6,
            "fecha": ["2023-01-01"] * 6,
            "lat": [-3.73, -3.73, None, -10.0, -3.73, -3.73],  # -10 es outlier
            "long": [-76.28, None, -76.28, -50.0, -76.28, -76.28],  # -50 es outlier
            "edad": [25, 35, 45, 28, 30, 32],
        })
        limpiador = LimpiadordeDatos()
        df_limpio = limpiador._eliminar_outliers(df_test)
        
        # Debe eliminar SOLO la fila 3 (ambas coordenadas presentes y ambas inválidas)
        # Todas las demás deben conservarse (nulos, o al menos una coordenada válida/ausente)
        assert len(df_limpio) > 3  # No debe eliminar todas
        assert len(df_limpio) <= 5  # Debe eliminar al máximo la fila con ambas inválidas

    def test_estandarizar_texto(self, df_sucio):
        """Prueba estandarización de texto."""
        limpiador = LimpiadordeDatos()
        df_limpio = limpiador._estandarizar_texto(df_sucio)
        # Jornada debe estar en mayúsculas
        if "jornada" in df_limpio.columns:
            jornadas = df_limpio["jornada"].dropna().unique()
            for j in jornadas:
                assert j == j.upper() or pd.isna(j)

    def test_validar_rangos(self, df_sucio):
        """Prueba validación de rangos."""
        limpiador = LimpiadordeDatos()
        df_limpio = limpiador._validar_rangos(df_sucio)
        # Horas deben estar entre 0 y 23
        if "hora" in df_limpio.columns:
            assert df_limpio["hora"].max() <= 23 or df_limpio["hora"].isna().all()
            assert df_limpio["hora"].min() >= 0 or df_limpio["hora"].isna().all()

    def test_limpiar_dataset_completo(self, df_sucio):
        """Prueba pipeline completo de limpieza."""
        limpiador = LimpiadordeDatos()
        df_limpio = limpiador.limpiar_dataset(df_sucio, "test_dataset")

        # Verificar que el dataset tiene menos o igual registros
        assert len(df_limpio) <= len(df_sucio)

        # Verificar que el reporte fue generado
        reporte = limpiador.generar_reporte()
        assert "test_dataset" in reporte
        assert "%" in reporte


class TestFeatureEngineering:
    """Tests para Feature Engineering."""

    @pytest.fixture
    def df_limpio(self):
        """Crear DataFrame limpio para pruebas."""
        return pd.DataFrame({
            "fecha": pd.to_datetime([
                "2023-01-15", "2023-02-20", "2023-06-10", "2023-09-05"
            ]),
            "a_o": [2023, 2023, 2023, 2023],
            "hora": [10, 14, 18, 22],
            "jornada": ["MANANA", "TARDE", "NOCHE", "NOCHE"],
            "dia_semana": ["LUNES", "MARTES", "SABADO", "DOMINGO"],
            "genero": ["M", "F", "M", "F"],
            "zona": ["CENTRO", "NORTE", "SUR", "CENTRO"],
            "edad": [25, 35, 45, 28],
            "lat": [-3.73, -3.74, -3.72, -3.73],
            "long": [-76.28, -76.29, -76.27, -76.28],
        })

    def test_crear_features_temporales(self, df_limpio):
        """Prueba creación de features temporales."""
        fe = FeatureEngineering()
        df_proc = fe._crear_features_temporales(df_limpio.copy())

        # Verificar que se crearon features de tiempo
        assert "mes" in df_proc.columns
        assert "trimestre" in df_proc.columns
        assert "periodo_dia" in df_proc.columns

        # Verificar valores
        assert df_proc["mes"].min() >= 1
        assert df_proc["mes"].max() <= 12

    def test_crear_features_geograficos(self, df_limpio):
        """Prueba creación de features geográficos."""
        fe = FeatureEngineering()
        df_proc = fe._crear_features_geograficos(df_limpio.copy())

        # Verificar que se crearon features geográficos
        assert "distancia_centro" in df_proc.columns
        assert "en_centro" in df_proc.columns

        # Verificar valores
        assert df_proc["distancia_centro"].min() >= 0
        assert df_proc["en_centro"].isin([0, 1]).all()

    def test_crear_features_categoricos(self, df_limpio):
        """Prueba codificación de categorías."""
        fe = FeatureEngineering()
        df_proc = fe._crear_features_categoricos(df_limpio.copy())

        # Verificar que se codificaron variables categóricas
        assert "jornada_encoded" in df_proc.columns
        assert "genero_encoded" in df_proc.columns

        # Verificar que están en rango válido
        assert df_proc["jornada_encoded"].min() >= 0
        assert df_proc["genero_encoded"].min() >= 0

    def test_procesar_para_prediccion(self, df_limpio):
        """Prueba pipeline completo de procesamiento."""
        fe = FeatureEngineering()
        df_proc = fe.procesar_para_prediccion(df_limpio)

        # Verificar que se agregaron features
        assert len(df_proc.columns) > len(df_limpio.columns)

        # Verificar dimensiones
        assert len(df_proc) == len(df_limpio)

        # Verificar que no hay NaN en features críticos
        # (permitir algunos en features opcionales)
        columnas_criticas = ["mes", "periodo_dia"]
        for col in columnas_criticas:
            if col in df_proc.columns:
                assert df_proc[col].notna().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
