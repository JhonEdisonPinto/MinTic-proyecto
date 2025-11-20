"""Módulo con funciones de procesamiento de datos.

Ejemplos mínimos para que el equipo tenga un punto de partida.
"""
from typing import Iterable
import pandas as pd


def read_csv_url(url: str, **kwargs) -> pd.DataFrame:
    """Leer CSV desde una URL pública y devolver un DataFrame.

    Args:
        url: URL del CSV.
        **kwargs: argumentos adicionales para pandas.read_csv.

    Returns:
        pd.DataFrame
    """
    return pd.read_csv(url, **kwargs)


def sample_counts(df: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    """Retorna las `n` categorías más frecuentes en `column`.

    Args:
        df: DataFrame de entrada.
        column: columna categórica a agrupar.
        n: número de filas a devolver.

    Returns:
        pd.DataFrame con conteos y porcentajes.
    """
    counts = df[column].value_counts(dropna=False).head(n).rename_axis(column).reset_index(name="count")
    counts["pct"] = counts["count"] / counts["count"].sum()
    return counts
