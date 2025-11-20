"""Pruebas unitarias básicas para `processor.py`.

Estas pruebas sirven como plantilla; el dataset real puede necesitar
mocking o fixtures más complejos.
"""
from mintic_project.processor import sample_counts
import pandas as pd


def test_sample_counts_basic():
    df = pd.DataFrame({"categoria": ["A", "B", "A", "C", None]})
    res = sample_counts(df, "categoria", n=3)
    assert "count" in res.columns
    assert res.iloc[0]["count"] == 2
