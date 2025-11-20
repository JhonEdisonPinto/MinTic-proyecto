"""setup.py para instalar el paquete del proyecto.

Comentarios en español: Este archivo permite instalar el paquete
con `pip install -e .` para desarrollo local.
"""
from setuptools import setup, find_packages

setup(
    name="mintic_project",
    version="0.1.0",
    description="Proyecto MinTIC - Analítica de datos abiertos de Colombia",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
    ],
    include_package_data=True,
    python_requires=">=3.11",
)
