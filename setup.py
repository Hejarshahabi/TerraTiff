"""Minimal setup.py for the TerraTiff package."""

import os
from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="terratiff",
    version="0.1.5",
    description="A lightweight, GDAL-free Python package for reading, writing, "
                "and exporting GeoTIFF raster files.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hejar Shahabi",
    license="Custom Non-Commercial",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "tifffile",
        "pyproj",
    ],
    extras_require={
        "dev": ["pytest"],
    },
)
