"""
Setup script for the raptor package.

Installation:
    pip install -e .               # Development (editable)
    pip install .                   # Standard install
"""
from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
long_description = ""
if os.path.exists(os.path.join(here, "README.md")):
    with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="raptor",
    version="0.0.1",
    author="Victor Alulema",
    author_email="victor.alulema@epn.edu.ec",
    description=(
        "Energy-optimal, regulation-aware eVTOL path planning "
        "for medical delivery in complex terrain (Quito, Ecuador)."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LUAS-EPN/uav-path-planning",
    packages=find_packages(),
    package_data={"raptor": ["../data/*.npz", "../data/*.json"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.5",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black>=22.0", "flake8>=5.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "uav", "evtol", "path-planning", "optimization",
        "dem", "terrain", "airspace", "medical-delivery", "quito",
        "differential-evolution", "energy-model", "RDAC-101",
    ],
    entry_points={
        "console_scripts": [
            "uav-experiments=scripts.run_experiments:main",
            "uav-scenarios=scripts.run_scenario_catalog:main",
        ],
    },
)
