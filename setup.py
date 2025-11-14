#!/usr/bin/env python
import sys
from setuptools import setup, find_packages
from pathlib import Path
import os


# legacy Install
if len(sys.argv) > 2 and sys.argv[2] in ["develop", "install"]:
    os.environ["UQPY_VERSION"] = sys.argv[1]
    del sys.argv[1]


if "UQPY_VERSION" in os.environ:
    version = os.environ["UQPY_VERSION"]
else:
    version = "4.2.1"

this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

setup(
    name="UQpy",
    version=version,
    url="https://github.com/SURGroup/UQpy",
    description="UQpy is a general purpose toolbox for Uncertainty Quantification",
    long_description=long_description,
    author="Michael D. Shields, Dimitris G. Giovanis, Audrey Olivier, Aakash Bangalore-Satish, Mohit Chauhan, "
    "Lohit Vandanapu, Ketson R.M. dos Santos",
    license="MIT",
    license_files=("LICENSE",),
    platforms=["OSX", "Windows", "Linux"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.pdf"]},
    python_requires=">3.9.0",
    install_requires=[
        "numpy>=2.0.0",
        "scipy>=1.13.0",
        "matplotlib>=3.9.0",
        "scikit-learn>=1.5.0",
        "fire>=0.6.0",
        "beartype>=0.18.5",
        "torch >= 2.2.2",
        "torchinfo >= 1.8.0",
    ],
    extras_require={
        "dev": [
            "pytest == 8.2.0",
            "pytest-cov == 5.0.0",
            "pylint == 3.1.0",
            "pytest-azurepipelines == 1.0.5",
            "pytest-cov == 5.0.0",
            "wheel == 0.43.0",
            "twine == 5.0.0",
            "sphinx_autodoc_typehints == 1.23.0",
            "sphinx_rtd_theme == 1.2.0",
            "sphinx_gallery == 0.13.0",
            "sphinxcontrib_bibtex == 2.5.0",
            "Sphinx==6.1.3",
            "hypothesis>=6.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Natural Language :: English",
    ],
)
