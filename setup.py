"""
Setup configuration for the denograd package.

This script uses setuptools to configure the package for distribution.
It reads the long description from README.md and defines package metadata,
dependencies, and classifiers.

Attributes:
    name (str): The name of the package ("denograd").
    version (str): The current version of the package ("0.1.2").
    author (str): The author of the package ("JJavier98").
    description (str): A brief summary of the package's purpose (Noise reduction framework).
    long_description (str): Detailed description read from README.md.
    long_description_content_type (str): Format of the long description ("text/markdown").
    py_modules (list): List of Python modules included in the package.
    classifiers (list): PyPI classifiers indicating language, license, and OS support.
    python_requires (str): Minimum Python version required (>=3.6).
    install_requires (list): List of external dependencies required to run the package.
"""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="denograd",
    version="1.0.0b1",
    author="JJavier98",
    description="Instance noise reduction framework based on Deep Learning gradients agnostic to \
        the network architecture.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["denograd"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "matplotlib",
        "torch",
        "ipython",
        "tqdm",
    ],
)
