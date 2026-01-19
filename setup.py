from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="denograd",
    version="0.1.1",
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
