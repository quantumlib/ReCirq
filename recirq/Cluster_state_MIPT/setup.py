"""Setup file for the measurement_entanglement module."""

from setuptools import setup, find_packages

setup(
    name="recirq-measurement-entanglement",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "cirq",
        "cirq-google",
        "torch",
        "numpy",
    ],
    python_requires=">=3.7",
) 