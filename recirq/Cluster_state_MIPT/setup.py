"""Setup file for the Cluster_state_MIPT module."""

from setuptools import setup, find_namespace_packages

setup(
    name="recirq-cluster-state",
    version="0.1.0",
    packages=find_namespace_packages(include=["recirq.*"]),
    install_requires=[
        "cirq>=0.13.0",
        "cirq-google>=0.13.0",
        "numpy>=1.19.0",
    ],
    python_requires=">=3.7",
) 