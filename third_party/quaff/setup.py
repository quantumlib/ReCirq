import pathlib

from setuptools import find_packages, setup


def _parse_requirements(path: pathlib.Path):
    lines = [line.strip() for line in path.read_text().splitlines() if line]
    return [line for line in lines if not line.startswith("#")]


install_requires = _parse_requirements(pathlib.Path("requirements.txt"))
setup(
    name="quaff",
    version="0.0.0",
    author="Bryan A. O'Gorman",
    install_requires=install_requires,
    license="Apache 2",
    long_description=open("README.md", encoding="utf-8").read(),
    packages=find_packages(),
)
