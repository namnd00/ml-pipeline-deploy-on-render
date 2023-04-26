# setup.py
from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="starter",
    version="0.0.0",
    description="Starter code.",
    author="Student",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
)
