# setup.py
from setuptools import setup, find_packages

setup(
    name="crypto-trading-api",
    version="7.7.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-multipart>=0.0.6",
    ],
    python_requires=">=3.8",
)