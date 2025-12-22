# setup.py
from setuptools import setup, find_packages

setup(
    name="crypto-trading-system",
    version="7.1.0",
    author="Crypto AI Trading System",
    description="سیستم تحلیل معاملاتی ارز دیجیتال با پشتیبانی از اسکالپ",
    packages=find_packages(),
    install_requires=[
        'fastapi==0.104.1',
        'uvicorn[standard]==0.24.0',
        'requests==2.31.0',
        'pydantic==2.5.0'
    ],
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)