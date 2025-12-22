# setup.py
"""
Setup configuration for Crypto AI Trading System
نسخه 7.3.0 - کامل و بهینه‌شده
"""

from setuptools import setup, find_packages
import os

# خواندن README برای long_description
def read_file(filename):
    """خواندن محتوای فایل"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ''

# خواندن requirements از فایل
def read_requirements(filename='requirements.txt'):
    """خواندن requirements از فایل"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f 
                    if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        # اگر فایل requirements.txt نبود، لیست دستی
        return [
            'fastapi==0.104.1',
            'uvicorn[standard]==0.24.0',
            'gunicorn==21.2.0',
            'pydantic==2.5.0',
            'pydantic-core==2.14.1',
            'requests==2.31.0',
            'httpx==0.25.1',
            'python-dotenv==1.0.0',
            'python-multipart==0.0.6',
            'slowapi==0.1.9',
            'pandas==2.1.3',
            'numpy==1.26.2',
            'psutil==5.9.6',
            'aiohttp==3.9.1'
        ]

setup(
    # ==============================================================================
    # اطلاعات پایه
    # ==============================================================================
    name="crypto-trading-system",
    version="7.3.0",
    author="Crypto AI Trading System",
    author_email="support@cryptotrading.example.com",  # ایمیل خود را بگذارید
    description="سیستم تحلیل معاملاتی ارز دیجیتال با پشتیبانی کامل از اسکالپ و سوئینگ",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    
    # ==============================================================================
    # URLs
    # ==============================================================================
    url="https://github.com/YOUR_USERNAME/crypto-trading-system",  # لینک GitHub خود
    project_urls={
        "Bug Tracker": "https://github.com/YOUR_USERNAME/crypto-trading-system/issues",
        "Documentation": "https://github.com/YOUR_USERNAME/crypto-trading-system/wiki",
        "Source Code": "https://github.com/YOUR_USERNAME/crypto-trading-system",
    },
    
    # ==============================================================================
    # Packages
    # ==============================================================================
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'examples']),
    include_package_data=True,
    
    # ==============================================================================
    # Dependencies
    # ==============================================================================
    install_requires=read_requirements(),
    
    # ==============================================================================
    # Extra Dependencies (اختیاری)
    # ==============================================================================
    extras_require={
        'dev': [
            'pytest>=7.4.3',
            'pytest-asyncio>=0.21.1',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ],
        'test': [
            'pytest>=7.4.3',
            'pytest-asyncio>=0.21.1',
            'httpx>=0.25.1',
        ],
        'docs': [
            'mkdocs>=1.5.0',
            'mkdocs-material>=9.0.0',
        ],
        'analysis': [
            'ta>=0.11.0',
            # 'ta-lib>=0.4.28',  # نیاز به نصب سیستمی
        ],
        'database': [
            'sqlalchemy>=2.0.23',
            'asyncpg>=0.29.0',
            'redis>=5.0.1',
        ],
        'monitoring': [
            'prometheus-client>=0.19.0',
            'sentry-sdk>=1.38.0',
        ]
    },
    
    # ==============================================================================
    # Entry Points (برای اجرای مستقیم)
    # ==============================================================================
    entry_points={
        'console_scripts': [
            'crypto-trading=api.main:main',  # اگر تابع main دارید
            'crypto-server=api.main:run_server',  # برای اجرای سرور
        ],
    },
    
    # ==============================================================================
    # Python Version
    # ==============================================================================
    python_requires='>=3.8',  # حداقل Python 3.8
    
    # ==============================================================================
    # Classifiers (برای PyPI)
    # ==============================================================================
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Framework :: FastAPI",
        "Natural Language :: Persian",
        "Natural Language :: English",
    ],
    
    # ==============================================================================
    # Keywords (برای جستجو در PyPI)
    # ==============================================================================
    keywords=[
        'crypto', 'trading', 'cryptocurrency', 'bitcoin', 'ethereum',
        'technical-analysis', 'scalping', 'swing-trading', 'api',
        'fastapi', 'binance', 'trading-bot', 'crypto-signals',
        'rsi', 'sma', 'macd', 'trading-strategy'
    ],
    
    # ==============================================================================
    # License
    # ==============================================================================
    license="MIT",
    
    # ==============================================================================
    # Additional Files
    # ==============================================================================
    package_data={
        'api': ['*.py'],
        '': ['README.md', 'LICENSE', '.env.example'],
    },
    
    # ==============================================================================
    # Zip Safe
    # ==============================================================================
    zip_safe=False,
)

# ==============================================================================
# توابع کمکی برای Entry Points
# ==============================================================================

def main():
    """تابع اصلی برای اجرای CLI"""
    import sys
    print("🚀 Crypto AI Trading System v7.3.0")
    print("Use: crypto-server to start the server")
    sys.exit(0)

def run_server():
    """اجرای سرور FastAPI"""
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Crypto Trading API on {host}:{port}")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )

if __name__ == '__main__':
    main()