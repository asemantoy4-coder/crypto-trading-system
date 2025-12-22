# api/config.py
"""
Configuration Management for Crypto AI Trading System v7.3.0
سیستم مدیریت تنظیمات با پشتیبانی از Environment Variables
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

# بارگذاری .env
load_dotenv()

# ==============================================================================
# Version Information
# ==============================================================================
VERSION = "7.3.0"
APP_NAME = "Crypto AI Trading System"
API_TITLE = f"{APP_NAME} v{VERSION}"
API_DESCRIPTION = """
🚀 سیستم تحلیل و معاملات ارز دیجیتال با هوش مصنوعی

ویژگی‌ها:
- تحلیل چند تایم‌فریمی (Multi-timeframe Analysis)
- سیگنال‌های اسکالپ (Scalp Signals: 1m/5m/15m)
- سیگنال‌های سوئینگ (Swing Signals: 1h/4h/1d)
- محدودیت نرخ درخواست (Rate Limiting)
- اعتبارسنجی ورودی (Input Validation)
- پشتیبانی از Binance و LBank API
"""

# ==============================================================================
# Application Settings
# ==============================================================================
class AppConfig:
    """تنظیمات اصلی برنامه"""
    
    # Basic Info
    VERSION: str = VERSION
    APP_NAME: str = APP_NAME
    TITLE: str = API_TITLE
    DESCRIPTION: str = API_DESCRIPTION
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    API_DIR: Path = Path(__file__).parent
    
    # URLs
    DOCS_URL: str = "/api/docs" if DEBUG else None
    REDOC_URL: str = "/api/redoc" if DEBUG else None
    OPENAPI_URL: str = "/api/openapi.json" if DEBUG else None
    
    @classmethod
    def is_production(cls) -> bool:
        """آیا در حالت production است؟"""
        return cls.ENVIRONMENT == "production"
    
    @classmethod
    def is_development(cls) -> bool:
        """آیا در حالت development است؟"""
        return cls.ENVIRONMENT == "development"

# ==============================================================================
# API Settings
# ==============================================================================
class APIConfig:
    """تنظیمات API"""
    
    # CORS
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: List[str] = ["*"]
    ALLOW_HEADERS: List[str] = ["*"]
    
    # Rate Limiting
    ENABLE_RATE_LIMIT: bool = os.getenv("ENABLE_RATE_LIMIT", "true").lower() == "true"
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "20/minute")
    RATE_LIMIT_STORAGE: str = "memory"  # یا "redis"
    
    # Timeouts
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 30))
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", 10))
    
    # Workers (برای Gunicorn)
    WEB_CONCURRENCY: int = int(os.getenv("WEB_CONCURRENCY", 2))
    
    @classmethod
    def get_cors_config(cls) -> Dict[str, Any]:
        """دریافت تنظیمات CORS"""
        return {
            "allow_origins": cls.ALLOWED_ORIGINS,
            "allow_credentials": cls.ALLOW_CREDENTIALS,
            "allow_methods": cls.ALLOW_METHODS,
            "allow_headers": cls.ALLOW_HEADERS,
        }

# ==============================================================================
# Trading Settings
# ==============================================================================
class TradingConfig:
    """تنظیمات معاملاتی"""
    
    # Default Symbols
    DEFAULT_SYMBOLS: List[str] = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", 
        "XRPUSDT", "ADAUSDT", "DOGEUSDT"
    ]
    
    # Timeframes
    SCALP_TIMEFRAMES: List[str] = ["1m", "5m", "15m"]
    SWING_TIMEFRAMES: List[str] = ["1h", "4h", "1d"]
    ALL_TIMEFRAMES: List[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    
    DEFAULT_TIMEFRAME: str = "5m"
    DEFAULT_LIMIT: int = 50
    
    # Analysis
    MIN_CONFIDENCE: float = 0.60  # حداقل اطمینان برای سیگنال
    RSI_PERIOD: int = 14
    SMA_PERIOD: int = 20
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    # Risk Management
    DEFAULT_STOP_LOSS_PERCENT: float = 2.0  # 2% استاپ لاس
    DEFAULT_TAKE_PROFIT_PERCENT: float = 5.0  # 5% تارگت
    SCALP_STOP_LOSS_PERCENT: float = 1.0  # 1% برای اسکالپ
    SCALP_TAKE_PROFIT_PERCENT: float = 2.0  # 2% برای اسکالپ

# ==============================================================================
# Exchange API Settings
# ==============================================================================
class ExchangeConfig:
    """تنظیمات صرافی‌ها"""
    
    # Binance
    BINANCE_API_KEY: Optional[str] = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET: Optional[str] = os.getenv("BINANCE_API_SECRET")
    BINANCE_BASE_URL: str = "https://api.binance.com"
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    
    # LBank
    LBANK_API_KEY: Optional[str] = os.getenv("LBANK_API_KEY")
    LBANK_API_SECRET: Optional[str] = os.getenv("LBANK_API_SECRET")
    LBANK_BASE_URL: str = "https://api.lbkex.com"
    
    # Priorities
    PRIMARY_EXCHANGE: str = "binance"
    FALLBACK_EXCHANGE: str = "lbank"
    
    @classmethod
    def has_binance_keys(cls) -> bool:
        """آیا کلیدهای Binance موجود است؟"""
        return bool(cls.BINANCE_API_KEY and cls.BINANCE_API_SECRET)
    
    @classmethod
    def has_lbank_keys(cls) -> bool:
        """آیا کلیدهای LBank موجود است؟"""
        return bool(cls.LBANK_API_KEY and cls.LBANK_API_SECRET)

# ==============================================================================
# Database Settings (برای آینده)
# ==============================================================================
class DatabaseConfig:
    """تنظیمات پایگاه داده"""
    
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    
    # SQLAlchemy
    SQLALCHEMY_ECHO: bool = AppConfig.DEBUG
    SQLALCHEMY_POOL_SIZE: int = 5
    SQLALCHEMY_MAX_OVERFLOW: int = 10
    
    @classmethod
    def is_database_enabled(cls) -> bool:
        """آیا دیتابیس فعال است؟"""
        return cls.DATABASE_URL is not None
    
    @classmethod
    def is_redis_enabled(cls) -> bool:
        """آیا Redis فعال است؟"""
        return cls.REDIS_URL is not None

# ==============================================================================
# Logging Settings
# ==============================================================================
class LoggingConfig:
    """تنظیمات لاگینگ"""
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # File Logging
    LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "logs/app.log")
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT: int = 5
    
    @classmethod
    def get_log_config(cls) -> Dict[str, Any]:
        """دریافت تنظیمات logging"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": cls.LOG_FORMAT,
                    "datefmt": cls.LOG_DATE_FORMAT,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": cls.LOG_LEVEL,
                },
            },
            "root": {
                "level": cls.LOG_LEVEL,
                "handlers": ["console"],
            },
        }

# ==============================================================================
# Security Settings
# ==============================================================================
class SecurityConfig:
    """تنظیمات امنیتی"""
    
    # Secret Key
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    
    # JWT (برای آینده)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # API Key (برای محدود کردن دسترسی)
    API_KEY_ENABLED: bool = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
    API_KEY: Optional[str] = os.getenv("API_KEY")
    
    # HTTPS
    FORCE_HTTPS: bool = os.getenv("FORCE_HTTPS", "false").lower() == "true"

# ==============================================================================
# Monitoring Settings (برای آینده)
# ==============================================================================
class MonitoringConfig:
    """تنظیمات مانیتورینگ"""
    
    # Sentry
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    SENTRY_ENVIRONMENT: str = AppConfig.ENVIRONMENT
    
    # Prometheus
    PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true"
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", 9090))
    
    @classmethod
    def is_sentry_enabled(cls) -> bool:
        """آیا Sentry فعال است؟"""
        return cls.SENTRY_DSN is not None

# ==============================================================================
# Feature Flags
# ==============================================================================
class FeatureFlags:
    """پرچم‌های ویژگی (برای فعال/غیرفعال کردن ویژگی‌ها)"""
    
    ENABLE_WEB_SEARCH: bool = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
    ENABLE_SCALP_SIGNALS: bool = os.getenv("ENABLE_SCALP_SIGNALS", "true").lower() == "true"
    ENABLE_SWING_SIGNALS: bool = os.getenv("ENABLE_SWING_SIGNALS", "true").lower() == "true"
    ENABLE_HISTORICAL_DATA: bool = os.getenv("ENABLE_HISTORICAL_DATA", "false").lower() == "true"
    ENABLE_BACKTESTING: bool = os.getenv("ENABLE_BACKTESTING", "false").lower() == "true"
    ENABLE_REAL_TRADING: bool = os.getenv("ENABLE_REAL_TRADING", "false").lower() == "true"

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_version() -> str:
    """دریافت نسخه برنامه"""
    return VERSION

def get_all_config() -> Dict[str, Any]:
    """دریافت تمام تنظیمات"""
    return {
        "version": VERSION,
        "app_name": APP_NAME,
        "title": API_TITLE,
        "description": API_DESCRIPTION,
        "environment": AppConfig.ENVIRONMENT,
        "debug": AppConfig.DEBUG,
        "api": {
            "cors_enabled": True,
            "rate_limit_enabled": APIConfig.ENABLE_RATE_LIMIT,
            "rate_limit": APIConfig.RATE_LIMIT,
            "allowed_origins": APIConfig.ALLOWED_ORIGINS,
        },
        "trading": {
            "default_symbols": TradingConfig.DEFAULT_SYMBOLS,
            "scalp_timeframes": TradingConfig.SCALP_TIMEFRAMES,
            "swing_timeframes": TradingConfig.SWING_TIMEFRAMES,
            "min_confidence": TradingConfig.MIN_CONFIDENCE,
        },
        "exchanges": {
            "primary": ExchangeConfig.PRIMARY_EXCHANGE,
            "fallback": ExchangeConfig.FALLBACK_EXCHANGE,
            "binance_configured": ExchangeConfig.has_binance_keys(),
            "lbank_configured": ExchangeConfig.has_lbank_keys(),
        },
        "features": {
            "web_search": FeatureFlags.ENABLE_WEB_SEARCH,
            "scalp_signals": FeatureFlags.ENABLE_SCALP_SIGNALS,
            "swing_signals": FeatureFlags.ENABLE_SWING_SIGNALS,
            "historical_data": FeatureFlags.ENABLE_HISTORICAL_DATA,
            "backtesting": FeatureFlags.ENABLE_BACKTESTING,
        },
        "monitoring": {
            "sentry_enabled": MonitoringConfig.is_sentry_enabled(),
            "prometheus_enabled": MonitoringConfig.PROMETHEUS_ENABLED,
        }
    }

def get_config_by_environment() -> Dict[str, Any]:
    """دریافت تنظیمات بر اساس محیط"""
    env = AppConfig.ENVIRONMENT
    
    configs = {
        "development": {
            "debug": True,
            "log_level": "DEBUG",
            "docs_enabled": True,
            "rate_limit": "100/minute",
        },
        "production": {
            "debug": False,
            "log_level": "INFO",
            "docs_enabled": False,
            "rate_limit": "20/minute",
        },
        "testing": {
            "debug": True,
            "log_level": "DEBUG",
            "docs_enabled": True,
            "rate_limit": "1000/minute",
        }
    }
    
    return configs.get(env, configs["production"])

def validate_config() -> Dict[str, Any]:
    """اعتبارسنجی تنظیمات"""
    issues = []
    warnings = []
    
    # چک کردن کلیدهای الزامی
    if AppConfig.ENVIRONMENT == "production":
        if SecurityConfig.SECRET_KEY == "your-secret-key-change-in-production":
            issues.append("SECRET_KEY must be changed in production!")
        
        if "*" in APIConfig.ALLOWED_ORIGINS:
            warnings.append("CORS allows all origins in production. Consider restricting.")
    
    # چک کردن صرافی‌ها
    if not ExchangeConfig.has_binance_keys() and not ExchangeConfig.has_lbank_keys():
        warnings.append("No exchange API keys configured. Using mock data only.")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }

def print_config_summary():
    """چاپ خلاصه تنظیمات"""
    print("=" * 60)
    print(f"🚀 {APP_NAME} v{VERSION}")
    print("=" * 60)
    print(f"Environment: {AppConfig.ENVIRONMENT}")
    print(f"Debug Mode: {AppConfig.DEBUG}")
    print(f"Host: {AppConfig.HOST}:{AppConfig.PORT}")
    print(f"CORS Origins: {', '.join(APIConfig.ALLOWED_ORIGINS)}")
    print(f"Rate Limiting: {APIConfig.RATE_LIMIT if APIConfig.ENABLE_RATE_LIMIT else 'Disabled'}")
    print(f"Log Level: {LoggingConfig.LOG_LEVEL}")
    print(f"Primary Exchange: {ExchangeConfig.PRIMARY_EXCHANGE}")
    print(f"Binance Keys: {'✅' if ExchangeConfig.has_binance_keys() else '❌'}")
    print(f"LBank Keys: {'✅' if ExchangeConfig.has_lbank_keys() else '❌'}")
    print("=" * 60)
    
    # اعتبارسنجی
    validation = validate_config()
    if not validation["valid"]:
        print("⚠️ Configuration Issues:")
        for issue in validation["issues"]:
            print(f"  ❌ {issue}")
    
    if validation["warnings"]:
        print("⚠️ Configuration Warnings:")
        for warning in validation["warnings"]:
            print(f"  ⚠️ {warning}")
    
    print("=" * 60)

# ==============================================================================
# Export
# ==============================================================================
__all__ = [
    # Version
    "VERSION",
    "get_version",
    "get_all_config",
    
    # Config Classes
    "AppConfig",
    "APIConfig",
    "TradingConfig",
    "ExchangeConfig",
    "DatabaseConfig",
    "LoggingConfig",
    "SecurityConfig",
    "MonitoringConfig",
    "FeatureFlags",
    
    # Helper Functions
    "get_config_by_environment",
    "validate_config",
    "print_config_summary",
]

# ==============================================================================
# Main (برای تست)
# ==============================================================================
if __name__ == "__main__":
    print_config_summary()
    
    # چاپ تنظیمات کامل
    import json
    print("\n📋 Full Configuration:")
    print(json.dumps(get_all_config(), indent=2))