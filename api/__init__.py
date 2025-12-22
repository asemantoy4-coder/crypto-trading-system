# api/__init__.py
"""
Crypto Trading System API Package
نسخه 7.3.0 با پشتیبانی کامل از سیگنال‌های اسکالپ
"""

import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# Import با مدیریت خطا
# ==============================================================================

# Import اصلی: FastAPI app
try:
    from .main import app
    logger.info("✅ Main app imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import main app: {e}")
    try:
        # Fallback: اگر در حالت standalone اجرا می‌شود
        from main import app
        logger.info("✅ Main app imported (standalone mode)")
    except ImportError:
        logger.error("❌ Could not import app at all")
        app = None

# Import توابع کمکی (اختیاری)
UTILS_AVAILABLE = False
try:
    from .utils import (
        get_market_data_with_fallback,
        analyze_with_multi_timeframe_strategy,
        calculate_24h_change_from_dataframe,
        calculate_simple_sma,
        calculate_simple_rsi
    )
    UTILS_AVAILABLE = True
    logger.info("✅ Utils functions imported")
except ImportError as e:
    logger.warning(f"⚠️ Utils not available: {e}")
    # توابع به صورت None می‌مانند
    get_market_data_with_fallback = None
    analyze_with_multi_timeframe_strategy = None
    calculate_24h_change_from_dataframe = None
    calculate_simple_sma = None
    calculate_simple_rsi = None

# Import توابع اضافی (اگر وجود دارند)
EXTRA_FUNCTIONS = False
try:
    from .utils import (
        calculate_macd_simple,
        analyze_trend_simple,
        analyze_scalp_conditions
    )
    EXTRA_FUNCTIONS = True
    logger.info("✅ Extra functions imported")
except ImportError:
    logger.debug("⚠️ Extra functions not available (optional)")
    calculate_macd_simple = None
    analyze_trend_simple = None
    analyze_scalp_conditions = None

# ==============================================================================
# Metadata
# ==============================================================================
__version__ = "7.3.0"
__author__ = "Crypto AI Trading System"
__description__ = "سیستم تحلیل معاملاتی ارز دیجیتال با پشتیبانی از اسکالپ"

# ==============================================================================
# __all__ - چه چیزهایی export می‌شوند
# ==============================================================================
__all__ = ['app']

# اضافه کردن توابع موجود به __all__
if UTILS_AVAILABLE:
    __all__.extend([
        'get_market_data_with_fallback',
        'analyze_with_multi_timeframe_strategy',
        'calculate_24h_change_from_dataframe',
        'calculate_simple_sma',
        'calculate_simple_rsi'
    ])

if EXTRA_FUNCTIONS:
    __all__.extend([
        'calculate_macd_simple',
        'analyze_trend_simple',
        'analyze_scalp_conditions'
    ])

# ==============================================================================
# Startup Message (فقط در حالت development)
# ==============================================================================
import os
if os.getenv("DEBUG", "false").lower() == "true":
    print("=" * 60)
    print(f"✅ Crypto Trading System API v{__version__} loaded")
    print(f"📊 Features: Scalp signals, Multi-timeframe analysis")
    print(f"🔧 Utils Available: {UTILS_AVAILABLE}")
    print(f"🔧 Extra Functions: {EXTRA_FUNCTIONS}")
    print("=" * 60)