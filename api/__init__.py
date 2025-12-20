# api/__init__.py
"""
Crypto Trading System API Package
نسخه 7.1.0 با پشتیبانی از سیگنال‌های اسکالپ
"""

# Import توابع مهم از ماژول‌ها
from .utils import (
    get_market_data_with_fallback,
    analyze_with_multi_timeframe_strategy,
    calculate_24h_change_from_dataframe,
    calculate_simple_sma,
    calculate_simple_rsi,
    calculate_macd_simple,
    analyze_trend_simple,
    analyze_scalp_conditions
)

from .main import app

__version__ = "7.1.0"
__author__ = "Crypto AI Trading System"
__description__ = "سیستم تحلیل معاملاتی ارز دیجیتال با پشتیبانی از اسکالپ"

__all__ = [
    # FastAPI app
    'app',
    
    # توابع اصلی
    'get_market_data_with_fallback',
    'analyze_with_multi_timeframe_strategy',
    'calculate_24h_change_from_dataframe',
    
    # توابع تحلیل تکنیکال
    'calculate_simple_sma',
    'calculate_simple_rsi',
    'calculate_macd_simple',
    'analyze_trend_simple',
    'analyze_scalp_conditions'
]

print(f"✅ Crypto Trading System API v{__version__} loaded")
print(f"📊 Features: Scalp signals, Multi-timeframe analysis, Real-time prices")