"""
Crypto Trading System API Package
نسخه 7.7.0 با پشتیبانی کامل از سیگنال‌های اسکالپ و Render deployment
"""

import logging
import os
import sys

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Import با مدیریت خطا
# ==============================================================================

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import اصلی: FastAPI app
try:
    from .main import app, API_VERSION
    logger.info(f"✅ Main app imported successfully - v{API_VERSION}")
except ImportError as e:
    logger.error(f"❌ Failed to import main app: {e}")
    try:
        # Fallback: اگر در حالت standalone اجرا می‌شود
        from main import app, API_VERSION
        logger.info(f"✅ Main app imported (standalone mode) - v{API_VERSION}")
    except ImportError:
        logger.error("❌ Could not import app at all")
        app = None
        API_VERSION = "Unknown"

# Import توابع کمکی (اختیاری)
UTILS_AVAILABLE = False
try:
    from .utils import (
        get_market_data_with_fallback,
        analyze_with_multi_timeframe_strategy,
        calculate_24h_change_from_dataframe,
        calculate_simple_sma,
        calculate_simple_rsi,
        calculate_rsi_series,
        detect_divergence,
        calculate_macd_simple,
        calculate_ichimoku_components,
        analyze_ichimoku_scalp_signal,
        get_ichimoku_scalp_signal,
        calculate_smart_entry,
        get_swing_high_low,
        get_support_resistance_levels,
        calculate_volatility,
        combined_analysis,
        generate_ichimoku_recommendation,
        get_fallback_signal,
        __version__ as utils_version
    )
    UTILS_AVAILABLE = True
    logger.info(f"✅ Utils functions imported - v{utils_version}")
except ImportError as e:
    logger.warning(f"⚠️ Utils not available: {e}")
    # توابع به صورت None می‌مانند
    get_market_data_with_fallback = None
    analyze_with_multi_timeframe_strategy = None
    calculate_24h_change_from_dataframe = None
    calculate_simple_sma = None
    calculate_simple_rsi = None
    calculate_rsi_series = None
    detect_divergence = None
    calculate_macd_simple = None
    calculate_ichimoku_components = None
    analyze_ichimoku_scalp_signal = None
    get_ichimoku_scalp_signal = None
    calculate_smart_entry = None
    get_swing_high_low = None
    get_support_resistance_levels = None
    calculate_volatility = None
    combined_analysis = None
    generate_ichimoku_recommendation = None
    get_fallback_signal = None
    utils_version = "Not Available"

# Import توابع اضافی (اگر وجود دارند)
EXTRA_FUNCTIONS = False
try:
    from .utils import (
        analyze_scalp_conditions,
        calculate_quality_line,
        calculate_golden_line
    )
    EXTRA_FUNCTIONS = True
    logger.info("✅ Extra functions imported")
except ImportError:
    logger.debug("⚠️ Extra functions not available (optional)")
    analyze_scalp_conditions = None
    calculate_quality_line = None
    calculate_golden_line = None

# ==============================================================================
# Metadata
# ==============================================================================
__version__ = API_VERSION if 'API_VERSION' in locals() else "7.7.0"
__author__ = "Crypto AI Trading System"
__description__ = "سیستم تحلیل معاملاتی ارز دیجیتال با پشتیبانی از اسکالپ و Render deployment"

# ==============================================================================
# __all__ - چه چیزهایی export می‌شوند
# ==============================================================================
__all__ = ['app', 'API_VERSION']

# اضافه کردن توابع موجود به __all__
if UTILS_AVAILABLE:
    __all__.extend([
        'get_market_data_with_fallback',
        'analyze_with_multi_timeframe_strategy',
        'calculate_24h_change_from_dataframe',
        'calculate_simple_sma',
        'calculate_simple_rsi',
        'calculate_rsi_series',
        'detect_divergence',
        'calculate_macd_simple',
        'calculate_ichimoku_components',
        'analyze_ichimoku_scalp_signal',
        'get_ichimoku_scalp_signal',
        'calculate_smart_entry',
        'get_swing_high_low',
        'get_support_resistance_levels',
        'calculate_volatility',
        'combined_analysis',
        'generate_ichimoku_recommendation',
        'get_fallback_signal'
    ])

if EXTRA_FUNCTIONS:
    __all__.extend([
        'analyze_scalp_conditions',
        'calculate_quality_line',
        'calculate_golden_line'
    ])

# ==============================================================================
# Helper function برای تست سریع
# ==============================================================================
def test_imports():
    """Test if all imports are working correctly."""
    print("=" * 60)
    print(f"🧪 Testing imports for Crypto Trading System v{__version__}")
    print("=" * 60)
    
    tests = {
        "FastAPI App": app is not None,
        "Utils Module": UTILS_AVAILABLE,
        "Extra Functions": EXTRA_FUNCTIONS
    }
    
    for test_name, result in tests.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
    
    if UTILS_AVAILABLE:
        print(f"\n📊 Utils Version: {utils_version}")
    
    print(f"🔧 Python Path: {sys.path[:2]}...")
    print("=" * 60)
    
    return all(tests.values())

# ==============================================================================
# Startup Message (فقط در حالت development)
# ==============================================================================
if os.getenv("DEBUG", "false").lower() == "true" or os.getenv("RENDER", "false").lower() == "false":
    print("=" * 60)
    print(f"🚀 Crypto Trading System API v{__version__}")
    print("=" * 60)
    print(f"📊 Features: Scalp signals, Multi-timeframe analysis")
    print(f"🔧 Utils Available: {UTILS_AVAILABLE}")
    print(f"🔧 Extra Functions: {EXTRA_FUNCTIONS}")
    print(f"🌍 Environment: {'Render' if os.getenv('RENDER') else 'Local'}")
    print("=" * 60)
    
    # Auto-run import test in debug mode
    if os.getenv("DEBUG", "false").lower() == "true":
        test_imports()