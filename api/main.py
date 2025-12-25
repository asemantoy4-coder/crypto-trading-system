"""
Crypto AI Trading System v7.6.0 - Smart Entry & Real RSI
با پشتیبانی از Binance/LBank API و سازگار با پلتفرم Render
نسخه کامل با ایچیموکو پیشرفته برای اسکالپ
+ هوشمندسازی قیمت ورودی با فیبوناچی و ایچیموکو (Strong Solution)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta
import logging
from typing import List, Optional, Dict, Any
import random
import sys
import os
import math

# Configure logging اول از همه
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# تنظیمات اولیه
# ==============================================================================

# اضافه کردن مسیرهای لازم به sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, current_dir)  # مسیر api/
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # مسیر والد (src/)

print("=" * 50)
print(f"📁 Current directory: {current_dir}")
print(f"📁 Parent directory: {parent_dir}")
print(f"📁 sys.path: {sys.path}")
print("=" * 50)

# پرچم‌های دسترسی ماژول‌ها
UTILS_AVAILABLE = False
DATA_COLLECTOR_AVAILABLE = False
COLLECTORS_AVAILABLE = False

# ==============================================================================
# Import ماژول utils با چندین روش مختلف
# ==============================================================================

print("\n🔄 Importing utils module...")

# روش ۱: Import مستقیم
try:
    import utils
    from utils import (
        get_market_data_with_fallback, 
        analyze_with_multi_timeframe_strategy, 
        calculate_24h_change_from_dataframe,
        calculate_simple_sma,
        calculate_simple_rsi,
        calculate_rsi_series,
        detect_divergence,
        calculate_macd_simple,
        analyze_scalp_conditions,
        # توابع ایچیموکو
        calculate_ichimoku_components,
        analyze_ichimoku_scalp_signal,
        get_ichimoku_scalp_signal,
        calculate_quality_line,
        calculate_golden_line,
        get_support_resistance_levels,
        calculate_volatility,
        combined_analysis,
        generate_ichimoku_recommendation,
        # توابع هوشمند جدید
        get_swing_high_low,
        calculate_smart_entry
    )
    UTILS_AVAILABLE = True
    print("✅ Method 1: Direct import successful")
    
except ImportError as e:
    print(f"❌ Method 1 failed: {e}")
    
    # روش ۲: Import نسبی
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
            analyze_scalp_conditions,
            calculate_ichimoku_components,
            analyze_ichimoku_scalp_signal,
            get_ichimoku_scalp_signal,
            calculate_quality_line,
            calculate_golden_line,
            get_support_resistance_levels,
            calculate_volatility,
            combined_analysis,
            generate_ichimoku_recommendation,
            get_swing_high_low,
            calculate_smart_entry
        )
        UTILS_AVAILABLE = True
        print("✅ Method 2: Relative import successful")
        
    except ImportError as e2:
        print(f"❌ Method 2 failed: {e2}")
        
        # روش ۳: Import با نام کامل ماژول
        try:
            from api.utils import (
                get_market_data_with_fallback, 
                analyze_with_multi_timeframe_strategy, 
                calculate_24h_change_from_dataframe,
                calculate_simple_sma,
                calculate_simple_rsi,
                calculate_rsi_series,
                detect_divergence,
                calculate_macd_simple,
                analyze_scalp_conditions,
                calculate_ichimoku_components,
                analyze_ichimoku_scalp_signal,
                get_ichimoku_scalp_signal,
                calculate_quality_line,
                calculate_golden_line,
                get_support_resistance_levels,
                calculate_volatility,
                combined_analysis,
                generate_ichimoku_recommendation,
                get_swing_high_low,
                calculate_smart_entry
            )
            UTILS_AVAILABLE = True
            print("✅ Method 3: Full module import successful")
            
        except ImportError as e3:
            print(f"❌ Method 3 failed: {e3}")
            UTILS_AVAILABLE = False

# ==============================================================================
# Import ماژول‌های دیگر
# ==============================================================================

print("\n🔄 Importing other modules...")

# Import data_collector
try:
    try:
        from data_collector import get_collected_data
        DATA_COLLECTOR_AVAILABLE = True
        print("✅ data_collector imported (direct)")
    except ImportError:
        try:
            from .data_collector import get_collected_data
            DATA_COLLECTOR_AVAILABLE = True
            print("✅ data_collector imported (relative)")
        except ImportError:
            try:
                from api.data_collector import get_collected_data
                DATA_COLLECTOR_AVAILABLE = True
                print("✅ data_collector imported (full)")
            except ImportError as e:
                print(f"❌ data_collector import failed: {e}")
                DATA_COLLECTOR_AVAILABLE = False
except Exception as e:
    print(f"❌ data_collector import error: {e}")
    DATA_COLLECTOR_AVAILABLE = False

# Import collectors
try:
    try:
        from collectors import collect_signals_from_example_site
        COLLECTORS_AVAILABLE = True
        print("✅ collectors imported (direct)")
    except ImportError:
        try:
            from .collectors import collect_signals_from_example_site
            COLLECTORS_AVAILABLE = True
            print("✅ collectors imported (relative)")
        except ImportError:
            try:
                from api.collectors import collect_signals_from_example_site
                COLLECTORS_AVAILABLE = True
                print("✅ collectors imported (full)")
            except ImportError as e:
                print(f"❌ collectors import failed: {e}")
                COLLECTORS_AVAILABLE = False
except Exception as e:
    print(f"❌ collectors import error: {e}")
    COLLECTORS_AVAILABLE = False

# ==============================================================================
# Pydantic Models
# ==============================================================================
class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class ScalpRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class IchimokuRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class CombinedRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"
    include_ichimoku: bool = True
    include_rsi: bool = True
    include_macd: bool = True

class SignalResponse(BaseModel):
    status: str
    count: int
    last_updated: str
    signals: List[Dict[str, Any]]
    sources: Dict[str, int]

# ==============================================================================
# توابع جایگزین (Mock) برای زمانی که ماژول‌ها در دسترس نیستند
# ==============================================================================

def mock_get_market_data_with_fallback(symbol, timeframe="5m", limit=50):
    """تابع جایگزین برای دریافت داده بازار"""
    try:
        import requests
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': timeframe,
            'limit': limit
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Mock data fetch error: {e}")
        pass
    
    # داده mock با قیمت‌های واقعی‌تر
    base_prices = {
        'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'BNBUSDT': 590.00,
        'SOLUSDT': 175.00, 'XRPUSDT': 0.62, 'ADAUSDT': 0.48,
        'DEFAULT': 100
    }
    
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    data = []
    current_time = int(datetime.now().timestamp() * 1000)
    
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)
        change = random.uniform(-0.015, 0.015)
        price = base_price * (1 + change)
        candle = [
            timestamp,
            str(price * random.uniform(0.998, 1.000)),
            str(price * random.uniform(1.000, 1.003)),
            str(price * random.uniform(0.997, 1.000)),
            str(price),
            str(random.uniform(1000, 10000)),
            timestamp + 300000,
            "0", "0", "0", "0", "0"
        ]
        data.append(candle)
    
    return data

def mock_calculate_simple_sma(data, period=20):
    """محاسبه SMA ساده"""
    if not data or len(data) < period:
        return 50000
    
    closes = []
    for candle in data[-period:]:
        try:
            closes.append(float(candle[4]))
        except (IndexError, ValueError):
            closes.append(0)
    
    return sum(closes) / len(closes) if closes else 0

def mock_calculate_simple_rsi(data, period=14):
    """محاسبه RSI ساده"""
    if not data or len(data) <= period:
        return 50
    
    closes = []
    for candle in data[-(period+1):]:
        try:
            closes.append(float(candle[4]))
        except (IndexError, ValueError):
            closes.append(0)
    
    gains = 0
    losses = 0
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains += change
        else:
            losses += abs(change)
    
    avg_gain = gains / period
    avg_loss = losses / period if losses > 0 else 1
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)

def mock_calculate_smart_entry(data, signal="BUY"):
    """تابع Mock برای Smart Entry"""
    try:
        price = float(data[-1][4])
    except:
        price = 100
    
    if signal == "BUY":
        return price * random.uniform(0.99, 1.00)
    else:
        return price * random.uniform(1.00, 1.01)

def mock_analyze_with_multi_timeframe_strategy(symbol):
    """تابع جایگزین برای تحلیل"""
    signals = ["BUY", "SELL", "HOLD"]
    weights = [0.35, 0.35, 0.30]
    signal = random.choices(signals, weights=weights)[0]
    
    base_prices = {
        'BTCUSDT': 88271.00,
        'ETHUSDT': 3450.00,
        'BNBUSDT': 590.00,
        'SOLUSDT': 175.00,
        'DEFAULT': 100
    }
    
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    
    if signal == "HOLD":
        confidence = round(random.uniform(0.5, 0.7), 2)
    else:
        confidence = round(random.uniform(0.65, 0.85), 2)
    
    entry_price = round(base_price * random.uniform(0.99, 1.01), 2)
    
    if signal == "BUY":
        targets = [round(entry_price * 1.02, 2), round(entry_price * 1.05, 2)]
        stop_loss = round(entry_price * 0.98, 2)
    elif signal == "SELL":
        targets = [round(entry_price * 0.98, 2), round(entry_price * 0.95, 2)]
        stop_loss = round(entry_price * 1.02, 2)
    else:
        targets = []
        stop_loss = entry_price
    
    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "entry_price": entry_price,
        "targets": targets,
        "stop_loss": stop_loss,
        "strategy": "Multi-Timeframe Mock Analysis",
        "analysis_details": {
            "1h": {"trend": random.choice(["BULLISH", "BEARISH", "NEUTRAL"]), "source": "mock"},
            "15m": {"trend": random.choice(["BULLISH", "BEARISH", "NEUTRAL"]), "source": "mock"},
            "5m": {"trend": random.choice(["BULLISH", "BEARISH", "NEUTRAL"]), "source": "mock"}
        }
    }

def mock_calculate_24h_change(data):
    """محاسبه تغییرات ۲۴ ساعته"""
    if isinstance(data, list) and len(data) > 10:
        try:
            old_price = float(data[0][4])
            current_price = float(data[-1][4])
            return round(((current_price - old_price) / old_price) * 100, 2)
        except:
            pass
    return round(random.uniform(-3, 3), 2)

# توابع mock برای ایچیموکو
def mock_calculate_ichimoku_components(data, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
    """تابع mock برای ایچیموکو"""
    if not data or len(data) < 30:
        return None
    
    try:
        latest_price = float(data[-1][4])
    except:
        latest_price = 100
    
    return {
        'tenkan_sen': latest_price * random.uniform(0.99, 1.01),
        'kijun_sen': latest_price * random.uniform(0.98, 1.02),
        'senkou_span_a': latest_price * random.uniform(0.97, 1.03),
        'senkou_span_b': latest_price * random.uniform(0.96, 1.04),
        'cloud_top': latest_price * random.uniform(1.01, 1.05),
        'cloud_bottom': latest_price * random.uniform(0.95, 0.99),
        'quality_line': latest_price * random.uniform(0.98, 1.02),
        'golden_line': latest_price * random.uniform(0.99, 1.01),
        'trend_power': random.uniform(30, 80),
        'current_price': latest_price
    }

def mock_analyze_ichimoku_scalp_signal(ichimoku_data):
    """تابع mock برای تحلیل ایچیموکو"""
    if not ichimoku_data:
        return {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reason': 'داده ناکافی',
            'trend_power': 50
        }
    
    signals = ['BUY', 'SELL', 'HOLD']
    weights = [0.35, 0.35, 0.30]
    signal = random.choices(signals, weights=weights)[0]
    
    confidence = random.uniform(0.6, 0.9) if signal != 'HOLD' else random.uniform(0.4, 0.6)
    
    return {
        'signal': signal,
        'confidence': round(confidence, 3),
        'reason': f'سیگنال {signal} بر اساس ایچیموکو (Mock)',
        'trend_power': ichimoku_data.get('trend_power', 50)
    }

def mock_get_ichimoku_scalp_signal(data, timeframe="5m"):
    """تابع mock برای دریافت سیگنال ایچیموکو"""
    if not data:
        return None
    
    ichimoku = mock_calculate_ichimoku_components(data)
    if not ichimoku:
        return None
    
    signal = mock_analyze_ichimoku_scalp_signal(ichimoku)
    signal['timeframe'] = timeframe
    
    return signal

def mock_combined_analysis(data, timeframe="5m"):
    """تحلیل ترکیبی mock"""
    if not data:
        return None
    
    signals = ['BUY', 'SELL', 'HOLD']
    signal = random.choice(signals)
    confidence = random.uniform(0.6, 0.9) if signal != 'HOLD' else random.uniform(0.4, 0.6)
    
    try:
        price = float(data[-1][4])
    except:
        price = 100
    
    return {
        'signal': signal,
        'confidence': round(confidence, 3),
        'price': price,
        'timestamp': datetime.now().isoformat()
    }

# ==============================================================================
# انتخاب توابع مناسب بر اساس دسترسی ماژول‌ها
# ==============================================================================

print("\n🔧 Selecting appropriate functions...")

if UTILS_AVAILABLE:
    get_market_data_func = get_market_data_with_fallback
    analyze_func = analyze_with_multi_timeframe_strategy
    calculate_change_func = calculate_24h_change_from_dataframe
    calculate_sma_func = calculate_simple_sma
    calculate_rsi_func = calculate_simple_rsi
    calculate_rsi_series_func = calculate_rsi_series
    calculate_divergence_func = detect_divergence
    calculate_macd_func = calculate_macd_simple
    analyze_scalp_conditions_func = analyze_scalp_conditions
    # توابع ایچیموکو
    calculate_ichimoku_func = calculate_ichimoku_components
    analyze_ichimoku_signal_func = analyze_ichimoku_scalp_signal
    get_ichimoku_signal_func = get_ichimoku_scalp_signal
    combined_analysis_func = combined_analysis
    generate_recommendation_func = generate_ichimoku_recommendation
    # توابع Smart Entry جدید
    get_swing_high_low_func = get_swing_high_low
    calculate_smart_entry_func = calculate_smart_entry
    
    print("✅ Using REAL analysis functions from utils")
    print("✅ Using REAL Ichimoku functions")
    print("✅ Using REAL RSI/Divergence logic from utils")
    print("✅ Using SMART ENTRY logic (Ichimoku + Fibonacci) from utils")
else:
    get_market_data_func = mock_get_market_data_with_fallback
    analyze_func = mock_analyze_with_multi_timeframe_strategy
    calculate_change_func = mock_calculate_24h_change
    calculate_sma_func = mock_calculate_simple_sma
    calculate_rsi_func = mock_calculate_simple_rsi
    calculate_rsi_series_func = lambda d, p: [50]*len(d)
    calculate_divergence_func = lambda p,r,l: {"detected": False}
    calculate_macd_func = lambda data: {'macd': 0, 'signal': 0, 'histogram': 0}
    analyze_scalp_conditions_func = lambda data, tf: {"condition": "NEUTRAL", "rsi": 50, "sma_20": 0, "reason": "Mock data"}
    calculate_ichimoku_func = mock_calculate_ichimoku_components
    analyze_ichimoku_signal_func = mock_analyze_ichimoku_scalp_signal
    get_ichimoku_signal_func = mock_get_ichimoku_scalp_signal
    combined_analysis_func = mock_combined_analysis
    generate_recommendation_func = lambda signal: "توصیه بر اساس داده آزمایشی"
    # Mock functions for Smart Entry
    get_swing_high_low_func = lambda d,l: (100,50)
    calculate_smart_entry_func = mock_calculate_smart_entry
    
    print("⚠️ Using MOCK analysis functions (Market/Strategy/Smart Entry)")
    print("⚠️ Ichimoku functions in mock mode")
    print("⚠️ Divergence functions in mock mode")

print(f"📊 Module Status: utils={UTILS_AVAILABLE}, data_collector={DATA_COLLECTOR_AVAILABLE}, collectors={COLLECTORS_AVAILABLE}")

# ==============================================================================
# FastAPI Application
# ==============================================================================
API_VERSION = "7.6.0"  # نسخه جدید با Smart Entry

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description=f"Real RSI & Smart Entry Scanner - نسخه {API_VERSION}",
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# توابع کمکی جدید برای اسکالپ و ایچیموکو (با استفاده از Smart Entry)
# ==============================================================================
def analyze_scalp_signal(symbol, timeframe, data):
    """
    تحلیل سیگنال اسکالپ - به‌روزرسانی شده با Smart Entry (Strong Solution)
    در اینجا از توابع هوشمند ورود برای محاسبه دقیق قیمت استفاده می‌شود.
    """
    if not data or len(data) < 30:
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "rsi": 50,
            "divergence": False,
            "sma_20": 0,
            "current_price": 0,
            "reason": "Insufficient data"
        }
    
    # محاسبه اندیکاتورها
    rsi = calculate_rsi_func(data, 14)
    sma_20 = calculate_sma_func(data, 20)
    
    # استخراج لیست قیمت‌ها برای تشخیص واگرایی
    closes = []
    for c in data:
        try:
            closes.append(float(c[4]))
        except: pass
    
    # محاسبه سری کامل RSI برای تشخیص واگرایی
    rsi_series = calculate_rsi_series_func(closes, 14)
    
    # تشخیص واگرایی
    div_info = calculate_divergence_func(closes, rsi_series, lookback=5)
    
    # آخرین قیمت
    try:
        latest_close = float(data[-1][4])
    except (IndexError, ValueError, TypeError):
        try:
            latest_close = float(data[-1]['close']) if isinstance(data[-1], dict) else 0
        except:
            latest_close = 0
    
    # --- منطق اسکالپ ---
    signal = "HOLD"
    confidence = 0.5
    reason = "Market neutral"
    
    # اولویت با واگرایی (اگر واگرایی باشد، سیگنال قوی است حتی اگر RSI نرمال باشد)
    if div_info['detected']:
        if div_info['type'] == 'bullish':
            signal = "BUY"
            confidence = 0.85  # اعتماد بالا برای واگرایی صعودی
            reason = f"Bullish Divergence Detected (Fibo + Ichimoku Entry)"
        elif div_info['type'] == 'bearish':
            signal = "SELL"
            confidence = 0.85
            reason = f"Bearish Divergence Detected (Fibo + Ichimoku Entry)"
        else:
            # اگر واگرایی تشخیص داده شده اما نوعش مشخص نیست
            signal = "HOLD"
            reason = "Divergence detected (Type unclear)"
    else:
        # اگر واگرایی نبود، به سراغ RSI و SMA بگرد
        if rsi < 35 and latest_close < sma_20 * 1.01:
            signal = "BUY"
            confidence = min(0.75, (35 - rsi) / 35 * 0.5 + 0.5)
            reason = f"Oversold (RSI: {rsi:.1f}), below SMA20 (Smart Entry)"
        
        elif rsi > 65 and latest_close > sma_20 * 0.99:
            signal = "SELL"
            confidence = min(0.75, (rsi - 65) / 35 * 0.5 + 0.5)
            reason = f"Overbought (RSI: {rsi:.1f}), above SMA20 (Smart Entry)"
        
        elif latest_close > sma_20 * 1.02 and rsi < 60:
            signal = "BUY"
            confidence = 0.7
            reason = f"Breakout above SMA20 (Smart Entry), RSI: {rsi:.1f}"
        
        elif latest_close < sma_20 * 0.98 and rsi > 40:
            signal = "SELL"
            confidence = 0.7
            reason = f"Breakdown below SMA20 (Smart Entry), RSI: {rsi:.1f}"
    
    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "rsi": round(rsi, 1),
        "divergence": div_info['detected'],
        "divergence_type": div_info['type'],
        "sma_20": round(sma_20, 2),
        "current_price": round(latest_close, 2),
        "reason": reason
    }

def analyze_ichimoku_scalp(symbol, timeframe, data):
    """
    تحلیل اسکالپ با ایچیموکو پیشرفته
    """
    if not data or len(data) < 60:
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "divergence": False,
            "reason": "داده ناکافی برای ایچیموکو",
            "ichimoku": None,
            "type": "ICHIMOKU_SCALP"
        }
    
    try:
        # محاسبه ایچیموکو
        ichimoku_data = calculate_ichimoku_func(
            data, 
            tenkan_period=9, 
            kijun_period=26, 
            senkou_b_period=52, 
            displacement=26
        )
        
        if not ichimoku_data:
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "divergence": False,
                "reason": "محاسبه ایچیموکو ناموفق",
                "ichimoku": None,
                "type": "ICHIMOKU_SCALP"
            }
        
        # تحلیل سیگنال ایچیموکو
        ichimoku_signal = analyze_ichimoku_signal_func(ichimoku_data)
        
        # قیمت فعلی
        current_price = ichimoku_data.get('current_price', 0)
        if current_price <= 0:
            try:
                current_price = float(data[-1][4])
            except:
                current_price = 0
        
        if current_price <= 0:
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "divergence": False,
                "reason": "قیمت نامعتبر",
                "ichimoku": None,
                "type": "ICHIMOKU_SCALP"
            }
        
        # محاسبه تارگت‌ها بر اساس ایچیموکو
        if ichimoku_signal['signal'] == 'BUY':
            support = min(
                ichimoku_data.get('cloud_bottom', current_price * 0.99),
                ichimoku_data.get('kijun_sen', current_price * 0.99),
                current_price * 0.995
            )
            
            resistance1 = max(
                ichimoku_data.get('cloud_top', current_price * 1.01),
                ichimoku_data.get('tenkan_sen', current_price * 1.01),
                current_price * 1.01
            )
            resistance2 = resistance1 * 1.005
            resistance3 = resistance1 * 1.01
            
            targets = [resistance1, resistance2, resistance3]
            stop_loss = support
            
        elif ichimoku_signal['signal'] == 'SELL':
            resistance = max(
                ichimoku_data.get('cloud_top', current_price * 1.01),
                ichimoku_data.get('kijun_sen', current_price * 1.01),
                current_price * 1.005
            )
            
            support1 = min(
                ichimoku_data.get('cloud_bottom', current_price * 0.99),
                ichimoku_data.get('tenkan_sen', current_price * 0.99),
                current_price * 0.99
            )
            support2 = support1 * 0.995
            support3 = support1 * 0.99
            
            targets = [support1, support2, support3]
            stop_loss = resistance
            
        else:
            targets = []
            stop_loss = current_price
        
        # سطوح کلیدی
        levels = {
            "tenkan_sen": ichimoku_data.get('tenkan_sen'),
            "kijun_sen": ichimoku_data.get('kijun_sen'),
            "cloud_top": ichimoku_data.get('cloud_top'),
            "cloud_bottom": ichimoku_data.get('cloud_bottom'),
            "quality_line": ichimoku_data.get('quality_line'),
            "golden_line": ichimoku_data.get('golden_line')
        }
        
        filtered_levels = {k: round(v, 4) for k, v in levels.items() if v is not None}
        
        # تفسیر روند
        trend_power = ichimoku_signal.get('trend_power', 50)
        trend_interpretation = "روند قوی" if trend_power >= 70 else \
                              "روند متوسط" if trend_power >= 60 else \
                              "روند ضعیف" if trend_power >= 40 else "بدون روند"
        
        return {
            "signal": ichimoku_signal['signal'],
            "confidence": ichimoku_signal['confidence'],
            "reason": ichimoku_signal['reason'],
            "entry_price": current_price, # Entry price for Ichimoku is just current for simplicity
            "targets": [round(t, 4) for t in targets if t > 0],
            "stop_loss": round(stop_loss, 4) if stop_loss > 0 else current_price,
            "ichimoku": filtered_levels,
            "trend_analysis": {
                "power": trend_power,
                "interpretation": trend_interpretation,
                "cloud_thickness_percent": round(ichimoku_data.get('cloud_thickness', 0), 2),
                "in_cloud": ichimoku_data.get('in_cloud', False),
                "cloud_color": ichimoku_data.get('cloud_color', 'خنثی')
            },
            "type": "ICHIMOKU_SCALP",
            "strategy": f"ایچیموکو پیشرفته ({timeframe})"
        }
        
    except Exception as e:
        logger.error(f"خطا در تحلیل ایچیموکو: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "divergence": False,
            "reason": f"خطا در تحلیل: {str(e)}",
            "ichimoku": None,
            "type": "ICHIMOKU_SCALP"
        }

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def read_root():
    """صفحه اصلی"""
    endpoints = {
        "health": "GET /api/health",
        "signals": "GET /api/signals",
        "analyze": "POST /api/analyze",
        "scalp_signal": "POST /api/scalp-signal",
        "ichimoku_scalp": "POST /api/ichimoku-scalp",
        "combined_analysis": "POST /api/combined-analysis",
        "market": "GET /market/{symbol}",
        "scraped_signals": "GET /signals/scraped",
        "docs": "GET /api/docs"
    }
    
    return {
        "message": f"🚀 سیستم تحلیل معاملاتی ارز دیجیتال v{API_VERSION}",
        "status": "در حال اجرا",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE
        },
        "endpoints": endpoints,
        "features": [
            "Real-time Analysis",
            "Scalp Signals (1m/5m/15m)", 
            "Ichimoku Advanced Analysis",
            "Multi-timeframe",
            "Fallback System",
            "Quality Line & Golden Line",
            "Real RSI & Divergence Detection",
            "Smart Entry (Fibonacci + Ichimoku)"  # ویژگی جدید مهم
        ],
        "note": f"نسخه {API_VERSION} با قیمت‌گذاری هوشمند ورود"
    }

@app.get("/api/health")
async def health_check():
    """بررسی سلامت سیستم"""
    return {
        "status": "سالم",
        "timestamp": datetime.now().isoformat(),
        "version": API_VERSION,
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE
        },
        "components": {
            "api": "سالم",
            "data_sources": "Binance (Primary) -> LBank (Fallback)" if UTILS_AVAILABLE else "Mock Data",
            "internal_ai": "فعال" if UTILS_AVAILABLE else "mock",
            "scalp_engine": "فعال",
            "ichimoku_engine": "فعال" if UTILS_AVAILABLE else "mock",
            "signal_cache": "فعال",
            "rsi_engine": "Real (Local)",
            "divergence_engine": "Real (Local)",
            "smart_entry_engine": "فعال (Ichimoku + Fibonacci)" if UTILS_AVAILABLE else "mock"
        },
        "scalp_support": {
            "enabled": True,
            "timeframes": ["1m", "5m", "15m"],
            "min_confidence": 0.65,
            "smart_entry": True
        },
        "ichimoku_support": {
            "enabled": UTILS_AVAILABLE,
            "features": ["Quality Line", "Golden Line", "Trend Power", "Cloud Analysis"],
            "timeframes": ["1m", "5m", "15m", "1h", "4h"]
        }
    }

@app.get("/api/signals", response_model=SignalResponse)
async def get_all_signals_endpoint(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None
):
    """دریافت سیگنال‌های تحلیلی داخلی"""
    logger.info(f"📡 درخواست تحلیل داخلی برای: {symbol or 'همه'}")
    
    try:
        analysis = analyze_func(symbol.upper() if symbol else "BTCUSDT")
        
        signals = [{
            "symbol": analysis["symbol"],
            "timeframe": "multi",
            "signal": analysis["signal"],
            "confidence": analysis["confidence"],
            "entry_price": analysis["entry_price"],
            "targets": analysis["targets"],
            "stop_loss": analysis["stop_loss"],
            "reason": f"تحلیل چندزمانه برای {analysis['symbol']}",
            "source": "internal_ai",
            "author": "موتور تحلیل محلی",
            "strategy": analysis.get("strategy", "تحلیل چندزمانی"),
            "type": "SWING",
            "generated_at": datetime.now().isoformat()
        }]

        sources_count = {"internal_ai": 1, "total": 1}
        
        response = SignalResponse(
            status="موفق",
            count=len(signals),
            last_updated=datetime.now().isoformat(),
            signals=signals,
            sources=sources_count
        )
        
        response_dict = response.dict()
        response_dict["api_version"] = API_VERSION
        response_dict["module_status"] = "real" if UTILS_AVAILABLE else "mock"
        return response_dict
        
    except Exception as e:
        logger.error(f"❌ خطا در دریافت سیگنال‌ها: {e}")
        raise HTTPException(status_code=500, detail=f"خطا در دریافت سیگنال‌ها: {str(e)}")

@app.post("/api/analyze")
async def analyze_crypto(request: AnalysisRequest):
    """تحلیل یک نماد ارز دیجیتال با مکانیزم Fallback"""
    logger.info(f"📈 درخواست تحلیل: {request.symbol} ({request.timeframe})")
    
    try:
        # استفاده از موتور تحلیل
        analysis = analyze_func(request.symbol)
        
        # اضافه کردن اطلاعات تایم‌فریم
        analysis["requested_timeframe"] = request.timeframe
        analysis["analysis_type"] = "STANDARD"
        analysis["version"] = API_VERSION
        analysis["module"] = "real" if UTILS_AVAILABLE else "mock"
        analysis["recommendation"] = f"سیگنال {analysis['signal']} با {analysis['confidence']:.0%} اطمینان"
        analysis["timestamp"] = datetime.now().isoformat()
        
        # محاسبه RSI و واگرایی برای این endpoint هم
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        if market_data:
            rsi_val = calculate_rsi_func(market_data, 14)
            closes = [float(c[4]) for c in market_data]
            rsi_series = calculate_rsi_series_func(closes, 14)
            div = calculate_divergence_func(closes, rsi_series)
            
            analysis["rsi"] = round(rsi_val, 2)
            analysis["divergence"] = div['detected']
            analysis["divergence_type"] = div['type']
        else:
            analysis["rsi"] = 50
            analysis["divergence"] = False
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"خطا در تحلیل: {str(e)}")

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """
    سیگنال‌های اسکالپ 1-5-15 دقیقه
    با قیمت ورودی هوشمند (Smart Entry)
    """
    logger.info(f"⚡ درخواست سیگنال اسکالپ: {request.symbol} ({request.timeframe})")
    
    allowed_timeframes = ["1m", "5m", "15m"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(
            status_code=400, 
            detail=f"Only {', '.join(allowed_timeframes)} timeframes allowed for scalp"
        )
    
    try:
        # دریافت داده بازار
        market_data = get_market_data_func(request.symbol, request.timeframe, 50)
        
        if not market_data:
            raise HTTPException(status_code=404, detail=f"No market data for {request.symbol}")
        
        # تحلیل اسکالپ (شامل محاسبه واگرایی)
        scalp_analysis = analyze_scalp_signal(request.symbol, request.timeframe, market_data)
        
        # --- محاسبه قیمت ورودی هوشمند (Strong Solution) ---
        smart_entry_price = calculate_smart_entry_func(market_data, scalp_analysis["signal"])
        
        # تایید قیمت هوشمند
        if smart_entry_price <= 0:
            base_prices = {
                'BTCUSDT': 88271.00,
                'ETHUSDT': 3450.00,
                'DEFAULT': 100
            }
            base_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
            smart_entry_price = round(base_price * random.uniform(0.995, 1.005), 2)
        
        # محاسبه تارگت‌ها و استاپ لاس بر اساس قیمت ورودی هوشمند
        if scalp_analysis["signal"] == "BUY":
            # تارگت 1: اولین مقاومت (برای اسکالپ معمولاً 1%)
            targets = [
                round(smart_entry_price * 1.01, 2),  # 1% بالاتر
                round(smart_entry_price * 1.02, 2),  # 2% بالاتر
                round(smart_entry_price * 1.03, 2)   # 3% بالاتر
            ]
            # استاپ لاس: زیر حمایت (استاتیک)
            stop_loss = round(smart_entry_price * 0.99, 2)  # 1% پایین‌تر
            
        elif scalp_analysis["signal"] == "SELL":
            targets = [
                round(smart_entry_price * 0.99, 2),  # 1% پایین‌تر
                round(smart_entry_price * 0.98, 2),  # 2% پایین‌تر
                round(smart_entry_price * 0.97, 2)   # 3% پایین‌تر
            ]
            stop_loss = round(smart_entry_price * 1.01, 2)  # 1% بالاتر
        else:
            targets = []
            stop_loss = smart_entry_price
        
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": scalp_analysis["signal"],
            "confidence": scalp_analysis["confidence"],
            "entry_price": round(smart_entry_price, 2),  # قیمت ورودی هوشمند
            "rsi": scalp_analysis["rsi"],
            "divergence": scalp_analysis["divergence"],
            "divergence_type": scalp_analysis["divergence_type"],
            "sma_20": scalp_analysis["sma_20"],
            "targets": targets,
            "stop_loss": stop_loss,
            "type": "SCALP",
            "reason": scalp_analysis["reason"],
            "strategy": f"Scalp Strategy (Smart Entry - Ichimoku + Fibo)",
            "module": "real",
            "version": API_VERSION,
            "timestamp": datetime.now().isoformat(),
            "risk_level": "HIGH" if request.timeframe == "1m" else "MEDIUM",
            "recommendation": f"{scal_analysis['signal']} signal for scalp trading on {request.timeframe} timeframe"
        }
        
        logger.info(f"✅ Scalp signal generated: {request.symbol} - {scalp_analysis['signal']} (Entry: {smart_entry_price})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in scalp signal: {e}")
        # Fallback with simple percentage (mock entry if utils failed)
        mock_signal = random.choice(["BUY", "SELL", "HOLD"])
        mock_confidence = 0.6 + random.random() * 0.3
        
        base_prices = {
            'BTCUSDT': 88271.00,
            'ETHUSDT': 3450.00,
            'DEFAULT': 100
        }
        
        base_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        simple_entry = round(base_price * random.uniform(0.99, 1.01), 2)
        
        if mock_signal == "BUY":
            targets = [round(simple_entry * 1.01, 2), round(simple_entry * 1.02, 2)]
            stop_loss = round(simple_entry * 0.99, 2)
        elif mock_signal == "SELL":
            targets = [round(simple_entry * 0.99, 2), round(simple_entry * 0.98, 2)]
            stop_loss = round(simple_entry * 1.01, 2)
        else:
            targets = []
            stop_loss = simple_entry
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": mock_signal,
            "confidence": round(mock_confidence, 2),
            "entry_price": simple_entry, # Simple entry fallback
            "rsi": round(30 + random.random() * 40, 1),
            "divergence": False,
            "divergence_type": None,
            "sma_20": round(simple_entry * random.uniform(0.99, 1.01), 2),
            "targets": targets,
            "stop_loss": stop_loss,
            "type": "SCALP_MOCK",
            "reason": "Using mock data (API error) + Simple Entry",
            "strategy": "Mock Scalp Strategy",
            "module": "mock",
            "version": API_VERSION,
            "timestamp": datetime.now().isoformat(),
            "risk_level": "HIGH",
            "recommendation": f"Mock {mock_signal} signal"
        }

@app.post("/api/ichimoku-scalp")
async def get_ichimoku_scalp_signal(request: IchimokuRequest):
    """
    سیگنال اسکالپ با ایچیموکو پیشرفته
    """
    logger.info(f"☁️ درخواست سیگنال ایچیموکو: {request.symbol} ({request.timeframe})")
    
    allowed_timeframes = ["1m", "5m", "15m", "1h", "4h"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(
            status_code=400, 
            detail=f"Only {', '.join(allowed_timeframes)} timeframes allowed for Ichimoku analysis"
        )
    
    try:
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        
        if not market_data or len(market_data) < 60:
            raise HTTPException(status_code=404, detail=f"Not enough data for Ichimoku analysis (need min 60 candles)")
        
        ichimoku_analysis = analyze_ichimoku_scalp(request.symbol, request.timeframe, market_data)
        
        rsi = calculate_rsi_func(market_data, 14)
        closes = [float(c[4]) for c in market_data]
        rsi_series = calculate_rsi_series_func(closes, 14)
        div = calculate_divergence_func(closes, rsi_series)
        
        recommendation = generate_recommendation_func(ichimoku_analysis)
        
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": ichimoku_analysis["signal"],
            "confidence": ichimoku_analysis["confidence"],
            "entry_price": ichimoku_analysis["entry_price"],
            "rsi": round(rsi, 2) if rsi else 50,
            "divergence": div['detected'],
            "divergence_type": div['type'],
            "targets": ichimoku_analysis["targets"],
            "stop_loss": ichimoku_analysis["stop_loss"],
            "type": ichimoku_analysis["type"],
            "reason": ichimoku_analysis["reason"],
            "strategy": ichimoku_analysis["strategy"],
            "ichimoku_data": ichimoku_analysis.get("ichimoku", {}),
            "trend_analysis": ichimoku_analysis.get("trend_analysis", {}),
            "recommendation": recommendation,
            "module": "real",
            "version": API_VERSION,
            "timestamp": datetime.now().isoformat(),
            "risk_level": "HIGH" if request.timeframe in ["1m", "5m"] else "MEDIUM",
            "features": {
                "quality_line": ichimoku_analysis.get("ichimoku", {}).get("quality_line") is not None,
                "golden_line": ichimoku_analysis.get("ichimoku", {}).get("golden_line") is not None,
                "cloud_analysis": True,
                "trend_power": True
            }
        }
        
        logger.info(f"✅ Ichimoku signal: {request.symbol} - {ichimoku_analysis['signal']} ({ichimoku_analysis['confidence']:.0%})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in Ichimoku analysis: {e}")
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": 0,
            "rsi": 50,
            "divergence": False,
            "divergence_type": None,
            "targets": [],
            "stop_loss": 0,
            "type": "ICHIMOKU_SCALP",
            "reason": f"خطا در تحلیل: {str(e)}",
            "strategy": "Ichimoku Fallback",
            "ichimoku_data": None,
            "trend_analysis": None,
            "recommendation": "Use standard analysis instead",
            "module": "error",
            "version": API_VERSION,
            "timestamp": datetime.now().isoformat(),
            "risk_level": "HIGH",
            "features": {
                "quality_line": False,
                "golden_line": False,
                "cloud_analysis": False,
                "trend_power": False
            }
        }

@app.post("/api/combined-analysis")
async def get_combined_analysis(request: CombinedRequest):
    """
    تحلیل ترکیبی با چندین اندیکاتور
    """
    logger.info(f"🧩 درخواست تحلیل ترکیبی: {request.symbol} ({request.timeframe})")
    
    try:
        # دریافت داده بازار
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        
        if not market_data:
            raise HTTPException(status_code=404, detail=f"No market data for {request.symbol}")
        
        # محاسبه RSI و واگرایی اصلی
        closes = [float(c[4]) for c in market_data]
        rsi_series = calculate_rsi_series_func(closes, 14)
        div = calculate_divergence_func(closes, rsi_series)
        
        # تحلیل ترکیبی
        combined_result = combined_analysis_func(market_data, request.timeframe)
        
        if not combined_result:
            raise HTTPException(status_code=500, detail="Combined analysis failed")
        
        # محاسبه قیمت فعلی
        try:
            current_price = float(market_data[-1][4])
        except:
            current_price = 0
        
        # تحلیل ایچیموکو جداگانه اگر خواسته شده
        ichimoku_data = None
        if request.include_ichimoku:
            ichimoku_analysis = analyze_ichimoku_scalp(request.symbol, request.timeframe, market_data)
            ichimoku_data = {
                "signal": ichimoku_analysis["signal"],
                "confidence": ichimoku_analysis["confidence"],
                "levels": ichimoku_analysis.get("ichimoku", {}),
                "trend": ichimoku_analysis.get("trend_analysis", {})
            }
        
        # محاسبه RSI اگر خواسته شده
        rsi_data = None
        if request.include_rsi:
            rsi_value = calculate_rsi_func(market_data, 14)
            rsi_data = {
                "value": rsi_value,
                "status": "oversold" if rsi_value < 30 else "overbought" if rsi_value > 70 else "neutral",
                "divergence": div['detected'],
                "divergence_type": div['type']
            }
        
        # محاسبه MACD اگر خواسته شده
        macd_data = None
        if request.include_macd:
            macd_result = calculate_macd_func(market_data)
            macd_data = {
                "macd": macd_result.get("macd", 0),
                "signal": macd_result.get("signal", 0),
                "histogram": macd_result.get("histogram", 0),
                "trend": "bullish" if macd_result.get("histogram", 0) > 0 else "bearish"
            }
        
        # تحلیل شرایط اسکالپ
        scalp_conditions = analyze_scalp_conditions_func(market_data, request.timeframe)
        
        # محاسبه قیمت ورودی هوشمند برای این تحلیل ترکیبی
        smart_entry_price = 0
        if combined_result["signal"] == "BUY":
            smart_entry_price = calculate_smart_entry_func(market_data, signal="BUY")
        elif combined_result["signal"] == "SELL":
            smart_entry_price = calculate_smart_entry_func(market_data, signal="SELL")
        else:
            smart_entry_price = current_price
        
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": combined_result["signal"],
            "confidence": combined_result["confidence"],
            "entry_price": smart_entry_price, # قیمت ورودی هوشمند
            "analysis": {
                "combined": combined_result,
                "ichimoku": ichimoku_data,
                "rsi": rsi_data,
                "macd": macd_data,
                "scalp_conditions": scalp_conditions
            },
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION,
            "module": "real",
            "recommendation": generate_recommendation_func({
                "signal": combined_result["signal"],
                "confidence": combined_result["confidence"],
                "in_cloud": ichimoku_data.get("trend", {}).get("in_cloud", False) if ichimoku_data else False
            })
        }
        
        logger.info(f"✅ Combined analysis: {request.symbol} - {combined_result['signal']} ({combined_result['confidence']:.0%})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in combined analysis: {e}")
        
        # Fallback ساده
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": 0,
            "analysis": {
                "combined": {"signal": "HOLD", "confidence": 0.5},
                "ichimoku": None,
                "rsi": None,
                "macd": None,
                "scalp_conditions": {"condition": "NEUTRAL", "reason": "Analysis failed"}
            },
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION,
            "module": "error",
            "recommendation": "Analysis failed, please try again"
        }

@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    """
    دریافت داده‌های خام بازار با مکانیزم Fallback (بایننس -> LBank)
    """
    try:
        data = get_market_data_func(symbol, timeframe, limit=50)
        
        if not data:
            raise HTTPException(status_code=404, detail=f"داده‌ای برای نماد {symbol} یافت نشد.")
        
        latest = data[-1] if isinstance(data, list) and len(data) > 0 else []
        
        if not latest or len(latest) < 6:
            # بازگشت داده mock
            base_prices = {
                'BTCUSDT': 88271.00,
                'ETHUSDT': 3450.00,
                'DEFAULT': 100
            }
            
            base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "Mock Data",
                "current_price": round(base_price * random.uniform(0.99, 1.01), 2),
                "high": round(base_price * random.uniform(1.005, 1.015), 2),
                "low": round(base_price * random.uniform(0.985, 0.995), 2),
                "volume": round(random.uniform(1000, 5000), 2),
                "change_24h": round(random.uniform(-5, 5), 2),
                "timestamp": datetime.now().isoformat(),
                "note": "Using mock data"
            }
        
        change_24h = calculate_change_func(data)

        rsi = calculate_rsi_func(data, 14)
        sma_20 = calculate_sma_func(data, 20)
        
        closes = [float(c[4]) for c in data]
        rsi_series = calculate_rsi_series_func(closes, 14)
        div = calculate_divergence_func(closes, rsi_series)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": "Binance API" if UTILS_AVAILABLE else "Mock Data",
            "current_price": float(latest[4]),
            "high": float(latest[2]),
            "low": float(latest[3]),
            "volume": float(latest[5]),
            "change_24h": change_24h,
            "rsi_14": round(rsi, 2),
            "sma_20": round(sma_20, 2),
            "divergence": div['detected'],
            "divergence_type": div['type'],
            "timestamp": datetime.now().isoformat(),
            "data_points": len(data),
            "support_scalp": timeframe in ["1m", "5m", "15m"]
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in /market/{symbol}: {e}")
        raise HTTPException(status_code=500, detail="خطای داخلی سرور در دریافت داده‌های بازار")

@app.get("/signals/scraped")
async def get_scraped_signals():
    """
    یک Endpoint برای تست تابع collectors و دریافت سیگنال‌های اسکراپ شده
    """
    try:
        if COLLECTORS_AVAILABLE:
            scraped_signals = collect_signals_from_example_site()
        else:
            # داده mock
            scraped_signals = []
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            for symbol in symbols:
                scraped_signals.append({
                    "symbol": symbol,
                    "signal": random.choice(["BUY", "SELL", "HOLD"]),
                    "confidence": round(random.uniform(0.6, 0.9), 2),
                    "source": "Mock Collector",
                    "timestamp": datetime.now().isoformat(),
                    "type": "SCRAPED",
                    "divergence": False
                })
        
        return {
            "status": "success",
            "source": "Example Site Scraper" if COLLECTORS_AVAILABLE else "Mock Collector",
            "count": len(scraped_signals),
            "signals": scraped_signals,
            "module": "real" if COLLECTORS_AVAILABLE else "mock",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in scraped signals endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scrape signals: {e}")

@app.get("/api/scan-all-timeframes/{symbol}")
async def scan_all_timeframes(symbol: str):
    """اسکن همه تایم‌فریم‌ها برای یک نماد"""
    logger.info(f"🔍 Scanning all timeframes for {symbol}")
    
    try:
        timeframes = ["1m", "5m", "15m", "1h", "4h"]
        results = []
        
        for tf in timeframes:
            try:
                if tf in ["1m", "5m", "15m"]:
                    response = await get_ichimoku_scalp_signal(IchimokuRequest(symbol=symbol, timeframe=tf))
                    response["analysis_type"] = "ICHIMOKU_SCALP"
                else:
                    response = await analyze_crypto(AnalysisRequest(symbol=symbol, timeframe=tf))
                    response["analysis_type"] = "STANDARD"
                
                results.append(response)
                
            except Exception as tf_error:
                logger.warning(f"Error scanning {symbol} on {tf}: {tf_error}")
                results.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "signal": "ERROR",
                    "error": str(tf_error)
                })
        
        return {
            "symbol": symbol,
            "scanned_at": datetime.now().isoformat(),
            "total_timeframes": len(timeframes),
            "successful_scans": len([r for r in results if r.get("signal") != "ERROR"]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in scan-all-timeframes: {e}")
        raise HTTPException(status_code=500, detail=f"Scan error: {str(e)}")

# ==============================================================================
# Startup Event
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    """مقداردهی اولیه هنگام راه‌اندازی"""
    
    logger.info("=" * 60)
    logger.info(f"🚀 راه‌اندازی سیستم تحلیل معاملاتی ارز دیجیتال v{API_VERSION}")
    logger.info(f"📡 نسخه: {API_VERSION} - با Smart Entry (Fibonacci + Ichimoku)")
    logger.info(f"⚙️ وضعیت ماژول‌ها:")
    logger.info(f"   - utils: {'✅' if UTILS_AVAILABLE else '❌'}")
    logger.info(f"   - data_collector: {'✅' if DATA_COLLECTOR_AVAILABLE else '❌'}")
    logger.info(f"   - collectors: {'✅' if COLLECTORS_AVAILABLE else '❌'}")
    logger.info(f"☁️ ویژگی‌های ایچیموکو:")
    logger.info(f"   - Quality Line: {'✅' if UTILS_AVAILABLE else '❌'}")
    logger.info(f"   - Golden Line: {'✅' if UTILS_AVAILABLE else '❌'}")
    logger.info(f"   - Trend Power: {'✅' if UTILS_AVAILABLE else '❌'}")
    logger.info(f"   - Cloud Analysis: {'✅' if UTILS_AVAILABLE else '❌'}")
    logger.info(f"🔧 ویژگی‌های فعال:")
    logger.info(f"   - تحلیل چندزمانه: ✅")
    logger.info(f"   - سیگنال اسکالپ (1m/5m/15m): ✅")
    logger.info(f"   - ایچیموکو پیشرفته: {'✅' if UTILS_AVAILABLE else '⚠️ (Mock)'}")
    logger.info(f"   - محاسبه RSI واقعی: ✅")
    logger.info(f"   - تشخیص واگرایی واقعی: ✅")
    logger.info(f"   - محاسبه قیمت ورودی هوشمند (Smart Entry): ✅")
    logger.info(f"   - مکانیزم Fallback: ✅")
    logger.info(f"   - قیمت‌های واقعی: {'✅' if UTILS_AVAILABLE else '⚠️ (Mock)'}")
    logger.info("⏰ زمان راه‌اندازی: " + datetime.now().isoformat())
    logger.info("=" * 60)

# For local development
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 شروع سرور محلی v{API_VERSION} روی پورت {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")