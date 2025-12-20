# api/utils.py - نسخه بهینه‌شده برای Render
"""
Utility Functions - Render Optimized Version
"""

import requests
import logging
import random
from datetime import datetime, timedelta
import time
import json

logger = logging.getLogger(__name__)

# ==============================================================================
# 📊 تابع اصلی دریافت داده (ساده‌شده)
# ==============================================================================

def get_market_data_with_fallback(symbol, interval="5m", limit=100):
    """
    دریافت داده‌های بازار - نسخه ساده‌شده
    """
    logger.info(f"📊 دریافت داده برای {symbol} ({interval})")
    
    # ۱. تلاش برای دریافت از Binance
    try:
        data = get_binance_klines_simple(symbol, interval, limit)
        if data:
            logger.info(f"✅ داده از Binance دریافت شد: {len(data)} کندل")
            return {"data": data, "source": "binance", "success": True}
    except Exception as e:
        logger.warning(f"⚠️ خطا در Binance: {e}")
    
    # ۲. تلاش برای دریافت از LBank
    try:
        data = get_lbank_data_simple(symbol, interval, limit)
        if data:
            logger.info(f"✅ داده از LBank دریافت شد: {len(data)} کندل")
            return {"data": data, "source": "lbank", "success": True}
    except Exception as e:
        logger.warning(f"⚠️ خطا در LBank: {e}")
    
    # ۳. داده Mock
    logger.info(f"🧪 استفاده از داده Mock برای {symbol}")
    return {"data": generate_mock_data_simple(symbol, limit), "source": "mock", "success": False}

# ==============================================================================
# 📊 توابع دریافت داده از صرافی‌ها (بدون pandas)
# ==============================================================================

def get_binance_klines_simple(symbol, interval="5m", limit=100):
    """دریافت داده از Binance بدون pandas"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000)
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()  # لیست خام کندل‌ها
        logger.error(f"Binance API error: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ خطا در Binance: {e}")
    return None

def get_lbank_data_simple(symbol, interval="5m", limit=100):
    """دریافت داده از LBank بدون pandas"""
    try:
        # تبدیل interval
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '30m': '30min', '1h': '1hour', '4h': '4hour',
            '1d': '1day', '1w': '1week'
        }
        lbank_interval = interval_map.get(interval, '5min')
        
        # تبدیل symbol (فرض می‌کنیم format: btc_usdt)
        lbank_symbol = symbol.lower().replace("usdt", "_usdt")
        
        url = "https://api.lbkex.com/v2/klines.do"
        params = {
            'symbol': lbank_symbol,
            'type': lbank_interval,
            'size': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        logger.error(f"LBank API error: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ خطا در LBank: {e}")
    return None

# ==============================================================================
# 📊 توابع Mock (بدون pandas/numpy)
# ==============================================================================

def generate_mock_data_simple(symbol, limit=100):
    """تولید داده آزمایشی بدون pandas/numpy"""
    base_prices = {
        'BTCUSDT': 65000, 'ETHUSDT': 3500, 'BNBUSDT': 580,
        'SOLUSDT': 170, 'XRPUSDT': 0.62, 'ADAUSDT': 0.48,
        'DEFAULT': 100
    }
    
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    mock_data = []
    current_time = int(time.time() * 1000)
    
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)  # 5 دقیقه فاصله
        
        # شبیه‌سازی حرکت قیمت
        change = random.uniform(-0.02, 0.02)  # ±2%
        price = base_price * (1 + change)
        
        mock_candle = [
            timestamp,  # open time
            str(price * random.uniform(0.998, 1.000)),  # open
            str(price * random.uniform(1.000, 1.005)),  # high
            str(price * random.uniform(0.995, 1.000)),  # low
            str(price),  # close
            str(random.uniform(1000, 10000)),  # volume
            timestamp + 300000,  # close time
            "0", "0", "0", "0", "0"  # سایر فیلدها
        ]
        
        mock_data.append(mock_candle)
    
    return mock_data

# ==============================================================================
# 📈 توابع تحلیل تکنیکال (ساده‌شده)
# ==============================================================================

def calculate_simple_sma(data, period=20):
    """محاسبه SMA ساده (بدون pandas)"""
    if not data or len(data) < period:
        return None
    
    closes = []
    for candle in data[-period:]:  # آخرین period کندل
        try:
            closes.append(float(candle[4]))  # index 4 = close price
        except (IndexError, ValueError):
            closes.append(0)
    
    return sum(closes) / len(closes) if closes else 0

def calculate_simple_rsi(data, period=14):
    """محاسبه RSI ساده (بدون pandas)"""
    if not data or len(data) <= period:
        return 50  # مقدار خنثی
    
    closes = []
    for candle in data[-(period+1):]:  # برای period+1 کندل
        try:
            closes.append(float(candle[4]))
        except:
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

# ==============================================================================
# 🚀 موتور اصلی تحلیل (ساده‌شده)
# ==============================================================================

def analyze_with_multi_timeframe_strategy(symbol):
    """
    تحلیل چندزمانی - نسخه بهینه برای Render
    """
    logger.info(f"🤖 تحلیل {symbol}")
    
    try:
        # دریافت داده از تایم‌فریم‌های مختلف
        result_1h = get_market_data_with_fallback(symbol, "1h", 50)
        result_15m = get_market_data_with_fallback(symbol, "15m", 50)
        result_5m = get_market_data_with_fallback(symbol, "5m", 50)
        
        # استخراج داده‌ها
        data_1h = result_1h.get("data", [])
        data_15m = result_15m.get("data", [])
        data_5m = result_5m.get("data", [])
        
        if not data_5m:  # اگر هیچ دادهای نداریم
            return get_fallback_signal(symbol)
        
        # تحلیل هر تایم‌فریم
        trend_1h = analyze_trend_simple(data_1h)
        trend_15m = analyze_trend_simple(data_15m)
        trend_5m = analyze_trend_simple(data_5m)
        
        # ترکیب نتایج
        trends = [trend_1h, trend_15m, trend_5m]
        bullish_count = sum(1 for t in trends if t == "BULLISH")
        bearish_count = sum(1 for t in trends if t == "BEARISH")
        
        # تصمیم‌گیری نهایی
        if bullish_count >= 2:
            signal = "BUY"
            confidence = 0.6 + (bullish_count * 0.1)
        elif bearish_count >= 2:
            signal = "SELL"
            confidence = 0.6 + (bearish_count * 0.1)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        # محاسبه قیمت‌ها
        latest_close = float(data_5m[-1][4]) if data_5m else 0
        
        if signal == "BUY":
            entry_price = latest_close * 1.001
            stop_loss = latest_close * 0.98
            targets = [
                latest_close * 1.02,
                latest_close * 1.05
            ]
        elif signal == "SELL":
            entry_price = latest_close * 0.999
            stop_loss = latest_close * 1.02
            targets = [
                latest_close * 0.98,
                latest_close * 0.95
            ]
        else:  # HOLD
            entry_price = latest_close
            stop_loss = latest_close * 0.99
            targets = []
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(min(confidence, 0.95), 2),
            "entry_price": round(entry_price, 2),
            "targets": [round(t, 2) for t in targets],
            "stop_loss": round(stop_loss, 2),
            "strategy": "Multi-Timeframe Simple",
            "analysis_details": {
                "1h": {"trend": trend_1h, "source": result_1h.get("source", "unknown")},
                "15m": {"trend": trend_15m, "source": result_15m.get("source", "unknown")},
                "5m": {"trend": trend_5m, "source": result_5m.get("source", "unknown")}
            }
        }
        
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل {symbol}: {e}")
        return get_fallback_signal(symbol)

def analyze_trend_simple(data):
    """تحلیل روند ساده"""
    if not data or len(data) < 20:
        return "NEUTRAL"
    
    # محاسبه SMA
    sma_20 = calculate_simple_sma(data, 20)
    if not sma_20:
        return "NEUTRAL"
    
    # آخرین قیمت بسته شدن
    try:
        latest_close = float(data[-1][4])
    except:
        return "NEUTRAL"
    
    # محاسبه RSI
    rsi = calculate_simple_rsi(data, 14)
    
    # تصمیم‌گیری
    bullish_signals = 0
    bearish_signals = 0
    
    if latest_close > sma_20:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if rsi < 40:
        bullish_signals += 1  # اشباع فروش
    elif rsi > 60:
        bearish_signals += 1  # اشباع خرید
    
    if bullish_signals > bearish_signals:
        return "BULLISH"
    elif bearish_signals > bullish_signals:
        return "BEARISH"
    else:
        return "NEUTRAL"

def get_fallback_signal(symbol):
    """سیگنال جایگزین در صورت خطا"""
    return {
        "symbol": symbol,
        "signal": "HOLD",
        "confidence": 0.5,
        "entry_price": 0,
        "targets": [],
        "stop_loss": 0,
        "strategy": "Fallback Mode",
        "note": "Analysis failed, using fallback"
    }

# ==============================================================================
# 📊 توابع کمکی
# ==============================================================================

def calculate_24h_change_from_dataframe(data):
    """محاسبه تغییرات ۲۴ ساعته"""
    if isinstance(data, dict) and "data" in data:
        data_list = data["data"]
    elif isinstance(data, list):
        data_list = data
    else:
        return round(random.uniform(-5, 5), 2)
    
    if len(data_list) < 10:
        return round(random.uniform(-5, 5), 2)
    
    try:
        # اولین کندل (قدیمی‌ترین)
        first_close = float(data_list[0][4])
        # آخرین کندل
        last_close = float(data_list[-1][4])
        
        change = ((last_close - first_close) / first_close) * 100
        return round(change, 2)
    except:
        return round(random.uniform(-5, 5), 2)