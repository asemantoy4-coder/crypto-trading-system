

بله، فایل `utils.py` شما را بررسی کردم. این فایل توابع پایه RSI و SMA را دارد، اما محاسبه **واگرایی (Divergence)** و محاسبه **سری کامل مقادیر RSI** (برای بررسی نقاط گذشته) را ندارد.

من کد زیر را با ادغام کردن منطق واقعی RSI و تشخیص واگرایی (که در مرحله قبل برای `main.py` نوشتم) در فایل `utils.py` بازنویسی کردم.

**تغییرات مهم در این نسخه:**
1.  اضافه شد `calculate_rsi_series`: این تابع لیست کامل RSI‌ها را محاسبه می‌کند.
2.  اضافه شد `detect_divergence`: این تابع پیک‌ها و دره‌ها را بررسی کرده و واگرایی صعودی/نزولی را پیدا می‌کند.
3.  اصلاح شد `calculate_simple_rsi`: اکنون از منطق سری RSI (Series) استفاده می‌کند که دقیق‌تر است.
4.  به‌روزرسانی `__all__`: توابع جدید به لیست صادرات اضافه شدند تا `main.py` بتواند آن‌ها را صدا بزند.

این فایل کامل `utils.py` اصلاح شده است:

```python
# api/utils.py - نسخه 7.5.0 با پشتیبانی از ایچیموکو پیشرفته، RSI واقعی و واگرایی
"""
Utility Functions - Render Optimized Version
با پشتیبانی کامل از تحلیل تکنیکال برای اسکالپ و سوئینگ
نسخه کامل با ایچیموکو پیشرفته، خطوط کیفیت و طلایی
+ تشخیص واقعی واگرایی (Divergence) و محاسبه سری RSI
"""

import requests
import logging
import random
from datetime import datetime, timedelta
import time
import json

logger = logging.getLogger(__name__)

# ==============================================================================
# 📊 تابع اصلی دریافت داده (ساده‌شده) - اصلاح شده برای سازگاری
# ==============================================================================

def get_market_data_with_fallback(symbol, interval="5m", limit=100, return_source=False):
    """
    دریافت داده‌های بازار - نسخه سازگار
    
    Parameters:
    -----------
    symbol : str
        نماد معاملاتی
    interval : str
        تایم‌فریم
    limit : int
        تعداد کندل‌ها
    return_source : bool
        اگر True باشد، دیکشنری با داده و source برمی‌گرداند
    
    Returns:
    --------
    list or dict
        لیست کندل‌ها یا دیکشنری با داده و source
    """
    logger.info(f"📊 دریافت داده برای {symbol} ({interval})")
    
    source = None
    data = None
    
    # ۱. تلاش برای دریافت از Binance
    try:
        data = get_binance_klines_simple(symbol, interval, limit)
        if data:
            logger.info(f"✅ داده از Binance دریافت شد: {len(data)} کندل")
            source = "binance"
    except Exception as e:
        logger.warning(f"⚠️ خطا در Binance: {e}")
    
    # ۲. اگر Binance جواب نداد، تلاش برای دریافت از LBank
    if not data:
        try:
            data = get_lbank_data_simple(symbol, interval, limit)
            if data:
                logger.info(f"✅ داده از LBank دریافت شد: {len(data)} کندل")
                source = "lbank"
        except Exception as e:
            logger.warning(f"⚠️ خطا در LBank: {e}")
    
    # ۳. اگر هیچ کدام جواب نداد، داده Mock
    if not data:
        logger.info(f"🧪 استفاده از داده Mock برای {symbol}")
        data = generate_mock_data_simple(symbol, limit)
        source = "mock"
    
    # بر اساس پارامتر return_source تصمیم بگیریم چه چیزی برگردانیم
    if return_source:
        return {
            "data": data,
            "source": source,
            "success": source != "mock"
        }
    else:
        return data

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
        'BTCUSDT': 88271.42, 'ETHUSDT': 3450.12, 'BNBUSDT': 590.54,
        'SOLUSDT': 175.98, 'XRPUSDT': 0.51234, 'ADAUSDT': 0.43210,
        'DOGEUSDT': 0.12345, 'SHIBUSDT': 0.00002345,
        'EURUSD': 1.08745, 'XAUUSD': 2387.65, 'PAXGUSDT': 2387.65,
        'DEFAULT': 100.50
    }
    
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    mock_data = []
    current_time = int(time.time() * 1000)
    
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)  # 5 دقیقه فاصله
        
        # شبیه‌سازی حرکت قیمت واقعی‌تر
        change = random.uniform(-0.015, 0.015)  # ±1.5%
        price = base_price * (1 + change)
        
        mock_candle = [
            timestamp,  # open time
            str(price * random.uniform(0.998, 1.000)),  # open
            str(price * random.uniform(1.000, 1.003)),  # high
            str(price * random.uniform(0.997, 1.000)),  # low
            str(price),  # close
            str(random.uniform(1000, 10000)),  # volume
            timestamp + 300000,  # close time
            "0", "0", "0", "0", "0"  # سایر فیلدها
        ]
        
        mock_data.append(mock_candle)
    
    return mock_data

# ==============================================================================
# 📈 تابع محاسبه RSI Series (واقعی) - جدید
# ==============================================================================

def calculate_rsi_series(closes, period=14):
    """
    محاسبه لیست کامل مقادیر RSI با استفاده از روش Wilder's Smoothing.
    ورودی: لیست قیمت‌های بسته شده (Closes).
    خروجی: لیست مقادیر RSI.
    """
    if len(closes) < period + 1:
        return [50] * len(closes)
    
    rsi_values = [50] * period  # پر کردن داده‌های اولی
    
    gains = 0.0
    losses = 0.0
    
    # محاسبه میانگین اولیه
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses += abs(change)
    
    avg_gain = gains / period
    avg_loss = losses / period
    
    # محاسبه اولین RSI
    if avg_loss == 0:
        rsi_values.append(100)
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    # محاسبه RSI برای بقیه کندل‌ها با روش Smoothing
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        
        if change > 0:
            gain = change
            loss = 0
        else:
            gain = 0
            loss = abs(change)
        
        # محاسبه میانگین‌های هموار شده
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            rsi_val = 100
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi_val)
    
    return rsi_values

def detect_divergence(prices, rsi_values, lookback=5):
    """
    تشخیص واگرایی صعودی و نزولی با بررسی پیک‌ها و دره‌ها.
    ورودی: لیست قیمت‌ها، لیست مقادیر RSI، تعداد کندل‌های بررسی (Lookback).
    خروجی: دیکشنری شامل وضعیت واگرایی.
    """
    divergence = {
        "detected": False,
        "type": None,  # "bullish" (صعودی), "bearish" (نزولی)
        "strength": None # "weak", "moderate", "strong"
    }
    
    if len(prices) < lookback * 3 or len(rsi_values) < lookback * 3:
        return divergence
    
    # تابع کمکی برای پیدا کردن قله‌ها و دره‌ها (Local Extrema)
    def find_pivots(data, window=3):
        pivots = []
        for i in range(window, len(data) - window):
            is_peak = True
            is_trough = True
            
            for j in range(1, window + 1):
                if data[i] <= data[i - j] or data[i] <= data[i + j]:
                    is_peak = False
                if data[i] >= data[i - j] or data[i] >= data[i + j]:
                    is_trough = False
            
            if is_peak:
                pivots.append({"index": i, "value": data[i], "type": "peak"})
            elif is_trough:
                pivots.append({"index": i, "value": data[i], "type": "trough"})
        return pivots

    price_pivots = find_pivots(prices, window=lookback)
    rsi_pivots = find_pivots(rsi_values, window=lookback)
    
    if len(price_pivots) < 2 or len(rsi_pivots) < 2:
        return divergence

    # بررسی واگرایی بر اساس آخرین پیک یا دره مشترک
    # ما آخرین پیوت مشابه را پیدا می‌کنیم و با پیوت قبل از آن مقایسه می‌کنیم
    
    # 1. پیدا کردن آخرین نقطه عطف مشترک (Peak یا Trough)
    last_price_pivot = price_pivots[-1]
    last_rsi_pivot = rsi_pivots[-1]
    
    # اگر نوع آخرین پیوت برابر نیست (مثلاً قیمت پیک است ولی RSI دره)، مقایسه معناداری ندارد
    if last_price_pivot['type'] != last_rsi_pivot['type']:
        return divergence

    # پیدا کردن پیوت قبل از آن
    prev_price_pivot = None
    prev_rsi_pivot = None
    
    # جستجو در پیوت‌های قبلی برای پیدا کردن همان نوع (Peak یا Trough)
    for pp in reversed(price_pivots[:-1]):
        if pp['type'] == last_price_pivot['type']:
            prev_price_pivot = pp
            break
    
    for rp in reversed(rsi_pivots[:-1]):
        if rp['type'] == last_rsi_pivot['type']:
            prev_rsi_pivot = rp
            break

    if not prev_price_pivot or not prev_rsi_pivot:
        return divergence

    # 2. بررسی شرایط واگرایی
    # اگر نوع پیوت 'peak' (قله) بود:
    if last_price_pivot['type'] == 'peak':
        # واگرایی نزولی (Bearish): قیمت سقف جدید زده (Higher High) اما RSI سقف پایین‌تر (Lower High) زده
        if last_price_pivot['value'] > prev_price_pivot['value'] and last_rsi_pivot['value'] < prev_rsi_pivot['value']:
            divergence["detected"] = True
            divergence["type"] = "bearish"
            divergence["strength"] = "strong"
            
    # اگر نوع پیوت 'trough' (دره) بود:
    elif last_price_pivot['type'] == 'trough':
        # واگرایی صعودی (Bullish): قیمت کف جدید زده (Lower Low) اما RSI کف بالاتر (Higher Low) زده
        if last_price_pivot['value'] < prev_price_pivot['value'] and last_rsi_pivot['value'] > prev_rsi_pivot['value']:
            divergence["detected"] = True
            divergence["type"] = "bullish"
            divergence["strength"] = "strong"

    return divergence

# ==============================================================================
# 📈 توابع تحلیل تکنیکال پایه (ساده‌شده) - اصلاح شده
# ==============================================================================

def calculate_simple_sma(data, period=20):
    """
    محاسبه SMA ساده (بدون pandas)
    
    Parameters:
    -----------
    data : list
        لیست کندل‌ها از API صرافی
    period : int
        دوره SMA (پیش‌فرض: 20)
    
    Returns:
    --------
    float or None
        مقدار SMA یا None اگر داده کافی نباشد
    """
    if not data or len(data) < period:
        return None
    
    closes = []
    for candle in data[-period:]:  # آخرین period کندل
        try:
            closes.append(float(candle[4]))  # index 4 = close price
        except (IndexError, ValueError, TypeError):
            closes.append(0)
    
    return sum(closes) / len(closes) if closes else 0

def calculate_simple_rsi(data, period=14):
    """
    محاسبه RSI ساده (بدون pandas) - به‌روزرسانی شده با RSI Series
    این تابع اکنون همان محاسبه واقعی سری RSI را استفاده می‌کند.
    
    Parameters:
    -----------
    data : list
        لیست کندل‌ها از API صرافی
    period : int
        دوره RSI (پیش‌فرض: 14)
    
    Returns:
    --------
    float
        مقدار RSI بین 0 تا 100
    """
    if not data or len(data) <= period:
        return 50  # مقدار خنثی
    
    # استخراج قیمت‌های بسته شده
    closes = []
    for candle in data[-(period+1):]:  # برای period+1 کندل
        try:
            closes.append(float(candle[4]))
        except (IndexError, ValueError, TypeError):
            closes.append(0)
    
    # استفاده از تابع سری RSI برای محاسبه دقیق
    rsi_series = calculate_rsi_series(closes, period)
    
    # برگرداندن آخرین مقدار
    return round(rsi_series[-1], 2)

def calculate_macd_simple(data, fast=12, slow=26, signal=9):
    """
    محاسبه MACD ساده (بدون pandas) - اصلاح شده
    
    Parameters:
    -----------
    data : list
        لیست کندل‌ها
    fast : int
        دوره EMA سریع
    slow : int
        دوره EMA کند
    signal : int
        دوره خط سیگنال
    
    Returns:
    --------
    dict
        {'macd': مقدار MACD, 'signal': خط سیگنال, 'histogram': هیستوگرام}
    """
    if not data or len(data) < slow + signal:
        return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    # محاسبه EMA سریع و کند
    closes = []
    for candle in data[-(slow + signal):]:
        try:
            closes.append(float(candle[4]))
        except (IndexError, ValueError, TypeError):
            continue
    
    if len(closes) < slow:
        return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    # محاسبه EMA واقعی
    def calculate_ema(prices, period):
        if not prices or len(prices) < period:
            return 0
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # SMA برای شروع
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    ema_fast = calculate_ema(closes[-fast:], fast)
    ema_slow = calculate_ema(closes, slow)
    
    macd_line = ema_fast - ema_slow
    
    # محاسبه خط سیگنال (EMA از MACD)
    # برای سادگی، از یک تقریب استفاده می‌کنیم
    macd_values = [macd_line]  # در واقع باید history داشته باشیم
    signal_line = macd_line * 0.9  # تقریب ساده
    
    histogram = macd_line - signal_line
    
    return {
        'macd': round(macd_line, 4),
        'signal': round(signal_line, 4),
        'histogram': round(histogram, 4)
    }

# ==============================================================================
# ☁️ توابع ایچیموکو پیشرفته برای اسکالپ (جدید)
# ==============================================================================

def calculate_ichimoku_components(data, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
    """
    محاسبه اجزای ایچیموکو کینکو هایو
    
    Parameters:
    -----------
    data : list
        لیست کندل‌ها
    tenkan_period : int
        دوره تنکان سن (خط تبدیل)
    kijun_period : int
        دوره کیجون سن (خط پایه)
    senkou_b_period : int
        دوره سنکو اسپن B
    displacement : int
        جابجایی ابر (پیش‌فرض ۲۶)
    
    Returns:
    --------
    dict
        تمام اجزای ایچیموکو
    """
    if not data or len(data) < max(kijun_period, senkou_b_period, displacement) + 10:
        return None
    
    # تبدیل داده به فرمت قابل پردازش
    highs = []
    lows = []
    closes = []
    
    for candle in data:
        try:
            highs.append(float(candle[2]))  # high
            lows.append(float(candle[3]))   # low
            closes.append(float(candle[4])) # close
        except (IndexError, ValueError, TypeError):
            continue
    
    if len(highs) < max(kijun_period, senkou_b_period) + displacement:
        return None
    
    # محاسبه تنکان سن (Tenkan-sen) - خط تبدیل
    tenkan_sen = calculate_ichimoku_line(highs, lows, tenkan_period)
    
    # محاسبه کیجون سن (Kijun-sen) - خط پایه
    kijun_sen = calculate_ichimoku_line(highs, lows, kijun_period)
    
    # سنکو اسپن A (Senkou Span A) - لبه پیشرو A
    senkou_span_a = []
    for i in range(len(tenkan_sen)):
        if i >= displacement:
            senkou_span_a.append((tenkan_sen[i] + kijun_sen[i]) / 2)
        else:
            senkou_span_a.append(None)
    
    # سنکو اسپن B (Senkou Span B) - لبه پیشرو B
    senkou_span_b = calculate_ichimoku_line(highs, lows, senkou_b_period)
    senkou_span_b = [None] * displacement + senkou_span_b[:-displacement] if len(senkou_span_b) > displacement else [None] * len(highs)
    
    # چیکو اسپن (Chikou Span) - خط تاخیری
    chikou_span = closes[:-displacement] + [None] * displacement if len(closes) > displacement else [None] * len(closes)
    
    # ابر کومو (Kumo Cloud)
    cloud_top = []
    cloud_bottom = []
    for i in range(len(senkou_span_a)):
        if senkou_span_a[i] is not None and senkou_span_b[i] is not None:
            cloud_top.append(max(senkou_span_a[i], senkou_span_b[i]))
            cloud_bottom.append(min(senkou_span_a[i], senkou_span_b[i]))
        else:
            cloud_top.append(None)
            cloud_bottom.append(None)
    
    # کیفیت لاین (Quality Line) - میانگین موزون
    quality_line = calculate_quality_line(closes, highs, lows, period=14)
    
    # خط طلایی (Golden Line) - ترکیب خاص
    golden_line = calculate_golden_line(tenkan_sen, kijun_sen, quality_line)
    
    # قدرت روند (Trend Power)
    trend_power = calculate_trend_power(tenkan_sen, kijun_sen, closes)
    
    return {
        'tenkan_sen': tenkan_sen[-1] if tenkan_sen else None,
        'kijun_sen': kijun_sen[-1] if kijun_sen else None,
        'senkou_span_a': senkou_span_a[-1] if senkou_span_a else None,
        'senkou_span_b': senkou_span_b[-1] if senkou_span_b else None,
        'chikou_span': chikou_span[-1] if chikou_span else None,
        'cloud_top': cloud_top[-1] if cloud_top else None,
        'cloud_bottom': cloud_bottom[-1] if cloud_bottom else None,
        'quality_line': quality_line[-1] if quality_line else None,
        'golden_line': golden_line[-1] if golden_line else None,
        'trend_power': trend_power,
        'in_cloud': cloud_bottom[-1] <= closes[-1] <= cloud_top[-1] if cloud_bottom[-1] and cloud_top[-1] and closes[-1] else False,
        'above_cloud': closes[-1] > cloud_top[-1] if cloud_top[-1] and closes[-1] else False,
        'below_cloud': closes[-1] < cloud_bottom[-1] if cloud_bottom[-1] and closes[-1] else False,
        'cloud_thickness': (cloud_top[-1] - cloud_bottom[-1]) / cloud_bottom[-1] * 100 if cloud_top[-1] and cloud_bottom[-1] and cloud_bottom[-1] > 0 else 0,
        'current_price': closes[-1] if closes else None
    }

def calculate_ichimoku_line(highs, lows, period):
    """محاسبه خط ایچیموکو (HH + LL) / 2"""
    result = []
    for i in range(len(highs)):
        if i >= period - 1:
            highest_high = max(highs[i-period+1:i+1])
            lowest_low = min(lows[i-period+1:i+1])
            result.append((highest_high + lowest_low) / 2)
        else:
            result.append(None)
    return result

def calculate_quality_line(closes, highs, lows, period=14):
    """
    خط کیفیت (Quality Line) - اندیکاتور اختصاصی
    ترکیبی از حجم، مومنتوم و نوسان
    """
    if len(closes) < period:
        return [None] * len(closes)
    
    quality = []
    for i in range(len(closes)):
        if i >= period - 1:
            # محاسبه میانگین وزنی با تاکید بر حرکات قوی
            weighted_sum = 0
            weight_sum = 0
            
            for j in range(period):
                idx = i - j
                if idx <= 0:
                    continue
                    
                price_change = abs(closes[idx] - closes[idx-1])
                range_size = highs[idx] - lows[idx] if highs[idx] > lows[idx] else 0.001
                
                # وزن بیشتر برای کندل‌های با رنج بزرگ
                weight = range_size / (closes[idx] + 0.001)
                weighted_sum += closes[idx] * weight
                weight_sum += weight
            
            quality.append(weighted_sum / weight_sum if weight_sum > 0 else closes[i])
        else:
            quality.append(None)
    
    return quality

def calculate_golden_line(tenkan_sen, kijun_sen, quality_line):
    """
    خط طلایی (Golden Line) - ترکیب پیشرفته
    فرمول: (تنکان × ۰.۴ + کیجون × ۰.۳ + کیفیت × ۰.۳)
    """
    if not tenkan_sen or not kijun_sen or not quality_line:
        return None
    
    golden = []
    min_len = min(len(tenkan_sen), len(kijun_sen), len(quality_line))
    
    for i in range(min_len):
        if tenkan_sen[i] is not None and kijun_sen[i] is not None and quality_line[i] is not None:
            # میانگین وزنی برای خط طلایی
            value = (tenkan_sen[i] *0.4 + 
                    kijun_sen[i] *0.3 + 
                    quality_line[i] *0.3)
            golden.append(value)
        else:
            golden.append(None)
    
    return golden

def calculate_trend_power(tenkan_sen, kijun_sen, closes):
    """
    محاسبه قدرت روند بر اساس ایچیموکو
    بازگشت: ۰ تا ۱۰۰ (قدرت روند)
    """
    if not tenkan_sen or not kijun_sen or not closes:
        return 50
    
    try:
        # بررسی داده‌های کافی
        valid_tenkan = [v for v in tenkan_sen if v is not None]
        valid_kijun = [v for v in kijun_sen if v is not None]
        
        if len(valid_tenkan) < 2 or len(valid_kijun) < 2:
            return 50
        
        # آخرین مقادیر
        last_tenkan = valid_tenkan[-1]
        last_kijun = valid_kijun[-1]
        last_close = closes[-1]
        
        if last_kijun == 0:
            return 50
        
        # ۱. فاصله تنکان-کیجون
        tk_distance = abs(last_tenkan - last_kijun) / last_kijun * 100
        
        # ۲. موقعیت قیمت نسبت به خطوط
        above_tenkan = 1 if last_close > last_tenkan else -1
        above_kijun = 1 if last_close > last_kijun else -1
        
        # ۳. زاویه روند (تقریبی)
        if len(valid_tenkan) > 5 and len(valid_kijun) > 5:
            tenkan_trend = (valid_tenkan[-1] - valid_tenkan[-5]) / valid_tenkan[-5] * 100 if valid_tenkan[-5] != 0 else 0
            kijun_trend = (valid_kijun[-1] - valid_kijun[-5]) / valid_kijun[-5] * 100 if valid_kijun[-5] != 0 else 0
            avg_trend = (tenkan_trend + kijun_trend) / 2
        else:
            avg_trend = 0
        
        # محاسبه نهایی قدرت روند
        trend_power = 50  # خنثی
        
        # اضافه کردن امتیازات
        trend_power += min(tk_distance * 2, 20)  # حداکثر ۲۰ امتیاز برای فاصله
        
        if above_tenkan == above_kijun == 1:
            trend_power += 15  # قیمت بالای هر دو خط
        elif above_tenkan == above_kijun == -1:
            trend_power -= 15  # قیمت زیر هر دو خط
        
        trend_power += avg_trend * 10  # جهت روند
        
        # محدود کردن به بازه ۰-۱۰۰
        trend_power = max(0, min(100, trend_power))
        
        return round(trend_power, 1)
    
    except Exception as e:
        logger.error(f"خطا در محاسبه قدرت روند: {e}")
        return 50

def analyze_ichimoku_scalp_signal(ichimoku_data):
    """
    تحلیل سیگنال اسکالپ با ایچیموکو پیشرفته
    
    Returns:
    --------
    dict
        سیگنال تحلیل
    """
    if not ichimoku_data:
        return {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reason': 'ایچیموکو داده ناکافی',
            'levels': {},
            'trend_power': 50
        }
    
    # استخراج داده‌ها
    tenkan = ichimoku_data.get('tenkan_sen')
    kijun = ichimoku_data.get('kijun_sen')
    cloud_top = ichimoku_data.get('cloud_top')
    cloud_bottom = ichimoku_data.get('cloud_bottom')
    quality_line = ichimoku_data.get('quality_line')
    golden_line = ichimoku_data.get('golden_line')
    trend_power = ichimoku_data.get('trend_power', 50)
    current_price = ichimoku_data.get('current_price')
    
    if None in [tenkan, kijun, current_price] or current_price <= 0:
        return {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reason': 'داده‌های ایچیموکو ناقص',
            'levels': {},
            'trend_power': trend_power
        }
    
    # محاسبه شرایط
    tenkan_above_kijun = tenkan > kijun
    price_above_tenkan = current_price > tenkan
    price_above_kijun = current_price > kijun
    price_above_cloud = current_price > cloud_top if cloud_top else False
    price_in_cloud = cloud_bottom <= current_price <= cloud_top if cloud_bottom and cloud_top else False
    
    # تحلیل سیگنال
    signal = 'HOLD'
    confidence = 0.5
    reason = "بازار خنثی"
    
    bullish_conditions = 0
    bearish_conditions = 0
    
    # شرایط خرید (Bullish)
    if tenkan_above_kijun:
        bullish_conditions += 1
        reason = "تنکان بالای کیجون"
    
    if price_above_tenkan and price_above_kijun:
        bullish_conditions += 1
        reason = "قیمت بالای هر دو خط"
    
    if price_above_cloud:
        bullish_conditions += 2  # وزن بیشتر
        reason = "قیمت بالای ابر کومو"
    
    # شرایط فروش (Bearish)
    if not tenkan_above_kijun:
        bearish_conditions += 1
        reason = "تنکان زیر کیجون"
    
    if not price_above_tenkan and not price_above_kijun:
        bearish_conditions += 1
        reason = "قیمت زیر هر دو خط"
    
    if cloud_bottom and current_price < cloud_bottom:
        bearish_conditions += 2
        reason = "قیمت زیر ابر کومو"
    
    # تصمیم‌گیری نهایی
    if bullish_conditions >= 3:
        signal = 'BUY'
        confidence = min(0.5 + (bullish_conditions * 0.1) + (trend_power / 200), 0.95)
        reason = f"{reason} (سیگنال خرید)"
    elif bearish_conditions >= 3:
        signal = 'SELL'
        confidence = min(0.5 + (bearish_conditions * 0.1) + ((100 - trend_power) / 200), 0.95)
        reason = f"{reason} (سیگنال فروش)"
    else:
        signal = 'HOLD'
        confidence = 0.5
    
    # اگر در ابر هستیم، اطمینان کم‌تر
    if price_in_cloud:
        confidence *= 0.7
        reason += " - درون ابر"
    
    # سطوح کلیدی
    levels = {
        'tenkan_sen': round(tenkan, 4),
        'kijun_sen': round(kijun, 4),
        'cloud_top': round(cloud_top, 4) if cloud_top else None,
        'cloud_bottom': round(cloud_bottom, 4) if cloud_bottom else None,
        'quality_line': round(quality_line, 4) if quality_line else None,
        'golden_line': round(golden_line, 4) if golden_line else None,
        'support_level': round(min(tenkan, kijun, cloud_bottom if cloud_bottom else tenkan), 4),
        'resistance_level': round(max(tenkan, kijun, cloud_top if cloud_top else kijun), 4),
        'current_price': round(current_price, 4)
    }
    
    # اگر خط طلایی داریم، سیگنال قوی‌تر
    if golden_line:
        if signal == 'BUY' and current_price > golden_line:
            confidence = min(confidence * 1.2, 0.95)
            reason += " + تأیید خط طلایی"
        elif signal == 'SELL' and current_price < golden_line:
            confidence = min(confidence * 1.2, 0.95)
            reason += " + تأیید خط طلایی"
    
    # اضافه کردن تفسیر قدرت روند
    trend_interpretation = "روند قوی" if trend_power >= 70 else \
                          "روند متوسط" if trend_power >= 60 else \
                          "روند ضعیف" if trend_power >= 40 else "بدون روند"
    
    return {
        'signal': signal,
        'confidence': round(confidence, 3),
        'reason': reason,
        'levels': levels,
        'trend_power': trend_power,
        'trend_interpretation': trend_interpretation,
        'cloud_thickness': ichimoku_data.get('cloud_thickness', 0),
        'in_cloud': price_in_cloud,
        'cloud_color': 'سبز' if cloud_top and cloud_bottom and cloud_top > cloud_bottom else 'قرمز'
    }

def get_ichimoku_scalp_signal(data, timeframe="5m"):
    """
    دریافت سیگنال اسکالپ با ایچیموکو
    (برای استفاده مستقیم در سایر ماژول‌ها)
    """
    try:
        if not data or len(data) < 60:
            return None
        
        # محاسبه ایچیموکو
        ichimoku = calculate_ichimoku_components(data)
        
        if not ichimoku:
            return None
        
        # تحلیل سیگنال
        signal = analyze_ichimoku_scalp_signal(ichimoku)
        
        # اضافه کردن اطلاعات تایم‌فریم
        signal['timeframe'] = timeframe
        
        return signal
        
    except Exception as e:
        logger.error(f"خطا در دریافت سیگنال ایچیموکو: {e}")
        return None

# ==============================================================================
# 🚀 موتور اصلی تحلیل (ساده‌شده)
# ==============================================================================

def analyze_with_multi_timeframe_strategy(symbol):
    """
    تحلیل چندزمانی - نسخه بهینه برای Render
    
    Parameters:
    -----------
    symbol : str
        نماد معاملاتی (مثلاً BTCUSDT)
    
    Returns:
    --------
    dict
        تحلیل کامل با سیگنال، اطمینان، قیمت ورود، تارگت‌ها و استاپ‌لاس
    """
    logger.info(f"🤖 تحلیل {symbol}")
    
    try:
        # دریافت داده از تایم‌فریم‌های مختلف
        result_1h = get_market_data_with_fallback(symbol, "1h", 50, return_source=True)
        result_15m = get_market_data_with_fallback(symbol, "15m", 50, return_source=True)
        result_5m = get_market_data_with_fallback(symbol, "5m", 50, return_source=True)
        
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
        try:
            latest_close = float(data_5m[-1][4])
        except (IndexError, ValueError, TypeError):
            latest_close = 100.0
        
        if latest_close <= 0:
            latest_close = 100.0
        
        # ✅ استفاده از تابع مرکزی برای محاسبه تارگت‌ها
        if signal == "BUY":
            entry_price = latest_close * 1.001
            stop_loss = latest_close * 0.98
            targets = [
                latest_close * 1.02,  # 2% بالاتر
                latest_close * 1.05   # 5% بالاتر
            ]
        elif signal == "SELL":
            entry_price = latest_close * 0.999
            stop_loss = latest_close * 1.02
            targets = [
                latest_close * 0.98,  # 2% پایین‌تر
                latest_close * 0.95   # 5% پایین‌تر
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
    """
    تحلیل روند ساده بر اساس SMA و RSI - اصلاح شده
    
    Parameters:
    -----------
    data : list
        لیست کندل‌ها
    
    Returns:
    --------
    str
        "BULLISH", "BEARISH", یا "NEUTRAL"
    """
    if not data or len(data) < 20:
        return "NEUTRAL"
    
    # محاسبه SMA
    sma_20 = calculate_simple_sma(data, 20)
    if sma_20 is None or sma_20 == 0:
        return "NEUTRAL"
    
    # آخرین قیمت بسته شدن
    try:
        latest_close = float(data[-1][4])
    except (IndexError, ValueError, TypeError):
        return "NEUTRAL"
    
    if latest_close <= 0:
        return "NEUTRAL"
    
    # محاسبه RSI (استفاده از نسخه جدید سری)
    rsi = calculate_simple_rsi(data, 14)
    
    # محاسبه MACD
    macd_data = calculate_macd_simple(data)
    
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
    
    if macd_data['histogram'] > 0:
        bullish_signals += 1
    elif macd_data['histogram'] < 0:
        bearish_signals += 1
    
    if bullish_signals > bearish_signals:
        return "BULLISH"
    elif bearish_signals > bullish_signals:
        return "BEARISH"
    else:
        return "NEUTRAL"

def get_fallback_signal(symbol):
    """
    سیگنال جایگزین در صورت خطا - اصلاح شده
    
    Parameters:
    -----------
    symbol : str
        نماد معاملاتی
    
    Returns:
    --------
    dict
        سیگنال fallback
    """
    # قیمت‌های پایه واقعی‌تر
    base_prices = {
        'BTCUSDT': 88271.42,
        'ETHUSDT': 3450.12,
        'BNBUSDT': 590.54,
        'SOLUSDT': 175.98,
        'DEFAULT': 100.50
    }
    
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    
    # شانس بیشتر برای HOLD
    signals = ["BUY", "SELL", "HOLD"]
    weights = [0.35, 0.35, 0.30]
    signal = random.choices(signals, weights=weights)[0]
    
    if signal == "HOLD":
        confidence = round(random.uniform(0.5, 0.7), 2)
    else:
        confidence = round(random.uniform(0.65, 0.85), 2)
    
    entry_price = round(base_price * random.uniform(0.99, 1.01), 2)
    
    if signal == "BUY":
        targets = [
            round(entry_price * 1.02, 2),  # 2% بالاتر
            round(entry_price * 1.05, 2)   # 5% بالاتر
        ]
        stop_loss = round(entry_price * 0.98, 2)  # 2% پایین‌تر
    elif signal == "SELL":
        targets = [
            round(entry_price * 0.98, 2),  # 2% پایین‌تر
            round(entry_price * 0.95, 2)   # 5% پایین‌تر
        ]
        stop_loss = round(entry_price * 1.02, 2)  # 2% بالاتر
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
        "strategy": "Fallback Mode",
        "note": "Analysis failed, using fallback"
    }

# ==============================================================================
# 📊 توابع کمکی
# ==============================================================================

def calculate_24h_change_from_dataframe(data):
    """
    محاسبه تغییرات ۲۴ ساعته
    
    Parameters:
    -----------
    data : list or dict
        داده‌های بازار
    
    Returns:
    --------
    float
        درصد تغییر
    """
    # اگر data دیکشنری است، استخراج کن
    if isinstance(data, dict) and "data" in data:
        data_list = data["data"]
    elif isinstance(data, list):
        data_list = data
    else:
        return round(random.uniform(-5, 5), 2)
    
    if not isinstance(data_list, list) or len(data_list) < 10:
        return round(random.uniform(-5, 5), 2)
    
    try:
        # اولین کندل (قدیمی‌ترین)
        first_close = float(data_list[0][4])
        # آخرین کندل
        last_close = float(data_list[-1][4])
        
        if first_close <= 0:
            return 0.0
        
        change = ((last_close - first_close) / first_close) * 100
        return round(change, 2)
    except (IndexError, ValueError, TypeError, ZeroDivisionError):
        return round(random.uniform(-5, 5), 2)

def analyze_scalp_conditions(data, timeframe):
    """
    تحلیل شرایط اسکالپ برای تایم‌فریم‌های کوتاه - اصلاح شده
    
    Parameters:
    -----------
    data : list
        داده‌های کندل
    timeframe : str
        تایم‌فریم (1m, 5m, 15m)
    
    Returns:
    --------
    dict
        تحلیل شرایط اسکالپ
    """
    if not data or len(data) < 20:
        return {
            "condition": "NEUTRAL",
            "rsi": 50,
            "sma_20": 0,
            "volatility": 0,
            "reason": "Insufficient data"
        }
    
    # محاسبه اندیکاتورها
    rsi = calculate_simple_rsi(data, 14)
    sma_20 = calculate_simple_sma(data, 20)
    
    # ✅ چک کردن None
    if sma_20 is None:
        sma_20 = 0
    
    try:
        latest_close = float(data[-1][4])
        prev_close = float(data[-2][4])
    except (IndexError, ValueError, TypeError):
        latest_close = 0
        prev_close = 0
    
    # نوسان‌پذیری
    volatility = abs((latest_close - prev_close) / prev_close * 100) if prev_close > 0 else 0
    
    # تحلیل شرایط
    condition = "NEUTRAL"
    reason = "Market in equilibrium"
    
    # ✅ چک کردن قیمت معتبر
    if latest_close <= 0 or sma_20 <= 0:
        return {
            "condition": "NEUTRAL",
            "rsi": round(rsi, 1),
            "sma_20": 0,
            "current_price": 0,
            "volatility": 0,
            "reason": "Invalid price data"
        }
    
    # شرایط خرید اسکالپ
    if rsi < 30 and latest_close < sma_20 * 1.01:
        condition = "BULLISH"
        reason = f"Oversold (RSI: {rsi:.1f}), price near SMA20"
    
    # شرایط فروش اسکالپ
    elif rsi > 70 and latest_close > sma_20 * 0.99:
        condition = "BEARISH"
        reason = f"Overbought (RSI: {rsi:.1f}), price near SMA20"
    
    # شرایط Breakout
    elif latest_close > sma_20 * 1.02 and rsi < 60:
        condition = "BULLISH"
        reason = f"Breakout above SMA20, RSI: {rsi:.1f}"
    
    elif latest_close < sma_20 * 0.98 and rsi > 40:
        condition = "BEARISH"
        reason = f"Breakdown below SMA20, RSI: {rsi:.1f}"
    
    # نوسان بالا (برای اسکالپ مناسب است)
    elif volatility > 1.0 and timeframe in ["1m", "5m"]:
        condition = "VOLATILE"
        reason = f"High volatility: {volatility:.2f}%"
    
    return {
        "condition": condition,
        "rsi": round(rsi, 1),
        "sma_20": round(sma_20, 2) if sma_20 else 0,
        "current_price": round(latest_close, 2),
        "volatility": round(volatility, 2),
        "reason": reason
    }

# ==============================================================================
# 🎯 توابع اضافی برای پشتیبانی بهتر
# ==============================================================================

def get_support_resistance_levels(data):
    """
    محاسبه سطوح حمایت و مقاومت ساده
    """
    if not data or len(data) < 20:
        return {"support": 0, "resistance": 0}
    
    highs = []
    lows = []
    
    for candle in data[-20:]:  # آخرین ۲۰ کندل
        try:
            highs.append(float(candle[2]))
            lows.append(float(candle[3]))
        except:
            continue
    
    if not highs or not lows:
        return {"support": 0, "resistance": 0}
    
    # ساده‌ترین روش: میانگین بالاها و پایین‌ها
    resistance = sum(highs) / len(highs)
    support = sum(lows) / len(lows)
    
    return {
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "range_percent": round((resistance - support) / support * 100, 2)
    }

def calculate_volatility(data, period=20):
    """
    محاسبه نوسان‌پذیری
    """
    if not data or len(data) < period:
        return 0
    
    changes = []
    for i in range(1, min(len(data), period)):
        try:
            current = float(data[-i][4])
            previous = float(data[-i-1][4])
            if previous > 0:
                change = abs(current - previous) / previous
                changes.append(change)
        except:
            continue
    
    if not changes:
        return 0
    
    avg_change = sum(changes) / len(changes)
    volatility = avg_change * 100  # به درصد
    
    return round(volatility, 2)

# ==============================================================================
# 🔧 توابع کمکی برای ایچیموکو
# ==============================================================================

def generate_ichimoku_recommendation(signal_data):
    """
    تولید توصیه معاملاتی بر اساس سیگنال ایچیموکو
    """
    signal = signal_data.get('signal', 'HOLD')
    confidence = signal_data.get('confidence', 0.5)
    in_cloud = signal_data.get('in_cloud', False)
    trend_power = signal_data.get('trend_power', 50)
    
    recommendations = {
        'BUY': {
            'high': 'سیگنال خرید قوی - ورود با حجم مناسب',
            'medium': 'سیگنال خرید متوسط - ورود با احتیاط',
            'low': 'سیگنال خرید ضعیف - منتظر تأیید بمانید'
        },
        'SELL': {
            'high': 'سیگنال فروش قوی - خروج یا Short',
            'medium': 'سیگنال فروش متوسط - کاهش موقعیت',
            'low': 'سیگنال فروش ضعیف - منتظر تأیید بمانید'
        },
        'HOLD': {
            'in_cloud': 'قیمت در ابر کومو - منتظر شکست بمانید',
            'low_conf': 'سیگنال نامشخص - از بازار دوری کنید',
            'default': 'موقعیت فعلی را حفظ کنید'
        }
    }
    
    if signal == 'BUY':
        if confidence > 0.7:
            return recommendations['BUY']['high']
        elif confidence > 0.6:
            return recommendations['BUY']['medium']
        else:
            return recommendations['BUY']['low']
    
    elif signal == 'SELL':
        if confidence > 0.7:
            return recommendations['SELL']['high']
        elif confidence > 0.6:
            return recommendations['SELL']['medium']
        else:
            return recommendations['SELL']['low']
    
    else:  # HOLD
        if in_cloud:
            return recommendations['HOLD']['in_cloud']
        elif confidence < 0.4 or trend_power < 30:
            return recommendations['HOLD']['low_conf']
        else:
            return recommendations['HOLD']['default']

# ==============================================================================
# 🎯 توابع ترکیبی برای تحلیل بهتر
# ==============================================================================

def combined_analysis(data, timeframe="5m"):
    """
    تحلیل ترکیبی با چندین اندیکاتور
    """
    if not data or len(data) < 30:
        return None
    
    # محاسبه همه اندیکاتورها
    results = {
        'rsi': calculate_simple_rsi(data, 14),
        'sma_20': calculate_simple_sma(data, 20),
        'macd': calculate_macd_simple(data),
        'ichimoku': get_ichimoku_scalp_signal(data, timeframe),
        'support_resistance': get_support_resistance_levels(data),
        'volatility': calculate_volatility(data, 20)
    }
    
    # تحلیل نهایی
    try:
        latest_price = float(data[-1][4])
    except:
        latest_price = 0
    
    # وزن‌دهی به سیگنال‌ها
    signals = {
        'buy': 0,
        'sell': 0,
        'hold': 0
    }
    
    # RSI
    if results['rsi'] < 30:
        signals['buy'] += 1.5
    elif results['rsi'] > 70:
        signals['sell'] += 1.5
    else:
        signals['hold'] += 1
    
    # SMA
    if latest_price > results['sma_20']:
        signals['buy'] += 1
    else:
        signals['sell'] += 1
    
    # MACD
    if results['macd']['histogram'] > 0:
        signals['buy'] += 1
    else:
        signals['sell'] += 1
    
    # ایچیموکو
    if results['ichimoku']:
        ich_signal = results['ichimoku'].get('signal', 'HOLD')
        if ich_signal == 'BUY':
            signals['buy'] += 2  # وزن بیشتر برای ایچیموکو
        elif ich_signal == 'SELL':
            signals['sell'] += 2
    
    # تصمیم نهایی
    final_signal = max(signals, key=signals.get)
    confidence = signals[final_signal] / sum(signals.values()) if sum(signals.values()) > 0 else 0.5
    
    return {
        'signal': final_signal.upper(),
        'confidence': round(confidence, 3),
        'details': results,
        'price': latest_price,
        'timestamp': datetime.now().isoformat()
    }

# ==============================================================================
# 📝 Version Info
# ==============================================================================

__version__ = "7.5.0"
__author__ = "Crypto AI Trading System"
__description__ = "سیستم تحلیل معاملاتی با پشتیبانی از ایچیموکو پیشرفته و واگرایی واقعی"

# ==============================================================================
# 📦 Export توابع
# ==============================================================================

__all__ = [
    # توابع اصلی
    'get_market_data_with_fallback',
    'analyze_with_multi_timeframe_strategy',
    
    # توابع تحلیل پایه
    'calculate_simple_sma',
    'calculate_simple_rsi', # به‌روزرسانی شده
    'calculate_rsi_series', # جدید - محاسبه لیست RSI
    'detect_divergence', # جدید - تشخیص واگرایی
    'calculate_macd_simple',
    'analyze_trend_simple',
    
    # توابع ایچیموکو
    'calculate_ichimoku_components',
    'analyze_ichimoku_scalp_signal',
    'get_ichimoku_scalp_signal',
    'calculate_quality_line',
    'calculate_golden_line',
    
    # توابع کمکی
    'calculate_24h_change_from_dataframe',
    'analyze_scalp_conditions',
    'get_support_resistance_levels',
    'calculate_volatility',
    'combined_analysis',
    'generate_ichimoku_recommendation',
    
    # توابع fallback
    'get_fallback_signal'
]

print(f"✅ utils.py v{__version__} loaded successfully!")
print(f"📊 Features: Ichimoku Advanced, Quality Line, Golden Line, Multi-Timeframe")
print(f"⚡ Scalp Support: 1m/5m/15m with Ichimoku")
print(f"🧠 RSI Engine: Real (Series Calculation)")
print(f"⚖️ Divergence Engine: Real (Peak/Trough Analysis)")
```