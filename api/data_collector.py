# api/data_collector.py - نسخه 7.4.0
"""
Data Collector - Lightweight version
"""

from datetime import datetime, timedelta
import random
import logging
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

# ==============================================================================
# Import مدیریت شده از utils
# ==============================================================================

# تلاش برای import توابع از utils با چند روش مختلف
def safe_import_utils():
    """Import امن توابع از utils"""
    functions = {}
    
    # لیست توابع مورد نیاز
    required_funcs = [
        'get_market_data_with_fallback',
        'calculate_simple_sma',
        'calculate_simple_rsi',
        'calculate_macd_simple',
        'get_ichimoku_scalp_signal',
        'calculate_ichimoku_components',
        'analyze_scalp_conditions'
    ]
    
    # روش ۱: import مستقیم از پوشه جاری
    try:
        import utils
        for func_name in required_funcs:
            if hasattr(utils, func_name):
                functions[func_name] = getattr(utils, func_name)
        if functions:
            logger.info("✅ utils imported directly")
            return functions
    except ImportError as e:
        logger.debug(f"Direct import failed: {e}")
    
    # روش ۲: import نسبی
    try:
        from . import utils as local_utils
        for func_name in required_funcs:
            if hasattr(local_utils, func_name):
                functions[func_name] = getattr(local_utils, func_name)
        if functions:
            logger.info("✅ utils imported relatively")
            return functions
    except ImportError as e:
        logger.debug(f"Relative import failed: {e}")
    
    # روش ۳: import با نام کامل
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        import api.utils as api_utils
        for func_name in required_funcs:
            if hasattr(api_utils, func_name):
                functions[func_name] = getattr(api_utils, func_name)
        if functions:
            logger.info("✅ utils imported via full path")
            return functions
    except ImportError as e:
        logger.debug(f"Full path import failed: {e}")
    
    # اگر هیچ‌کدام جواب نداد، توابع mock برگردان
    logger.warning("⚠️ Could not import utils, using mock functions")
    return create_mock_functions()

def create_mock_functions():
    """ایجاد توابع mock برای وقتی که import شکست می‌خورد"""
    
    def mock_get_market_data(symbol, timeframe="5m", limit=50):
        """تابع mock برای دریافت داده بازار"""
        data = []
        base_price = 88271.00 if symbol.upper() == "BTCUSDT" else 3450.00 if symbol.upper() == "ETHUSDT" else 100.00
        current_time = int(datetime.now().timestamp() * 1000)
        
        for i in range(limit):
            timestamp = current_time - (i * 5 * 60 * 1000)
            price = base_price * (1 + random.uniform(-0.02, 0.02))
            
            candle = [
                timestamp,
                str(price * random.uniform(0.998, 1.002)),
                str(price * random.uniform(1.000, 1.004)),
                str(price * random.uniform(0.996, 1.000)),
                str(price),
                str(random.uniform(1000, 10000)),
                timestamp + 300000,
                "0", "0", "0", "0", "0"
            ]
            data.append(candle)
        
        return data
    
    def mock_calculate_sma(data, period=20):
        """تابع mock برای SMA"""
        if not data or len(data) < period:
            return 50000
        return sum(float(candle[4]) for candle in data[-period:]) / period
    
    def mock_calculate_rsi(data, period=14):
        """تابع mock برای RSI"""
        return 50 + random.uniform(-20, 20)
    
    def mock_calculate_macd(data):
        """تابع mock برای MACD"""
        return {'macd': 0, 'signal': 0, 'histogram': random.uniform(-10, 10)}
    
    def mock_get_ichimoku_signal(data, timeframe="5m"):
        """تابع mock برای ایچیموکو"""
        signals = ['BUY', 'SELL', 'HOLD']
        weights = [0.35, 0.35, 0.30]
        signal = random.choices(signals, weights=weights)[0]
        
        return {
            'signal': signal,
            'confidence': random.uniform(0.6, 0.9) if signal != 'HOLD' else random.uniform(0.4, 0.6),
            'reason': f'سیگنال {signal} (Mock)',
            'timeframe': timeframe
        }
    
    def mock_calculate_ichimoku(data):
        """تابع mock برای محاسبه ایچیموکو"""
        try:
            price = float(data[-1][4])
        except:
            price = 100
        
        return {
            'tenkan_sen': price * random.uniform(0.99, 1.01),
            'kijun_sen': price * random.uniform(0.98, 1.02),
            'cloud_top': price * random.uniform(1.01, 1.05),
            'cloud_bottom': price * random.uniform(0.95, 0.99),
            'trend_power': random.uniform(30, 80)
        }
    
    def mock_analyze_scalp_conditions(data, timeframe):
        """تابع mock برای تحلیل اسکالپ"""
        return {
            "condition": random.choice(["BULLISH", "BEARISH", "NEUTRAL"]),
            "rsi": 30 + random.random() * 40,
            "reason": "تحلیل آزمایشی"
        }
    
    return {
        'get_market_data_with_fallback': mock_get_market_data,
        'calculate_simple_sma': mock_calculate_sma,
        'calculate_simple_rsi': mock_calculate_rsi,
        'calculate_macd_simple': mock_calculate_macd,
        'get_ichimoku_scalp_signal': mock_get_ichimoku_signal,
        'calculate_ichimoku_components': mock_calculate_ichimoku,
        'analyze_scalp_conditions': mock_analyze_scalp_conditions
    }

# بارگذاری توابع
utils_funcs = safe_import_utils()

# اختصاص توابع به متغیرهای جهانی
get_market_data_with_fallback = utils_funcs.get('get_market_data_with_fallback')
calculate_simple_sma = utils_funcs.get('calculate_simple_sma')
calculate_simple_rsi = utils_funcs.get('calculate_simple_rsi')
calculate_macd_simple = utils_funcs.get('calculate_macd_simple')
get_ichimoku_scalp_signal = utils_funcs.get('get_ichimoku_scalp_signal')
calculate_ichimoku_components = utils_funcs.get('calculate_ichimoku_components')
analyze_scalp_conditions = utils_funcs.get('analyze_scalp_conditions')

# ==============================================================================
# توابع اصلی
# ==============================================================================

def get_collected_data(symbols=None, timeframe="5m", limit=50, include_analysis=False):
    """
    دریافت داده جمع‌آوری شده با پشتیبانی از تحلیل پیشرفته
    
    Parameters:
    -----------
    symbols : list or None
        لیست نمادها (پیش‌فرض: BTCUSDT, ETHUSDT)
    timeframe : str
        تایم‌فریم (پیش‌فرض: 5m)
    limit : int
        تعداد کندل‌ها (پیش‌فرض: 50)
    include_analysis : bool
        آیا تحلیل تکنیکال هم شامل شود؟
    
    Returns:
    --------
    dict
        داده‌های جمع‌آوری شده
    """
    logger.info(f"📊 جمع‌آوری داده برای {symbols or ['BTCUSDT', 'ETHUSDT']} ({timeframe})")
    
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # اضافه کردن یک نماد بیشتر
    
    results = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "timeframe": timeframe,
        "symbols_analyzed": [],
        "price_data": {},
        "technical_analysis": {} if include_analysis else None,
        "market_metrics": {
            "total_market_cap": 1.8e12,
            "btc_dominance": 52.5,
            "volume_24h": 75.3e9,
            "fear_greed_index": random.randint(40, 70)
        },
        "summary": {
            "symbols_collected": 0,
            "total_data_points": 0,
            "analysis_included": include_analysis
        }
    }
    
    total_data_points = 0
    
    for symbol in symbols[:5]:  # حداکثر ۵ نماد برای جلوگیری از overload
        try:
            symbol_upper = symbol.upper()
            
            # دریافت داده بازار
            market_data = get_market_data_with_fallback(symbol_upper, timeframe, limit)
            
            if not market_data:
                logger.warning(f"⚠️ No data for {symbol_upper}")
                continue
            
            # آخرین قیمت
            try:
                latest_price = float(market_data[-1][4])
                latest_high = float(market_data[-1][2])
                latest_low = float(market_data[-1][3])
                volume = float(market_data[-1][5])
            except (IndexError, ValueError, TypeError):
                continue
            
            # ذخیره داده قیمت
            results["price_data"][symbol_upper] = {
                "price": latest_price,
                "high": latest_high,
                "low": latest_low,
                "volume": volume,
                "data_points": len(market_data),
                "timeframe": timeframe,
                "last_updated": datetime.now().isoformat()
            }
            
            total_data_points += len(market_data)
            
            # اگر تحلیل درخواست شده
            if include_analysis and len(market_data) >= 20:
                analysis = perform_technical_analysis(symbol_upper, timeframe, market_data)
                if analysis:
                    if "technical_analysis" not in results or results["technical_analysis"] is None:
                        results["technical_analysis"] = {}
                    results["technical_analysis"][symbol_upper] = analysis
            
            results["symbols_analyzed"].append(symbol_upper)
            
        except Exception as e:
            logger.error(f"❌ Error collecting data for {symbol}: {e}")
            continue
    
    # به‌روزرسانی خلاصه
    results["summary"]["symbols_collected"] = len(results["symbols_analyzed"])
    results["summary"]["total_data_points"] = total_data_points
    
    # اگر هیچ داده‌ای جمع‌آوری نشد
    if not results["symbols_analyzed"]:
        results["status"] = "partial"
        results["note"] = "Limited data collected, using fallback"
        
        # داده‌های fallback
        for symbol in symbols[:3]:
            base_price = 88271.00 if "BTC" in symbol.upper() else \
                        3450.00 if "ETH" in symbol.upper() else \
                        590.00 if "BNB" in symbol.upper() else 100.00
            
            results["price_data"][symbol.upper()] = {
                "price": base_price * random.uniform(0.99, 1.01),
                "high": base_price * random.uniform(1.005, 1.015),
                "low": base_price * random.uniform(0.985, 0.995),
                "volume": random.uniform(1000, 5000),
                "data_points": 50,
                "timeframe": timeframe,
                "last_updated": datetime.now().isoformat(),
                "note": "Fallback data"
            }
            results["symbols_analyzed"].append(symbol.upper())
    
    logger.info(f"✅ Collected data for {len(results['symbols_analyzed'])} symbols, {total_data_points} data points")
    return results

def perform_technical_analysis(symbol, timeframe, market_data):
    """
    انجام تحلیل تکنیکال بر روی داده‌های بازار
    """
    try:
        analysis = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "indicators": {},
            "signals": [],
            "recommendation": "HOLD"
        }
        
        # محاسبه اندیکاتورهای پایه
        if calculate_simple_sma:
            sma_20 = calculate_simple_sma(market_data, 20)
            sma_50 = calculate_simple_sma(market_data, 50)
            analysis["indicators"]["sma"] = {
                "sma_20": round(sma_20, 4) if sma_20 else None,
                "sma_50": round(sma_50, 4) if sma_50 else None,
                "trend": "bullish" if sma_20 and sma_50 and sma_20 > sma_50 else "bearish" if sma_20 and sma_50 else "neutral"
            }
        
        if calculate_simple_rsi:
            rsi = calculate_simple_rsi(market_data, 14)
            analysis["indicators"]["rsi"] = {
                "value": round(rsi, 2),
                "status": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            }
        
        if calculate_macd_simple:
            macd = calculate_macd_simple(market_data)
            analysis["indicators"]["macd"] = macd
        
        # تحلیل ایچیموکو (برای تایم‌فریم‌های کوتاه)
        if get_ichimoku_scalp_signal and timeframe in ["1m", "5m", "15m"]:
            ichimoku_signal = get_ichimoku_scalp_signal(market_data, timeframe)
            if ichimoku_signal:
                analysis["indicators"]["ichimoku"] = {
                    "signal": ichimoku_signal.get("signal"),
                    "confidence": ichimoku_signal.get("confidence"),
                    "reason": ichimoku_signal.get("reason")
                }
                
                # اضافه کردن به سیگنال‌ها
                if ichimoku_signal.get("signal") in ["BUY", "SELL"]:
                    analysis["signals"].append({
                        "type": "ICHIMOKU",
                        "signal": ichimoku_signal.get("signal"),
                        "confidence": ichimoku_signal.get("confidence"),
                        "reason": ichimoku_signal.get("reason")
                    })
        
        # تحلیل شرایط اسکالپ
        if analyze_scalp_conditions and timeframe in ["1m", "5m", "15m"]:
            scalp_analysis = analyze_scalp_conditions(market_data, timeframe)
            analysis["indicators"]["scalp"] = scalp_analysis
            
            if scalp_analysis.get("condition") in ["BULLISH", "BEARISH"]:
                signal_type = "BUY" if scalp_analysis["condition"] == "BULLISH" else "SELL"
                analysis["signals"].append({
                    "type": "SCALP",
                    "signal": signal_type,
                    "confidence": 0.6,  # اطمینان متوسط برای اسکالپ
                    "reason": scalp_analysis.get("reason")
                })
        
        # تصمیم‌گیری نهایی بر اساس سیگنال‌ها
        if analysis["signals"]:
            buy_signals = [s for s in analysis["signals"] if s["signal"] == "BUY"]
            sell_signals = [s for s in analysis["signals"] if s["signal"] == "SELL"]
            
            if buy_signals and len(buy_signals) > len(sell_signals):
                analysis["recommendation"] = "BUY"
                # میانگین اطمینان سیگنال‌های خرید
                avg_confidence = sum(s["confidence"] for s in buy_signals) / len(buy_signals)
                analysis["confidence"] = round(avg_confidence, 3)
            elif sell_signals and len(sell_signals) > len(buy_signals):
                analysis["recommendation"] = "SELL"
                avg_confidence = sum(s["confidence"] for s in sell_signals) / len(sell_signals)
                analysis["confidence"] = round(avg_confidence, 3)
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Error in technical analysis for {symbol}: {e}")
        return None

def get_market_overview(timeframe="5m"):
    """
    دریافت نمای کلی بازار
    """
    logger.info(f"🌐 دریافت نمای کلی بازار ({timeframe})")
    
    # نمادهای اصلی برای تحلیل
    major_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    
    # جمع‌آوری داده با تحلیل
    data = get_collected_data(
        symbols=major_symbols,
        timeframe=timeframe,
        limit=30,
        include_analysis=True
    )
    
    # تحلیل کلی بازار
    market_sentiment = analyze_market_sentiment(data)
    
    overview = {
        "timestamp": datetime.now().isoformat(),
        "timeframe": timeframe,
        "market_status": "open",
        "market_sentiment": market_sentiment,
        "top_gainers": [],
        "top_losers": [],
        "most_active": [],
        "summary": {
            "total_symbols": len(data.get("symbols_analyzed", [])),
            "with_analysis": data.get("technical_analysis") is not None,
            "data_points": data.get("summary", {}).get("total_data_points", 0)
        }
    }
    
    # محاسبه تغییرات و فعالیت
    if data.get("price_data"):
        price_changes = []
        for symbol, price_info in data["price_data"].items():
            # شبیه‌سازی تغییر قیمت
            change_percent = random.uniform(-3, 3)
            price_changes.append({
                "symbol": symbol,
                "price": price_info.get("price", 0),
                "change_percent": round(change_percent, 2),
                "volume": price_info.get("volume", 0)
            })
        
        # مرتب‌سازی
        if price_changes:
            price_changes.sort(key=lambda x: x["change_percent"], reverse=True)
            overview["top_gainers"] = price_changes[:3]
            overview["top_losers"] = sorted(price_changes[-3:], key=lambda x: x["change_percent"])
            
            price_changes.sort(key=lambda x: x["volume"], reverse=True)
            overview["most_active"] = price_changes[:3]
    
    # اگر تحلیل تکنیکال داریم، سیگنال‌های قوی را اضافه کن
    strong_signals = []
    if data.get("technical_analysis"):
        for symbol, analysis in data["technical_analysis"].items():
            if analysis.get("confidence", 0) > 0.7:
                strong_signals.append({
                    "symbol": symbol,
                    "signal": analysis.get("recommendation"),
                    "confidence": analysis.get("confidence"),
                    "timeframe": timeframe
                })
    
    if strong_signals:
        overview["strong_signals"] = strong_signals
    
    return overview

def analyze_market_sentiment(data):
    """
    تحلیل احساسات بازار بر اساس داده‌های جمع‌آوری شده
    """
    sentiment_score = 50  # خنثی
    
    try:
        # بررسی سیگنال‌ها در تحلیل تکنیکال
        if data.get("technical_analysis"):
            buy_count = 0
            sell_count = 0
            total_symbols = len(data["technical_analysis"])
            
            for symbol, analysis in data["technical_analysis"].items():
                recommendation = analysis.get("recommendation", "HOLD")
                if recommendation == "BUY":
                    buy_count += 1
                elif recommendation == "SELL":
                    sell_count += 1
            
            if total_symbols > 0:
                buy_ratio = buy_count / total_symbols
                sell_ratio = sell_count / total_symbols
                
                if buy_ratio > 0.6:
                    sentiment_score = 75  # صعودی قوی
                elif buy_ratio > 0.4:
                    sentiment_score = 65  # صعودی متوسط
                elif sell_ratio > 0.6:
                    sentiment_score = 25  # نزولی قوی
                elif sell_ratio > 0.4:
                    sentiment_score = 35  # نزولی متوسط
        
        # بررسی Fear & Greed Index
        fear_greed = data.get("market_metrics", {}).get("fear_greed_index", 50)
        sentiment_score = (sentiment_score + fear_greed) / 2
        
    except Exception as e:
        logger.error(f"❌ Error in sentiment analysis: {e}")
    
    # تفسیر امتیاز
    if sentiment_score >= 70:
        return {"score": round(sentiment_score, 1), "text": "صعودی قوی", "color": "green"}
    elif sentiment_score >= 60:
        return {"score": round(sentiment_score, 1), "text": "صعودی", "color": "light_green"}
    elif sentiment_score >= 40:
        return {"score": round(sentiment_score, 1), "text": "خنثی", "color": "yellow"}
    elif sentiment_score >= 30:
        return {"score": round(sentiment_score, 1), "text": "نزولی", "color": "orange"}
    else:
        return {"score": round(sentiment_score, 1), "text": "نزولی قوی", "color": "red"}

# ==============================================================================
# توابع کمکی
# ==============================================================================

def get_symbol_info(symbol):
    """دریافت اطلاعات یک نماد خاص"""
    symbol_upper = symbol.upper()
    
    # اطلاعات پایه نمادها
    symbol_info = {
        "BTCUSDT": {
            "name": "Bitcoin",
            "sector": "Cryptocurrency",
            "market_cap": 1.1e12,
            "description": "اولین و بزرگترین ارز دیجیتال"
        },
        "ETHUSDT": {
            "name": "Ethereum",
            "sector": "Cryptocurrency",
            "market_cap": 450e9,
            "description": "پلتفرم قراردادهای هوشمند"
        },
        "BNBUSDT": {
            "name": "Binance Coin",
            "sector": "Exchange Token",
            "market_cap": 90e9,
            "description": "توکن بومی صرافی بایننس"
        },
        "SOLUSDT": {
            "name": "Solana",
            "sector": "Cryptocurrency",
            "market_cap": 75e9,
            "description": "پلتفرم بلاکچین سریع"
        },
        "XRPUSDT": {
            "name": "Ripple",
            "sector": "Cryptocurrency",
            "market_cap": 40e9,
            "description": "پروتکل پرداخت بین‌بانکی"
        }
    }
    
    return symbol_info.get(symbol_upper, {
        "name": symbol_upper.replace("USDT", ""),
        "sector": "Cryptocurrency",
        "market_cap": 1e9,
        "description": "ارز دیجیتال"
    })

# ==============================================================================
# Export توابع
# ==============================================================================

__all__ = [
    'get_collected_data',
    'get_market_overview',
    'perform_technical_analysis',
    'analyze_market_sentiment',
    'get_symbol_info'
]

print(f"✅ data_collector.py loaded - Version 7.4.0")
print(f"📊 Features: Market data collection, Technical analysis, Ichimoku support")
