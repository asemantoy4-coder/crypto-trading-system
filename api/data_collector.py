# api/data_collector.py
"""
Data Collector - نسخه کامل و بهینه‌شده
جمع‌آوری داده از چندین منبع با مکانیزم Fallback
نسخه 7.3.0
"""

from datetime import datetime, timedelta
import logging
import random
from typing import List, Dict, Optional, Any
import time

logger = logging.getLogger(__name__)

# ==============================================================================
# تنظیمات پیش‌فرض
# ==============================================================================
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
DEFAULT_TIMEFRAME = "5m"
DEFAULT_LIMIT = 50

# ==============================================================================
# تابع اصلی جمع‌آوری داده
# ==============================================================================

def get_collected_data(
    symbols: Optional[List[str]] = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    limit: int = DEFAULT_LIMIT,
    include_metrics: bool = True,
    include_signals: bool = False
) -> Dict[str, Any]:
    """
    جمع‌آوری داده از منابع مختلف
    
    Parameters:
    -----------
    symbols : List[str], optional
        لیست نمادهای معاملاتی
    timeframe : str
        تایم‌فریم (1m, 5m, 15m, 1h, 4h, 1d)
    limit : int
        تعداد کندل‌ها
    include_metrics : bool
        شامل کردن متریک‌های بازار
    include_signals : bool
        شامل کردن سیگنال‌های تحلیلی
    
    Returns:
    --------
    Dict[str, Any]
        دیکشنری شامل تمام داده‌های جمع‌آوری شده
    """
    start_time = time.time()
    
    if not symbols:
        symbols = DEFAULT_SYMBOLS
    
    logger.info(f"📊 جمع‌آوری داده برای {len(symbols)} نماد...")
    
    # جمع‌آوری داده قیمت
    price_data = collect_price_data(symbols, timeframe, limit)
    
    # متریک‌های بازار
    market_metrics = {}
    if include_metrics:
        market_metrics = collect_market_metrics()
    
    # سیگنال‌های تحلیلی
    signals = {}
    if include_signals:
        signals = collect_analysis_signals(symbols)
    
    # خلاصه
    execution_time = round(time.time() - start_time, 2)
    
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "execution_time": execution_time,
        "config": {
            "symbols": symbols,
            "timeframe": timeframe,
            "limit": limit
        },
        "price_data": price_data,
        "market_metrics": market_metrics if include_metrics else None,
        "signals": signals if include_signals else None,
        "summary": {
            "total_symbols": len(symbols),
            "successful_collections": len([p for p in price_data.values() if p.get("success")]),
            "failed_collections": len([p for p in price_data.values() if not p.get("success")]),
            "data_points_collected": sum(p.get("data_points", 0) for p in price_data.values()),
            "sources_used": list(set(p.get("source") for p in price_data.values()))
        }
    }
    
    logger.info(f"✅ جمع‌آوری کامل شد در {execution_time} ثانیه")
    
    return result

# ==============================================================================
# توابع کمکی جمع‌آوری
# ==============================================================================

def collect_price_data(
    symbols: List[str],
    timeframe: str,
    limit: int
) -> Dict[str, Dict[str, Any]]:
    """
    جمع‌آوری داده قیمت برای همه نمادها
    
    Returns:
    --------
    Dict[str, Dict]
        دیکشنری با کلید نماد و مقدار داده قیمت
    """
    price_data = {}
    
    for symbol in symbols:
        try:
            # تلاش برای دریافت داده واقعی
            data = fetch_symbol_data(symbol, timeframe, limit)
            
            if data and len(data) > 0:
                # پردازش داده
                processed = process_price_data(symbol, data)
                price_data[symbol] = {
                    "success": True,
                    "source": processed["source"],
                    "data_points": len(data),
                    "latest_price": processed["latest_price"],
                    "high_24h": processed["high_24h"],
                    "low_24h": processed["low_24h"],
                    "volume_24h": processed["volume_24h"],
                    "change_24h": processed["change_24h"],
                    "updated_at": datetime.now().isoformat()
                }
            else:
                # داده mock
                price_data[symbol] = generate_mock_price_data(symbol)
                
        except Exception as e:
            logger.error(f"❌ خطا در جمع‌آوری {symbol}: {e}")
            price_data[symbol] = {
                "success": False,
                "error": str(e),
                "source": "error"
            }
    
    return price_data

def fetch_symbol_data(symbol: str, timeframe: str, limit: int) -> Optional[List]:
    """
    دریافت داده از API (با استفاده از utils.py)
    """
    try:
        # استفاده از تابع utils
        from .utils import get_market_data_with_fallback
        
        result = get_market_data_with_fallback(
            symbol=symbol,
            interval=timeframe,
            limit=limit,
            return_source=True
        )
        
        if isinstance(result, dict):
            return result.get("data")
        else:
            return result
            
    except ImportError:
        logger.warning("⚠️ utils.py not available, using mock data")
        return None
    except Exception as e:
        logger.error(f"❌ خطا در fetch: {e}")
        return None

def process_price_data(symbol: str, data: List) -> Dict[str, Any]:
    """
    پردازش داده خام قیمت
    """
    if not data or len(data) == 0:
        return generate_mock_price_data(symbol)
    
    try:
        # آخرین کندل
        latest = data[-1]
        latest_price = float(latest[4])  # close price
        
        # محاسبه high/low/volume از تمام داده
        highs = [float(candle[2]) for candle in data]
        lows = [float(candle[3]) for candle in data]
        volumes = [float(candle[5]) for candle in data]
        
        high_24h = max(highs)
        low_24h = min(lows)
        volume_24h = sum(volumes)
        
        # محاسبه تغییرات
        first_price = float(data[0][4])
        change_24h = ((latest_price - first_price) / first_price * 100) if first_price > 0 else 0
        
        return {
            "source": "real_api",
            "latest_price": round(latest_price, 2),
            "high_24h": round(high_24h, 2),
            "low_24h": round(low_24h, 2),
            "volume_24h": round(volume_24h, 2),
            "change_24h": round(change_24h, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ خطا در پردازش: {e}")
        return generate_mock_price_data(symbol)

def generate_mock_price_data(symbol: str) -> Dict[str, Any]:
    """
    تولید داده mock برای زمانی که API در دسترس نیست
    """
    base_prices = {
        'BTCUSDT': 88271.42,
        'ETHUSDT': 3450.12,
        'BNBUSDT': 590.54,
        'SOLUSDT': 175.98,
        'XRPUSDT': 0.51234,
        'ADAUSDT': 0.43210,
        'DEFAULT': 100.00
    }
    
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    
    # شبیه‌سازی نوسانات
    variation = random.uniform(-0.05, 0.05)  # ±5%
    latest_price = base_price * (1 + variation)
    
    return {
        "success": True,
        "source": "mock",
        "data_points": 50,
        "latest_price": round(latest_price, 2),
        "high_24h": round(latest_price * 1.03, 2),
        "low_24h": round(latest_price * 0.97, 2),
        "volume_24h": round(random.uniform(100000, 1000000), 2),
        "change_24h": round(variation * 100, 2),
        "updated_at": datetime.now().isoformat()
    }

def collect_market_metrics() -> Dict[str, Any]:
    """
    جمع‌آوری متریک‌های کلی بازار
    """
    try:
        # می‌توانید از API های مثل CoinGecko یا CoinMarketCap استفاده کنید
        # فعلاً داده mock
        
        return {
            "total_market_cap": round(random.uniform(1.8e12, 2.2e12), 2),
            "total_volume_24h": round(random.uniform(70e9, 90e9), 2),
            "btc_dominance": round(random.uniform(50, 55), 2),
            "eth_dominance": round(random.uniform(15, 20), 2),
            "defi_market_cap": round(random.uniform(50e9, 80e9), 2),
            "stablecoin_market_cap": round(random.uniform(150e9, 180e9), 2),
            "active_cryptocurrencies": random.randint(8000, 10000),
            "active_markets": random.randint(35000, 40000),
            "market_cap_change_24h": round(random.uniform(-3, 3), 2),
            "updated_at": datetime.now().isoformat(),
            "source": "mock"  # تغییر به "coingecko" یا "coinmarketcap" وقتی واقعی شد
        }
        
    except Exception as e:
        logger.error(f"❌ خطا در جمع‌آوری متریک‌ها: {e}")
        return {
            "error": str(e),
            "source": "error"
        }

def collect_analysis_signals(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    جمع‌آوری سیگنال‌های تحلیلی برای نمادها
    """
    signals = {}
    
    for symbol in symbols:
        try:
            # استفاده از تابع تحلیل از utils.py
            from .utils import analyze_with_multi_timeframe_strategy
            
            analysis = analyze_with_multi_timeframe_strategy(symbol)
            
            signals[symbol] = {
                "signal": analysis.get("signal"),
                "confidence": analysis.get("confidence"),
                "entry_price": analysis.get("entry_price"),
                "targets": analysis.get("targets"),
                "stop_loss": analysis.get("stop_loss"),
                "strategy": analysis.get("strategy"),
                "timestamp": datetime.now().isoformat()
            }
            
        except ImportError:
            logger.warning(f"⚠️ Cannot analyze {symbol}: utils not available")
            signals[symbol] = generate_mock_signal(symbol)
        except Exception as e:
            logger.error(f"❌ خطا در تحلیل {symbol}: {e}")
            signals[symbol] = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    return signals

def generate_mock_signal(symbol: str) -> Dict[str, Any]:
    """
    تولید سیگنال mock
    """
    signals = ["BUY", "SELL", "HOLD"]
    signal = random.choice(signals)
    confidence = round(random.uniform(0.5, 0.85), 2)
    
    base_prices = {
        'BTCUSDT': 88271.42,
        'ETHUSDT': 3450.12,
        'DEFAULT': 100.00
    }
    
    price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    
    return {
        "signal": signal,
        "confidence": confidence,
        "entry_price": round(price, 2),
        "targets": [round(price * 1.02, 2), round(price * 1.05, 2)] if signal == "BUY" else [],
        "stop_loss": round(price * 0.98, 2) if signal == "BUY" else round(price * 1.02, 2),
        "strategy": "Mock Signal",
        "timestamp": datetime.now().isoformat(),
        "source": "mock"
    }

# ==============================================================================
# توابع اضافی
# ==============================================================================

def get_market_overview(top_n: int = 10) -> Dict[str, Any]:
    """
    نمای کلی بازار - top N نماد
    
    Parameters:
    -----------
    top_n : int
        تعداد نمادهای برتر
    
    Returns:
    --------
    Dict
        اطلاعات نمای کلی
    """
    symbols = DEFAULT_SYMBOLS[:top_n]
    
    data = get_collected_data(
        symbols=symbols,
        include_metrics=True,
        include_signals=True
    )
    
    # اضافه کردن رتبه‌بندی
    if data["price_data"]:
        # مرتب‌سازی بر اساس تغییرات 24 ساعته
        sorted_symbols = sorted(
            data["price_data"].items(),
            key=lambda x: x[1].get("change_24h", 0),
            reverse=True
        )
        
        data["rankings"] = {
            "top_gainers": [(s, d["change_24h"]) for s, d in sorted_symbols[:5] if d.get("success")],
            "top_losers": [(s, d["change_24h"]) for s, d in sorted_symbols[-5:] if d.get("success")],
        }
    
    return data

def collect_historical_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1d"
) -> Dict[str, Any]:
    """
    جمع‌آوری داده تاریخی
    
    Parameters:
    -----------
    symbol : str
        نماد معاملاتی
    start_date : datetime
        تاریخ شروع
    end_date : datetime
        تاریخ پایان
    timeframe : str
        تایم‌فریم
    
    Returns:
    --------
    Dict
        داده تاریخی
    """
    # محاسبه تعداد روزها
    days = (end_date - start_date).days
    
    logger.info(f"📊 جمع‌آوری {days} روز داده تاریخی برای {symbol}")
    
    # فعلاً mock - می‌توانید با API واقعی جایگزین کنید
    return {
        "symbol": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timeframe": timeframe,
        "days": days,
        "status": "mock",
        "note": "Historical data collection not implemented yet"
    }

# ==============================================================================
# Export
# ==============================================================================

__all__ = [
    'get_collected_data',
    'collect_price_data',
    'collect_market_metrics',
    'collect_analysis_signals',
    'get_market_overview',
    'collect_historical_data'
]

if __name__ == "__main__":
    # تست
    print("🧪 Testing data_collector...")
    
    result = get_collected_data(
        symbols=["BTCUSDT", "ETHUSDT"],
        include_metrics=True,
        include_signals=True
    )
    
    print(f"✅ Status: {result['status']}")
    print(f"📊 Symbols: {result['summary']['total_symbols']}")
    print(f"⏱️ Execution time: {result['execution_time']}s")
    print(f"💰 BTC Price: ${result['price_data']['BTCUSDT']['latest_price']}")