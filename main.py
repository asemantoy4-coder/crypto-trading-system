import os
import sys
import time
import uvicorn
import logging
import threading
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException
from typing import List, Optional
import numpy as np
from pydantic import BaseModel
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

# اضافه کردن مسیر جاری به پایتون برای پیدا کردن فایل‌ها
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# CONFIGURATION
# ==============================================================================

API_VERSION = "8.5.2"
DEBUG_MODE = os.environ.get("DEBUG", "False").lower() == "true"

# لاگینگ بهینه
logging.basicConfig(
    level=logging.INFO if not DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CryptoAIScalper")

# ==============================================================================
# TELEGRAM SETTINGS
# ==============================================================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8066443971:AAFBvYtLTdQIrLe07CJ-X18UyaPi3Dpb5zo")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "@AsemanSignals")

def send_telegram_auto(text: str):
    """ارسال خودکار پیام به تلگرام"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not configured")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(
            url, 
            json={
                "chat_id": TELEGRAM_CHAT_ID, 
                "text": text, 
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }, 
            timeout=5
        )
        if response.status_code == 200:
            logger.debug(f"✅ Telegram message sent: {text[:50]}...")
        else:
            logger.error(f"❌ Telegram error {response.status_code}: {response.text}")
    except requests.exceptions.Timeout:
        logger.warning("⏱️ Telegram timeout")
    except Exception as e:
        logger.error(f"❌ Telegram error: {e}")

# ==============================================================================
# PRICE CACHE (برای کاهش فشار API)
# ==============================================================================

price_cache = {
    'btc': {'price': 0, 'timestamp': 0, 'change': 0, 'volume': 0}
}
cache_lock = threading.Lock()

def update_price_cache():
    """آپدیت دوره‌ای کش قیمت‌ها"""
    try:
        import requests as req
        
        # گرفتن قیمت BTC
        price_response = req.get(
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", 
            timeout=2
        )
        
        if price_response.status_code != 200:
            return
        
        price_data = price_response.json()
        current_price = float(price_data['price'])
        
        # گرفتن اطلاعات 24h
        ticker_response = req.get(
            "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT",
            timeout=2
        )
        
        change_percent = 0
        volume = 0
        
        if ticker_response.status_code == 200:
            ticker_data = ticker_response.json()
            change_percent = float(ticker_data.get('priceChangePercent', 0))
            volume = float(ticker_data.get('volume', 0))
        
        with cache_lock:
            price_cache['btc']['price'] = current_price
            price_cache['btc']['change'] = change_percent
            price_cache['btc']['volume'] = volume
            price_cache['btc']['timestamp'] = time.time()
            
        if DEBUG_MODE:
            logger.debug(f"💰 Cache updated: ${current_price:,.0f} ({change_percent:+.2f}%)")
            
    except Exception as e:
        logger.debug(f"⚠️ Cache update failed: {e}")

def get_cached_btc_price():
    """دریافت قیمت کش شده BTC"""
    with cache_lock:
        return price_cache['btc'].copy()

# ==============================================================================
# MODULE IMPORTS (شرطی)
# ==============================================================================

HAS_PANDAS = False
HAS_PANDAS_TA = False
UTILS_AVAILABLE = False
COLLECTORS_AVAILABLE = False
HAS_TDR_ATR = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    logger.warning("⚠️ Pandas not available")

try:
    import pandas_ta
    HAS_PANDAS_TA = True
except ImportError:
    logger.warning("⚠️ Pandas_TA not available")

try:
    from utils import (
        format_binance_price,
        get_enhanced_scalp_signal,
        get_market_data_with_fallback,
        get_momentum_persian_msg
    )
    UTILS_AVAILABLE = True
    logger.info("✅ Utils module loaded")
except ImportError as e:
    logger.error(f"❌ Utils Import Error: {e}")
    UTILS_AVAILABLE = False

try:
    from scalper_engine import ScalperEngine
    COLLECTORS_AVAILABLE = True
    HAS_TDR_ATR = hasattr(ScalperEngine, 'calculate_tdr_advanced')
    logger.info("✅ ScalperEngine loaded")
except ImportError as e:
    logger.error(f"❌ Error importing ScalperEngine: {e}")
    class ScalperEngine:
        @staticmethod
        def calculate_tdr_advanced(data):
            return 0.5
        @staticmethod
        def get_ai_confirmation(*args, **kwargs):
            return "AI not available"

# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="Crypto AI Scalper",
    description="Professional Scalping & Trading Analysis API",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==============================================================================
# DATA MODELS
# ==============================================================================

class ScalpRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    use_ai: bool = False

class IchimokuRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"

class SignalDetail(BaseModel):
    symbol: str
    signal: str
    entry_price: float
    stop_loss: float
    targets: List[float]
    momentum_score: float
    user_message: str
    is_risky_for_retail: bool
    execution_type: str
    signal_id: str
    timestamp: str

# ==============================================================================
# SCHEDULER FUNCTIONS (فوق‌سبک)
# ==============================================================================

def golden_hour_job():
    """
    نسخه فوق‌سبک: فقط از کش می‌خواند، API نمی‌زند
    """
    try:
        # تنظیم تایم‌زون ایران
        tehran_tz = pytz.timezone('Asia/Tehran')
        now = datetime.now(tehran_tz)
        
        # لاگ فقط در حالت دیباگ
        if DEBUG_MODE:
            logger.debug(f"🕒 Scheduler check: {now.strftime('%H:%M')} Tehran")
        
        # فقط سه‌شنبه تا پنج‌شنبه
        if now.weekday() not in [1, 2, 3]:
            return
        
        # فقط ساعات طلایی
        if not ((10 <= now.hour < 12) or (17 <= now.hour < 19)):
            return
        
        # گرفتن قیمت از کش (بدون API call)
        btc_data = get_cached_btc_price()
        price = btc_data['price']
        change = btc_data['change']
        
        if price <= 0:
            logger.warning("⚠️ Invalid price in cache, skipping")
            return
        
        # آماده‌سازی پیام
        change_icon = "📈" if change >= 0 else "📉"
        change_text = f"{change:+.2f}%" if change != 0 else "بدون تغییر"
        
        msg_lines = [
            f"🔔 <b>بازار زنده آسمان</b>",
            f"⏰ {now.strftime('%H:%M')} تهران",
            f"₿ بیت‌کوین: <code>{price:,.0f}$</code>",
            f"{change_icon} تغییر: {change_text}",
            "",
            f"💡 <i>برای تحلیل عمیق به پنل وب مراجعه کنید</i>",
            f"🔄 @AsemanSignals"
        ]
        
        send_telegram_auto("\n".join(msg_lines))
        logger.info(f"✅ Light signal sent: ${price:,.0f} ({change_text})")
        
    except Exception as e:
        logger.error(f"❌ Error in golden_hour_job: {e}")

def hourly_market_update():
    """آپدیت ساعتی (فوق‌سبک)"""
    try:
        tehran_tz = pytz.timezone('Asia/Tehran')
        now = datetime.now(tehran_tz)
        
        # فقط ساعات فعال
        if not (8 <= now.hour < 22):
            return
        
        # گرفتن از کش
        btc_data = get_cached_btc_price()
        if btc_data['price'] <= 0:
            return
        
        # پیام ساده
        msg = f"""
📊 <b>آپدیت بازار</b>
⏰ {now.strftime('%H:%M')} تهران
₿ BTC: <code>{btc_data['price']:,.0f}$</code>

🔄 @AsemanSignals
        """
        send_telegram_auto(msg.strip())
        
    except Exception as e:
        logger.debug(f"⚠️ Hourly update error: {e}")

def cache_warmup():
    """گرم کردن کش در ابتدای راه‌اندازی"""
    logger.info("🔥 Warming up price cache...")
    update_price_cache()
    logger.info("✅ Cache warmed up")

# ==============================================================================
# PERFORMANCE MONITOR
# ==============================================================================

class PerformanceMonitor:
    def __init__(self):
        self.request_times = []
        self.request_count = 0
    
    def record_request(self, processing_time: float):
        self.request_times.append(processing_time)
        self.request_count += 1
        if len(self.request_times) > 100:
            self.request_times.pop(0)

performance_monitor = PerformanceMonitor()

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    """صفحه اصلی"""
    btc_data = get_cached_btc_price()
    
    return {
        "status": "Online",
        "message": "Crypto AI Scalper System",
        "version": API_VERSION,
        "modules": {
            "utils": UTILS_AVAILABLE,
            "pandas": HAS_PANDAS,
            "pandas_ta": HAS_PANDAS_TA,
            "scalper_engine": COLLECTORS_AVAILABLE,
        },
        "market": {
            "btc_price": btc_data['price'],
            "btc_change": btc_data['change'],
            "cache_age": round(time.time() - btc_data['timestamp'], 1)
        },
        "endpoints": {
            "health": "/health",
            "analyze": "POST /analyze",
            "market_scan": "/v1/market-scan",
            "telegram_test": "/v1/telegram-test",
            "performance": "/v1/performance"
        },
        "server_time": datetime.now(timezone.utc).isoformat(),
        "tehran_time": datetime.now(pytz.timezone('Asia/Tehran')).strftime('%Y-%m-%d %H:%M:%S')
    }

@app.get("/health")
async def health():
    """چک سلامت سیستم"""
    btc_data = get_cached_btc_price()
    
    return {
        "status": "Healthy",
        "modules": {
            "utils": UTILS_AVAILABLE,
            "pandas": HAS_PANDAS,
            "pandas_ta": HAS_PANDAS_TA,
            "scalper_engine": COLLECTORS_AVAILABLE,
        },
        "telegram": {
            "configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
            "token_length": len(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else 0
        },
        "cache": {
            "btc_price": btc_data['price'],
            "age_seconds": round(time.time() - btc_data['timestamp'], 1),
            "is_fresh": (time.time() - btc_data['timestamp']) < 300
        },
        "scheduler": {
            "status": "running" if 'scheduler' in globals() and scheduler.running else "unknown",
            "jobs": len(scheduler.get_jobs()) if 'scheduler' in globals() else 0
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/analyze")
async def analyze(request: ScalpRequest):
    """تحلیل اصلی بازار"""
    start_time = time.time()
    
    if not UTILS_AVAILABLE:
        raise HTTPException(status_code=503, detail="تحلیلگر در دسترس نیست")
    
    try:
        # دریافت داده‌های بازار
        data = get_market_data_with_fallback(request.symbol, request.timeframe, 100)
        if not data:
            return {
                "signal": "HOLD", 
                "message": "خطا در دریافت داده‌ها",
                "symbol": request.symbol,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        
        # تحلیل سیگنال
        result = get_enhanced_scalp_signal(data, request.symbol, request.timeframe)
        
        # افزودن اطلاعات اضافی
        processing_time = round((time.time() - start_time) * 1000, 2)
        result.update({
            "processing_time_ms": processing_time,
            "version": API_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_hit": False
        })
        
        logger.info(f"✅ Analysis: {request.symbol} -> {result.get('signal', 'UNKNOWN')} ({processing_time}ms)")
        
        # ارسال به تلگرام برای سیگنال‌های قوی
        if result.get("signal") != "HOLD" and result.get("confidence", 0) > 0.7:
            try:
                telegram_msg = f"""
🔔 <b>سیگنال {request.symbol}</b>
🎯 {result.get('signal')} | اطمینان: {result.get('confidence', 0)*100:.1f}%
💰 قیمت: {result.get('current_price', 0):,.2f}$
⏰ TF: {request.timeframe}

💡 {result.get('momentum_message', '')}
🔄 @AsemanSignals
                """
                send_telegram_auto(telegram_msg.strip())
            except:
                pass
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Analysis error for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"خطای تحلیل: {str(e)[:80]}"
        )

@app.post("/v1/analyze")
async def analyze_pair(request: ScalpRequest):
    """Endpoint سازگاری"""
    return await analyze(request)

@app.get("/v1/market-scan")
async def market_scanner():
    """اسکنر بازار"""
    try:
        if not UTILS_AVAILABLE:
            return {
                "status": "warning",
                "message": "ماژول تحلیل در دسترس نیست",
                "data": [],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        results = []
        
        for symbol in symbols:
            try:
                data = get_market_data_with_fallback(symbol, "1h", 50)
                if data and len(data) >= 20:
                    result = get_enhanced_scalp_signal(data, symbol, "1h")
                    if result:
                        results.append({
                            "symbol": symbol,
                            "signal": result.get("signal", "HOLD"),
                            "confidence": result.get("confidence", 0),
                            "price": result.get("current_price", 0)
                        })
            except Exception as e:
                logger.debug(f"Scan error for {symbol}: {e}")
                continue
        
        return {
            "status": "success",
            "data": results,
            "count": len(results),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market scanner error: {e}")
        return {
            "status": "error",
            "message": str(e)[:100],
            "data": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/v1/telegram-test")
async def telegram_test():
    """تست تلگرام"""
    test_msg = f"""
✅ <b>تست اتصال تلگرام</b>
🕒 {datetime.now(pytz.timezone('Asia/Tehran')).strftime('%H:%M:%S')}
🌐 نسخه: {API_VERSION}
📊 وضعیت: آنلاین

این پیام تست است.
🔄 @AsemanSignals
    """
    
    try:
        send_telegram_auto(test_msg.strip())
        return {
            "status": "success",
            "message": "پیام تست ارسال شد",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"خطا: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/v1/performance")
async def get_performance_stats():
    """آمار عملکرد"""
    if performance_monitor.request_times:
        avg_time = np.mean(performance_monitor.request_times) * 1000
        max_time = np.max(performance_monitor.request_times) * 1000
        min_time = np.min(performance_monitor.request_times) * 1000
    else:
        avg_time = max_time = min_time = 0
    
    return {
        "requests": {
            "total": performance_monitor.request_count,
            "last_100_avg_ms": round(avg_time, 2),
            "last_100_max_ms": round(max_time, 2),
            "last_100_min_ms": round(min_time, 2)
        },
        "cache": {
            "btc_price": price_cache['btc']['price'],
            "age_minutes": round((time.time() - price_cache['btc']['timestamp']) / 60, 1)
        },
        "memory_mb": round(os.sys.getsizeof({}) / 1024 / 1024, 2),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ==============================================================================
# MIDDLEWARE
# ==============================================================================

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    performance_monitor.record_request(process_time)
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response

# ==============================================================================
# STARTUP AND SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """راه‌اندازی سیستم"""
    logger.info(f"🚀 Starting Crypto AI Scalper v{API_VERSION}")
    
    # گرم کردن کش
    cache_warmup()
    
    # راه‌اندازی زمان‌بند
    global scheduler
    try:
        scheduler = BackgroundScheduler()
        
        # آپدیت کش: هر 5 دقیقه
        scheduler.add_job(update_price_cache, 'interval', minutes=5, id='cache_update')
        
        # سیگنال طلایی: هر 30 دقیقه در ساعات خاص
        scheduler.add_job(golden_hour_job, 'interval', minutes=30, id='golden_hour')
        
        # آپدیت ساعتی: هر 2 ساعت
        scheduler.add_job(hourly_market_update, 'interval', hours=2, id='hourly_update')
        
        scheduler.start()
        logger.info(f"✅ Scheduler started with {len(scheduler.get_jobs())} jobs")
        
    except Exception as e:
        logger.error(f"❌ Failed to start scheduler: {e}")
        scheduler = None
    
    # نمایش وضعیت
    btc_data = get_cached_btc_price()
    
    print(f"\n{'='*60}")
    print(f"CRYPTO AI SCALPER v{API_VERSION}")
    print(f"{'='*60}")
    print(f"Status:        ✅ ONLINE")
    print(f"BTC Price:     ${btc_data['price']:,.0f} ({btc_data['change']:+.2f}%)")
    print(f"Utils:         {'✅' if UTILS_AVAILABLE else '❌'}")
    print(f"Telegram:      {'✅' if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else '❌'}")
    print(f"Scheduler:     {'✅' if scheduler and scheduler.running else '❌'}")
    print(f"{'='*60}")
    print(f"API Docs:      /docs")
    print(f"Health:        /health")
    print(f"Analyze:       POST /analyze")
    print(f"Golden Hours:  Tue-Thu 10-12 & 17-19 Tehran")
    print(f"{'='*60}\n")
    
    # پیام شروع به تلگرام
    try:
        start_msg = f"""
🚀 <b>ربات آسمان فعال شد</b>
🕒 {datetime.now(pytz.timezone('Asia/Tehran')).strftime('%H:%M')}
🌐 v{API_VERSION}
💰 BTC: ${btc_data['price']:,.0f}

✅ سیستم آماده ارائه خدمات
🔄 @AsemanSignals
        """
        send_telegram_auto(start_msg.strip())
    except:
        pass
    
    logger.info("✅ Startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """خاموش کردن سیستم"""
    logger.info("🛑 Shutting down...")
    
    # توقف زمان‌بند
    if 'scheduler' in globals() and scheduler:
        scheduler.shutdown()
        logger.info("✅ Scheduler stopped")
    
    # پیام خاموش شدن
    try:
        shutdown_msg = f"""
🛑 <b>ربات آسمان خاموش شد</b>
🕒 {datetime.now(pytz.timezone('Asia/Tehran')).strftime('%H:%M')}

📊 تا راه‌اندازی مجدد...
🔄 @AsemanSignals
        """
        send_telegram_auto(shutdown_msg.strip())
    except:
        pass
    
    logger.info("✅ Shutdown completed")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info" if not DEBUG_MODE else "debug",
        access_log=False,
        server_header=False
    )