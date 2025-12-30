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
from fastapi.responses import HTMLResponse

# اضافه کردن مسیر جاری به پایتون برای پیدا کردن فایل‌ها
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# CONFIGURATION
# ==============================================================================

API_VERSION = "8.5.3"
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
    targets_formatted: Optional[List[str]] = None
    profit_percentages: Optional[List[float]] = None
    risk_reward: Optional[float] = None
    momentum_score: float
    user_message: str
    is_risky_for_retail: bool
    execution_type: str
    signal_id: str
    timestamp: str
    confidence: float
    timeframe: str

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
            "performance": "/v1/performance",
            "signal_details": "POST /v1/signal-details",
            "signal_html": "GET /v1/signal-html/{symbol}"
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
        
        # اطمینان از وجود کلیدهای ضروری
        required_keys = ['targets', 'stop_loss', 'entry_price', 'confidence']
        for key in required_keys:
            if key not in result:
                result[key] = [] if key == 'targets' else 0
        
        # فرمت کردن تارگت‌ها برای نمایش بهتر
        if 'targets' in result and result['targets']:
            # اگر کمتر از 3 تارگت داریم، کامل کنیم
            while len(result['targets']) < 3:
                if result['targets']:
                    result['targets'].append(result['targets'][-1] * 1.01)  # 1% افزایش
                else:
                    result['targets'] = [0, 0, 0]
        
        # فرمت کردن اعداد برای نمایش
        entry_price = result.get('entry_price', 0)
        formatted_entry = f"{entry_price:.2f}" if entry_price >= 1 else f"{entry_price:.5f}"
        formatted_sl = f"{result.get('stop_loss', 0):.2f}" if result.get('stop_loss', 0) >= 1 else f"{result.get('stop_loss', 0):.5f}"
        
        formatted_targets = []
        profit_percentages = []
        
        if result['targets']:
            for target in result['targets'][:3]:
                formatted_target = f"{target:.2f}" if target >= 1 else f"{target:.5f}"
                formatted_targets.append(formatted_target)
                
                if entry_price > 0:
                    profit_pct = ((target - entry_price) / entry_price) * 100
                    profit_percentages.append(round(profit_pct, 2))
                else:
                    profit_percentages.append(0)
        
        result.update({
            "processing_time_ms": processing_time,
            "version": API_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_hit": False,
            "timeframe": request.timeframe,
            "entry_price_formatted": formatted_entry,
            "stop_loss_formatted": formatted_sl,
            "targets_formatted": formatted_targets,
            "profit_percentages": profit_percentages
        })
        
        logger.info(f"✅ Analysis: {request.symbol} -> {result.get('signal', 'UNKNOWN')} ({processing_time}ms)")
        
        # ارسال به تلگرام برای سیگنال‌های قوی
        if result.get("signal") != "HOLD" and result.get("confidence", 0) > 0.6:
            try:
                targets = result.get("targets", [])
                sl = result.get("stop_loss", 0)
                entry_price = result.get("entry_price", 0)
                
                # فرمت کردن متن برای نمایش ۳ تارگت
                target_text = ""
                if targets and len(targets) >= 3:
                    # آیکون‌ها و برچسب‌ها
                    icons = ["🎯", "🚀", "💎"]
                    labels = ["T1 (Safe)", "T2 (Liq)", "T3 (SMC)"]
                    
                    for i in range(min(3, len(targets))):
                        icon = icons[i] if i < len(icons) else "📍"
                        label = labels[i] if i < len(labels) else f"T{i+1}"
                        target_value = targets[i]
                        
                        # محاسبه درصد سود نسبت به ورود
                        if entry_price > 0:
                            profit_pct = ((target_value - entry_price) / entry_price) * 100
                            target_text += f"{icon} <b>{label}:</b> <code>{target_value:.5f}</code> ({profit_pct:+.2f}%)\n"
                        else:
                            target_text += f"{icon} <b>{label}:</b> <code>{target_value:.5f}</code>\n"
                
                # محاسبه ریسک به ریوارد اگر تارگت و استاپ وجود دارد
                rr_text = ""
                if entry_price > 0 and sl > 0 and targets and len(targets) > 0:
                    risk = abs(entry_price - sl)
                    if risk > 0:
                        avg_target = sum(targets[:min(3, len(targets))]) / min(3, len(targets))
                        reward = abs(avg_target - entry_price)
                        rr_ratio = reward / risk if risk > 0 else 0
                        rr_text = f"📊 R/R: 1:{rr_ratio:.1f}\n"
                
                telegram_msg = f"""
🔔 <b>سیگنال جدید: {request.symbol}</b>
📈 جهت: <b>{result.get('signal')}</b>
💰 ورود: <code>{entry_price:.5f}</code>
🛑 استاپ: <code>{sl:.5f}</code>

{target_text}
{rr_text}📈 اطمینان: {result.get('confidence', 0)*100:.1f}%
💡 {result.get('momentum_message', '')}
⏰ تایم‌فریم: {request.timeframe}
🔄 @AsemanSignals
                """
                send_telegram_auto(telegram_msg.strip())
                
                logger.info(f"📤 Telegram alert sent for {request.symbol}")
                
            except Exception as telegram_error:
                logger.error(f"❌ Telegram alert failed: {telegram_error}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Analysis error for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"خطای تحلیل: {str(e)[:80]}"
        )

@app.post("/v1/signal-details")
async def get_signal_details(request: ScalpRequest):
    """دریافت جزئیات کامل سیگنال با تارگت‌های فرمت شده"""
    result = await analyze(request)
    
    if result.get("signal") == "HOLD":
        return result
    
    # محاسبه ریسک به ریوارد
    risk_reward = None
    entry_price = result.get("entry_price", 0)
    sl = result.get("stop_loss", 0)
    targets = result.get("targets", [])
    
    if entry_price > 0 and sl > 0 and targets:
        risk = abs(entry_price - sl)
        if risk > 0:
            avg_target = sum(targets[:min(3, len(targets))]) / min(3, len(targets))
            reward = abs(avg_target - entry_price)
            risk_reward = round(reward / risk, 2)
    
    # ساخت ساختار نمایش تارگت‌ها
    targets_display = []
    for i in range(3):
        if i < len(targets):
            target = targets[i]
            profit_pct = ((target - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            
            targets_display.append({
                "target": target,
                "formatted": f"{target:.2f}" if target >= 1 else f"{target:.5f}",
                "profit_percent": round(profit_pct, 2),
                "icon": ["🎯", "🚀", "💎"][i],
                "label": ["Target 1 (Safe)", "Target 2 (Liquidity)", "Target 3 (SMC)"][i]
            })
        else:
            targets_display.append({
                "target": 0,
                "formatted": "N/A",
                "profit_percent": 0,
                "icon": ["🎯", "🚀", "💎"][i],
                "label": ["Target 1 (Safe)", "Target 2 (Liquidity)", "Target 3 (SMC)"][i]
            })
    
    # اضافه کردن اطلاعات فرمت شده به نتیجه
    result.update({
        "risk_reward": risk_reward,
        "targets_display": targets_display,
        "summary": {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": result.get('signal'),
            "confidence": f"{result.get('confidence', 0)*100:.1f}%",
            "entry_price": result.get('entry_price'),
            "stop_loss": result.get('stop_loss'),
            "risk_reward": f"1:{risk_reward}" if risk_reward else "N/A"
        }
    })
    
    return result

@app.post("/v1/analyze")
async def analyze_pair(request: ScalpRequest):
    """Endpoint سازگاری"""
    return await analyze(request)

@app.post("/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """Legacy endpoint for backward compatibility"""
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
        
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
        results = []
        
        for symbol in symbols:
            try:
                data = get_market_data_with_fallback(symbol, "1h", 50)
                if data and len(data) >= 20:
                    result = get_enhanced_scalp_signal(data, symbol, "1h")
                    if result and result.get("signal") != "HOLD":
                        results.append({
                            "symbol": symbol,
                            "signal": result.get("signal", "HOLD"),
                            "confidence": result.get("confidence", 0),
                            "price": result.get("current_price", 0),
                            "targets": result.get("targets", [])[:3]
                        })
            except Exception as e:
                logger.debug(f"Scan error for {symbol}: {e}")
                continue
        
        # ارسال بهترین سیگنال به تلگرام
        if results:
            best_signal = max(results, key=lambda x: x.get("confidence", 0))
            if best_signal.get("confidence", 0) > 0.7:
                try:
                    scan_msg = f"""
📊 <b>اسکن بازار آسمان</b>
🏆 بهترین سیگنال: {best_signal['symbol']}
🎯 جهت: {best_signal['signal']}
📈 اطمینان: {best_signal['confidence']*100:.1f}%
💰 قیمت: {best_signal['price']:,.2f}$

🔄 @AsemanSignals
                    """
                    send_telegram_auto(scan_msg.strip())
                except:
                    pass
        
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
        "modules": {
            "utils": UTILS_AVAILABLE,
            "pandas": HAS_PANDAS,
            "pandas_ta": HAS_PANDAS_TA,
            "scalper_engine": COLLECTORS_AVAILABLE
        },
        "memory_mb": round(os.sys.getsizeof({}) / 1024 / 1024, 2),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ==============================================================================
# HTML ENDPOINT برای نمایش زیبا
# ==============================================================================

@app.get("/v1/signal-html/{symbol}")
async def get_signal_html(symbol: str, timeframe: str = "5m"):
    """دریافت سیگنال به فرمت HTML برای نمایش در وب"""
    
    # دریافت سیگنال
    request = ScalpRequest(symbol=symbol, timeframe=timeframe)
    result = await analyze(request)
    
    if result.get("signal") == "HOLD":
        html_content = f"""
        <div class="signal-card hold">
            <h3>📊 {symbol} - {timeframe}</h3>
            <p class="hold-text">⏸️ حالت انتظار - فعلاً سیگنالی وجود ندارد</p>
        </div>
        """
    else:
        targets = result.get("targets", [])
        entry = result.get("entry_price", 0)
        sl = result.get("stop_loss", 0)
        
        # ساخت HTML برای تارگت‌ها
        targets_html = ""
        if targets:
            for i in range(min(3, len(targets))):
                target = targets[i]
                icon = ["🎯", "🚀", "💎"][i]
                label = ["تارگت ۱ (ایمن)", "تارگت ۲ (نقدینگی)", "تارگت ۳ (SMC)"][i]
                
                if entry > 0:
                    profit = ((target - entry) / entry) * 100
                    profit_class = "profit-positive" if profit > 0 else "profit-negative"
                    profit_text = f"<span class='{profit_class}'>({profit:+.2f}%)</span>"
                else:
                    profit_text = ""
                
                targets_html += f"""
                <div class="target-row">
                    <span class="target-icon">{icon}</span>
                    <span class="target-label">{label}:</span>
                    <span class="target-value">{target:.5f}</span>
                    {profit_text}
                </div>
                """
        
        html_content = f"""
        <div class="signal-card {result.get('signal', '').lower()}">
            <h3>🔔 سیگنال {symbol} - {timeframe}</h3>
            <div class="signal-header">
                <span class="signal-direction">{result.get('signal')}</span>
                <span class="confidence">{result.get('confidence', 0)*100:.1f}% اطمینان</span>
            </div>
            
            <div class="price-info">
                <div class="price-row">
                    <span>💰 قیمت ورود:</span>
                    <span class="price-value">{entry:.5f}</span>
                </div>
                <div class="price-row">
                    <span>🛑 استاپ لاس:</span>
                    <span class="price-value">{sl:.5f}</span>
                </div>
            </div>
            
            <div class="targets-section">
                <h4>🎯 تارگت‌ها:</h4>
                {targets_html}
            </div>
            
            <div class="message">
                <p>💡 {result.get('momentum_message', '')}</p>
            </div>
            
            <div class="footer">
                <span class="timestamp">{result.get('timestamp', '')}</span>
                <span class="badge">@AsemanSignals</span>
            </div>
        </div>
        """
    
    # استایل CSS
    style = """
    <style>
    .signal-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 400px;
        margin: 20px auto;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid #2d3748;
    }
    
    .signal-card.buy {
        border-left: 5px solid #10b981;
    }
    
    .signal-card.sell {
        border-left: 5px solid #ef4444;
    }
    
    .signal-card.hold {
        border-left: 5px solid #f59e0b;
    }
    
    .signal-card h3 {
        margin: 0 0 15px 0;
        color: #e2e8f0;
        font-size: 18px;
    }
    
    .signal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid #2d3748;
    }
    
    .signal-direction {
        font-size: 20px;
        font-weight: bold;
        padding: 5px 15px;
        border-radius: 25px;
    }
    
    .signal-card.buy .signal-direction {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }
    
    .signal-card.sell .signal-direction {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    
    .confidence {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        padding: 5px 10px;
        border-radius: 10px;
        font-size: 14px;
    }
    
    .price-info {
        margin-bottom: 20px;
    }
    
    .price-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        padding: 8px 0;
        border-bottom: 1px dashed #4a5568;
    }
    
    .price-value {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #fbbf24;
    }
    
    .targets-section {
        background: rgba(0,0,0,0.2);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .targets-section h4 {
        margin-top: 0;
        color: #cbd5e0;
        font-size: 16px;
    }
    
    .target-row {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        padding: 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
    }
    
    .target-icon {
        margin-right: 10px;
        font-size: 18px;
    }
    
    .target-label {
        flex-grow: 1;
        color: #a0aec0;
    }
    
    .target-value {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #fbbf24;
        margin-right: 10px;
    }
    
    .profit-positive {
        color: #10b981;
        font-weight: bold;
    }
    
    .profit-negative {
        color: #ef4444;
        font-weight: bold;
    }
    
    .message {
        background: rgba(59, 130, 246, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-right: 3px solid #3b82f6;
    }
    
    .message p {
        margin: 0;
        color: #cbd5e0;
        line-height: 1.5;
    }
    
    .footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 15px;
        border-top: 1px solid #2d3748;
        font-size: 12px;
        color: #718096;
    }
    
    .badge {
        background: rgba(139, 92, 246, 0.2);
        color: #8b5cf6;
        padding: 4px 10px;
        border-radius: 5px;
    }
    
    .hold-text {
        text-align: center;
        padding: 30px;
        color: #f59e0b;
        font-size: 16px;
    }
    </style>
    """
    
    return HTMLResponse(content=style + html_content)

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
    cache_warmup()
    global scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_price_cache, 'interval', minutes=5)
    scheduler.add_job(golden_hour_job, 'interval', minutes=30)
    scheduler.start()
    logger.info(f"🚀 System v{API_VERSION} Online")
    
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
    print(f"{'='*60}\n")
    
    # ارسال پیام شروع
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))