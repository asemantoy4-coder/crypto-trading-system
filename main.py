import os
import sys
import time
import uvicorn
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Final
import numpy as np
from pydantic import BaseModel

# اضافه کردن مسیر جاری به پایتون برای پیدا کردن فایل‌ها
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# CONFIGURATION (با type hints)
# ==============================================================================

# استفاده از Final برای ثابت‌ها (پایتون 3.8+)
API_VERSION: Final[str] = "8.5.1"
DEBUG_MODE: bool = os.environ.get("DEBUG", "False").lower() == "true"

# لاگینگ بهینه برای مصرف کمتر منابع
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger("CryptoAIScalper")

# ==============================================================================
# MODULE IMPORTS (با type hints)
# ==============================================================================

# متغیرهای وضعیت ماژول‌ها
HAS_PANDAS: bool = False
HAS_PANDAS_TA: bool = False
UTILS_AVAILABLE: bool = False
COLLECTORS_AVAILABLE: bool = False
HAS_TDR_ATR: bool = False

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

# ایمپورت ایمن از utils.py
try:
    # نام توابع را دقیقاً با آنچه در utils دارید ست کنید
    from utils import (
        format_binance_price,
        get_enhanced_scalp_signal,
        get_market_data_with_fallback,
        get_momentum_persian_msg  # نام تابع اصلاح شد
    )
    UTILS_AVAILABLE = True
    logger.info("✅ Utils module loaded successfully")
except ImportError as e:
    logger.error(f"❌ Utils Import Error: {e}")
    UTILS_AVAILABLE = False

# بررسی ماژول ScalperEngine
class FallbackScalperEngine:
    """کلاس جایگزین در صورت عدم وجود ScalperEngine"""
    @staticmethod
    def calculate_tdr_advanced(data):
        return 0.5
    
    @staticmethod
    def get_ai_confirmation(*args, **kwargs):
        return "AI not available"
    
    @staticmethod
    def calculate_atr_advanced(data):
        return 0.0, 0.0, 50.0

# متغیر global برای ScalperEngine
ScalperEngine = FallbackScalperEngine

try:
    from scalper_engine import ScalperEngine as RealScalperEngine
    COLLECTORS_AVAILABLE = True
    logger.info("✅ ScalperEngine loaded successfully")
    HAS_TDR_ATR = hasattr(RealScalperEngine, 'calculate_tdr_advanced')
    # استفاده از نسخه واقعی اگر import شد
    ScalperEngine = RealScalperEngine
except ImportError as e:
    logger.error(f"❌ Error importing ScalperEngine: {e}")

# ==============================================================================
# FASTAPI APP
# ==============================================================================

app: FastAPI = FastAPI(
    title="Crypto AI Scalper",
    description="Professional Scalping & Trading Analysis API",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==============================================================================
# MODELS (Pydantic Models)
# ==============================================================================

class ScalpRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    use_ai: bool = False  # تغییر پیش‌فرض به False برای سادگی

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
# PERFORMANCE MONITOR
# ==============================================================================

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.request_times: List[float] = []
    
    def record_request(self, processing_time: float) -> None:
        self.request_times.append(processing_time)
        if len(self.request_times) > 50:
            self.request_times.pop(0)

# ایجاد instance از PerformanceMonitor
performance_monitor: PerformanceMonitor = PerformanceMonitor()

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root() -> dict:
    return {
        "status": "Online",
        "msg": "System is running on Free Tier",
        "version": API_VERSION,
        "modules": {
            "utils": UTILS_AVAILABLE,
            "pandas": HAS_PANDAS,
            "pandas_ta": HAS_PANDAS_TA,
            "scalper_engine": COLLECTORS_AVAILABLE,
            "tdr_atr": HAS_TDR_ATR
        },
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "market_scan": "/v1/market-scan",
            "performance": "/v1/performance"
        }
    }

@app.get("/health")
async def health() -> dict:
    return {
        "status": "Healthy",
        "pandas": HAS_PANDAS,
        "utils": UTILS_AVAILABLE,
        "pandas_ta": HAS_PANDAS_TA,
        "scalper_engine": COLLECTORS_AVAILABLE,
        "tdr_atr": HAS_TDR_ATR,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/analyze")
async def analyze(request: ScalpRequest) -> dict:
    """
    تحلیل اصلی بازار با استفاده از ماژول utils
    """
    start_time = time.time()
    logger.info(f"📊 Analysis Request: {request.symbol} [{request.timeframe}]")
    
    if not UTILS_AVAILABLE:
        raise HTTPException(status_code=503, detail="تحلیلگر آماده نیست")
    
    try:
        # دریافت داده‌های بازار
        data = get_market_data_with_fallback(request.symbol, request.timeframe, 100)
        if not data:
            return {"signal": "HOLD", "message": "خطا در دریافت دیتا"}
        
        # تحلیل سیگنال
        result = get_enhanced_scalp_signal(data, request.symbol, request.timeframe)
        
        # افزودن اطلاعات زمان
        if result and "timestamp" not in result:
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # افزودن زمان پردازش
        processing_time = round((time.time() - start_time) * 1000, 2)
        if result:
            result["processing_time_ms"] = processing_time
            result["version"] = API_VERSION
        
        logger.info(f"✅ Analysis completed in {processing_time}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Analysis Error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"خطا در تحلیل: {str(e)[:100]}"
        )

# ==============================================================================
# COMPATIBILITY ENDPOINTS
# ==============================================================================

@app.post("/v1/analyze")
async def analyze_pair(request: ScalpRequest) -> dict:
    """
    نسخه قدیمی تحلیل برای سازگاری
    """
    return await analyze(request)

@app.post("/scalp-signal")
async def get_scalp_signal(request: ScalpRequest) -> dict:
    """Legacy endpoint for backward compatibility"""
    return await analyze(request)

# ==============================================================================
# MARKET SCANNER
# ==============================================================================

@app.get("/v1/market-scan")
async def market_scanner() -> dict:
    """
    اسکنر بازار برای پیشنهادات لحظه‌ای
    """
    try:
        if not UTILS_AVAILABLE:
            return {
                "status": "warning",
                "message": "ماژول تحلیل در دسترس نیست",
                "data": [],
                "server_time": datetime.now(timezone.utc).isoformat()
            }
        
        # نمادهای محبوب برای اسکن
        popular_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        top_picks = []
        
        for symbol in popular_symbols:
            try:
                data = get_market_data_with_fallback(symbol, "1h", 50)
                if data and len(data) >= 20:
                    result = get_enhanced_scalp_signal(data, symbol, "1h")
                    
                    if result:
                        top_picks.append({
                            "symbol": symbol,
                            "signal": result.get("signal", "HOLD"),
                            "confidence": result.get("confidence", 0.5),
                            "message": result.get("momentum_message", ""),
                            "price": result.get("current_price", 0)
                        })
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        return {
            "status": "success",
            "data": top_picks,
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "total_scanned": len(popular_symbols),
            "successful_scans": len(top_picks)
        }
        
    except Exception as e:
        logger.error(f"Market scanner error: {e}")
        return {
            "status": "error",
            "message": f"خطا در اسکن بازار: {str(e)[:100]}",
            "data": [],
            "server_time": datetime.now(timezone.utc).isoformat()
        }

# ==============================================================================
# MIDDLEWARE
# ==============================================================================

@app.middleware("http")
async def monitor_performance(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time
    performance_monitor.record_request(processing_time)
    
    # Add performance headers
    response.headers["X-Processing-Time"] = str(round(processing_time * 1000, 2))
    response.headers["X-API-Version"] = API_VERSION
    
    return response

@app.get("/v1/performance")
async def get_performance_stats() -> dict:
    """Get system performance statistics"""
    if performance_monitor.request_times:
        avg_latency = np.mean(performance_monitor.request_times) * 1000
    else:
        avg_latency = 0
    
    return {
        "average_latency_ms": round(avg_latency, 2),
        "total_requests": len(performance_monitor.request_times),
        "modules": {
            "utils": UTILS_AVAILABLE,
            "pandas": HAS_PANDAS,
            "pandas_ta": HAS_PANDAS_TA,
            "scalper_engine": COLLECTORS_AVAILABLE,
            "tdr_atr": HAS_TDR_ATR
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ==============================================================================
# STARTUP AND MAIN
# ==============================================================================

@app.on_event("startup")
async def startup_event() -> None:
    """Startup event handler"""
    logger.info(f"🚀 Starting Crypto AI Scalper v{API_VERSION}")
    logger.info(f"📦 Utils Available: {UTILS_AVAILABLE}")
    logger.info(f"📦 Pandas TA: {HAS_PANDAS_TA}")
    logger.info(f"📦 ScalperEngine: {COLLECTORS_AVAILABLE}")
    logger.info(f"📦 TDR ATR: {HAS_TDR_ATR}")
    
    print(f"\n{'=' * 50}")
    print(f"CRYPTO AI SCALPER v{API_VERSION}")
    print(f"{'=' * 50}")
    print("Status: ✅ Online")
    print(f"Utils Module: {'✅ Available' if UTILS_AVAILABLE else '❌ Not Available'}")
    print(f"Pandas TA: {'✅ Available' if HAS_PANDAS_TA else '❌ Not Available'}")
    print(f"Scalper Engine: {'✅ Available' if COLLECTORS_AVAILABLE else '❌ Not Available'}")
    print(f"{'=' * 50}")
    print(f"API Documentation: /docs")
    print(f"Health Check: /health")
    print(f"Main Endpoint: POST /analyze")
    print(f"{'=' * 50}\n")
    
    logger.info("✅ System startup completed successfully!")

# ==============================================================================
# EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # تنظیم پورت برای Render و داکر
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=False  # غیرفعال کردن access log برای مصرف کمتر منابع
    )