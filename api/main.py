"""
Crypto AI Trading System v7.0 - Render Optimized & Final Version
با پشتیبانی از Binance/LBank API و سازگار با پلتفرم Render
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

# ==============================================================================
# Import ماژول‌های دیگر و Routerها
# ==============================================================================
# اضافه کردن ریشه به sys.path برای import کردن config
sys.path.append('.')
try:
    from config import get_version, get_all_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Import ماژول‌های کمکی
try:
    from utils import get_market_data_with_fallback, analyze_with_multi_timeframe_strategy, calculate_24h_change_from_dataframe
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    
try:
    from data_collector import get_collected_data
    DATA_COLLECTOR_AVAILABLE = True
except ImportError:
    DATA_COLLECTOR_AVAILABLE = False
    
try:
    from collectors import collect_signals_from_example_site
    COLLECTORS_AVAILABLE = True
except ImportError:
    COLLECTORS_AVAILABLE = False

# Import Routerهای ماژولار (اگر وجود دارند)
try:
    from . import auto_learning
    AUTO_LEARNING_AVAILABLE = True
except ImportError:
    AUTO_LEARNING_AVAILABLE = False
    
try:
    from . import model_trainer
    MODEL_TRAINER_AVAILABLE = True
except ImportError:
    MODEL_TRAINER_AVAILABLE = False

# ==============================================================================
# Configure logging
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# Pydantic Models
# ==============================================================================
class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class SignalResponse(BaseModel):
    status: str
    count: int
    last_updated: str
    signals: List[Dict[str, Any]]
    sources: Dict[str, int]

# ==============================================================================
# توابع جایگزین برای زمانی که ماژول‌ها در دسترس نیستند
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
    except:
        pass
    
    # داده mock
    data = []
    base_price = 50000 if "BTC" in symbol else 3000
    current_time = int(datetime.now().timestamp() * 1000)
    
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)
        price = base_price * (1 + random.uniform(-0.02, 0.02))
        
        data.append([
            timestamp,
            str(price * 0.998),
            str(price * 1.005),
            str(price * 0.995),
            str(price),
            str(random.uniform(1000, 10000)),
            timestamp + 300000,
            "0", "0", "0", "0", "0"
        ])
    
    return data

def mock_analyze_with_multi_timeframe_strategy(symbol):
    """تابع جایگزین برای تحلیل"""
    signals = ["BUY", "SELL", "HOLD"]
    signal = random.choice(signals)
    
    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": round(random.uniform(0.6, 0.9), 2),
        "entry_price": round(random.uniform(50000, 51000), 2),
        "targets": [
            round(random.uniform(52000, 53000), 2),
            round(random.uniform(54000, 55000), 2)
        ],
        "stop_loss": round(random.uniform(48000, 49000), 2),
        "strategy": "Multi-Timeframe Mock Analysis"
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
    return round(random.uniform(-5, 5), 2)

# انتخاب تابع مناسب
if UTILS_AVAILABLE:
    get_market_data_func = get_market_data_with_fallback
    analyze_func = analyze_with_multi_timeframe_strategy
    calculate_change_func = calculate_24h_change_from_dataframe
else:
    get_market_data_func = mock_get_market_data_with_fallback
    analyze_func = mock_analyze_with_multi_timeframe_strategy
    calculate_change_func = mock_calculate_24h_change

# ==============================================================================
# FastAPI Application
# ==============================================================================
API_VERSION = get_version() if HAS_CONFIG else "7.0.0"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description=f"Multi-source signal API with Fallback (Binance -> LBank) - نسخه {API_VERSION}",
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
# افزودن Routerهای ماژولار (اگر موجود باشند)
# ==============================================================================
if AUTO_LEARNING_AVAILABLE:
    app.include_router(auto_learning.router)
    logger.info("✅ Auto Learning router added")

if MODEL_TRAINER_AVAILABLE:
    app.include_router(model_trainer.router)
    logger.info("✅ Model Trainer router added")

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def read_root():
    """صفحه اصلی - با استفاده از نسخه از config"""
    version = get_version() if HAS_CONFIG else "7.0.0"
    all_config = get_all_config() if HAS_CONFIG else {}
    
    endpoints = {
        "health": "GET /api/health",
        "signals": "GET /api/signals",
        "analyze": "POST /api/analyze",
        "market": "GET /market/{symbol}",
        "scraped_signals": "GET /signals/scraped",
        "docs": "GET /api/docs"
    }
    
    # اضافه کردن endpoints ماژولار اگر موجود باشند
    if AUTO_LEARNING_AVAILABLE:
        endpoints["auto_learning"] = "GET /auto-learn/status"
    if MODEL_TRAINER_AVAILABLE:
        endpoints["model_trainer"] = "GET /models/status"
    
    return {
        "message": f"🚀 سیستم تحلیل معاملاتی ارز دیجیتال v{version}",
        "status": "در حال اجرا",
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "config_status": "فعال" if HAS_CONFIG else "fallback",
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE,
            "auto_learning": AUTO_LEARNING_AVAILABLE,
            "model_trainer": MODEL_TRAINER_AVAILABLE
        },
        "endpoints": endpoints,
        "data_sources": ["Binance API", "LBank API", "GitHub", "تحلیل داخلی"],
        "note": f"نسخه {version} با مکانیزم جایگزینی (Binance -> LBank)",
        **({"config_info": all_config} if all_config else {})
    }

@app.get("/api/health")
async def health_check():
    """بررسی سلامت سیستم"""
    version = get_version() if HAS_CONFIG else "7.0.0"
    return {
        "status": "سالم",
        "timestamp": datetime.now().isoformat(),
        "version": version,
        "config_module": "فعال" if HAS_CONFIG else "fallback",
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE,
            "auto_learning": AUTO_LEARNING_AVAILABLE,
            "model_trainer": MODEL_TRAINER_AVAILABLE
        },
        "components": {
            "api": "سالم",
            "data_sources": "Binance (Primary) -> LBank (Fallback)",
            "internal_ai": "فعال" if UTILS_AVAILABLE else "mock",
            "signal_cache": "فعال"
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
        # تولید سیگنال داخلی با استفاده از موتور تحلیل
        analysis = analyze_func(symbol.upper() if symbol else "BTCUSDT")
        
        signals = [{
            "symbol": analysis["symbol"],
            "timeframe": "5m",
            "signal": analysis["signal"],
            "confidence": analysis["confidence"],
            "entry_price": analysis["entry_price"],
            "targets": analysis["targets"],
            "stop_loss": analysis["stop_loss"],
            "reason": f"تحلیل داخلی MTF برای {analysis['symbol']}",
            "source": "internal_ai",
            "author": "موتور تحلیل محلی",
            "strategy": analysis.get("strategy", "تحلیل چندزمانی"),
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
        response_dict["api_version"] = get_version() if HAS_CONFIG else "7.0.0"
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
        
        version = get_version() if HAS_CONFIG else "7.0.0"
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": analysis["signal"],
            "confidence": analysis["confidence"],
            "entry_price": analysis["entry_price"],
            "targets": analysis["targets"],
            "stop_loss": analysis["stop_loss"],
            "version": version,
            "strategy": analysis.get("strategy", "تحلیل چندزمانی"),
            "module": "real" if UTILS_AVAILABLE else "mock",
            "recommendation": f"سیگنال {analysis['signal']} با {analysis['confidence']:.0%} اطمینان",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"خطا در تحلیل: {str(e)}")

@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    """
    دریافت داده‌های خام بازار با مکانیزم Fallback (بایننس -> LBank)
    """
    try:
        # استفاده از تابع با مکانیزم جایگزینی
        data = get_market_data_func(symbol, timeframe, limit=50)
        
        if not data:
            raise HTTPException(status_code=404, detail=f"داده‌ای برای نماد {symbol} یافت نشد.")
        
        # آخرین کندل
        latest = data[-1] if isinstance(data, list) and len(data) > 0 else []
        
        if not latest or len(latest) < 6:
            # بازگشت داده mock
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "Mock Data",
                "current_price": round(random.uniform(50000, 51000), 2),
                "high": round(random.uniform(51000, 52000), 2),
                "low": round(random.uniform(49000, 50000), 2),
                "volume": round(random.uniform(1000, 5000), 2),
                "change_24h": round(random.uniform(-5, 5), 2),
                "timestamp": datetime.now().isoformat(),
                "note": "Using mock data"
            }
        
        # محاسبه تغییرات ۲۴ ساعته
        change_24h = calculate_change_func(data)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": "Binance API" if UTILS_AVAILABLE else "Mock Data",
            "current_price": float(latest[4]),
            "high": float(latest[2]),
            "low": float(latest[3]),
            "volume": float(latest[5]),
            "change_24h": change_24h,
            "timestamp": datetime.now().isoformat(),
            "data_points": len(data)
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
                    "timestamp": datetime.now().isoformat()
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

# ==============================================================================
# Startup Event
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    """مقداردهی اولیه هنگام راه‌اندازی"""
    version = get_version() if HAS_CONFIG else "7.0.0"
    
    logger.info("=" * 60)
    logger.info(f"🚀 راه‌اندازی سیستم تحلیل معاملاتی ارز دیجیتال v{version}")
    logger.info(f"📡 نسخه: {version} - با مکانیزم جایگزینی (Binance -> LBank)")
    logger.info(f"⚙️ وضعیت config: {'فعال' if HAS_CONFIG else 'fallback'}")
    logger.info(f"🔧 ماژول‌های فعال:")
    logger.info(f"   - utils: {'✅' if UTILS_AVAILABLE else '❌'}")
    logger.info(f"   - data_collector: {'✅' if DATA_COLLECTOR_AVAILABLE else '❌'}")
    logger.info(f"   - collectors: {'✅' if COLLECTORS_AVAILABLE else '❌'}")
    logger.info(f"   - auto_learning: {'✅' if AUTO_LEARNING_AVAILABLE else '❌'}")
    logger.info(f"   - model_trainer: {'✅' if MODEL_TRAINER_AVAILABLE else '❌'}")
    logger.info("⏰ زمان راه‌اندازی: " + datetime.now().isoformat())
    logger.info("=" * 60)

# For local development
if __name__ == "__main__":
    version = get_version() if HAS_CONFIG else "7.0.0"
    logger.info(f"🚀 شروع سرور محلی v{version} روی پورت 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")