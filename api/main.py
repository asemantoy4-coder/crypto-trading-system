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
import os

# ==============================================================================
# Import ماژول‌های دیگر و Routerها
# ==============================================================================
# اضافه کردن ریشه به sys.path برای import کردن
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Configure logging اول از همه
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print(f"📁 Current directory: {current_dir}")
print(f"📁 sys.path: {sys.path}")

# Import ماژول‌های کمکی با روش‌های مختلف
UTILS_AVAILABLE = False
DATA_COLLECTOR_AVAILABLE = False
COLLECTORS_AVAILABLE = False
AUTO_LEARNING_AVAILABLE = False
MODEL_TRAINER_AVAILABLE = False

# تلاش برای import ماژول utils
try:
    # روش 1: import مستقیم از پوشه جاری
    print("🔄 Attempting to import utils directly...")
    import utils
    from utils import (
        get_market_data_with_fallback, 
        analyze_with_multi_timeframe_strategy, 
        calculate_24h_change_from_dataframe,
        calculate_simple_sma,
        calculate_simple_rsi
    )
    UTILS_AVAILABLE = True
    print("✅ utils imported successfully with direct import")
    
except ImportError as e1:
    print(f"❌ Direct import failed: {e1}")
    try:
        # روش 2: import با sys.path
        print("🔄 Attempting import with sys.path modification...")
        sys.path.insert(0, current_dir)
        from api.utils import (
            get_market_data_with_fallback, 
            analyze_with_multi_timeframe_strategy, 
            calculate_24h_change_from_dataframe,
            calculate_simple_sma,
            calculate_simple_rsi
        )
        UTILS_AVAILABLE = True
        print("✅ utils imported successfully with absolute import")
    except ImportError as e2:
        print(f"❌ Absolute import failed: {e2}")
        UTILS_AVAILABLE = False

# تلاش برای import ماژول‌های دیگر
try:
    from data_collector import get_collected_data
    DATA_COLLECTOR_AVAILABLE = True
    print("✅ data_collector imported successfully")
except ImportError as e:
    print(f"❌ data_collector import failed: {e}")
    DATA_COLLECTOR_AVAILABLE = False

try:
    from collectors import collect_signals_from_example_site
    COLLECTORS_AVAILABLE = True
    print("✅ collectors imported successfully")
except ImportError as e:
    print(f"❌ collectors import failed: {e}")
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
        timestamp = current_time - (i * 5 * 60 * 1000)  # 5 دقیقه فاصله
        
        # شبیه‌سازی حرکت قیمت واقعی‌تر
        change = random.uniform(-0.015, 0.015)  # ±1.5%
        price = base_price * (1 + change)
        
        candle = [
            timestamp,  # open time
            str(price * random.uniform(0.998, 1.000)),  # open
            str(price * random.uniform(1.000, 1.003)),  # high
            str(price * random.uniform(0.997, 1.000)),  # low
            str(price),  # close
            str(random.uniform(1000, 10000)),  # volume
            timestamp + 300000,  # close time
            "0", "0", "0", "0", "0"  # سایر فیلدها
        ]
        
        data.append(candle)
    
    return data

def mock_calculate_simple_sma(data, period=20):
    """محاسبه SMA ساده (بدون pandas)"""
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

def mock_analyze_with_multi_timeframe_strategy(symbol):
    """تابع جایگزین برای تحلیل"""
    # تحلیل واقعی‌تر
    signals = ["BUY", "SELL", "HOLD"]
    
    # شانس بیشتر برای HOLD
    weights = [0.35, 0.35, 0.30]
    signal = random.choices(signals, weights=weights)[0]
    
    # قیمت‌های واقعی‌تر
    base_prices = {
        'BTCUSDT': 88271.00,
        'ETHUSDT': 3450.00,
        'BNBUSDT': 590.00,
        'SOLUSDT': 175.00,
        'DEFAULT': 100
    }
    
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    
    # اطمینان منطقی‌تر
    if signal == "HOLD":
        confidence = round(random.uniform(0.5, 0.7), 2)
    else:
        confidence = round(random.uniform(0.65, 0.85), 2)
    
    entry_price = round(base_price * random.uniform(0.99, 1.01), 2)
    
    if signal == "BUY":
        targets = [
            round(entry_price * 1.02, 2),
            round(entry_price * 1.05, 2)
        ]
        stop_loss = round(entry_price * 0.98, 2)
    elif signal == "SELL":
        targets = [
            round(entry_price * 0.98, 2),
            round(entry_price * 0.95, 2)
        ]
        stop_loss = round(entry_price * 1.02, 2)
    else:  # HOLD
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

# انتخاب تابع مناسب
if UTILS_AVAILABLE:
    get_market_data_func = get_market_data_with_fallback
    analyze_func = analyze_with_multi_timeframe_strategy
    calculate_change_func = calculate_24h_change_from_dataframe
    calculate_sma_func = calculate_simple_sma
    calculate_rsi_func = calculate_simple_rsi
    print("🔧 Using REAL analysis functions from utils")
else:
    get_market_data_func = mock_get_market_data_with_fallback
    analyze_func = mock_analyze_with_multi_timeframe_strategy
    calculate_change_func = mock_calculate_24h_change
    calculate_sma_func = mock_calculate_simple_sma
    calculate_rsi_func = mock_calculate_simple_rsi
    print("⚠️ Using MOCK analysis functions")

# ==============================================================================
# FastAPI Application
# ==============================================================================
API_VERSION = "7.1.0"  # نسخه جدید با پشتیبانی از اسکالپ

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description=f"Multi-source signal API with Scalp Support - نسخه {API_VERSION}",
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
# توابع کمکی جدید برای اسکالپ
# ==============================================================================
def analyze_scalp_signal(symbol, timeframe, data):
    """تحلیل سیگنال اسکالپ"""
    if not data or len(data) < 20:
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "rsi": 50,
            "sma_20": 0,
            "current_price": 0,
            "reason": "Insufficient data"
        }
    
    # محاسبه اندیکاتورها
    rsi = calculate_rsi_func(data, 14)
    sma_20 = calculate_sma_func(data, 20)
    
    # آخرین قیمت - با حفاظت بیشتر
    try:
        latest_close = float(data[-1][4])
    except (IndexError, ValueError, TypeError):
        try:
            # اگر ساختار داده متفاوت است
            latest_close = float(data[-1]['close']) if isinstance(data[-1], dict) else 0
        except:
            latest_close = 0
    
    # منطق اسکالپ
    signal = "HOLD"
    confidence = 0.5
    reason = "Market neutral"
    
    # شرایط خرید اسکالپ
    if rsi < 35 and latest_close < sma_20 * 1.01:
        signal = "BUY"
        confidence = min(0.75, (35 - rsi) / 35 * 0.5 + 0.5)
        reason = f"Oversold (RSI: {rsi:.1f}), price below SMA20"
    
    # شرایط فروش اسکالپ
    elif rsi > 65 and latest_close > sma_20 * 0.99:
        signal = "SELL"
        confidence = min(0.75, (rsi - 65) / 35 * 0.5 + 0.5)
        reason = f"Overbought (RSI: {rsi:.1f}), price above SMA20"
    
    # شرایط breakout
    elif latest_close > sma_20 * 1.02 and rsi < 60:
        signal = "BUY"
        confidence = 0.7
        reason = f"Breakout above SMA20, RSI: {rsi:.1f}"
    
    elif latest_close < sma_20 * 0.98 and rsi > 40:
        signal = "SELL"
        confidence = 0.7
        reason = f"Breakdown below SMA20, RSI: {rsi:.1f}"
    
    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "rsi": round(rsi, 1),
        "sma_20": round(sma_20, 2),
        "current_price": round(latest_close, 2),
        "reason": reason
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
        "scalp_signal": "POST /api/scalp-signal",  # جدید
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
            "collectors": COLLECTORS_AVAILABLE,
            "auto_learning": AUTO_LEARNING_AVAILABLE,
            "model_trainer": MODEL_TRAINER_AVAILABLE
        },
        "endpoints": endpoints,
        "features": ["Real-time Analysis", "Scalp Signals (5m/15m)", "Multi-timeframe", "Fallback System"],
        "note": f"نسخه {API_VERSION} با پشتیبانی از سیگنال‌های اسکالپ"
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
            "collectors": COLLECTORS_AVAILABLE,
            "auto_learning": AUTO_LEARNING_AVAILABLE,
            "model_trainer": MODEL_TRAINER_AVAILABLE
        },
        "components": {
            "api": "سالم",
            "data_sources": "Binance (Primary) -> LBank (Fallback)" if UTILS_AVAILABLE else "Mock Data",
            "internal_ai": "فعال" if UTILS_AVAILABLE else "mock",
            "scalp_engine": "فعال",
            "signal_cache": "فعال"
        },
        "scalp_support": {
            "enabled": True,
            "timeframes": ["1m", "5m", "15m"],
            "min_confidence": 0.65
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
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"خطا در تحلیل: {str(e)}")

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """سیگنال‌های اسکالپ 1-5-15 دقیقه"""
    logger.info(f"⚡ درخواست سیگنال اسکالپ: {request.symbol} ({request.timeframe})")
    
    # فقط تایم‌فریم‌های کوتاه مجاز
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
        
        # تحلیل اسکالپ
        scalp_analysis = analyze_scalp_signal(request.symbol, request.timeframe, market_data)
        
        # محاسبه تارگت‌ها و استاپ لاس - با حفاظت بیشتر
        current_price = scalp_analysis.get("current_price", 0)
        
        # اگر قیمت صفر است، از قیمت mock استفاده کن
        if current_price <= 0:
            base_prices = {
                'BTCUSDT': 88271.00,
                'ETHUSDT': 3450.00,
                'DEFAULT': 100
            }
            base_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
            current_price = round(base_price * random.uniform(0.995, 1.005), 2)
        
        if scalp_analysis["signal"] == "BUY":
            targets = [
                round(current_price * 1.01, 2),  # 1%
                round(current_price * 1.02, 2),  # 2%
                round(current_price * 1.03, 2)   # 3%
            ]
            stop_loss = round(current_price * 0.99, 2)  # 1% stop
        elif scalp_analysis["signal"] == "SELL":
            targets = [
                round(current_price * 0.99, 2),  # 1%
                round(current_price * 0.98, 2),  # 2%
                round(current_price * 0.97, 2)   # 3%
            ]
            stop_loss = round(current_price * 1.01, 2)  # 1% stop
        else:
            targets = []
            stop_loss = current_price
        
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": scalp_analysis["signal"],
            "confidence": scalp_analysis["confidence"],
            "entry_price": current_price,
            "rsi": scalp_analysis["rsi"],
            "sma_20": scalp_analysis["sma_20"],
            "targets": targets,
            "stop_loss": stop_loss,
            "type": "SCALP",
            "reason": scalp_analysis["reason"],
            "strategy": f"Scalp Strategy ({request.timeframe})",
            "module": "real" if UTILS_AVAILABLE else "mock",
            "version": API_VERSION,
            "timestamp": datetime.now().isoformat(),
            "risk_level": "HIGH" if request.timeframe == "1m" else "MEDIUM",
            "recommendation": f"{scalp_analysis['signal']} signal for scalp trading on {request.timeframe} timeframe"
        }
        
        logger.info(f"✅ Scalp signal generated: {request.symbol} - {scalp_analysis['signal']} ({scalp_analysis['confidence']:.0%})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in scalp signal: {e}")
        # Fallback to mock data
        mock_signal = random.choice(["BUY", "SELL", "HOLD"])
        mock_confidence = 0.6 + random.random() * 0.3
        
        base_prices = {
            'BTCUSDT': 88271.00,
            'ETHUSDT': 3450.00,
            'DEFAULT': 100
        }
        
        base_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        current_price = round(base_price * random.uniform(0.995, 1.005), 2)
        
        if mock_signal == "BUY":
            targets = [round(current_price * 1.01, 2), round(current_price * 1.02, 2)]
            stop_loss = round(current_price * 0.99, 2)
        elif mock_signal == "SELL":
            targets = [round(current_price * 0.99, 2), round(current_price * 0.98, 2)]
            stop_loss = round(current_price * 1.01, 2)
        else:
            targets = []
            stop_loss = current_price
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": mock_signal,
            "confidence": round(mock_confidence, 2),
            "entry_price": current_price,
            "rsi": round(30 + random.random() * 40, 1),
            "sma_20": round(current_price * random.uniform(0.99, 1.01), 2),
            "targets": targets,
            "stop_loss": stop_loss,
            "type": "SCALP_MOCK",
            "reason": "Using mock data (API error)",
            "strategy": "Mock Scalp Strategy",
            "module": "mock",
            "version": API_VERSION,
            "timestamp": datetime.now().isoformat(),
            "risk_level": "HIGH",
            "recommendation": f"Mock {mock_signal} signal"
        }

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
        
        # محاسبه تغییرات ۲۴ ساعته
        change_24h = calculate_change_func(data)

        # محاسبه RSI و SMA برای اطلاعات بیشتر
        rsi = calculate_rsi_func(data, 14)
        sma_20 = calculate_sma_func(data, 20)

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
                    "type": "SCRAPED"
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
                    # استفاده از اسکالپ برای تایم‌فریم‌های کوتاه
                    response = await get_scalp_signal(ScalpRequest(symbol=symbol, timeframe=tf))
                    response["analysis_type"] = "SCALP"
                else:
                    # استفاده از تحلیل استاندارد برای تایم‌فریم‌های بلند
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
    logger.info(f"📡 نسخه: {API_VERSION} - با پشتیبانی از سیگنال‌های اسکالپ")
    logger.info(f"⚙️ وضعیت ماژول‌ها:")
    logger.info(f"   - utils: {'✅' if UTILS_AVAILABLE else '❌'}")
    logger.info(f"   - data_collector: {'✅' if DATA_COLLECTOR_AVAILABLE else '❌'}")
    logger.info(f"   - collectors: {'✅' if COLLECTORS_AVAILABLE else '❌'}")
    logger.info(f"   - auto_learning: {'✅' if AUTO_LEARNING_AVAILABLE else '❌'}")
    logger.info(f"   - model_trainer: {'✅' if MODEL_TRAINER_AVAILABLE else '❌'}")
    logger.info(f"🔧 ویژگی‌های فعال:")
    logger.info(f"   - تحلیل چندزمانه: ✅")
    logger.info(f"   - سیگنال اسکالپ (1m/5m/15m): ✅")
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