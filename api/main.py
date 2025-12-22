"""
Crypto AI Trading System v7.3.0 - Complete Version with Config
با پشتیبانی از Binance/LBank API و سازگار با پلتفرم Render
نسخه کامل با استفاده از config.py برای مدیریت تنظیمات
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from datetime import datetime, timedelta
import logging
from typing import List, Optional, Dict, Any
import random
import sys
import os

# ==============================================================================
# Import Config
# ==============================================================================
from config import (
    AppConfig,
    APIConfig,
    TradingConfig,
    ExchangeConfig,
    LoggingConfig,
    FeatureFlags,
    get_all_config,
    print_config_summary,
    validate_config
)

# ==============================================================================
# Logging Configuration
# ==============================================================================
logging.basicConfig(
    level=getattr(logging, LoggingConfig.LOG_LEVEL),
    format=LoggingConfig.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Import ماژول‌های کمکی
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

logger.info(f"📁 Current directory: {current_dir}")

# وضعیت ماژول‌ها
UTILS_AVAILABLE = False
DATA_COLLECTOR_AVAILABLE = False
COLLECTORS_AVAILABLE = False

# تلاش برای import utils
try:
    from utils import (
        get_market_data_with_fallback,
        analyze_with_multi_timeframe_strategy,
        calculate_24h_change_from_dataframe,
        calculate_simple_sma,
        calculate_simple_rsi
    )
    UTILS_AVAILABLE = True
    logger.info("✅ utils imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ utils not available: {e}")
    UTILS_AVAILABLE = False

# تلاش برای import سایر ماژول‌ها
try:
    from data_collector import get_collected_data
    DATA_COLLECTOR_AVAILABLE = True
    logger.info("✅ data_collector imported")
except ImportError:
    logger.warning("⚠️ data_collector not available")

try:
    from collectors import collect_signals_from_example_site
    COLLECTORS_AVAILABLE = True
    logger.info("✅ collectors imported")
except ImportError:
    logger.warning("⚠️ collectors not available")

# ==============================================================================
# Pydantic Models با Validation
# ==============================================================================
class AnalysisRequest(BaseModel):
    """درخواست تحلیل با اعتبارسنجی"""
    symbol: str
    timeframe: str = TradingConfig.DEFAULT_TIMEFRAME

    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Symbol must be at least 3 characters')
        v = v.strip().upper()
        if not v.replace('USDT', '').replace('BUSD', '').replace('BTC', '').isalnum():
            raise ValueError('Invalid symbol format')
        return v

    @validator('timeframe')
    def validate_timeframe(cls, v):
        if v not in TradingConfig.ALL_TIMEFRAMES:
            raise ValueError(f'Timeframe must be one of {TradingConfig.ALL_TIMEFRAMES}')
        return v

class ScalpRequest(BaseModel):
    """درخواست سیگنال اسکالپ"""
    symbol: str
    timeframe: str = "5m"

    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Symbol must be at least 3 characters')
        return v.strip().upper()

    @validator('timeframe')
    def validate_timeframe(cls, v):
        if v not in TradingConfig.SCALP_TIMEFRAMES:
            raise ValueError(f'Scalp timeframe must be one of {TradingConfig.SCALP_TIMEFRAMES}')
        return v

class SignalResponse(BaseModel):
    """پاسخ سیگنال‌ها"""
    status: str
    count: int
    last_updated: str
    signals: List[Dict[str, Any]]
    sources: Dict[str, int]

# ==============================================================================
# توابع Mock (جایگزین)
# ==============================================================================
def mock_get_market_data_with_fallback(symbol, timeframe="5m", limit=50):
    """دریافت داده mock بازار"""
    try:
        import requests
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol.upper(), 'interval': timeframe, 'limit': limit}
        response = requests.get(url, params=params, timeout=APIConfig.API_TIMEOUT)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.debug(f"Binance API error, using mock: {e}")

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
            timestamp, str(price * random.uniform(0.998, 1.000)),
            str(price * random.uniform(1.000, 1.003)),
            str(price * random.uniform(0.997, 1.000)),
            str(price), str(random.uniform(1000, 10000)),
            timestamp + 300000, "0", "0", "0", "0", "0"
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
        except (IndexError, ValueError, TypeError):
            closes.append(0)

    return sum(closes) / len(closes) if closes else 0

def mock_calculate_simple_rsi(data, period=14):
    """محاسبه RSI با رفع باگ division by zero"""
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
    avg_loss = losses / period if losses > 0 else 0.0001

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)

def mock_analyze_with_multi_timeframe_strategy(symbol):
    """تحلیل Mock با منطق واقعی‌تر"""
    signals = ["BUY", "SELL", "HOLD"]
    weights = [0.35, 0.35, 0.30]
    signal = random.choices(signals, weights=weights)[0]

    base_prices = {
        'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'BNBUSDT': 590.00,
        'SOLUSDT': 175.00, 'DEFAULT': 100
    }

    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])

    if signal == "HOLD":
        confidence = round(random.uniform(0.5, 0.7), 2)
    else:
        confidence = round(random.uniform(TradingConfig.MIN_CONFIDENCE, 0.85), 2)

    entry_price = round(base_price * random.uniform(0.99, 1.01), 2)

    targets_data = calculate_targets_and_stop(signal, entry_price, "STANDARD")

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "entry_price": entry_price,
        **targets_data,
        "strategy": "Multi-Timeframe Mock Analysis",
        "analysis_details": {
            "1h": {"trend": random.choice(["BULLISH", "BEARISH", "NEUTRAL"]), "source": "mock"},
            "15m": {"trend": random.choice(["BULLISH", "BEARISH", "NEUTRAL"]), "source": "mock"},
            "5m": {"trend": random.choice(["BULLISH", "BEARISH", "NEUTRAL"]), "source": "mock"}
        }
    }

def mock_calculate_24h_change(data):
    """محاسبه تغییرات 24 ساعته"""
    if isinstance(data, list) and len(data) > 10:
        try:
            old_price = float(data[0][4])
            current_price = float(data[-1][4])
            return round(((current_price - old_price) / old_price) * 100, 2)
        except:
            pass
    return round(random.uniform(-3, 3), 2)

# ==============================================================================
# انتخاب توابع مناسب
# ==============================================================================
if UTILS_AVAILABLE:
    get_market_data_func = get_market_data_with_fallback
    analyze_func = analyze_with_multi_timeframe_strategy
    calculate_change_func = calculate_24h_change_from_dataframe
    calculate_sma_func = calculate_simple_sma
    calculate_rsi_func = calculate_simple_rsi
    logger.info("🔧 Using REAL functions from utils")
else:
    get_market_data_func = mock_get_market_data_with_fallback
    analyze_func = mock_analyze_with_multi_timeframe_strategy
    calculate_change_func = mock_calculate_24h_change
    calculate_sma_func = mock_calculate_simple_sma
    calculate_rsi_func = mock_calculate_simple_rsi
    logger.info("⚠️ Using MOCK functions")

# ==============================================================================
# تابع مرکزی برای محاسبه تارگت‌ها و استاپ
# ==============================================================================
def calculate_targets_and_stop(signal: str, entry_price: float, trade_type: str = "STANDARD") -> dict:
    """
    محاسبه دقیق تارگت‌ها و استاپ لاس با استفاده از TradingConfig
    """
    if entry_price <= 0:
        logger.warning("Invalid entry price, using default")
        entry_price = 100

    if trade_type == "SCALP":
        stop_percent = TradingConfig.SCALP_STOP_LOSS_PERCENT / 100
        profit_percent = TradingConfig.SCALP_TAKE_PROFIT_PERCENT / 100
        
        if signal == "BUY":
            return {
                "targets": [
                    round(entry_price * (1 + profit_percent), 2),
                    round(entry_price * (1 + profit_percent * 2), 2),
                    round(entry_price * (1 + profit_percent * 3), 2)
                ],
                "stop_loss": round(entry_price * (1 - stop_percent), 2)
            }
        elif signal == "SELL":
            return {
                "targets": [
                    round(entry_price * (1 - profit_percent), 2),
                    round(entry_price * (1 - profit_percent * 2), 2),
                    round(entry_price * (1 - profit_percent * 3), 2)
                ],
                "stop_loss": round(entry_price * (1 + stop_percent), 2)
            }
    else:  # STANDARD
        stop_percent = TradingConfig.DEFAULT_STOP_LOSS_PERCENT / 100
        profit_percent = TradingConfig.DEFAULT_TAKE_PROFIT_PERCENT / 100
        
        if signal == "BUY":
            return {
                "targets": [
                    round(entry_price * (1 + profit_percent / 2), 2),
                    round(entry_price * (1 + profit_percent), 2)
                ],
                "stop_loss": round(entry_price * (1 - stop_percent), 2)
            }
        elif signal == "SELL":
            return {
                "targets": [
                    round(entry_price * (1 - profit_percent / 2), 2),
                    round(entry_price * (1 - profit_percent), 2)
                ],
                "stop_loss": round(entry_price * (1 + stop_percent), 2)
            }

    return {"targets": [], "stop_loss": entry_price}

# ==============================================================================
# تابع تحلیل اسکالپ
# ==============================================================================
def analyze_scalp_signal(symbol, timeframe, data):
    """تحلیل سیگنال اسکالپ"""
    if not data or len(data) < 20:
        return {
            "signal": "HOLD", "confidence": 0.5, "rsi": 50,
            "sma_20": 0, "current_price": 0, "reason": "Insufficient data"
        }

    rsi = calculate_rsi_func(data, TradingConfig.RSI_PERIOD)
    sma_20 = calculate_sma_func(data, TradingConfig.SMA_PERIOD)

    try:
        latest_close = float(data[-1][4])
    except (IndexError, ValueError, TypeError):
        latest_close = 0

    signal = "HOLD"
    confidence = 0.5
    reason = "Market neutral"

    if rsi < 35 and latest_close < sma_20 * 1.01:
        signal = "BUY"
        confidence = min(0.75, (35 - rsi) / 35 * 0.5 + 0.5)
        reason = f"Oversold (RSI: {rsi:.1f}), price below SMA20"
    elif rsi > 65 and latest_close > sma_20 * 0.99:
        signal = "SELL"
        confidence = min(0.75, (rsi - 65) / 35 * 0.5 + 0.5)
        reason = f"Overbought (RSI: {rsi:.1f}), price above SMA20"
    elif latest_close > sma_20 * 1.02 and rsi < 60:
        signal = "BUY"
        confidence = 0.7
        reason = f"Breakout above SMA20, RSI: {rsi:.1f}"
    elif latest_close < sma_20 * 0.98 and rsi > 40:
        signal = "SELL"
        confidence = 0.7
        reason = f"Breakdown below SMA20, RSI: {rsi:.1f}"

    return {
        "signal": signal, "confidence": round(confidence, 2),
        "rsi": round(rsi, 1), "sma_20": round(sma_20, 2),
        "current_price": round(latest_close, 2), "reason": reason
    }

# ==============================================================================
# FastAPI Application با Config
# ==============================================================================
app = FastAPI(
    title=AppConfig.TITLE,
    description=AppConfig.DESCRIPTION,
    version=AppConfig.VERSION,
    docs_url=AppConfig.DOCS_URL,
    redoc_url=AppConfig.REDOC_URL,
    openapi_url=AppConfig.OPENAPI_URL
)

# CORS با استفاده از Config
app.add_middleware(
    CORSMiddleware,
    **APIConfig.get_cors_config()
)

# Rate Limiting
if APIConfig.ENABLE_RATE_LIMIT:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info(f"✅ Rate limiting enabled: {APIConfig.RATE_LIMIT}")
else:
    limiter = None
    logger.warning("⚠️ Rate limiting disabled")

# ==============================================================================
# API Endpoints
# ==============================================================================
@app.get("/")
async def read_root():
    """صفحه اصلی"""
    return {
        "message": f"🚀 {AppConfig.APP_NAME} v{AppConfig.VERSION}",
        "status": "running",
        "version": AppConfig.VERSION,
        "environment": AppConfig.ENVIRONMENT,
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE
        },
        "endpoints": {
            "health": "GET /api/health",
            "config": "GET /api/config",
            "signals": "GET /api/signals",
            "analyze": "POST /api/analyze",
            "scalp_signal": "POST /api/scalp-signal",
            "market": "GET /market/{symbol}",
            "scan_all": "GET /api/scan-all-timeframes/{symbol}",
            "docs": "GET /api/docs" if AppConfig.DEBUG else None
        },
        "features": {
            "scalp_signals": FeatureFlags.ENABLE_SCALP_SIGNALS,
            "swing_signals": FeatureFlags.ENABLE_SWING_SIGNALS,
            "rate_limiting": APIConfig.ENABLE_RATE_LIMIT,
            "real_trading": FeatureFlags.ENABLE_REAL_TRADING
        }
    }

@app.get("/api/health")
async def health_check():
    """بررسی سلامت سیستم"""
    try:
        import psutil
        system_info = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    except ImportError:
        system_info = {"status": "psutil not available"}

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": AppConfig.VERSION,
        "environment": AppConfig.ENVIRONMENT,
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE
        },
        "system": system_info,
        "config": {
            "rate_limit": APIConfig.RATE_LIMIT if APIConfig.ENABLE_RATE_LIMIT else "disabled",
            "cors_origins": len(APIConfig.ALLOWED_ORIGINS),
            "exchanges": {
                "binance": ExchangeConfig.has_binance_keys(),
                "lbank": ExchangeConfig.has_lbank_keys(),
                "primary": ExchangeConfig.PRIMARY_EXCHANGE
            }
        },
        "features": {
            "rate_limiting": APIConfig.ENABLE_RATE_LIMIT,
            "scalp_signals": FeatureFlags.ENABLE_SCALP_SIGNALS,
            "swing_signals": FeatureFlags.ENABLE_SWING_SIGNALS,
            "validation": True
        }
    }

@app.get("/api/config")
async def get_config():
    """دریافت تنظیمات سیستم"""
    if not AppConfig.DEBUG and AppConfig.is_production():
        raise HTTPException(status_code=403, detail="Config endpoint disabled in production")
    
    config = get_all_config()
    validation = validate_config()
    
    return {
        **config,
        "validation": validation
    }

@app.get("/api/signals", response_model=SignalResponse)
async def get_all_signals_endpoint(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None
):
    """دریافت سیگنال‌های تحلیلی داخلی"""
    logger.info(f"📡 Signal request: {symbol or 'all'}")

    try:
        target_symbol = symbol.upper() if symbol else TradingConfig.DEFAULT_SYMBOLS[0]
        analysis = analyze_func(target_symbol)

        signals = [{
            "symbol": analysis["symbol"],
            "timeframe": "multi",
            "signal": analysis["signal"],
            "confidence": analysis["confidence"],
            "entry_price": analysis["entry_price"],
            "targets": analysis["targets"],
            "stop_loss": analysis["stop_loss"],
            "reason": f"Multi-timeframe analysis for {analysis['symbol']}",
            "source": "internal_ai",
            "author": "Local Analysis Engine",
            "strategy": analysis.get("strategy", "Multi-Timeframe"),
            "type": "SWING",
            "generated_at": datetime.now().isoformat()
        }]

        response = SignalResponse(
            status="success",
            count=len(signals),
            last_updated=datetime.now().isoformat(),
            signals=signals,
            sources={"internal_ai": 1, "total": 1}
        )

        response_dict = response.dict()
        response_dict["api_version"] = AppConfig.VERSION
        return response_dict

    except Exception as e:
        logger.error(f"❌ Error getting signals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving signals")

@app.post("/api/analyze")
async def analyze_crypto(request: Request, data: AnalysisRequest):
    """تحلیل یک نماد ارز دیجیتال"""
    # Apply rate limiting if enabled
    if APIConfig.ENABLE_RATE_LIMIT and limiter:
        try:
            await limiter.limit(APIConfig.RATE_LIMIT)(request)
        except:
            pass
    
    logger.info(f"📈 Analysis: {data.symbol} ({data.timeframe})")

    try:
        analysis = analyze_func(data.symbol)

        targets_data = calculate_targets_and_stop(
            signal=analysis["signal"],
            entry_price=analysis["entry_price"],
            trade_type="STANDARD"
        )

        analysis.update(targets_data)
        analysis["requested_timeframe"] = data.timeframe
        analysis["analysis_type"] = "STANDARD"
        analysis["version"] = AppConfig.VERSION
        analysis["timestamp"] = datetime.now().isoformat()
        analysis["config"] = {
            "min_confidence": TradingConfig.MIN_CONFIDENCE,
            "stop_loss_percent": TradingConfig.DEFAULT_STOP_LOSS_PERCENT,
            "take_profit_percent": TradingConfig.DEFAULT_TAKE_PROFIT_PERCENT
        }

        return analysis

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis error")

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: Request, data: ScalpRequest):
    """سیگنال‌های اسکالپ"""
    if not FeatureFlags.ENABLE_SCALP_SIGNALS:
        raise HTTPException(status_code=403, detail="Scalp signals are disabled")
    
    # Apply rate limiting
    if APIConfig.ENABLE_RATE_LIMIT and limiter:
        try:
            await limiter.limit(APIConfig.RATE_LIMIT)(request)
        except:
            pass
    
    logger.info(f"⚡ Scalp: {data.symbol} ({data.timeframe})")

    try:
        market_data = get_market_data_func(data.symbol, data.timeframe, TradingConfig.DEFAULT_LIMIT)

        if not market_data:
            raise HTTPException(status_code=404, detail=f"No data for {data.symbol}")

        scalp_analysis = analyze_scalp_signal(data.symbol, data.timeframe, market_data)
        current_price = scalp_analysis.get("current_price", 0)

        if current_price <= 0:
            base_prices = {'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'DEFAULT': 100}
            base_price = base_prices.get(data.symbol.upper(), base_prices['DEFAULT'])
            current_price = round(base_price * random.uniform(0.995, 1.005), 2)

        targets_data = calculate_targets_and_stop(
            signal=scalp_analysis["signal"],
            entry_price=current_price,
            trade_type="SCALP"
        )

        return {
            "symbol": data.symbol,
            "timeframe": data.timeframe,
            "signal": scalp_analysis["signal"],
            "confidence": scalp_analysis["confidence"],
            "entry_price": current_price,
            "rsi": scalp_analysis["rsi"],
            "sma_20": scalp_analysis["sma_20"],
            **targets_data,
            "type": "SCALP",
            "reason": scalp_analysis["reason"],
            "strategy": f"Scalp Strategy ({data.timeframe})",
            "version": AppConfig.VERSION,
            "timestamp": datetime.now().isoformat(),
            "risk_level": "HIGH" if data.timeframe == "1m" else "MEDIUM",
            "config": {
                "stop_loss_percent": TradingConfig.SCALP_STOP_LOSS_PERCENT,
                "take_profit_percent": TradingConfig.SCALP_TAKE_PROFIT_PERCENT
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Scalp error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Scalp signal error")

@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    """دریافت داده‌های بازار"""
    try:
        data = get_market_data_func(symbol, timeframe, limit=TradingConfig.DEFAULT_LIMIT)

        if not data:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        latest = data[-1] if isinstance(data, list) and len(data) > 0 else []

        if not latest or len(latest) < 6:
            base_prices = {'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'DEFAULT': 100}
            base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "Mock Data",
                "current_price": round(base_price * random.uniform(0.99, 1.01), 2),
                "high": round(base_price * 1.01, 2),
                "low": round(base_price * 0.99, 2),
                "volume": round(random.uniform(1000, 5000), 2),
                "change_24h": round(random.uniform(-5, 5), 2),
                "timestamp": datetime.now().isoformat()
            }

        change_24h = calculate_change_func(data)
        rsi = calculate_rsi_func(data, TradingConfig.RSI_PERIOD)
        sma_20 = calculate_sma_func(data, TradingConfig.SMA_PERIOD)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": "Binance API" if UTILS_AVAILABLE else "Mock",
            "current_price": float(latest[4]),
            "high": float(latest[2]),
            "low": float(latest[3]),
            "volume": float(latest[5]),
            "change_24h": change_24h,
            "rsi_14": round(rsi, 2),
            "sma_20": round(sma_20, 2),
            "timestamp": datetime.now().isoformat(),
            "data_points": len(data)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market data error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Market data error")

@app.get("/api/scan-all-timeframes/{symbol}")
async def scan_all_timeframes(symbol: str):
    """اسکن همه تایم‌فریم‌ها"""
    logger.info(f"🔍 Scanning {symbol}")

    try:
        timeframes = TradingConfig.SCALP_TIMEFRAMES + TradingConfig.SWING_TIMEFRAMES[:2]
        results = []

        for tf in timeframes:
            try:
                if tf in TradingConfig.SCALP_TIMEFRAMES and FeatureFlags.ENABLE_SCALP_SIGNALS:
                    req = ScalpRequest(symbol=symbol, timeframe=tf)
                    response = await get_scalp_signal(Request(scope={"type": "http"}), req)
                    response["analysis_type"] = "SCALP"
                elif tf in TradingConfig.SWING_TIMEFRAMES and FeatureFlags.ENABLE_SWING_SIGNALS:
                    req = AnalysisRequest(symbol=symbol, timeframe=tf)
                    response = await analyze_crypto(Request(scope={"type": "http"}), req)
                    response["analysis_type"] = "