"""
Crypto AI Trading System v7.7.0 - RENDER FIXED VERSION
Optimized for Render deployment with corrected imports and stable calculations.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import time
import random
import math

# ==============================================================================
# Path and Import Setup - CRITICAL FOR RENDER
# ==============================================================================

# Get current directory (where main.py is located - should be in api/ folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add to sys.path for proper imports
sys.path.insert(0, current_dir)  # Add api/ to path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # Add parent to path

print("=" * 60)
print("CRYPTO AI TRADING SYSTEM - INITIALIZATION")
print("=" * 60)
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
print("=" * 60)

# ==============================================================================
# Configure Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Import FastAPI and Dependencies
# ==============================================================================

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    print("✅ FastAPI dependencies imported successfully")
except ImportError as e:
    print(f"❌ Failed to import FastAPI dependencies: {e}")
    raise

# ==============================================================================
# Import Utils Module - RENDER COMPATIBLE
# ==============================================================================

print("\n[1/3] Importing utils module...")
UTILS_AVAILABLE = False
utils = None

try:
    # Strategy 1: Direct import (main.py and utils.py are in same directory)
    import utils
    UTILS_AVAILABLE = True
    print("✅ Strategy 1: Direct import successful")
except ImportError as e:
    print(f"❌ Strategy 1 failed: {str(e)[:100]}")
    
    try:
        # Strategy 2: Relative import
        from . import utils
        UTILS_AVAILABLE = True
        print("✅ Strategy 2: Relative import successful")
    except ImportError as e:
        print(f"❌ Strategy 2 failed: {str(e)[:100]}")
        
        try:
            # Strategy 3: Import from file path
            import importlib.util
            utils_path = os.path.join(current_dir, "utils.py")
            
            if os.path.exists(utils_path):
                print(f"📄 Found utils.py at: {utils_path}")
                spec = importlib.util.spec_from_file_location("utils", utils_path)
                utils_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(utils_module)
                sys.modules["utils"] = utils_module
                utils = utils_module
                UTILS_AVAILABLE = True
                print("✅ Strategy 3: Loaded from file directly")
            else:
                print(f"❌ utils.py not found at: {utils_path}")
        except Exception as e:
            print(f"❌ Strategy 3 failed: {str(e)[:100]}")

if UTILS_AVAILABLE and utils:
    print(f"📦 Utils module loaded successfully")
    if hasattr(utils, '__version__'):
        print(f"   Version: {utils.__version__}")
else:
    print("⚠️ Utils module NOT available. Will use mock functions.")

# ==============================================================================
# Import Individual Functions from Utils
# ==============================================================================

print("\n[2/3] Importing individual functions...")

# Define mock functions first (as fallback)
def mock_get_market_data_with_fallback(symbol, interval="5m", limit=50, return_source=False):
    """Mock market data generator."""
    base_prices = {
        'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'DOGEUSDT': 0.12116,
        'ALGOUSDT': 0.1187, 'DEFAULT': 100.0
    }
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    data = []
    current_time = int(time.time() * 1000)
    
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)
        change = random.uniform(-0.015, 0.015)
        price = base_price * (1 + change)
        candle = [
            timestamp,
            str(price * random.uniform(0.998, 1.000)),
            str(price * random.uniform(1.000, 1.003)),
            str(price * random.uniform(0.997, 1.000)),
            str(price),
            str(random.uniform(1000, 10000)),
            timestamp + 300000, "0", "0", "0", "0", "0"
        ]
        data.append(candle)
    
    if return_source:
        return {"data": data, "source": "mock", "success": False}
    return data

def mock_calculate_simple_rsi(data, period=14):
    return round(random.uniform(30, 70), 2)

def mock_calculate_simple_sma(data, period=20):
    if not data or len(data) < period:
        return 0
    try:
        closes = [float(c[4]) for c in data[-period:]]
        return sum(closes) / len(closes)
    except:
        return 0

def mock_detect_divergence(prices, rsi_values, lookback=5):
    return {"detected": False, "type": "none", "strength": None}

def mock_calculate_smart_entry(data, signal="BUY", strategy="ICHIMOKU_FIBO"):
    try:
        base_price = float(data[-1][4]) if data else 0
        if signal == "BUY":
            return base_price * 0.999
        elif signal == "SELL":
            return base_price * 1.001
        return base_price
    except:
        return 0

def mock_analyze_with_multi_timeframe_strategy(symbol):
    signals = ["BUY", "SELL", "HOLD"]
    weights = [0.35, 0.35, 0.30]
    signal = random.choices(signals, weights=weights)[0]
    base_prices = {'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'DOGEUSDT': 0.12116, 'DEFAULT': 100}
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    entry_price = base_price * random.uniform(0.99, 1.01)
    
    if signal == "BUY":
        targets = [entry_price * 1.01, entry_price * 1.015, entry_price * 1.02]
        stop_loss = entry_price * 0.99
    elif signal == "SELL":
        targets = [entry_price * 0.99, entry_price * 0.985, entry_price * 0.98]
        stop_loss = entry_price * 1.01
    else:
        targets = [entry_price * 1.005, entry_price * 1.01, entry_price * 1.015]
        stop_loss = entry_price * 0.995
    
    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": round(random.uniform(0.6, 0.85), 2),
        "entry_price": round(entry_price, 8),
        "targets": [round(t, 8) for t in targets],
        "stop_loss": round(stop_loss, 8),
        "strategy": "Mock Analysis"
    }

# Now try to import real functions or use mocks
if UTILS_AVAILABLE:
    try:
        # Import all required functions from utils
        get_market_data_with_fallback = utils.get_market_data_with_fallback
        analyze_with_multi_timeframe_strategy = utils.analyze_with_multi_timeframe_strategy
        calculate_simple_rsi = utils.calculate_simple_rsi
        calculate_simple_sma = utils.calculate_simple_sma
        calculate_rsi_series = utils.calculate_rsi_series
        detect_divergence = utils.detect_divergence
        calculate_macd_simple = utils.calculate_macd_simple
        calculate_ichimoku_components = utils.calculate_ichimoku_components
        analyze_ichimoku_scalp_signal = utils.analyze_ichimoku_scalp_signal
        get_ichimoku_scalp_signal = utils.get_ichimoku_scalp_signal
        calculate_smart_entry = utils.calculate_smart_entry
        get_swing_high_low = utils.get_swing_high_low
        get_support_resistance_levels = utils.get_support_resistance_levels
        calculate_volatility = utils.calculate_volatility
        combined_analysis = utils.combined_analysis
        generate_ichimoku_recommendation = utils.generate_ichimoku_recommendation
        get_fallback_signal = utils.get_fallback_signal
        
        print("✅ All real functions imported from utils")
        USING_REAL_FUNCTIONS = True
        
    except AttributeError as e:
        print(f"⚠️ Failed to import some functions from utils: {e}")
        print("   Falling back to mock functions")
        USING_REAL_FUNCTIONS = False
else:
    USING_REAL_FUNCTIONS = False

if not USING_REAL_FUNCTIONS:
    # Use mock functions
    get_market_data_func = mock_get_market_data_with_fallback
    analyze_func = mock_analyze_with_multi_timeframe_strategy
    calculate_rsi_func = mock_calculate_simple_rsi
    calculate_sma_func = mock_calculate_simple_sma
    calculate_rsi_series_func = lambda closes, period=14: [random.uniform(30, 70) for _ in closes]
    calculate_divergence_func = mock_detect_divergence
    calculate_macd_func = lambda data, **kwargs: {'macd': 0, 'signal': 0, 'histogram': 0}
    calculate_ichimoku_func = lambda data, **kwargs: {'current_price': float(data[-1][4]) if data else 0}
    analyze_ichimoku_signal_func = lambda data: {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Mock'}
    get_ichimoku_signal_func = lambda data, timeframe=None: None
    calculate_smart_entry_func = mock_calculate_smart_entry
    get_swing_high_low_func = lambda data, period=20: (100, 50)
    get_support_resistance_levels_func = lambda data: {"support": 0, "resistance": 0, "range_percent": 0}
    calculate_volatility_func = lambda data, period=20: 1.0
    combined_analysis_func = lambda data, timeframe=None: None
    generate_recommendation_func = lambda signal_data: "Hold (Mock)"
    print("⚠️ Using MOCK functions for all calculations")
else:
    # Use real functions
    get_market_data_func = get_market_data_with_fallback
    analyze_func = analyze_with_multi_timeframe_strategy
    calculate_rsi_func = calculate_simple_rsi
    calculate_sma_func = calculate_simple_sma
    calculate_rsi_series_func = calculate_rsi_series
    calculate_divergence_func = detect_divergence
    calculate_macd_func = calculate_macd_simple
    calculate_ichimoku_func = calculate_ichimoku_components
    analyze_ichimoku_signal_func = analyze_ichimoku_scalp_signal
    get_ichimoku_signal_func = get_ichimoku_scalp_signal
    calculate_smart_entry_func = calculate_smart_entry
    get_swing_high_low_func = get_swing_high_low
    get_support_resistance_levels_func = get_support_resistance_levels
    calculate_volatility_func = calculate_volatility
    combined_analysis_func = combined_analysis
    generate_recommendation_func = generate_ichimoku_recommendation
    print("✅ Using REAL functions for all calculations")

# ==============================================================================
# Pydantic Models
# ==============================================================================

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class ScalpRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class IchimokuRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class CombinedRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"
    include_ichimoku: bool = True
    include_rsi: bool = True
    include_macd: bool = True

class SignalResponse(BaseModel):
    status: str
    count: int
    last_updated: str
    signals: list
    sources: dict

# ==============================================================================
# Helper Functions
# ==============================================================================

def analyze_scalp_signal(symbol, timeframe, data):
    """Analyze scalp signal based on RSI and divergence."""
    if not data or len(data) < 30:
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "rsi": 50,
            "divergence": False,
            "sma_20": 0,
            "reason": "Insufficient data",
            "current_price": 0
        }
    
    try:
        rsi = calculate_rsi_func(data, 14)
        sma_20 = calculate_sma_func(data, 20)
        closes = [float(c[4]) for c in data[-30:]]
        rsi_series = calculate_rsi_series_func(closes, 14)
        div_info = calculate_divergence_func(closes, rsi_series, lookback=5)
        latest_close = closes[-1] if closes else 0
        
        signal, confidence, reason = "HOLD", 0.5, "Neutral"
        
        if div_info['detected']:
            if div_info['type'] == 'bullish':
                signal, confidence, reason = "BUY", 0.85, "Bullish divergence detected"
            elif div_info['type'] == 'bearish':
                signal, confidence, reason = "SELL", 0.85, "Bearish divergence detected"
        else:
            if rsi < 35 and latest_close < sma_20:
                signal, confidence, reason = "BUY", 0.75, f"Oversold (RSI: {rsi:.1f}) & below SMA20"
            elif rsi > 65 and latest_close > sma_20:
                signal, confidence, reason = "SELL", 0.75, f"Overbought (RSI: {rsi:.1f}) & above SMA20"
            elif rsi < 40:
                signal, confidence, reason = "BUY", 0.65, f"Near oversold (RSI: {rsi:.1f})"
            elif rsi > 60:
                signal, confidence, reason = "SELL", 0.65, f"Near overbought (RSI: {rsi:.1f})"
        
        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "rsi": round(rsi, 1),
            "divergence": div_info['detected'],
            "divergence_type": div_info['type'],
            "sma_20": round(sma_20, 8) if sma_20 else 0,
            "current_price": round(latest_close, 8),
            "reason": reason
        }
    except Exception as e:
        logger.error(f"Error in analyze_scalp_signal: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "rsi": 50,
            "divergence": False,
            "sma_20": 0,
            "reason": f"Error: {str(e)[:50]}",
            "current_price": 0
        }

def calculate_targets_and_stop_loss(entry_price, signal, risk_level="MEDIUM"):
    """Calculate consistent targets and stop loss."""
    if entry_price <= 0:
        return [0, 0, 0], 0, [0, 0, 0], 0
    
    if signal == "BUY":
        if risk_level == "HIGH":
            targets = [1.015, 1.025, 1.035]
            stop_loss = 0.985
        elif risk_level == "MEDIUM":
            targets = [1.005, 1.010, 1.015]
            stop_loss = 0.995
        else:  # LOW
            targets = [1.003, 1.006, 1.009]
            stop_loss = 0.997
    elif signal == "SELL":
        if risk_level == "HIGH":
            targets = [0.985, 0.975, 0.965]
            stop_loss = 1.015
        elif risk_level == "MEDIUM":
            targets = [0.995, 0.990, 0.985]
            stop_loss = 1.005
        else:  # LOW
            targets = [0.997, 0.994, 0.991]
            stop_loss = 1.003
    else:  # HOLD
        targets = [1.002, 1.004, 1.006]
        stop_loss = 0.998
    
    targets_prices = [round(entry_price * t, 8) for t in targets]
    stop_loss_price = round(entry_price * stop_loss, 8)
    
    targets_percent = [round(((t - entry_price) / entry_price) * 100, 2) for t in targets_prices]
    stop_loss_percent = round(((stop_loss_price - entry_price) / entry_price) * 100, 2)
    
    return targets_prices, stop_loss_price, targets_percent, stop_loss_percent

# ==============================================================================
# FastAPI App
# ==============================================================================

API_VERSION = "7.7.0-RENDER-FIXED"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description="Fixed for Render deployment with stable calculations",
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
# API Endpoints
# ==============================================================================

@app.get("/")
async def read_root():
    return {
        "message": f"Crypto AI Trading System v{API_VERSION}",
        "status": "Active",
        "version": API_VERSION,
        "utils_available": UTILS_AVAILABLE,
        "using_real_functions": USING_REAL_FUNCTIONS,
        "endpoints": {
            "/api/health": "Health check",
            "/api/scalp-signal": "Get scalp signal (POST)",
            "/api/ichimoku-scalp": "Get ichimoku signal (POST)",
            "/api/analyze": "General analysis (POST)",
            "/market/{symbol}": "Market data (GET)"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "Healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "utils": UTILS_AVAILABLE,
            "using_real_functions": USING_REAL_FUNCTIONS
        },
        "system": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform
        }
    }

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """Get scalp trading signal with proper target calculations."""
    logger.info(f"Scalp signal request: {request.symbol} ({request.timeframe})")
    
    try:
        # Get market data
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        
        if not market_data or len(market_data) < 20:
            logger.warning(f"Insufficient data for {request.symbol}")
            market_data = mock_get_market_data_with_fallback(request.symbol, request.timeframe, 100)
        
        # Analyze signal
        scalp_analysis = analyze_scalp_signal(request.symbol, request.timeframe, market_data)
        
        # Calculate entry price
        try:
            entry_price = calculate_smart_entry_func(market_data, scalp_analysis["signal"])
            if entry_price <= 0:
                entry_price = scalp_analysis.get("current_price", 0)
        except:
            try:
                entry_price = float(market_data[-1][4])
            except:
                entry_price = 0
        
        # Determine risk level
        rsi = scalp_analysis["rsi"]
        confidence = scalp_analysis["confidence"]
        risk_level = "MEDIUM"
        
        if (rsi > 80 or rsi < 20) and confidence > 0.8:
            risk_level = "HIGH"
        elif (rsi > 70 or rsi < 30) and confidence > 0.7:
            risk_level = "MEDIUM"
        elif confidence < 0.5:
            risk_level = "LOW"
        
        # Calculate targets
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_targets_and_stop_loss(
            entry_price, scalp_analysis["signal"], risk_level
        )
        
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": scalp_analysis["signal"],
            "confidence": scalp_analysis["confidence"],
            "entry_price": round(entry_price, 8),
            "rsi": scalp_analysis["rsi"],
            "divergence": scalp_analysis["divergence"],
            "divergence_type": scalp_analysis.get("divergence_type", "none"),
            "sma_20": scalp_analysis.get("sma_20", 0),
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "risk_level": risk_level,
            "reason": scalp_analysis["reason"],
            "strategy": "Scalp Analysis",
            "generated_at": datetime.now().isoformat(),
            "data_source": "real" if USING_REAL_FUNCTIONS else "mock"
        }
        
        logger.info(f"Generated {response['signal']} signal for {response['symbol']}")
        return response
        
    except Exception as e:
        logger.error(f"Error in scalp signal: {e}")
        # Fallback response
        fallback_signal = random.choice(["BUY", "SELL", "HOLD"])
        base_prices = {'DOGEUSDT': 0.12116, 'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'DEFAULT': 100}
        base_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        entry_price = round(base_price * random.uniform(0.99, 1.01), 8)
        
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_targets_and_stop_loss(
            entry_price, fallback_signal, "MEDIUM"
        )
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": fallback_signal,
            "confidence": 0.5,
            "entry_price": entry_price,
            "rsi": round(random.uniform(30, 70), 1),
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "risk_level": "MEDIUM",
            "reason": f"System error - using fallback",
            "strategy": "Fallback Mode",
            "generated_at": datetime.now().isoformat(),
            "data_source": "error_fallback"
        }

@app.post("/api/ichimoku-scalp")
async def get_ichimoku_scalp_signal(request: IchimokuRequest):
    """Get Ichimoku-based trading signal."""
    try:
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        
        if not market_data or len(market_data) < 60:
            raise HTTPException(status_code=404, detail="Not enough data for Ichimoku analysis")
        
        # Get Ichimoku data
        ichimoku_data = calculate_ichimoku_func(market_data)
        if not ichimoku_data:
            raise HTTPException(status_code=500, detail="Ichimoku calculation failed")
        
        # Analyze signal
        ichimoku_signal = analyze_ichimoku_signal_func(ichimoku_data)
        current_price = ichimoku_data.get('current_price', 0)
        
        if current_price <= 0:
            try:
                current_price = float(market_data[-1][4])
            except:
                current_price = 0
        
        # Calculate targets
        risk_level = "HIGH" if ichimoku_signal['confidence'] > 0.8 else "MEDIUM" if ichimoku_signal['confidence'] > 0.6 else "LOW"
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_targets_and_stop_loss(
            current_price, ichimoku_signal['signal'], risk_level
        )
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": ichimoku_signal['signal'],
            "confidence": ichimoku_signal['confidence'],
            "entry_price": round(current_price, 8),
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "rsi": calculate_rsi_func(market_data, 14),
            "reason": ichimoku_signal['reason'],
            "recommendation": generate_recommendation_func(ichimoku_signal),
            "risk_level": risk_level,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Ichimoku signal: {e}")
        raise HTTPException(status_code=500, detail=f"Ichimoku analysis error: {str(e)[:100]}")

@app.post("/api/analyze")
async def analyze_crypto(request: AnalysisRequest):
    """General analysis endpoint."""
    try:
        analysis = analyze_func(request.symbol)
        analysis["version"] = API_VERSION
        
        # Add market data
        market_data = get_market_data_func(request.symbol, request.timeframe, 50)
        if market_data:
            analysis["current_price"] = float(market_data[-1][4]) if market_data else 0
            analysis["rsi"] = calculate_rsi_func(market_data, 14)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in analyze_crypto: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)[:100]}")

@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    """Get market data for a symbol."""
    try:
        data = get_market_data_func(symbol, timeframe, 50)
        
        if not data:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "mock",
                "current_price": 0,
                "change_24h": 0,
                "rsi_14": 50,
                "sma_20": 0
            }
        
        latest = data[-1]
        current_price = float(latest[4])
        
        # Calculate 24h change (simplified)
        if len(data) > 10:
            old_price = float(data[0][4])
            change_24h = ((current_price - old_price) / old_price) * 100
        else:
            change_24h = 0
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": "real" if USING_REAL_FUNCTIONS else "mock",
            "current_price": current_price,
            "high": float(latest[2]),
            "low": float(latest[3]),
            "change_24h": round(change_24h, 2),
            "rsi_14": calculate_rsi_func(data, 14),
            "sma_20": calculate_sma_func(data, 20),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in market data: {e}")
        raise HTTPException(status_code=500, detail=f"Market data error: {str(e)[:100]}")

@app.get("/api/signals", response_model=SignalResponse)
async def get_all_signals_endpoint(symbol: Optional[str] = None):
    """Get all signals."""
    try:
        target_symbol = symbol.upper() if symbol else "BTCUSDT"
        analysis = analyze_func(target_symbol)
        
        signals = [{
            "symbol": analysis["symbol"],
            "signal": analysis["signal"],
            "confidence": analysis["confidence"],
            "entry_price": analysis["entry_price"],
            "targets": analysis["targets"],
            "stop_loss": analysis["stop_loss"],
            "generated_at": datetime.now().isoformat()
        }]
        
        return SignalResponse(
            status="Success",
            count=1,
            last_updated=datetime.now().isoformat(),
            signals=signals,
            sources={"internal": 1, "total": 1}
        )
        
    except Exception as e:
        logger.error(f"Error in get_all_signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# Startup Event
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info(f"🚀 Starting Crypto AI Trading System v{API_VERSION}")
    logger.info(f"📦 Utils Available: {UTILS_AVAILABLE}")
    logger.info(f"📦 Using Real Functions: {USING_REAL_FUNCTIONS}")
    logger.info(f"📦 Python Path: {sys.path}")
    logger.info("✅ System startup completed successfully!")

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"\n{'=' * 60}")
    print(f"🚀 Starting Crypto AI Trading System v{API_VERSION}")
    print(f"🌐 Server: {host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/api/docs")
    print(f"❤️  Health: http://{host}:{port}/api/health")
    print(f"📦 Utils: {UTILS_AVAILABLE}")
    print(f"🔧 Real Functions: {USING_REAL_FUNCTIONS}")
    print(f"{'=' * 60}\n")
    
    uvicorn.run(
        "main:app",  # Note: This should be "api.main:app" if running from parent directory
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )