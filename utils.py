"""
Crypto AI Trading System v7.6.6 (COMPLETELY FIXED TARGETS)
Fixed:
1. Correct targets and stop loss calculation for all signals
2. Proper smart entry price formatting
3. Fixed percentage calculations
4. Enhanced debugging logs
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
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# Path Setup
# ==============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, current_dir)  # Add api/ to path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # Add src/ to path

print("=" * 50)
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")

# ==============================================================================
# Module Availability Flags
# ==============================================================================

UTILS_AVAILABLE = False
DATA_COLLECTOR_AVAILABLE = False
COLLECTORS_AVAILABLE = False

# ==============================================================================
# Import Utils Module
# ==============================================================================

print("Importing utils module...")

try:
    import utils
    from utils import (
        get_market_data_with_fallback, 
        analyze_with_multi_timeframe_strategy, 
        calculate_24h_change_from_dataframe,
        calculate_simple_sma,
        calculate_simple_rsi,
        calculate_rsi_series,
        detect_divergence,
        calculate_macd_simple,
        analyze_scalp_conditions,
        calculate_ichimoku_components,
        analyze_ichimoku_scalp_signal,
        get_ichimoku_scalp_signal,
        calculate_quality_line,
        calculate_golden_line,
        get_support_resistance_levels,
        calculate_volatility,
        combined_analysis,
        generate_ichimoku_recommendation,
        get_swing_high_low,
        calculate_smart_entry
    )
    UTILS_AVAILABLE = True
    print("SUCCESS: Utils imported directly")
except ImportError as e:
    print(f"FAILED: Direct import: {e}")
    try:
        from .utils import (
            get_market_data_with_fallback, 
            analyze_with_multi_timeframe_strategy, 
            calculate_24h_change_from_dataframe,
            calculate_simple_sma,
            calculate_simple_rsi,
            calculate_rsi_series,
            detect_divergence,
            calculate_macd_simple,
            analyze_scalp_conditions,
            calculate_ichimoku_components,
            analyze_ichimoku_scalp_signal,
            get_ichimoku_scalp_signal,
            calculate_quality_line,
            calculate_golden_line,
        )
        UTILS_AVAILABLE = True
        print("SUCCESS: Utils imported relatively")
    except ImportError as e:
        print(f"FAILED: Relative import: {e}")
        UTILS_AVAILABLE = False

# ==============================================================================
# Import Other Modules
# ==============================================================================

print("Importing other modules...")

# Data Collector
try:
    try:
        from data_collector import get_collected_data
        DATA_COLLECTOR_AVAILABLE = True
        print("SUCCESS: data_collector imported (direct)")
    except ImportError:
        try:
            from .data_collector import get_collected_data
            DATA_COLLECTOR_AVAILABLE = True
            print("SUCCESS: data_collector imported (relative)")
        except ImportError:
            try:
                from api.data_collector import get_collected_data
                DATA_COLLECTOR_AVAILABLE = True
                print("SUCCESS: data_collector imported (full)")
            except ImportError as e:
                print(f"FAILED: data_collector import: {e}")
                DATA_COLLECTOR_AVAILABLE = False
except Exception as e:
    print(f"ERROR: data_collector: {e}")
    DATA_COLLECTOR_AVAILABLE = False

# Collectors
try:
    try:
        from collectors import collect_signals_from_example_site
        COLLECTORS_AVAILABLE = True
        print("SUCCESS: collectors imported (direct)")
    except ImportError:
        try:
            from .collectors import collect_signals_from_example_site
            COLLECTORS_AVAILABLE = True
            print("SUCCESS: collectors imported (relative)")
        except ImportError:
            try:
                from api.collectors import collect_signals_from_example_site
                COLLECTORS_AVAILABLE = True
                print("SUCCESS: collectors imported (full)")
            except ImportError as e:
                print(f"FAILED: collectors import: {e}")
                COLLECTORS_AVAILABLE = False
except Exception as e:
    print(f"ERROR: collectors: {e}")
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
    signals: List[Dict[str, Any]]
    sources: Dict[str, int]

# ==============================================================================
# Mock Functions (Fixed Signatures - CORRECTED)
# ==============================================================================

def mock_get_market_data_with_fallback(symbol, interval="5m", limit=50):
    """Mock data generator with real price logic."""
    try:
        import requests
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200: return response.json()
    except: pass
    
    # Use utils mock generator if available
    if UTILS_AVAILABLE:
        return utils.generate_mock_data_simple(symbol, limit)
        
    base_prices = {'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'BNBUSDT': 590.00, 'SOLUSDT': 175.00, 
                   'ALGOUSDT': 0.1187, 'DEFAULT': 100}
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    data = []
    current_time = int(datetime.now().timestamp() * 1000)
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)
        change = random.uniform(-0.015, 0.015)
        price = base_price * (1 + change)
        candle = [timestamp, str(price), str(price), str(price), str(price), 
                  str(random.uniform(1000, 10000)), timestamp + 300000, 
                  "0", "0", "0", "0", "0"]
        data.append(candle)
    return data

def mock_calculate_simple_sma(data, period=20): 
    if not data or len(data) < period:
        return 0
    try:
        closes = [float(c[4]) for c in data[-period:]]
        return sum(closes) / len(closes)
    except:
        return 50000

def mock_calculate_simple_rsi(data, period=14): 
    return random.uniform(30, 70)

def mock_calculate_rsi_series(data, period=14): 
    return [random.uniform(30, 70) for _ in range(len(data) if data else 0)]

def mock_detect_divergence(prices, rsi_values, lookback=5): 
    """FIXED: Added lookback parameter"""
    return {"detected": False, "type": "none"}

def mock_calculate_macd_simple(data, **kwargs): 
    return {'macd': random.uniform(-1, 1), 'signal': random.uniform(-1, 1), 'histogram': random.uniform(-0.5, 0.5)}

def mock_analyze_scalp_conditions(data, timeframe): 
    return {"condition": "NEUTRAL", "rsi": random.uniform(30, 70), "sma_20": 0}

def mock_calculate_ichimoku_components(data, **kwargs): 
    try: 
        return {'current_price': float(data[-1][4])} if data else {'current_price': 100}
    except: 
        return {'current_price': 100}

def mock_analyze_ichimoku_scalp_signal(ichimoku_data): 
    signal = random.choice(['BUY', 'SELL', 'HOLD'])
    confidence = random.uniform(0.6, 0.9) if signal != 'HOLD' else random.uniform(0.4, 0.6)
    return {'signal': signal, 'confidence': confidence, 'reason': 'Mock analysis'}

def mock_get_ichimoku_scalp_signal(data, timeframe="5m"): 
    return None

def mock_get_support_resistance_levels(data):
    return {"support": 0, "resistance": 0, "range_percent": 0}

def mock_calculate_volatility(data, period=20):
    return random.uniform(0.5, 3.0)

def mock_calculate_24h_change(data): 
    return random.uniform(-5, 5)

def mock_get_swing_high_low(data, period=20): 
    """FIXED: Correct signature"""
    if not data or len(data) < period:
        return 100, 50
    try:
        highs = [float(c[2]) for c in data[-period:]]
        lows = [float(c[3]) for c in data[-period:]]
        return max(highs) if highs else 100, min(lows) if lows else 50
    except:
        return 100, 50

def mock_generate_ichimoku_recommendation(signal): 
    recommendations = {
        'BUY': ['Strong Buy', 'Medium Buy', 'Weak Buy'],
        'SELL': ['Strong Sell', 'Medium Sell', 'Weak Sell'],
        'HOLD': ['Wait in cloud', 'Stay away', 'Hold']
    }
    signal_type = signal.get('signal', 'HOLD')
    return random.choice(recommendations.get(signal_type, ['Hold']))

def mock_calculate_smart_entry(data, signal="BUY"):
    """Mock Smart Entry - FIXED with proper values."""
    try: 
        if not data:
            return 100.0  # مقدار پیش‌فرض
            
        base_price = float(data[-1][4])
        logger.info(f"📊 Mock Smart Entry - Base Price: {base_price}, Signal: {signal}")
        
        # اطمینان از اینکه قیمت منطقی است
        if base_price > 1000000 or base_price < 0.0001:  # اگر قیمت غیرمنطقی است
            base_prices = {'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'ALGOUSDT': 0.1187}
            base_price = base_prices.get('BTCUSDT', 100.0)
            
        if signal == "BUY":
            price = base_price * random.uniform(0.995, 0.999)  # کمی زیر قیمت فعلی
        elif signal == "SELL":
            price = base_price * random.uniform(1.001, 1.005)  # کمی بالای قیمت فعلی
        else:
            price = base_price
            
        logger.info(f"📊 Mock Smart Entry - Result: {price}")
        return round(price, 2)
    except Exception as e:
        logger.error(f"❌ Error in mock_calculate_smart_entry: {e}")
        return 100.0  # مقدار پیش‌فرض

def mock_combined_analysis(data, timeframe): 
    return None

def mock_analyze_with_multi_timeframe_strategy(symbol):
    """Mock Analysis."""
    data = mock_get_market_data_with_fallback(symbol, "5m", 50)
    signals = ["BUY", "SELL", "HOLD"]
    weights = [0.35, 0.35, 0.30]
    signal = random.choices(signals, weights=weights)[0]
    base_prices = {'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'ALGOUSDT': 0.1187, 'DEFAULT': 100}
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    try: 
        latest_close = float(data[-1][4]) if data else base_price
    except: 
        latest_close = base_price
    
    # Use mock smart entry
    smart_entry = mock_calculate_smart_entry(data, signal)
    
    if signal == "HOLD": 
        confidence = round(random.uniform(0.5, 0.7), 2)
    else: 
        confidence = round(random.uniform(0.65, 0.85), 2)
    
    # Fixed targets and stop loss calculation with 3 targets
    if signal == "BUY": 
        targets = [
            round(smart_entry * 1.01, 2),   # +1%
            round(smart_entry * 1.015, 2),  # +1.5%
            round(smart_entry * 1.02, 2)    # +2%
        ]
        stop_loss = round(smart_entry * 0.99, 2)   # -1%
    elif signal == "SELL": 
        targets = [
            round(smart_entry * 0.99, 2),   # -1%
            round(smart_entry * 0.985, 2),  # -1.5%
            round(smart_entry * 0.98, 2)    # -2%
        ]
        stop_loss = round(smart_entry * 1.01, 2)   # +1%
    else: 
        targets = [
            round(smart_entry * 1.005, 2),  # +0.5%
            round(smart_entry * 1.01, 2),   # +1%
            round(smart_entry * 1.015, 2)   # +1.5%
        ]
        stop_loss = round(smart_entry * 0.995, 2)  # -0.5%
    
    logger.info(f"🎯 Mock Analysis - Signal: {signal}, Entry: {smart_entry}, Targets: {targets}, SL: {stop_loss}")
    
    return {
        "symbol": symbol, 
        "signal": signal, 
        "confidence": confidence, 
        "entry_price": smart_entry, 
        "targets": targets, 
        "stop_loss": stop_loss, 
        "strategy": "Mock Multi-Timeframe"
    }

# ==============================================================================
# Function Selection Logic
# ==============================================================================

if UTILS_AVAILABLE:
    get_market_data_func = get_market_data_with_fallback
    analyze_func = analyze_with_multi_timeframe_strategy
    calculate_change_func = calculate_24h_change_from_dataframe
    calculate_sma_func = calculate_simple_sma
    calculate_rsi_func = calculate_simple_rsi
    calculate_rsi_series_func = calculate_rsi_series
    calculate_divergence_func = detect_divergence
    calculate_macd_func = calculate_macd_simple
    analyze_scalp_conditions_func = analyze_scalp_conditions
    calculate_ichimoku_func = calculate_ichimoku_components
    analyze_ichimoku_signal_func = analyze_ichimoku_scalp_signal
    get_ichimoku_signal_func = get_ichimoku_scalp_signal
    combined_analysis_func = combined_analysis
    generate_recommendation_func = generate_ichimoku_recommendation
    get_swing_high_low_func = get_swing_high_low
    calculate_smart_entry_func = calculate_smart_entry
    get_support_resistance_levels_func = get_support_resistance_levels
    calculate_volatility_func = calculate_volatility
    print("USING: REAL UTILS")
else:
    get_market_data_func = mock_get_market_data_with_fallback
    analyze_func = mock_analyze_with_multi_timeframe_strategy
    calculate_change_func = mock_calculate_24h_change
    calculate_sma_func = mock_calculate_simple_sma
    calculate_rsi_func = mock_calculate_simple_rsi
    calculate_rsi_series_func = mock_calculate_rsi_series
    calculate_divergence_func = mock_detect_divergence  # FIXED: Using function not lambda
    calculate_macd_func = mock_calculate_macd_simple
    analyze_scalp_conditions_func = mock_analyze_scalp_conditions
    calculate_ichimoku_func = mock_calculate_ichimoku_components
    analyze_ichimoku_signal_func = mock_analyze_ichimoku_scalp_signal
    get_ichimoku_signal_func = mock_get_ichimoku_scalp_signal
    combined_analysis_func = mock_combined_analysis
    generate_recommendation_func = mock_generate_ichimoku_recommendation
    get_swing_high_low_func = mock_get_swing_high_low  # FIXED: Using function
    calculate_smart_entry_func = mock_calculate_smart_entry
    get_support_resistance_levels_func = mock_get_support_resistance_levels
    calculate_volatility_func = mock_calculate_volatility
    print("USING: MOCK UTILS (Fixed)")

# ==============================================================================
# Internal Helper Logic
# ==============================================================================

def analyze_scalp_signal(symbol, timeframe, data):
    """Internal logic for Scalp Signal (RSI + Divergence)."""
    if not data or len(data) < 30:
        return {
            "signal": "HOLD", 
            "confidence": 0.5, 
            "rsi": 50, 
            "divergence": False,
            "sma_20": 0,
            "reason": "Insufficient data"
        }
    
    rsi = calculate_rsi_func(data, 14)
    sma_20 = calculate_sma_func(data, 20)
    closes = [float(c[4]) for c in data[-30:]]  # Use last 30 candles
    rsi_series = calculate_rsi_series_func(closes, 14)
    div_info = calculate_divergence_func(closes, rsi_series, lookback=5)  # FIXED: Correct signature
    latest_close = closes[-1] if closes else 0
    
    signal, confidence, reason = "HOLD", 0.5, "Neutral"
    
    # Improved signal logic
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

def analyze_ichimoku_scalp(symbol, timeframe, data):
    """Internal logic for Ichimoku Scalp."""
    if not data or len(data) < 60:
        return {
            "signal": "HOLD", 
            "confidence": 0.5, 
            "divergence": False, 
            "reason": "Insufficient Ichimoku data"
        }
    
    ichimoku_data = calculate_ichimoku_func(data)
    if not ichimoku_data: 
        return {
            "signal": "HOLD", 
            "confidence": 0.5, 
            "divergence": False, 
            "reason": "Ichimoku calculation failed"
        }
    
    ichimoku_signal = analyze_ichimoku_signal_func(ichimoku_data)
    current_price = ichimoku_data.get('current_price', 0)
    
    if current_price <= 0: 
        return {
            "signal": "HOLD", 
            "confidence": 0.5, 
            "divergence": False, 
            "reason": "Invalid Price"
        }
    
    # Calculate proper targets and stop loss based on signal
    if ichimoku_signal['signal'] == 'BUY':
        # For BUY: targets above entry, stop loss below
        targets = [
            round(current_price * 1.01, 8),   # +1%
            round(current_price * 1.015, 8),  # +1.5%
            round(current_price * 1.02, 8)    # +2%
        ]
        stop_loss = round(current_price * 0.99, 8)  # -1%
    elif ichimoku_signal['signal'] == 'SELL':
        # For SELL: targets below entry, stop loss above
        targets = [
            round(current_price * 0.99, 8),   # -1%
            round(current_price * 0.985, 8),  # -1.5%
            round(current_price * 0.98, 8)    # -2%
        ]
        stop_loss = round(current_price * 1.01, 8)  # +1%
    else: 
        # For HOLD
        targets = [
            round(current_price * 1.005, 8),  # +0.5%
            round(current_price * 1.01, 8),   # +1%
            round(current_price * 1.015, 8)   # +1.5%
        ]
        stop_loss = round(current_price * 0.995, 8)  # -0.5%
    
    levels = {
        'tenkan_sen': ichimoku_data.get('tenkan_sen'), 
        'kijun_sen': ichimoku_data.get('kijun_sen'), 
        'cloud_top': ichimoku_data.get('cloud_top'), 
        'cloud_bottom': ichimoku_data.get('cloud_bottom'), 
        'quality_line': ichimoku_data.get('quality_line'), 
        'golden_line': ichimoku_data.get('golden_line')
    }
    trend_interpretation = "Strong" if ichimoku_data.get('trend_power', 50) >= 70 else "Medium"
    
    return {
        "signal": ichimoku_signal['signal'], 
        "confidence": ichimoku_signal['confidence'], 
        "reason": ichimoku_signal['reason'], 
        "entry_price": current_price, 
        "targets": targets, 
        "stop_loss": stop_loss, 
        "ichimoku": levels, 
        "trend_analysis": {
            "power": ichimoku_data.get('trend_power'), 
            "interpretation": trend_interpretation
        }, 
        "type": "ICHIMOKU_SCALP"
    }

# ==============================================================================
# FastAPI App
# ==============================================================================

API_VERSION = "7.6.6-FIXED"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}", 
    description="Fixed Targets & Stop Loss Calculation", 
    version=API_VERSION, 
    docs_url="/api/docs"
)
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
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
        "endpoints": {
            "/api/health": "Health check",
            "/api/scalp-signal": "Scalp signal (POST)",
            "/api/ichimoku-scalp": "Ichimoku signal (POST)",
            "/api/analyze": "General analysis (POST)",
            "/market/{symbol}": "Market data (GET)"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "Healthy", 
        "version": API_VERSION, 
        "modules": {
            "utils": UTILS_AVAILABLE, 
            "data_collector": DATA_COLLECTOR_AVAILABLE, 
            "collectors": COLLECTORS_AVAILABLE
        }
    }

@app.get("/api/signals", response_model=SignalResponse)
async def get_all_signals_endpoint(symbol: Optional[str] = None):
    try:
        analysis = analyze_func(symbol.upper() if symbol else "BTCUSDT")
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_crypto(request: AnalysisRequest):
    try:
        analysis = analyze_func(request.symbol)
        analysis["version"] = API_VERSION
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        if market_data:
            analysis["rsi"] = calculate_rsi_func(market_data, 14)
            closes = [float(c[4]) for c in market_data]
            rsi_series = calculate_rsi_series_func(closes, 14)
            div = calculate_divergence_func(closes, rsi_series, lookback=5)
            analysis["divergence"] = div['detected']
            analysis["divergence_type"] = div['type']
        return analysis
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """Scalp Signal Endpoint with Smart Entry - FIXED TARGETS VERSION."""
    allowed_timeframes = ["1m", "5m", "15m"]
    if request.timeframe not in allowed_timeframes: 
        raise HTTPException(status_code=400, detail="Invalid timeframe")
    
    try:
        logger.info(f"⚡ Scalp signal request: {request.symbol} ({request.timeframe})")
        
        market_data = get_market_data_func(request.symbol, request.timeframe, 50)
        if not market_data: 
            raise HTTPException(status_code=404, detail="No market data")
        
        # DEBUG: Log market data
        try:
            latest_price = float(market_data[-1][4]) if market_data else 0
            logger.info(f"📊 Market Data - Latest Price: {latest_price}, Candles: {len(market_data)}")
        except:
            logger.warning("⚠️ Could not read market data price")
        
        scalp_analysis = analyze_scalp_signal(request.symbol, request.timeframe, market_data)
        smart_entry_price = calculate_smart_entry_func(market_data, scalp_analysis["signal"])
        
        if smart_entry_price <= 0:
            try: 
                smart_entry_price = float(market_data[-1][4])
            except: 
                smart_entry_price = scalp_analysis.get("current_price", 0)
        
        # Ensure price is reasonable
        if smart_entry_price > 1000000 or smart_entry_price < 0.0001:
            logger.warning(f"⚠️ Unreasonable price detected: {smart_entry_price}, using fallback")
            base_prices = {'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'ALGOUSDT': 0.1187, 'DEFAULT': 100}
            smart_entry_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        
        # DEBUG: Log the signal and entry price
        logger.info(f"🔍 Signal Analysis:")
        logger.info(f"  - Signal: {scalp_analysis['signal']}")
        logger.info(f"  - Entry Price: {smart_entry_price}")
        logger.info(f"  - RSI: {scalp_analysis['rsi']}")
        logger.info(f"  - SMA20: {scalp_analysis.get('sma_20', 0)}")
        logger.info(f"  - Divergence: {scalp_analysis['divergence']}")
        
        # FIXED: Calculate proper targets and stop loss with different percentages
        targets = []
        stop_loss = smart_entry_price
        
        if scalp_analysis["signal"] == "BUY":
            # For BUY: TP above entry, SL below entry
            targets = [
                round(smart_entry_price * 1.01, 2),   # TP1: +1%
                round(smart_entry_price * 1.015, 2),  # TP2: +1.5%
                round(smart_entry_price * 1.02, 2)    # TP3: +2%
            ]
            stop_loss = round(smart_entry_price * 0.99, 2)   # SL: -1%
            logger.info(f"📈 BUY Signal:")
            logger.info(f"  - Targets: {targets}")
            logger.info(f"  - Stop Loss: {stop_loss}")
            
        elif scalp_analysis["signal"] == "SELL":
            # For SELL: TP below entry, SL above entry
            targets = [
                round(smart_entry_price * 0.99, 2),   # TP1: -1%
                round(smart_entry_price * 0.985, 2),  # TP2: -1.5%
                round(smart_entry_price * 0.98, 2)    # TP3: -2%
            ]
            stop_loss = round(smart_entry_price * 1.01, 2)   # SL: +1%
            logger.info(f"📉 SELL Signal:")
            logger.info(f"  - Targets: {targets}")
            logger.info(f"  - Stop Loss: {stop_loss}")
            
        else: 
            # For HOLD signal - Conservative targets
            targets = [
                round(smart_entry_price * 1.005, 2),  # +0.5%
                round(smart_entry_price * 1.01, 2),   # +1%
                round(smart_entry_price * 1.015, 2)   # +1.5%
            ]
            stop_loss = round(smart_entry_price * 0.995, 2)  # -0.5%
            logger.info(f"⏸️ HOLD Signal:")
            logger.info(f"  - Targets: {targets}")
            logger.info(f"  - Stop Loss: {stop_loss}")
        
        # Calculate percentages for display
        try:
            tp1_percent = round(((targets[0] - smart_entry_price) / smart_entry_price) * 100, 2)
            tp2_percent = round(((targets[1] - smart_entry_price) / smart_entry_price) * 100, 2)
            tp3_percent = round(((targets[2] - smart_entry_price) / smart_entry_price) * 100, 2)
            sl_percent = round(((stop_loss - smart_entry_price) / smart_entry_price) * 100, 2)
        except:
            tp1_percent, tp2_percent, tp3_percent, sl_percent = 0, 0, 0, 0
        
        # Determine risk level based on RSI and confidence
        risk_level = "LOW"
        rsi = scalp_analysis["rsi"]
        confidence = scalp_analysis["confidence"]
        
        if (rsi > 75 or rsi < 25) and confidence > 0.8:
            risk_level = "HIGH"
        elif (rsi > 70 or rsi < 30) and confidence > 0.7:
            risk_level = "MEDIUM"
        
        # Format response with proper field names
        response = {
            "symbol": request.symbol, 
            "timeframe": request.timeframe, 
            "signal": scalp_analysis["signal"], 
            "confidence": scalp_analysis["confidence"], 
            "entry_price": round(float(smart_entry_price), 2), 
            "rsi": scalp_analysis["rsi"], 
            "divergence": scalp_analysis["divergence"], 
            "divergence_type": scalp_analysis.get("divergence_type", "none"),
            "sma_20": round(float(scalp_analysis.get("sma_20", 0)), 2),
            "targets": [round(float(t), 2) for t in targets], 
            "stop_loss": round(float(stop_loss), 2),
            "targets_percent": [tp1_percent, tp2_percent, tp3_percent],
            "stop_loss_percent": sl_percent,
            "risk_level": risk_level,
            "reason": scalp_analysis["reason"], 
            "strategy": "Scalp Smart Entry",
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Generated scalp signal:")
        logger.info(f"  - Symbol: {response['symbol']}")
        logger.info(f"  - Signal: {response['signal']}")
        logger.info(f"  - Confidence: {response['confidence']}")
        logger.info(f"  - Entry: ${response['entry_price']}")
        logger.info(f"  - Targets: {response['targets']} ({response['targets_percent']}%)")
        logger.info(f"  - Stop Loss: ${response['stop_loss']} ({response['stop_loss_percent']}%)")
        logger.info(f"  - Risk Level: {response['risk_level']}")
        
        return response
        
    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"❌ Error in scalp signal: {e}", exc_info=True)
        # Fallback response
        mock_signal = random.choice(["BUY", "SELL", "HOLD"])
        base_prices = {'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'ALGOUSDT': 0.1187, 'DEFAULT': 100}
        base_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        entry_price = round(base_price * random.uniform(0.99, 1.01), 2)
        
        if mock_signal == "BUY":
            targets = [
                round(entry_price * 1.01, 2),   # +1%
                round(entry_price * 1.015, 2),  # +1.5%
                round(entry_price * 1.02, 2)    # +2%
            ]
            stop_loss = round(entry_price * 0.99, 2)   # -1%
        elif mock_signal == "SELL":
            targets = [
                round(entry_price * 0.99, 2),   # -1%
                round(entry_price * 0.985, 2),  # -1.5%
                round(entry_price * 0.98, 2)    # -2%
            ]
            stop_loss = round(entry_price * 1.01, 2)   # +1%
        else:
            targets = [
                round(entry_price * 1.005, 2),  # +0.5%
                round(entry_price * 1.01, 2),   # +1%
                round(entry_price * 1.015, 2)   # +1.5%
            ]
            stop_loss = round(entry_price * 0.995, 2)  # -0.5%
        
        # Calculate percentages for fallback
        tp1_percent = round(((targets[0] - entry_price) / entry_price) * 100, 2)
        tp2_percent = round(((targets[1] - entry_price) / entry_price) * 100, 2)
        tp3_percent = round(((targets[2] - entry_price) / entry_price) * 100, 2)
        sl_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
        
        # Determine risk level for fallback
        risk_level = "MEDIUM" if mock_signal != "HOLD" else "LOW"
        
        return {
            "symbol": request.symbol, 
            "timeframe": request.timeframe, 
            "signal": mock_signal, 
            "confidence": 0.6, 
            "entry_price": entry_price, 
            "rsi": random.uniform(30, 70), 
            "targets": targets, 
            "stop_loss": stop_loss,
            "targets_percent": [tp1_percent, tp2_percent, tp3_percent],
            "stop_loss_percent": sl_percent,
            "risk_level": risk_level,
            "reason": "Analysis Error - Using fallback",
            "strategy": "Fallback Mode",
            "sma_20": round(entry_price * random.uniform(0.99, 1.01), 8)
        }

@app.post("/api/ichimoku-scalp")
async def get_ichimoku_scalp_signal(request: IchimokuRequest):
    """Ichimoku Scalp Endpoint."""
    allowed_timeframes = ["1m", "5m", "15m", "1h", "4h"]
    if request.timeframe not in allowed_timeframes: 
        raise HTTPException(status_code=400, detail="Invalid timeframe")
    
    try:
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        if not market_data or len(market_data) < 60: 
            raise HTTPException(status_code=404, detail="Not enough Ichimoku data")
        
        ichimoku_analysis = analyze_ichimoku_scalp(request.symbol, request.timeframe, market_data)
        rsi = calculate_rsi_func(market_data, 14)
        closes = [float(c[4]) for c in market_data]
        rsi_series = calculate_rsi_series_func(closes, 14)
        div = calculate_divergence_func(closes, rsi_series, lookback=5)
        recommendation = generate_recommendation_func(ichimoku_analysis)
        
        # Calculate percentages
        entry_price = ichimoku_analysis["entry_price"]
        targets = ichimoku_analysis["targets"]
        stop_loss = ichimoku_analysis["stop_loss"]
        
        tp_percents = [
            round(((target - entry_price) / entry_price) * 100, 2) 
            for target in targets[:3]
        ] if len(targets) >= 3 else [0, 0, 0]
        
        sl_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
        
        # Determine risk level
        risk_level = "LOW"
        confidence = ichimoku_analysis["confidence"]
        if confidence > 0.8:
            risk_level = "HIGH"
        elif confidence > 0.6:
            risk_level = "MEDIUM"
        
        return {
            "symbol": request.symbol, 
            "timeframe": request.timeframe, 
            "signal": ichimoku_analysis["signal"], 
            "confidence": ichimoku_analysis["confidence"], 
            "entry_price": round(float(entry_price), 2), 
            "targets": [round(float(t), 2) for t in targets], 
            "stop_loss": round(float(stop_loss), 2),
            "targets_percent": tp_percents,
            "stop_loss_percent": sl_percent,
            "rsi": round(rsi, 2), 
            "divergence": div['detected'], 
            "divergence_type": div['type'], 
            "reason": ichimoku_analysis["reason"], 
            "strategy": f"Ichimoku Scalp ({request.timeframe})", 
            "recommendation": recommendation,
            "trend_power": ichimoku_analysis.get("trend_analysis", {}).get("power", 50),
            "risk_level": risk_level
        }
    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Error in Ichimoku: {e}")
        return {
            "symbol": request.symbol, 
            "timeframe": request.timeframe, 
            "signal": "HOLD", 
            "confidence": 0.5, 
            "reason": "Ichimoku Error"
        }

@app.post("/api/combined-analysis")
async def get_combined_analysis(request: CombinedRequest):
    """Combined Analysis Endpoint."""
    try:
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        if not market_data: 
            raise HTTPException(status_code=404, detail="No data")
        
        results = {
            'rsi': calculate_rsi_func(market_data, 14),  # FIXED: Using calculate_rsi_func
            'sma_20': calculate_sma_func(market_data, 20), 
            'macd': calculate_macd_func(market_data),  # FIXED: Using calculate_macd_func
            'ichimoku': get_ichimoku_signal_func(market_data, request.timeframe), 
            'support_resistance': get_support_resistance_levels_func(market_data), 
            'volatility': calculate_volatility_func(market_data, 20)
        }
        
        latest_price = float(market_data[-1][4])
        signals = {'buy': 0, 'sell': 0, 'hold': 0}
        
        if results['rsi'] < 30: 
            signals['buy'] += 1.5
        elif results['rsi'] > 70: 
            signals['sell'] += 1.5
        else: 
            signals['hold'] += 1
            
        if latest_price > results['sma_20']: 
            signals['buy'] += 1
        else: 
            signals['sell'] += 1
            
        if results['macd']['histogram'] > 0: 
            signals['buy'] += 1
        else: 
            signals['sell'] += 1
            
        if results['ichimoku']:
            ich_signal = results['ichimoku'].get('signal', 'HOLD')
            if ich_signal == 'BUY': 
                signals['buy'] += 2
            elif ich_signal == 'SELL': 
                signals['sell'] += 2
        
        final_signal = max(signals, key=signals.get)
        confidence = signals[final_signal] / sum(signals.values()) if sum(signals.values()) > 0 else 0.5
        
        return {
            'signal': final_signal.upper(), 
            'confidence': round(confidence, 3), 
            'details': results, 
            'price': latest_price, 
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    try:
        data = get_market_data_func(symbol, timeframe, limit=50)
        if not data: 
            raise HTTPException(status_code=404, detail="No data")
        
        latest = data[-1] if isinstance(data, list) and len(data) > 0 else []
        if not latest or len(latest) < 6: 
            return {
                "symbol": symbol, 
                "timeframe": timeframe, 
                "source": "Mock", 
                "current_price": 100, 
                "change_24h": 0
            }
        
        change_24h = calculate_change_func(data)
        rsi = calculate_rsi_func(data, 14)  # FIXED: Using calculate_rsi_func
        sma_20 = calculate_sma_func(data, 20)
        closes = [float(c[4]) for c in data]
        rsi_series = calculate_rsi_series_func(closes, 14)
        div = calculate_divergence_func(closes, rsi_series, lookback=5)
        
        return {
            "symbol": symbol, 
            "timeframe": timeframe, 
            "source": "Binance" if UTILS_AVAILABLE else "Mock", 
            "current_price": float(latest[4]), 
            "high": float(latest[2]), 
            "low": float(latest[3]), 
            "change_24h": change_24h, 
            "rsi_14": round(rsi, 2), 
            "sma_20": round(sma_20, 2), 
            "divergence": div['detected'], 
            "divergence_type": div['type']
        }
    except HTTPException: 
        raise
    except Exception as e: 
        raise HTTPException(status_code=500, detail="Market Data Error")

@app.get("/signals/scraped")
async def get_scraped_signals():
    """Scraped Signals Endpoint."""
    try:
        if COLLECTORS_AVAILABLE: 
            scraped_signals = collect_signals_from_example_site()
        else: 
            scraped_signals = [{
                "symbol": "BTCUSDT", 
                "signal": random.choice(["BUY", "SELL", "HOLD"]), 
                "confidence": 0.7
            }]
        return {
            "status": "success", 
            "count": len(scraped_signals), 
            "signals": scraped_signals
        }
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scan-all-timeframes/{symbol}")
async def scan_all_timeframes(symbol: str):
    """Scan All Timeframes Endpoint."""
    try:
        timeframes = ["1m", "5m", "15m", "1h", "4h"]
        results = []
        for tf in timeframes:
            try:
                if tf in ["1m", "5m", "15m"]: 
                    response = await get_ichimoku_scalp_signal(
                        IchimokuRequest(symbol=symbol, timeframe=tf)
                    )
                else: 
                    response = await analyze_crypto(
                        AnalysisRequest(symbol=symbol, timeframe=tf)
                    )
                response["timeframe"] = tf
                results.append(response)
            except Exception: 
                results.append({
                    "symbol": symbol, 
                    "timeframe": tf, 
                    "signal": "ERROR", 
                    "error": "Analysis failed"
                })
        return {
            "symbol": symbol, 
            "scanned_at": datetime.now().isoformat(), 
            "total": len(timeframes), 
            "successful": len([r for r in results if r.get("signal") != "ERROR"]), 
            "results": results
        }
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting v{API_VERSION}")
    logger.info(f"Utils Available: {UTILS_AVAILABLE}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)