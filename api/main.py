"""
Crypto AI Trading System v8.0.0 - Optimized for Render
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import logging
from typing import List, Optional, Dict, Any
import random
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# Imports
# ==============================================================================

# Try to import utils
try:
    from utils import (
        get_market_data_with_fallback,
        calculate_simple_sma,
        calculate_simple_rsi,
        detect_divergence,
        calculate_rsi_series,
        calculate_smart_entry,
        get_swing_high_low,
        analyze_ichimoku_scalp_signal,
        calculate_ichimoku_components,
        get_ichimoku_scalp_signal
    )
    UTILS_AVAILABLE = True
    logger.info("✅ Utils module loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Utils import failed: {e}")
    UTILS_AVAILABLE = False

# ==============================================================================
# Pydantic Models
# ==============================================================================

class ScalpRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class IchimokuRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

# ==============================================================================
# Mock Functions (Fallback)
# ==============================================================================

def mock_get_market_data(symbol, interval="5m", limit=50):
    """Mock market data generator"""
    base_prices = {
        'BTCUSDT': random.uniform(85000, 95000),
        'ETHUSDT': random.uniform(3000, 4000),
        'ALGOUSDT': random.uniform(0.10, 0.15),
        'BNBUSDT': random.uniform(550, 650),
        'SOLUSDT': random.uniform(150, 200)
    }
    
    base_price = base_prices.get(symbol.upper(), 100)
    data = []
    current_time = int(datetime.now().timestamp() * 1000)
    
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)
        change = random.uniform(-0.02, 0.02)
        price = base_price * (1 + change)
        
        candle = [
            timestamp,
            str(price * random.uniform(0.99, 1.00)),
            str(price * random.uniform(1.00, 1.01)),
            str(price * random.uniform(0.99, 1.00)),
            str(price),
            str(random.uniform(1000, 10000)),
            timestamp + 300000,
            "0", "0", "0", "0", "0"
        ]
        data.append(candle)
    
    return data

def mock_calculate_sma(data, period=20):
    if not data or len(data) < period:
        return 0
    try:
        closes = [float(c[4]) for c in data[-period:]]
        return sum(closes) / len(closes)
    except:
        return 0

def mock_calculate_rsi(data, period=14):
    return random.uniform(30, 70)

# ==============================================================================
# Function Selection
# ==============================================================================

if UTILS_AVAILABLE:
    get_market_data_func = get_market_data_with_fallback
    calculate_sma_func = calculate_simple_sma
    calculate_rsi_func = calculate_simple_rsi
    calculate_divergence_func = detect_divergence
    calculate_rsi_series_func = calculate_rsi_series
    calculate_smart_entry_func = calculate_smart_entry
    get_swing_high_low_func = get_swing_high_low
    analyze_ichimoku_signal_func = analyze_ichimoku_scalp_signal
    calculate_ichimoku_func = calculate_ichimoku_components
    get_ichimoku_signal_func = get_ichimoku_scalp_signal
else:
    get_market_data_func = mock_get_market_data
    calculate_sma_func = mock_calculate_sma
    calculate_rsi_func = mock_calculate_rsi
    calculate_divergence_func = lambda p, r, l=5: {"detected": False, "type": "none"}
    calculate_rsi_series_func = lambda d, p=14: [50] * len(d) if d else []
    calculate_smart_entry_func = lambda d, s="BUY": float(d[-1][4]) if d else 0
    get_swing_high_low_func = lambda d, p=20: (100, 50)
    analyze_ichimoku_signal_func = lambda d: {"signal": "HOLD", "confidence": 0.5}
    calculate_ichimoku_func = lambda d: {"current_price": float(d[-1][4]) if d else 0}
    get_ichimoku_signal_func = lambda d, t="5m": None

# ==============================================================================
# FastAPI App
# ==============================================================================

app = FastAPI(
    title="Crypto AI Trading System",
    description="Professional Trading Signals API",
    version="8.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Helper Functions
# ==============================================================================

def analyze_scalp_signal(symbol, timeframe, data):
    """Analyze scalp signal"""
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
    closes = [float(c[4]) for c in data[-30:]]
    rsi_series = calculate_rsi_series_func(closes, 14)
    div_info = calculate_divergence_func(closes, rsi_series, lookback=5)
    latest_close = closes[-1] if closes else 0
    
    signal, confidence, reason = "HOLD", 0.5, "Neutral"
    
    if div_info['detected']:
        if div_info['type'] == 'bullish':
            signal, confidence, reason = "BUY", 0.85, "Bullish divergence"
        elif div_info['type'] == 'bearish':
            signal, confidence, reason = "SELL", 0.85, "Bearish divergence"
    else:
        if rsi < 35 and latest_close < sma_20:
            signal, confidence, reason = "BUY", 0.75, f"Oversold (RSI: {rsi:.1f})"
        elif rsi > 65 and latest_close > sma_20:
            signal, confidence, reason = "SELL", 0.75, f"Overbought (RSI: {rsi:.1f})"
    
    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "rsi": round(rsi, 1),
        "divergence": div_info['detected'],
        "divergence_type": div_info['type'],
        "sma_20": round(sma_20, 8),
        "current_price": round(latest_close, 8),
        "reason": reason
    }

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def root():
    return {
        "message": "Crypto AI Trading System API",
        "status": "active",
        "version": "8.0.0",
        "endpoints": {
            "/api/health": "Health check",
            "/api/scalp-signal": "Scalp signal (POST)",
            "/api/ichimoku-scalp": "Ichimoku signal (POST)",
            "/market/{symbol}": "Market data (GET)",
            "/api/docs": "API documentation"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "utils_available": UTILS_AVAILABLE,
        "environment": os.getenv("RENDER", "development")
    }

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """Get scalp trading signal"""
    try:
        logger.info(f"📡 Scalp signal request: {request.symbol} ({request.timeframe})")
        
        # Validate timeframe
        allowed_timeframes = ["1m", "5m", "15m"]
        if request.timeframe not in allowed_timeframes:
            raise HTTPException(status_code=400, detail="Invalid timeframe")
        
        # Get market data
        market_data = get_market_data_func(request.symbol, request.timeframe, 50)
        if not market_data:
            raise HTTPException(status_code=404, detail="No market data")
        
        # Analyze signal
        analysis = analyze_scalp_signal(request.symbol, request.timeframe, market_data)
        
        # Calculate smart entry
        smart_entry = calculate_smart_entry_func(market_data, analysis["signal"])
        if smart_entry <= 0:
            try:
                smart_entry = float(market_data[-1][4])
            except:
                smart_entry = analysis.get("current_price", 0)
        
        # Calculate targets and stop loss
        if analysis["signal"] == "BUY":
            targets = [
                round(smart_entry * 1.01, 8),
                round(smart_entry * 1.015, 8),
                round(smart_entry * 1.02, 8)
            ]
            stop_loss = round(smart_entry * 0.99, 8)
        elif analysis["signal"] == "SELL":
            targets = [
                round(smart_entry * 0.99, 8),
                round(smart_entry * 0.985, 8),
                round(smart_entry * 0.98, 8)
            ]
            stop_loss = round(smart_entry * 1.01, 8)
        else:
            targets = [
                round(smart_entry * 1.005, 8),
                round(smart_entry * 1.01, 8),
                round(smart_entry * 1.015, 8)
            ]
            stop_loss = round(smart_entry * 0.995, 8)
        
        # Calculate percentages
        tp_percents = [
            round(((t - smart_entry) / smart_entry) * 100, 2)
            for t in targets[:3]
        ]
        sl_percent = round(((stop_loss - smart_entry) / smart_entry) * 100, 2)
        
        # Determine risk level
        risk_level = "LOW"
        if analysis["rsi"] > 75 or analysis["rsi"] < 25:
            risk_level = "HIGH"
        elif analysis["rsi"] > 70 or analysis["rsi"] < 30:
            risk_level = "MEDIUM"
        
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": analysis["signal"],
            "confidence": analysis["confidence"],
            "entry_price": round(smart_entry, 8),
            "rsi": analysis["rsi"],
            "sma_20": analysis["sma_20"],
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": tp_percents,
            "stop_loss_percent": sl_percent,
            "risk_level": risk_level,
            "reason": analysis["reason"],
            "divergence": analysis["divergence"],
            "divergence_type": analysis["divergence_type"],
            "generated_at": datetime.now().isoformat(),
            "strategy": "AI Scalp Strategy"
        }
        
        logger.info(f"✅ Signal generated: {response['signal']} for {response['symbol']}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in scalp signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ichimoku-scalp")
async def get_ichimoku_signal(request: IchimokuRequest):
    """Get Ichimoku cloud signal"""
    try:
        # Validate timeframe
        allowed_timeframes = ["5m", "15m", "1h", "4h"]
        if request.timeframe not in allowed_timeframes:
            raise HTTPException(status_code=400, detail="Invalid timeframe")
        
        # Get market data
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        if not market_data or len(market_data) < 60:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        # Get Ichimoku signal
        ichimoku_data = calculate_ichimoku_func(market_data)
        if not ichimoku_data:
            raise HTTPException(status_code=500, detail="Ichimoku calculation failed")
        
        signal_info = analyze_ichimoku_signal_func(ichimoku_data)
        current_price = ichimoku_data.get('current_price', 0)
        
        # Calculate targets
        if signal_info['signal'] == 'BUY':
            targets = [
                round(current_price * 1.01, 8),
                round(current_price * 1.02, 8),
                round(current_price * 1.03, 8)
            ]
            stop_loss = round(current_price * 0.99, 8)
        elif signal_info['signal'] == 'SELL':
            targets = [
                round(current_price * 0.99, 8),
                round(current_price * 0.98, 8),
                round(current_price * 0.97, 8)
            ]
            stop_loss = round(current_price * 1.01, 8)
        else:
            targets = [round(current_price * 1.01, 8)]
            stop_loss = round(current_price * 0.99, 8)
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": signal_info['signal'],
            "confidence": signal_info['confidence'],
            "entry_price": round(current_price, 8),
            "targets": targets,
            "stop_loss": stop_loss,
            "reason": signal_info.get('reason', 'Ichimoku analysis'),
            "generated_at": datetime.now().isoformat(),
            "strategy": "Ichimoku Cloud Strategy"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in Ichimoku signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    """Get market data"""
    try:
        data = get_market_data_func(symbol, timeframe, 50)
        if not data:
            raise HTTPException(status_code=404, detail="No data")
        
        latest = data[-1]
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": float(latest[4]),
            "high": float(latest[2]),
            "low": float(latest[3]),
            "volume": float(latest[5]),
            "timestamp": datetime.now().isoformat(),
            "source": "Binance" if UTILS_AVAILABLE else "Mock"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting market data: {e}")
        raise HTTPException(status_code=500, detail="Market data error")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Trading API starting up...")
    logger.info(f"📊 Utils available: {UTILS_AVAILABLE}")
    logger.info(f"🌐 Environment: {os.getenv('RENDER', 'development')}")

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)