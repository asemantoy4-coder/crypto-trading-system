"""
Crypto AI Trading System v7.6.8 - Render Fixed
Simplified version without circular imports
"""

from fastapi import FastAPI, HTTPException
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
# Simple Utility Functions (No external imports to avoid circular imports)
# ==============================================================================

def calculate_targets_and_stop_loss(entry_price, signal):
    """Calculate 3 targets and stop loss with percentages."""
    try:
        entry = float(entry_price)
        
        if signal == "BUY":
            targets = [
                round(entry * 1.01, 2),   # +1%
                round(entry * 1.015, 2),  # +1.5%
                round(entry * 1.02, 2)    # +2%
            ]
            stop_loss = round(entry * 0.99, 2)   # -1%
        elif signal == "SELL":
            targets = [
                round(entry * 0.99, 2),   # -1%
                round(entry * 0.985, 2),  # -1.5%
                round(entry * 0.98, 2)    # -2%
            ]
            stop_loss = round(entry * 1.01, 2)   # +1%
        else:  # HOLD
            targets = [
                round(entry * 1.005, 2),  # +0.5%
                round(entry * 1.01, 2),   # +1%
                round(entry * 1.015, 2)   # +1.5%
            ]
            stop_loss = round(entry * 0.995, 2)  # -0.5%
        
        # Calculate percentages
        tp_percents = [
            round(((t - entry) / entry) * 100, 2) for t in targets
        ]
        sl_percent = round(((stop_loss - entry) / entry) * 100, 2)
        
        return {
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": tp_percents,
            "stop_loss_percent": sl_percent
        }
    except Exception as e:
        logger.error(f"Error in calculate_targets_and_stop_loss: {e}")
        # Default values
        return {
            "targets": [100.0, 101.0, 102.0],
            "stop_loss": 99.0,
            "targets_percent": [1.0, 1.5, 2.0],
            "stop_loss_percent": -1.0
        }

def format_price(price):
    """Format price for display."""
    try:
        num = float(price)
        if num >= 1000:
            return f"${num:,.0f}"
        elif num >= 1:
            return f"${num:,.2f}"
        else:
            return f"${num:.6f}"
    except:
        return "$0.00"

def get_mock_market_data(symbol, interval="5m", limit=100):
    """Mock market data generator."""
    base_prices = {
        'BTCUSDT': 88271.42,
        'ETHUSDT': 3450.12,
        'BNBUSDT': 590.54,
        'SOLUSDT': 175.98,
        'DEFAULT': 100.0
    }
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    
    data = []
    import time
    current_time = int(time.time() * 1000)
    
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)
        change = random.uniform(-0.01, 0.01)
        price = base_price * (1 + change)
        
        candle = [
            timestamp,
            str(price * 0.998),  # open
            str(price * 1.002),  # high
            str(price * 0.997),  # low
            str(price),          # close
            str(random.uniform(1000, 10000)),  # volume
            timestamp + 300000,
            "0", "0", "0", "0", "0"
        ]
        data.append(candle)
    
    return data

def mock_analyze_with_multi_timeframe_strategy(symbol):
    """Mock analysis function."""
    base_prices = {
        'BTCUSDT': 88271.42,
        'ETHUSDT': 3450.12,
        'BNBUSDT': 590.54,
        'SOLUSDT': 175.98,
        'DEFAULT': 100.0
    }
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    
    signals = ["BUY", "SELL", "HOLD"]
    weights = [0.4, 0.4, 0.2]
    signal = random.choices(signals, weights=weights)[0]
    
    if signal == "HOLD":
        confidence = round(random.uniform(0.5, 0.7), 2)
    else:
        confidence = round(random.uniform(0.7, 0.9), 2)
    
    entry_price = round(base_price * random.uniform(0.99, 1.01), 2)
    
    # Calculate targets
    calc_result = calculate_targets_and_stop_loss(entry_price, signal)
    
    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "entry_price": entry_price,
        "entry_price_formatted": format_price(entry_price),
        "targets": calc_result["targets"],
        "targets_formatted": [format_price(t) for t in calc_result["targets"]],
        "stop_loss": calc_result["stop_loss"],
        "stop_loss_formatted": format_price(calc_result["stop_loss"]),
        "targets_percent": calc_result["targets_percent"],
        "stop_loss_percent": calc_result["stop_loss_percent"],
        "strategy": "Multi-Timeframe Analysis",
        "reason": "Technical analysis based on multiple timeframes"
    }

def mock_calculate_simple_rsi(data, period=14):
    """Mock RSI calculation."""
    return round(random.uniform(30, 70), 1)

def mock_calculate_simple_sma(data, period=20):
    """Mock SMA calculation."""
    if not data or len(data) < period:
        return 100.0
    try:
        closes = [float(c[4]) for c in data[-period:]]
        return round(sum(closes) / len(closes), 2)
    except:
        return 100.0

def mock_calculate_smart_entry(data, signal="BUY"):
    """Mock smart entry calculation."""
    try:
        if not data:
            base_prices = {'BTCUSDT': 88271.42, 'DEFAULT': 100.0}
            base_price = base_prices.get('BTCUSDT', 100.0)
        else:
            base_price = float(data[-1][4])
        
        if signal == "BUY":
            return round(base_price * random.uniform(0.995, 0.999), 2)
        elif signal == "SELL":
            return round(base_price * random.uniform(1.001, 1.005), 2)
        else:
            return round(base_price, 2)
    except:
        return 100.0

# ==============================================================================
# Try to import real utils (optional)
# ==============================================================================

REAL_UTILS_AVAILABLE = False
try:
    # Try to import utils from the same directory
    import utils
    REAL_UTILS_AVAILABLE = True
    logger.info("✅ Real utils imported successfully")
    
    # Use real functions if available
    get_market_data_func = utils.get_market_data_with_fallback
    analyze_func = utils.analyze_with_multi_timeframe_strategy
    calculate_rsi_func = utils.calculate_simple_rsi
    calculate_sma_func = utils.calculate_simple_sma
    calculate_smart_entry_func = utils.calculate_smart_entry
    
except ImportError as e:
    logger.warning(f"⚠️ Using mock utils: {e}")
    REAL_UTILS_AVAILABLE = False
    
    # Use mock functions
    get_market_data_func = get_mock_market_data
    analyze_func = mock_analyze_with_multi_timeframe_strategy
    calculate_rsi_func = mock_calculate_simple_rsi
    calculate_sma_func = mock_calculate_simple_sma
    calculate_smart_entry_func = mock_calculate_smart_entry

# ==============================================================================
# Pydantic Models
# ==============================================================================

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class ScalpRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

# ==============================================================================
# FastAPI App
# ==============================================================================

API_VERSION = "7.6.8-SIMPLE"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description="Simple version for Render deployment",
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url=None
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
async def root():
    return {
        "message": f"Crypto AI Trading System v{API_VERSION}",
        "status": "Running",
        "real_utils": REAL_UTILS_AVAILABLE,
        "endpoints": {
            "/": "This page",
            "/api/health": "Health check",
            "/api/test": "Test formatting",
            "/api/analyze": "Analysis (POST)",
            "/api/scalp-signal": "Scalp signal (POST)"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "real_utils": REAL_UTILS_AVAILABLE,
        "render": True
    }

@app.get("/api/test")
async def test_format():
    """Test endpoint to verify everything works."""
    test_price = 88271.42
    calc_result = calculate_targets_and_stop_loss(test_price, "BUY")
    
    return {
        "test": "Number Formatting Test",
        "original_price": test_price,
        "formatted_price": format_price(test_price),
        "targets_raw": calc_result["targets"],
        "targets_formatted": [format_price(t) for t in calc_result["targets"]],
        "stop_loss_raw": calc_result["stop_loss"],
        "stop_loss_formatted": format_price(calc_result["stop_loss"]),
        "targets_percent": calc_result["targets_percent"],
        "stop_loss_percent": calc_result["stop_loss_percent"],
        "api_version": API_VERSION,
        "utils": REAL_UTILS_AVAILABLE
    }

@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    """General analysis endpoint."""
    try:
        logger.info(f"Analyzing {request.symbol} ({request.timeframe})")
        
        # Get analysis
        analysis = analyze_func(request.symbol)
        
        # Ensure we have all required fields
        if "targets_formatted" not in analysis:
            analysis["targets_formatted"] = [format_price(t) for t in analysis.get("targets", [])]
        
        if "entry_price_formatted" not in analysis:
            analysis["entry_price_formatted"] = format_price(analysis.get("entry_price", 0))
        
        if "stop_loss_formatted" not in analysis:
            analysis["stop_loss_formatted"] = format_price(analysis.get("stop_loss", 0))
        
        # Add metadata
        analysis["generated_at"] = datetime.now().isoformat()
        analysis["version"] = API_VERSION
        analysis["timeframe"] = request.timeframe
        analysis["real_analysis"] = REAL_UTILS_AVAILABLE
        
        logger.info(f"Analysis complete: {analysis['signal']} for {analysis['symbol']}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        
        # Fallback
        base_prices = {
            'BTCUSDT': 88271.42,
            'ETHUSDT': 3450.12,
            'BNBUSDT': 590.54,
            'SOLUSDT': 175.98,
            'DEFAULT': 100.0
        }
        entry_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        calc_result = calculate_targets_and_stop_loss(entry_price, "HOLD")
        
        return {
            "symbol": request.symbol.upper(),
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": entry_price,
            "entry_price_formatted": format_price(entry_price),
            "targets": calc_result["targets"],
            "targets_formatted": [format_price(t) for t in calc_result["targets"]],
            "stop_loss": calc_result["stop_loss"],
            "stop_loss_formatted": format_price(calc_result["stop_loss"]),
            "targets_percent": calc_result["targets_percent"],
            "stop_loss_percent": calc_result["stop_loss_percent"],
            "reason": f"Analysis error",
            "strategy": "Fallback",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION,
            "timeframe": request.timeframe,
            "real_analysis": False
        }

@app.post("/api/scalp-signal")
async def scalp_signal(request: ScalpRequest):
    """Scalp signal endpoint."""
    allowed_timeframes = ["1m", "5m", "15m"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(400, f"Invalid timeframe. Use: {allowed_timeframes}")
    
    try:
        logger.info(f"Scalp signal for {request.symbol} ({request.timeframe})")
        
        # Get market data
        market_data = get_market_data_func(request.symbol, request.timeframe, 50)
        
        # Calculate indicators
        rsi = calculate_rsi_func(market_data, 14)
        sma_20 = calculate_sma_func(market_data, 20)
        
        # Get current price
        try:
            current_price = float(market_data[-1][4]) if market_data else 100.0
        except:
            current_price = 100.0
        
        # Determine signal based on RSI
        signal = "HOLD"
        confidence = 0.5
        reason = "Neutral market"
        
        if rsi < 30:
            signal = "BUY"
            confidence = 0.85
            reason = f"Oversold (RSI: {rsi:.1f})"
        elif rsi > 70:
            signal = "SELL"
            confidence = 0.85
            reason = f"Overbought (RSI: {rsi:.1f})"
        elif rsi < 40:
            signal = "BUY"
            confidence = 0.7
            reason = f"Near oversold (RSI: {rsi:.1f})"
        elif rsi > 60:
            signal = "SELL"
            confidence = 0.7
            reason = f"Near overbought (RSI: {rsi:.1f})"
        
        # Get smart entry
        entry_price = calculate_smart_entry_func(market_data, signal)
        if entry_price <= 0:
            entry_price = current_price
        
        # Calculate targets
        calc_result = calculate_targets_and_stop_loss(entry_price, signal)
        
        # Risk level
        risk_level = "MEDIUM"
        if (rsi < 25 or rsi > 75) and confidence > 0.8:
            risk_level = "HIGH"
        elif confidence < 0.6:
            risk_level = "LOW"
        
        response = {
            "symbol": request.symbol.upper(),
            "timeframe": request.timeframe,
            "signal": signal,
            "confidence": round(confidence, 2),
            "entry_price": round(entry_price, 2),
            "entry_price_formatted": format_price(entry_price),
            "rsi": round(rsi, 1),
            "sma_20": round(sma_20, 2),
            "targets": calc_result["targets"],
            "targets_formatted": [format_price(t) for t in calc_result["targets"]],
            "stop_loss": calc_result["stop_loss"],
            "stop_loss_formatted": format_price(calc_result["stop_loss"]),
            "targets_percent": calc_result["targets_percent"],
            "stop_loss_percent": calc_result["stop_loss_percent"],
            "risk_level": risk_level,
            "reason": reason,
            "strategy": "RSI Scalp",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION,
            "real_data": REAL_UTILS_AVAILABLE
        }
        
        logger.info(f"Scalp signal generated: {signal} for {request.symbol}")
        
        return response
        
    except Exception as e:
        logger.error(f"Scalp signal error: {e}")
        
        # Fallback
        base_prices = {
            'BTCUSDT': 88271.42,
            'ETHUSDT': 3450.12,
            'BNBUSDT': 590.54,
            'SOLUSDT': 175.98,
            'DEFAULT': 100.0
        }
        entry_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        calc_result = calculate_targets_and_stop_loss(entry_price, "HOLD")
        
        return {
            "symbol": request.symbol.upper(),
            "timeframe": request.timeframe,
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": entry_price,
            "entry_price_formatted": format_price(entry_price),
            "rsi": 50.0,
            "sma_20": entry_price,
            "targets": calc_result["targets"],
            "targets_formatted": [format_price(t) for t in calc_result["targets"]],
            "stop_loss": calc_result["stop_loss"],
            "stop_loss_formatted": format_price(calc_result["stop_loss"]),
            "targets_percent": calc_result["targets_percent"],
            "stop_loss_percent": calc_result["stop_loss_percent"],
            "risk_level": "LOW",
            "reason": "Error in analysis",
            "strategy": "Fallback",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION,
            "real_data": False
        }

# ==============================================================================
# Startup
# ==============================================================================

@app.on_event("startup")
async def startup():
    logger.info(f"🚀 Starting v{API_VERSION}")
    logger.info(f"📊 Real utils: {REAL_UTILS_AVAILABLE}")

# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)