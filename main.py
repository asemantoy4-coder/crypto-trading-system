"""
Crypto AI Trading System v7.6.10 - Fixed Circular Import
Main API file
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import logging
from typing import List, Optional, Dict, Any
import random
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# Import utils (should work now)
# ==============================================================================

print("Importing utils module...")

try:
    from utils import (
        get_market_data_with_fallback,
        analyze_with_multi_timeframe_strategy,
        calculate_simple_rsi,
        calculate_simple_sma,
        calculate_smart_entry,
        get_ichimoku_scalp_signal,
        generate_ichimoku_recommendation,
        calculate_24h_change_from_dataframe,
        get_support_resistance_levels
    )
    UTILS_AVAILABLE = True
    print("✅ SUCCESS: Utils imported")
except ImportError as e:
    print(f"❌ FAILED: {e}")
    UTILS_AVAILABLE = False

# ==============================================================================
# Helper Functions
# ==============================================================================

def calculate_targets_and_stop_loss(entry_price, signal):
    """Calculate targets and stop loss."""
    try:
        entry = float(entry_price)
        
        if signal == "BUY":
            targets = [
                round(entry * 1.01, 2),   # +1%
                round(entry * 1.015, 2),  # +1.5%
                round(entry * 1.02, 2)    # +2%
            ]
            stop_loss = round(entry * 0.99, 2)
        elif signal == "SELL":
            targets = [
                round(entry * 0.99, 2),   # -1%
                round(entry * 0.985, 2),  # -1.5%
                round(entry * 0.98, 2)    # -2%
            ]
            stop_loss = round(entry * 1.01, 2)
        else:  # HOLD
            targets = [
                round(entry * 1.005, 2),  # +0.5%
                round(entry * 1.01, 2),   # +1%
                round(entry * 1.015, 2)   # +1.5%
            ]
            stop_loss = round(entry * 0.995, 2)
        
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
    except:
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

# ==============================================================================
# FastAPI App
# ==============================================================================

API_VERSION = "7.6.10-FIXED"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description="Fixed circular import issue",
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
async def root():
    return {
        "message": f"Crypto AI Trading System v{API_VERSION}",
        "status": "Running",
        "utils": UTILS_AVAILABLE,
        "endpoints": {
            "/": "This page",
            "/api/health": "Health check",
            "/api/test": "Test endpoint",
            "/api/analyze": "Analysis (POST)",
            "/api/scalp-signal": "Scalp signal (POST)"
        }
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": API_VERSION,
        "utils": UTILS_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/test")
async def test():
    """Test endpoint."""
    return {
        "status": "ok",
        "message": "API is working",
        "utils": UTILS_AVAILABLE,
        "test_price": 88271.42,
        "formatted": format_price(88271.42)
    }

@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    """Analysis endpoint."""
    try:
        if not UTILS_AVAILABLE:
            raise Exception("Utils not available")
        
        analysis = analyze_with_multi_timeframe_strategy(request.symbol)
        
        # Add formatted prices
        analysis["entry_price_formatted"] = format_price(analysis["entry_price"])
        analysis["targets_formatted"] = [format_price(t) for t in analysis["targets"]]
        analysis["stop_loss_formatted"] = format_price(analysis["stop_loss"])
        
        # Add metadata
        analysis["generated_at"] = datetime.now().isoformat()
        analysis["version"] = API_VERSION
        
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
        calc = calculate_targets_and_stop_loss(entry_price, "HOLD")
        
        return {
            "symbol": request.symbol,
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": entry_price,
            "entry_price_formatted": format_price(entry_price),
            "targets": calc["targets"],
            "targets_formatted": [format_price(t) for t in calc["targets"]],
            "stop_loss": calc["stop_loss"],
            "stop_loss_formatted": format_price(calc["stop_loss"]),
            "targets_percent": calc["targets_percent"],
            "stop_loss_percent": calc["stop_loss_percent"],
            "reason": "Utils not available",
            "strategy": "Fallback",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION
        }

@app.post("/api/scalp-signal")
async def scalp_signal(request: ScalpRequest):
    """Scalp signal endpoint."""
    try:
        if not UTILS_AVAILABLE:
            raise Exception("Utils not available")
        
        # Get market data
        market_data = get_market_data_with_fallback(request.symbol, request.timeframe, 50)
        
        # Calculate indicators
        rsi = calculate_simple_rsi(market_data, 14)
        sma_20 = calculate_simple_sma(market_data, 20)
        
        # Determine signal
        signal = "HOLD"
        confidence = 0.5
        reason = "Neutral"
        
        if rsi < 30:
            signal = "BUY"
            confidence = 0.85
            reason = f"Oversold (RSI: {rsi})"
        elif rsi > 70:
            signal = "SELL"
            confidence = 0.85
            reason = f"Overbought (RSI: {rsi})"
        
        # Get entry price
        entry_price = calculate_smart_entry(market_data, signal)
        if entry_price <= 0:
            entry_price = float(market_data[-1][4]) if market_data else 100.0
        
        # Calculate targets
        calc = calculate_targets_and_stop_loss(entry_price, signal)
        
        response = {
            "symbol": request.symbol.upper(),
            "timeframe": request.timeframe,
            "signal": signal,
            "confidence": round(confidence, 2),
            "entry_price": round(entry_price, 2),
            "entry_price_formatted": format_price(entry_price),
            "rsi": rsi,
            "sma_20": sma_20,
            "targets": calc["targets"],
            "targets_formatted": [format_price(t) for t in calc["targets"]],
            "stop_loss": calc["stop_loss"],
            "stop_loss_formatted": format_price(calc["stop_loss"]),
            "targets_percent": calc["targets_percent"],
            "stop_loss_percent": calc["stop_loss_percent"],
            "reason": reason,
            "strategy": "RSI Scalp",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Scalp error: {e}")
        
        # Fallback
        base_prices = {
            'BTCUSDT': 88271.42,
            'ETHUSDT': 3450.12,
            'BNBUSDT': 590.54,
            'SOLUSDT': 175.98,
            'DEFAULT': 100.0
        }
        entry_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        calc = calculate_targets_and_stop_loss(entry_price, "HOLD")
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": entry_price,
            "entry_price_formatted": format_price(entry_price),
            "rsi": 50.0,
            "sma_20": entry_price,
            "targets": calc["targets"],
            "targets_formatted": [format_price(t) for t in calc["targets"]],
            "stop_loss": calc["stop_loss"],
            "stop_loss_formatted": format_price(calc["stop_loss"]),
            "targets_percent": calc["targets_percent"],
            "stop_loss_percent": calc["stop_loss_percent"],
            "reason": "Error in analysis",
            "strategy": "Fallback",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)