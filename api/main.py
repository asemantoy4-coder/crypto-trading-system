"""
Crypto AI Trading System v7.6.7 - Fixed Targets Display
Main entry point for Render deployment
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
# Path Setup
# ==============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, current_dir)  # Add api/ to path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # Add parent to path

print("=" * 50)
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")

# ==============================================================================
# Import Utils Module
# ==============================================================================

print("Importing utils module...")

UTILS_AVAILABLE = False
try:
    from utils import (
        get_market_data_with_fallback,
        analyze_with_multi_timeframe_strategy,
        calculate_simple_rsi,
        calculate_simple_sma,
        calculate_rsi_series,
        detect_divergence,
        calculate_smart_entry,
        get_ichimoku_scalp_signal,
        generate_ichimoku_recommendation,
        calculate_ichimoku_components,
        analyze_ichimoku_scalp_signal,
        calculate_macd_simple
    )
    UTILS_AVAILABLE = True
    print("✅ SUCCESS: Utils imported directly")
except ImportError as e:
    print(f"❌ FAILED: Direct import: {e}")
    try:
        from .utils import (
            get_market_data_with_fallback,
            analyze_with_multi_timeframe_strategy,
            calculate_simple_rsi,
            calculate_simple_sma,
            calculate_rsi_series,
            detect_divergence,
            calculate_smart_entry,
            get_ichimoku_scalp_signal,
            generate_ichimoku_recommendation
        )
        UTILS_AVAILABLE = True
        print("✅ SUCCESS: Utils imported relatively")
    except ImportError as e:
        print(f"❌ FAILED: Relative import: {e}")
        UTILS_AVAILABLE = False

# ==============================================================================
# Mock Functions (if utils not available)
# ==============================================================================

if not UTILS_AVAILABLE:
    print("⚠️ Using mock functions")
    
    def get_market_data_with_fallback(symbol, interval="5m", limit=100):
        """Mock market data."""
        logger.info(f"Mock: Fetching data for {symbol} ({interval})")
        base_prices = {'BTCUSDT': 88271.42, 'ETHUSDT': 3450.12, 'BNBUSDT': 590.54, 'SOLUSDT': 175.98, 'DEFAULT': 100.0}
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
                str(price * 0.998),
                str(price * 1.002),
                str(price * 0.997),
                str(price),
                str(random.uniform(1000, 10000)),
                timestamp + 300000,
                "0", "0", "0", "0", "0"
            ]
            data.append(candle)
        return data
    
    def analyze_with_multi_timeframe_strategy(symbol):
        """Mock analysis."""
        base_prices = {'BTCUSDT': 88271.42, 'ETHUSDT': 3450.12, 'BNBUSDT': 590.54, 'SOLUSDT': 175.98, 'DEFAULT': 100.0}
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
        if signal == "BUY":
            targets = [
                round(entry_price * 1.01, 2),
                round(entry_price * 1.015, 2),
                round(entry_price * 1.02, 2)
            ]
            stop_loss = round(entry_price * 0.99, 2)
        elif signal == "SELL":
            targets = [
                round(entry_price * 0.99, 2),
                round(entry_price * 0.985, 2),
                round(entry_price * 0.98, 2)
            ]
            stop_loss = round(entry_price * 1.01, 2)
        else:
            targets = [
                round(entry_price * 1.005, 2),
                round(entry_price * 1.01, 2),
                round(entry_price * 1.015, 2)
            ]
            stop_loss = round(entry_price * 0.995, 2)
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "entry_price": entry_price,
            "targets": targets,
            "stop_loss": stop_loss,
            "strategy": "Mock Multi-Timeframe"
        }
    
    def calculate_simple_rsi(data, period=14):
        return round(random.uniform(30, 70), 1)
    
    def calculate_simple_sma(data, period=20):
        if not data or len(data) < period:
            return 100.0
        try:
            closes = [float(c[4]) for c in data[-period:]]
            return round(sum(closes) / len(closes), 2)
        except:
            return 100.0
    
    def calculate_rsi_series(data, period=14):
        return [50.0] * (len(data) if data else 0)
    
    def detect_divergence(prices, rsi_values, lookback=5):
        return {"detected": False, "type": "none"}
    
    def calculate_smart_entry(data, signal="BUY"):
        try:
            base_price = float(data[-1][4]) if data else 100.0
            if signal == "BUY":
                return round(base_price * random.uniform(0.995, 0.999), 2)
            elif signal == "SELL":
                return round(base_price * random.uniform(1.001, 1.005), 2)
            else:
                return round(base_price, 2)
        except:
            return 100.0
    
    def get_ichimoku_scalp_signal(data, timeframe="5m"):
        return None
    
    def generate_ichimoku_recommendation(signal):
        return "Hold"

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
# Helper Functions
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
            "targets_percent": [0.0, 1.0, 2.0],
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
# FastAPI App
# ==============================================================================

API_VERSION = "7.6.7-FIXED-TARGETS"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description="Fixed Targets Display Version",
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
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
        "status": "Running on Render",
        "utils_available": UTILS_AVAILABLE,
        "endpoints": {
            "/": "This page",
            "/api/health": "Health check",
            "/api/test-format": "Test number formatting",
            "/api/analyze": "General analysis (POST)",
            "/api/scalp-signal": "Scalp signal (POST)",
            "/api/ichimoku-scalp": "Ichimoku signal (POST)",
            "/api/docs": "Swagger documentation"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "utils": UTILS_AVAILABLE,
        "environment": os.environ.get("RENDER", "local")
    }

@app.get("/api/test-format")
async def test_format():
    """Test endpoint to verify number formatting."""
    test_price = 88271.42
    
    # Test all signals
    buy_result = calculate_targets_and_stop_loss(test_price, "BUY")
    sell_result = calculate_targets_and_stop_loss(test_price, "SELL")
    hold_result = calculate_targets_and_stop_loss(test_price, "HOLD")
    
    return {
        "test": "Number Formatting Test",
        "original_price": test_price,
        "formatted_price": format_price(test_price),
        "buy_signal": {
            "signal": "BUY",
            "entry": test_price,
            "targets": buy_result["targets"],
            "targets_formatted": [format_price(t) for t in buy_result["targets"]],
            "stop_loss": buy_result["stop_loss"],
            "stop_loss_formatted": format_price(buy_result["stop_loss"]),
            "percentages": {
                "targets": buy_result["targets_percent"],
                "stop_loss": buy_result["stop_loss_percent"]
            }
        },
        "sell_signal": {
            "signal": "SELL",
            "entry": test_price,
            "targets": sell_result["targets"],
            "targets_formatted": [format_price(t) for t in sell_result["targets"]],
            "stop_loss": sell_result["stop_loss"],
            "stop_loss_formatted": format_price(sell_result["stop_loss"]),
            "percentages": {
                "targets": sell_result["targets_percent"],
                "stop_loss": sell_result["stop_loss_percent"]
            }
        },
        "api_info": {
            "version": API_VERSION,
            "utils": UTILS_AVAILABLE,
            "endpoint": "/api/test-format"
        }
    }

@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    """General analysis endpoint."""
    try:
        logger.info(f"🔍 Analysis request: {request.symbol} ({request.timeframe})")
        
        # Get analysis from utils
        analysis = analyze_with_multi_timeframe_strategy(request.symbol)
        
        # Ensure we have proper targets (always 3 targets)
        if "targets" not in analysis or len(analysis.get("targets", [])) != 3:
            calc_result = calculate_targets_and_stop_loss(
                analysis.get("entry_price", 100.0),
                analysis.get("signal", "HOLD")
            )
            analysis["targets"] = calc_result["targets"]
            analysis["stop_loss"] = calc_result["stop_loss"]
        
        # Calculate percentages if not present
        if "targets_percent" not in analysis:
            entry = analysis.get("entry_price", 100.0)
            targets = analysis.get("targets", [])
            analysis["targets_percent"] = [
                round(((t - entry) / entry) * 100, 2) for t in targets
            ] if len(targets) == 3 else [1.0, 1.5, 2.0]
        
        if "stop_loss_percent" not in analysis:
            entry = analysis.get("entry_price", 100.0)
            stop_loss = analysis.get("stop_loss", 99.0)
            analysis["stop_loss_percent"] = round(((stop_loss - entry) / entry) * 100, 2)
        
        # Add formatted prices for frontend
        analysis["entry_price_formatted"] = format_price(analysis.get("entry_price", 0))
        analysis["targets_formatted"] = [format_price(t) for t in analysis.get("targets", [])]
        analysis["stop_loss_formatted"] = format_price(analysis.get("stop_loss", 0))
        
        # Add metadata
        analysis["generated_at"] = datetime.now().isoformat()
        analysis["version"] = API_VERSION
        analysis["timeframe"] = request.timeframe
        
        logger.info(f"✅ Analysis complete: {analysis['signal']} for {analysis['symbol']}")
        logger.info(f"   Entry: {analysis['entry_price_formatted']}")
        logger.info(f"   Targets: {analysis['targets_formatted']}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}", exc_info=True)
        
        # Fallback response
        base_prices = {
            'BTCUSDT': 88271.42,
            'ETHUSDT': 3450.12,
            'BNBUSDT': 590.54,
            'SOLUSDT': 175.98,
            'DEFAULT': 100.0
        }
        symbol = request.symbol.upper()
        entry_price = base_prices.get(symbol, base_prices['DEFAULT'])
        
        calc_result = calculate_targets_and_stop_loss(entry_price, "HOLD")
        
        return {
            "symbol": symbol,
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
            "reason": f"Analysis error: {str(e)[:100]}",
            "strategy": "Fallback",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION,
            "timeframe": request.timeframe
        }

@app.post("/api/scalp-signal")
async def scalp_signal(request: ScalpRequest):
    """Scalp signal endpoint with RSI-based signals."""
    allowed_timeframes = ["1m", "5m", "15m"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Allowed: {allowed_timeframes}")
    
    try:
        logger.info(f"⚡ Scalp signal request: {request.symbol} ({request.timeframe})")
        
        # Get market data
        market_data = get_market_data_with_fallback(
            request.symbol,
            request.timeframe,
            50
        )
        
        if not market_data:
            raise HTTPException(status_code=404, detail="No market data available")
        
        # Calculate indicators
        rsi = calculate_simple_rsi(market_data, 14)
        sma_20 = calculate_simple_sma(market_data, 20)
        
        # Get current price
        try:
            current_price = float(market_data[-1][4])
        except:
            current_price = 100.0
        
        # Determine signal based on RSI
        signal = "HOLD"
        confidence = 0.5
        reason = "Neutral market conditions"
        
        if rsi < 30:
            signal = "BUY"
            confidence = 0.85
            reason = f"Strong oversold (RSI: {rsi:.1f})"
        elif rsi > 70:
            signal = "SELL"
            confidence = 0.85
            reason = f"Strong overbought (RSI: {rsi:.1f})"
        elif rsi < 35:
            signal = "BUY"
            confidence = 0.75
            reason = f"Oversold (RSI: {rsi:.1f})"
        elif rsi > 65:
            signal = "SELL"
            confidence = 0.75
            reason = f"Overbought (RSI: {rsi:.1f})"
        elif rsi < 40:
            signal = "BUY"
            confidence = 0.65
            reason = f"Near oversold (RSI: {rsi:.1f})"
        elif rsi > 60:
            signal = "SELL"
            confidence = 0.65
            reason = f"Near overbought (RSI: {rsi:.1f})"
        
        # Get smart entry price
        entry_price = calculate_smart_entry(market_data, signal)
        if entry_price <= 0:
            entry_price = current_price
        
        # Ensure reasonable price
        if entry_price > 1000000 or entry_price < 0.0001:
            base_prices = {
                'BTCUSDT': 88271.42,
                'ETHUSDT': 3450.12,
                'BNBUSDT': 590.54,
                'SOLUSDT': 175.98,
                'DEFAULT': 100.0
            }
            entry_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        
        # Calculate targets and stop loss
        calc_result = calculate_targets_and_stop_loss(entry_price, signal)
        
        # Determine risk level
        risk_level = "MEDIUM"
        if (rsi < 25 or rsi > 75) and confidence > 0.8:
            risk_level = "HIGH"
        elif confidence < 0.6:
            risk_level = "LOW"
        
        # Check for divergence
        closes = [float(c[4]) for c in market_data[-30:]] if len(market_data) >= 30 else []
        rsi_series = calculate_rsi_series(market_data, 14)[-30:] if market_data else []
        div_info = detect_divergence(closes, rsi_series, lookback=5)
        
        # Build response
        response = {
            "symbol": request.symbol.upper(),
            "timeframe": request.timeframe,
            "signal": signal,
            "confidence": round(float(confidence), 2),
            "entry_price": round(float(entry_price), 2),
            "entry_price_formatted": format_price(entry_price),
            "rsi": round(float(rsi), 1),
            "sma_20": round(float(sma_20), 2) if sma_20 else 0,
            "divergence": div_info.get("detected", False),
            "divergence_type": div_info.get("type", "none"),
            "targets": [float(t) for t in calc_result["targets"]],
            "targets_formatted": [format_price(t) for t in calc_result["targets"]],
            "stop_loss": float(calc_result["stop_loss"]),
            "stop_loss_formatted": format_price(calc_result["stop_loss"]),
            "targets_percent": calc_result["targets_percent"],
            "stop_loss_percent": calc_result["stop_loss_percent"],
            "risk_level": risk_level,
            "reason": reason,
            "strategy": "RSI Scalp with Smart Entry",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION
        }
        
        logger.info(f"✅ Scalp signal generated:")
        logger.info(f"   Signal: {response['signal']} (Confidence: {response['confidence']})")
        logger.info(f"   Entry: {response['entry_price_formatted']}")
        logger.info(f"   Targets: {response['targets_formatted']} ({response['targets_percent']}%)")
        logger.info(f"   Stop Loss: {response['stop_loss_formatted']} ({response['stop_loss_percent']}%)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Scalp signal error: {e}", exc_info=True)
        
        # Fallback response
        base_prices = {
            'BTCUSDT': 88271.42,
            'ETHUSDT': 3450.12,
            'BNBUSDT': 590.54,
            'SOLUSDT': 175.98,
            'DEFAULT': 100.0
        }
        symbol = request.symbol.upper()
        entry_price = base_prices.get(symbol, base_prices['DEFAULT'])
        
        calc_result = calculate_targets_and_stop_loss(entry_price, "HOLD")
        
        return {
            "symbol": symbol,
            "timeframe": request.timeframe,
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": entry_price,
            "entry_price_formatted": format_price(entry_price),
            "rsi": 50.0,
            "sma_20": entry_price,
            "divergence": False,
            "divergence_type": "none",
            "targets": calc_result["targets"],
            "targets_formatted": [format_price(t) for t in calc_result["targets"]],
            "stop_loss": calc_result["stop_loss"],
            "stop_loss_formatted": format_price(calc_result["stop_loss"]),
            "targets_percent": calc_result["targets_percent"],
            "stop_loss_percent": calc_result["stop_loss_percent"],
            "risk_level": "LOW",
            "reason": f"Error: {str(e)[:100]}",
            "strategy": "Fallback Mode",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION
        }

@app.post("/api/ichimoku-scalp")
async def ichimoku_scalp(request: IchimokuRequest):
    """Ichimoku Kinko Hyo scalp signal."""
    allowed_timeframes = ["5m", "15m", "1h", "4h"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Allowed: {allowed_timeframes}")
    
    try:
        logger.info(f"☁️ Ichimoku request: {request.symbol} ({request.timeframe})")
        
        # Get market data
        market_data = get_market_data_with_fallback(
            request.symbol,
            request.timeframe,
            100
        )
        
        if not market_data or len(market_data) < 60:
            raise HTTPException(status_code=404, detail="Not enough data for Ichimoku analysis")
        
        # Get Ichimoku signal
        ichimoku_signal = get_ichimoku_scalp_signal(market_data, request.timeframe)
        
        if not ichimoku_signal:
            ichimoku_signal = {
                "signal": "HOLD",
                "confidence": 0.5,
                "reason": "Ichimoku analysis unavailable",
                "trend_power": 50
            }
        
        # Get current price
        try:
            current_price = float(market_data[-1][4])
        except:
            current_price = 100.0
        
        # Calculate targets
        calc_result = calculate_targets_and_stop_loss(
            current_price,
            ichimoku_signal.get("signal", "HOLD")
        )
        
        # Get recommendation
        recommendation = generate_ichimoku_recommendation(ichimoku_signal)
        
        # Build response
        response = {
            "symbol": request.symbol.upper(),
            "timeframe": request.timeframe,
            "signal": ichimoku_signal.get("signal", "HOLD"),
            "confidence": round(float(ichimoku_signal.get("confidence", 0.5)), 2),
            "entry_price": round(float(current_price), 2),
            "entry_price_formatted": format_price(current_price),
            "targets": [float(t) for t in calc_result["targets"]],
            "targets_formatted": [format_price(t) for t in calc_result["targets"]],
            "stop_loss": float(calc_result["stop_loss"]),
            "stop_loss_formatted": format_price(calc_result["stop_loss"]),
            "targets_percent": calc_result["targets_percent"],
            "stop_loss_percent": calc_result["stop_loss_percent"],
            "reason": ichimoku_signal.get("reason", "No specific reason"),
            "recommendation": recommendation,
            "trend_power": ichimoku_signal.get("trend_power", 50),
            "strategy": f"Ichimoku Kinko Hyo ({request.timeframe})",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION
        }
        
        logger.info(f"✅ Ichimoku signal: {response['signal']} (Confidence: {response['confidence']})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ichimoku error: {e}")
        
        # Fallback
        base_prices = {
            'BTCUSDT': 88271.42,
            'ETHUSDT': 3450.12,
            'BNBUSDT': 590.54,
            'SOLUSDT': 175.98,
            'DEFAULT': 100.0
        }
        symbol = request.symbol.upper()
        entry_price = base_prices.get(symbol, base_prices['DEFAULT'])
        
        calc_result = calculate_targets_and_stop_loss(entry_price, "HOLD")
        
        return {
            "symbol": symbol,
            "timeframe": request.timeframe,
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
            "reason": f"Ichimoku analysis error: {str(e)[:100]}",
            "recommendation": "Hold - Analysis failed",
            "trend_power": 50,
            "strategy": "Fallback",
            "generated_at": datetime.now().isoformat(),
            "version": API_VERSION
        }

@app.get("/api/debug/{symbol}")
async def debug_symbol(symbol: str):
    """Debug endpoint to see raw data."""
    try:
        market_data = get_market_data_with_fallback(symbol, "5m", 10)
        
        debug_info = {
            "symbol": symbol.upper(),
            "data_points": len(market_data) if market_data else 0,
            "utils_available": UTILS_AVAILABLE,
            "api_version": API_VERSION,
            "sample_data": None
        }
        
        if market_data and len(market_data) > 0:
            debug_info["sample_data"] = {
                "first_candle": market_data[0] if len(market_data) > 0 else None,
                "last_candle": market_data[-1] if len(market_data) > 0 else None,
                "last_close": float(market_data[-1][4]) if market_data else 0,
                "rsi": calculate_simple_rsi(market_data, 14),
                "sma_20": calculate_simple_sma(market_data, 20)
            }
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}

# ==============================================================================
# Startup Event
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 Starting Crypto AI Trading System v{API_VERSION}")
    logger.info(f"📦 Utils Available: {UTILS_AVAILABLE}")
    logger.info(f"🌐 Service ready at: http://0.0.0.0:{os.environ.get('PORT', 8000)}")

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )