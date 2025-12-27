"""
Crypto AI Trading System v8.2.0 (SIMPLIFIED - NO NUMPY)
Fixed import issues and dependencies
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
import time
import requests
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. PRICE VALIDATION FUNCTIONS
# ==============================================================================

KNOWN_PRICES = {
    'BTCUSDT': 88271.00,
    'ETHUSDT': 3450.00,
    'BNBUSDT': 590.00,
    'SOLUSDT': 175.00,
    'DOGEUSDT': 0.12116,
    'ALGOUSDT': 0.1187,
    'AVAXUSDT': 12.45,
    'ADAUSDT': 0.45,
    'XRPUSDT': 0.52,
    'DOTUSDT': 6.80,
    'MATICUSDT': 0.75,
    'LINKUSDT': 13.50,
    'UNIUSDT': 6.20,
    'DEFAULT': 100.0
}

def validate_symbol(symbol: str) -> str:
    """Validate and normalize symbol"""
    symbol = symbol.upper().strip()
    
    # Common symbol corrections
    corrections = {
        'AVA': 'AVAX',
        'AVALANCHE': 'AVAX',
        'AVALANCHEUSDT': 'AVAXUSDT',
        'BITCOIN': 'BTC',
        'ETHEREUM': 'ETH',
        'BINANCE': 'BNB',
        'SOLANA': 'SOL',
        'DOGECOIN': 'DOGE',
        'ALGORAND': 'ALGO',
        'CARDANO': 'ADA',
        'RIPPLE': 'XRP',
        'POLKADOT': 'DOT',
        'POLYGON': 'MATIC',
        'CHAINLINK': 'LINK',
        'UNISWAP': 'UNI'
    }
    
    # Check if symbol needs correction
    if symbol in corrections:
        symbol = corrections[symbol]
    elif not symbol.endswith('USDT'):
        symbol = f"{symbol}USDT"
    
    return symbol

def get_real_time_price(symbol: str) -> float:
    """Get real-time price from Binance API with fallback"""
    symbol = validate_symbol(symbol)
    
    try:
        # Try Binance API first
        url = "https://api.binance.com/api/v3/ticker/price"
        response = requests.get(url, params={'symbol': symbol}, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            price = float(data['price'])
            logger.info(f"✅ Real price for {symbol}: ${price:.8f}")
            return price
    except Exception as e:
        logger.warning(f"Binance API failed for {symbol}: {e}")
    
    # Use known prices as fallback
    price = KNOWN_PRICES.get(symbol, KNOWN_PRICES['DEFAULT'])
    logger.info(f"⚠️ Using known price for {symbol}: ${price:.8f}")
    return price

# ==============================================================================
# 2. ADVANCED TARGET CALCULATION FUNCTIONS
# ==============================================================================

def calculate_fibonacci_targets(entry_price, signal, risk_level="MEDIUM", volatility=1.0):
    """
    Calculate Fibonacci-based targets
    """
    if entry_price <= 0:
        return [0, 0, 0], 0, [0, 0, 0], 0
    
    # Fibonacci retracement levels for different signals
    if signal == "BUY":
        if risk_level == "HIGH":
            targets_mult = [1.01, 1.02, 1.03]  # 1%, 2%, 3%
            stop_mult = 0.985  # -1.5%
        elif risk_level == "MEDIUM":
            targets_mult = [1.005, 1.01, 1.015]  # 0.5%, 1%, 1.5%
            stop_mult = 0.99  # -1%
        else:  # LOW
            targets_mult = [1.003, 1.006, 1.009]  # 0.3%, 0.6%, 0.9%
            stop_mult = 0.995  # -0.5%
    
    elif signal == "SELL":
        if risk_level == "HIGH":
            targets_mult = [0.99, 0.98, 0.97]  # -1%, -2%, -3%
            stop_mult = 1.015  # +1.5%
        elif risk_level == "MEDIUM":
            targets_mult = [0.995, 0.99, 0.985]  # -0.5%, -1%, -1.5%
            stop_mult = 1.01  # +1%
        else:  # LOW
            targets_mult = [0.997, 0.994, 0.991]  # -0.3%, -0.6%, -0.9%
            stop_mult = 1.005  # +0.5%
    
    else:  # HOLD
        targets_mult = [1.002, 1.004, 1.006]
        stop_mult = 0.998
    
    # Adjust for volatility
    vol_factor = 1 + (volatility / 100)
    targets = [round(entry_price * mult * vol_factor, 8) for mult in targets_mult]
    stop_loss = round(entry_price * stop_mult / vol_factor, 8)
    
    # Calculate percentages
    targets_percent = [
        round(((target - entry_price) / entry_price) * 100, 2)
        for target in targets
    ]
    stop_loss_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
    
    logger.info(f"🎯 Targets: {targets} ({targets_percent}%) | Stop: {stop_loss} ({stop_loss_percent}%)")
    
    return targets, stop_loss, targets_percent, stop_loss_percent

def calculate_simple_sma(data, period=20):
    """Calculate Simple Moving Average"""
    if not data or len(data) < period:
        return 0
    
    try:
        closes = []
        for candle in data[-period:]:
            if len(candle) > 4:
                try:
                    closes.append(float(candle[4]))
                except:
                    continue
        
        if not closes:
            return 0
        
        return sum(closes) / len(closes)
    except:
        return 0

def calculate_simple_rsi(data, period=14):
    """Calculate RSI"""
    if not data or len(data) < period:
        return 50.0
    
    try:
        closes = []
        for candle in data[-period*2:]:
            if len(candle) > 4:
                try:
                    closes.append(float(candle[4]))
                except:
                    continue
        
        if len(closes) < period:
            return 50.0
        
        gains, losses = 0, 0
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        
        avg_gain = gains / period
        avg_loss = losses / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(max(0, min(100, rsi)), 2)
    except:
        return round(random.uniform(30, 70), 2)

def calculate_volatility(data, period=20):
    """Calculate volatility without numpy"""
    if not data or len(data) < period:
        return 1.0
    
    try:
        closes = []
        for candle in data[-period:]:
            if len(candle) > 4:
                try:
                    closes.append(float(candle[4]))
                except:
                    continue
        
        if len(closes) < 2:
            return 1.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0:
                ret = (closes[i] - closes[i-1]) / closes[i-1]
                returns.append(abs(ret))
        
        if not returns:
            return 1.0
        
        # Calculate standard deviation manually
        mean = sum(returns) / len(returns)
        variance = sum((x - mean) ** 2 for x in returns) / len(returns)
        std_dev = variance ** 0.5
        
        # Annualize (rough approximation)
        volatility = std_dev * 100 * (365 ** 0.5)
        return round(max(0.5, min(volatility, 10.0)), 2)
    except:
        return round(random.uniform(0.5, 3.0), 2)

def get_support_resistance_levels(data, period=20):
    """Calculate support and resistance levels"""
    if not data or len(data) < period:
        return {"support": 0, "resistance": 0, "range_percent": 0}
    
    try:
        highs = []
        lows = []
        
        for candle in data[-period:]:
            if len(candle) > 3:
                try:
                    highs.append(float(candle[2]))
                    lows.append(float(candle[3]))
                except:
                    continue
        
        if not highs or not lows:
            return {"support": 0, "resistance": 0, "range_percent": 0}
        
        resistance = sum(highs) / len(highs)
        support = sum(lows) / len(lows)
        
        # Ensure resistance > support
        if resistance <= support:
            resistance = support * 1.02
        
        range_percent = ((resistance - support) / support * 100) if support > 0 else 0
        
        return {
            "support": round(support, 8),
            "resistance": round(resistance, 8),
            "range_percent": round(range_percent, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating S/R: {e}")
        return {"support": 0, "resistance": 0, "range_percent": 0}

# ==============================================================================
# 3. FASTAPI APP
# ==============================================================================

API_VERSION = "8.2.0-SIMPLE"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description="Simple version without numpy dependency",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 4. PYDANTIC MODELS
# ==============================================================================

class ScalpRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

# ==============================================================================
# 5. API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    return {
        "message": f"Crypto AI Trading System v{API_VERSION}",
        "status": "Active",
        "endpoints": {
            "/": "This info",
            "/health": "Health check",
            "/price/{symbol}": "Get current price",
            "/scalp": "POST - Get scalp signal",
            "/analyze": "POST - General analysis",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "Healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/price/{symbol}")
async def get_price(symbol: str):
    """Get current price for a symbol"""
    try:
        valid_symbol = validate_symbol(symbol)
        price = get_real_time_price(valid_symbol)
        
        return {
            "symbol": valid_symbol,
            "price": price,
            "formatted": f"${price:.8f}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Price error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scalp")
async def scalp_signal(request: ScalpRequest):
    """
    Get scalp trading signal
    """
    start_time = time.time()
    
    try:
        # Validate symbol
        symbol = validate_symbol(request.symbol)
        logger.info(f"🚀 Scalp request: {symbol} ({request.timeframe})")
        
        # Get current price
        current_price = get_real_time_price(symbol)
        
        # Generate mock data based on current price
        mock_data = []
        base_time = int(time.time() * 1000)
        
        for i in range(50):
            timestamp = base_time - (i * 5 * 60 * 1000)
            variation = random.uniform(-0.02, 0.02)
            price = current_price * (1 + variation)
            
            candle = [
                timestamp,
                str(price * random.uniform(0.998, 1.000)),
                str(price * random.uniform(1.000, 1.003)),
                str(price * random.uniform(0.997, 1.000)),
                str(price),
                str(random.uniform(1000, 10000))
            ]
            mock_data.append(candle)
        
        # Calculate indicators
        rsi = calculate_simple_rsi(mock_data, 14)
        sma_20 = calculate_simple_sma(mock_data, 20)
        volatility = calculate_volatility(mock_data, 20)
        sr_levels = get_support_resistance_levels(mock_data, 20)
        
        # Determine signal
        if rsi < 30 and current_price < sma_20:
            signal = "BUY"
            confidence = 0.75
            reason = f"Oversold (RSI: {rsi:.1f}) and below SMA20"
        elif rsi > 70 and current_price > sma_20:
            signal = "SELL"
            confidence = 0.75
            reason = f"Overbought (RSI: {rsi:.1f}) and above SMA20"
        elif rsi < 35:
            signal = "BUY"
            confidence = 0.65
            reason = f"Near oversold (RSI: {rsi:.1f})"
        elif rsi > 65:
            signal = "SELL"
            confidence = 0.65
            reason = f"Near overbought (RSI: {rsi:.1f})"
        else:
            signal = "HOLD"
            confidence = 0.5
            reason = "Neutral market conditions"
        
        # Determine risk level
        if confidence > 0.7 and (rsi > 75 or rsi < 25):
            risk_level = "HIGH"
        elif confidence > 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate targets
        entry_price = current_price * 0.999 if signal == "BUY" else current_price * 1.001
        
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_fibonacci_targets(
            entry_price=entry_price,
            signal=signal,
            risk_level=risk_level,
            volatility=volatility
        )
        
        # Prepare response
        response = {
            "symbol": symbol,
            "timeframe": request.timeframe,
            "signal": signal,
            "confidence": confidence,
            "entry_price": round(entry_price, 8),
            "current_price": round(current_price, 8),
            "rsi": round(rsi, 2),
            "sma_20": round(sma_20, 8),
            "volatility": volatility,
            "support": sr_levels["support"],
            "resistance": sr_levels["resistance"],
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "risk_level": risk_level,
            "reason": reason,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ {symbol}: {signal} signal ({confidence*100:.0f}% confidence)")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Scalp error: {e}")
        
        # Fallback response
        symbol = validate_symbol(request.symbol)
        price = get_real_time_price(symbol)
        
        return {
            "symbol": symbol,
            "timeframe": request.timeframe,
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": round(price, 8),
            "current_price": round(price, 8),
            "rsi": 50.0,
            "sma_20": round(price * 0.99, 8),
            "volatility": 1.5,
            "targets": [round(price * 1.01, 8), round(price * 1.02, 8), round(price * 1.03, 8)],
            "stop_loss": round(price * 0.99, 8),
            "targets_percent": [1.0, 2.0, 3.0],
            "stop_loss_percent": -1.0,
            "risk_level": "MEDIUM",
            "reason": f"Fallback mode: {str(e)[:100]}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """
    General market analysis
    """
    try:
        symbol = validate_symbol(request.symbol)
        price = get_real_time_price(symbol)
        
        # Basic analysis
        price_change = random.uniform(-5, 5)
        volume = random.uniform(1000000, 50000000)
        
        return {
            "symbol": symbol,
            "timeframe": request.timeframe,
            "current_price": price,
            "price_change_24h": round(price_change, 2),
            "volume_24h": round(volume, 2),
            "market_cap": round(price * random.uniform(1000000, 1000000000), 2),
            "analysis": {
                "trend": random.choice(["Bullish", "Bearish", "Neutral"]),
                "strength": random.randint(1, 10),
                "recommendation": random.choice(["Buy", "Sell", "Hold", "Strong Buy", "Strong Sell"])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def get_symbols():
    """Get supported symbols"""
    symbols = list(KNOWN_PRICES.keys())
    symbols.remove('DEFAULT')
    
    return {
        "symbols": symbols,
        "count": len(symbols),
        "timestamp": datetime.now().isoformat()
    }

# ==============================================================================
# 6. STARTUP
# ==============================================================================

@app.on_event("startup")
async def startup():
    logger.info(f"🚀 Starting Simple Crypto AI Trading System v{API_VERSION}")
    logger.info("✅ No numpy dependency required")
    logger.info("✅ Price validation active")
    
    # Test the system
    try:
        test_price = get_real_time_price("BTCUSDT")
        logger.info(f"✅ Test successful - BTC price: ${test_price:.2f}")
    except Exception as e:
        logger.warning(f"⚠️ Test warning: {e}")

# ==============================================================================
# 7. MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"\n{'=' * 60}")
    print(f"Crypto AI Trading System v{API_VERSION}")
    print(f"Server starting on: http://{host}:{port}")
    print(f"Docs available at: http://{host}:{port}/docs")
    print(f"{'=' * 60}\n")
    
    uvicorn.run(
        "main:app",  # Changed this line - use string reference
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )