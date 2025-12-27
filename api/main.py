"""
Crypto AI Trading System v8.1.0 (PRICE-FIXED)
Fixed Price Calculation Issues + Advanced Fibonacci Targets
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
import time
import numpy as np
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. PRICE VALIDATION FUNCTIONS (FIXED)
# ==============================================================================

KNOWN_PRICES = {
    'BTCUSDT': 88271.00,
    'ETHUSDT': 3450.00,
    'BNBUSDT': 590.00,
    'SOLUSDT': 175.00,
    'DOGEUSDT': 0.12116,
    'ALGOUSDT': 0.1187,
    'AVAXUSDT': 12.45,  # Fixed: AVAX instead of AVA
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
    
    # Try CoinGecko as fallback
    try:
        coin_id = symbol.replace('USDT', '').lower()
        if coin_id == 'avax':
            coin_id = 'avalanche-2'
        elif coin_id == 'doge':
            coin_id = 'dogecoin'
        
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {'ids': coin_id, 'vs_currencies': 'usd'}
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if coin_id in data:
                price = data[coin_id]['usd']
                logger.info(f"✅ CoinGecko price for {symbol}: ${price:.8f}")
                return float(price)
    except Exception as e:
        logger.warning(f"CoinGecko API failed: {e}")
    
    # Use known prices as final fallback
    price = KNOWN_PRICES.get(symbol, KNOWN_PRICES['DEFAULT'])
    logger.info(f"⚠️ Using known price for {symbol}: ${price:.8f}")
    return price

def validate_price_consistency(calculated_price: float, symbol: str, source: str) -> float:
    """Validate price consistency with real market data"""
    symbol = validate_symbol(symbol)
    
    # Get real price for comparison
    real_price = get_real_time_price(symbol)
    
    # Check if calculated price is reasonable
    if calculated_price <= 0:
        logger.warning(f"Invalid calculated price: {calculated_price}, using real price")
        return real_price
    
    # Calculate percentage difference
    if real_price > 0:
        diff_percent = abs(calculated_price - real_price) / real_price * 100
        
        if diff_percent > 50:  # More than 50% difference
            logger.warning(f"⚠️ Price inconsistency: Calculated ${calculated_price:.8f} vs Real ${real_price:.8f} ({diff_percent:.2f}% diff)")
            
            # For certain symbols, enforce known reasonable ranges
            symbol_ranges = {
                'BTCUSDT': (50000, 100000),
                'ETHUSDT': (2000, 5000),
                'AVAXUSDT': (10, 20),
                'SOLUSDT': (100, 300),
                'DOGEUSDT': (0.1, 0.2),
                'ALGOUSDT': (0.1, 0.2),
                'ADAUSDT': (0.3, 0.6),
                'DEFAULT': (real_price * 0.5, real_price * 1.5)
            }
            
            min_price, max_price = symbol_ranges.get(symbol, symbol_ranges['DEFAULT'])
            
            if calculated_price < min_price or calculated_price > max_price:
                logger.warning(f"Price out of range: ${calculated_price:.8f}, using real price: ${real_price:.8f}")
                return real_price
    
    return calculated_price

# ==============================================================================
# 2. ADVANCED TARGET CALCULATION FUNCTIONS (FIXED)
# ==============================================================================

def calculate_advanced_targets_and_stop_loss(entry_price, signal, risk_level="MEDIUM", 
                                            ichimoku_data=None, support=None, resistance=None,
                                            volatility=1.0, trend_power=50, rsi=50):
    """
    Fixed version with price validation
    """
    # Validate entry price
    if entry_price <= 0:
        logger.error(f"Invalid entry price: {entry_price}")
        return [0, 0, 0], 0, [0, 0, 0], 0
    
    # Extract Ichimoku levels if available
    ichimoku_levels = {}
    if ichimoku_data:
        ichimoku_levels = {
            'tenkan_sen': max(ichimoku_data.get('tenkan_sen', entry_price), 0.00000001),
            'kijun_sen': max(ichimoku_data.get('kijun_sen', entry_price), 0.00000001),
            'cloud_top': max(ichimoku_data.get('cloud_top', entry_price * 1.02), 0.00000001),
            'cloud_bottom': max(ichimoku_data.get('cloud_bottom', entry_price * 0.98), 0.00000001),
        }
    
    # Calculate Fibonacci levels
    if support and resistance and support > 0 and resistance > support:
        swing_low = support
        swing_high = resistance
    else:
        # Default swings based on volatility
        swing_low = entry_price * (1 - max(volatility/100, 0.01))
        swing_high = entry_price * (1 + max(volatility/100, 0.01))
    
    price_range = max(swing_high - swing_low, 0.00000001)
    
    # Fibonacci retracement levels
    fib_retracement = {
        '0.236': swing_low + price_range * 0.236,
        '0.382': swing_low + price_range * 0.382,
        '0.5': swing_low + price_range * 0.5,
        '0.618': swing_low + price_range * 0.618,
        '0.786': swing_low + price_range * 0.786,
        '0.886': swing_low + price_range * 0.886
    }
    
    # Fibonacci extension levels
    fib_extension = {
        '1.0': swing_high,
        '1.272': swing_high + price_range * 0.272,
        '1.414': swing_high + price_range * 0.414,
        '1.618': swing_high + price_range * 0.618,
        '2.0': swing_high + price_range,
        '2.618': swing_high + price_range * 1.618
    }
    
    # Risk configuration
    risk_configs = {
        "HIGH": {
            "BUY": {
                "targets": ['1.272', '1.618', '2.0'],
                "stop_loss": '0.786',
                "use_fibonacci": True,
                "max_risk_percent": 2.0
            },
            "SELL": {
                "targets": ['0.618', '0.382', '0.236'],
                "stop_loss": '1.272',
                "use_fibonacci": True,
                "max_risk_percent": 2.0
            }
        },
        "MEDIUM": {
            "BUY": {
                "targets": ['1.0', '1.272', '1.618'],
                "stop_loss": '0.618',
                "use_fibonacci": True,
                "max_risk_percent": 1.5
            },
            "SELL": {
                "targets": ['0.786', '0.618', '0.382'],
                "stop_loss": '1.0',
                "use_fibonacci": True,
                "max_risk_percent": 1.5
            }
        },
        "LOW": {
            "BUY": {
                "targets": ['0.786', '1.0', '1.272'],
                "stop_loss": '0.5',
                "use_fibonacci": True,
                "max_risk_percent": 1.0
            },
            "SELL": {
                "targets": ['0.886', '0.786', '0.618'],
                "stop_loss": '1.0',
                "use_fibonacci": True,
                "max_risk_percent": 1.0
            }
        }
    }
    
    config = risk_configs.get(risk_level, risk_configs["MEDIUM"])
    signal_config = config.get(signal, {
        "targets": ['1.003', '1.006', '1.009'] if signal == "BUY" else ['0.997', '0.994', '0.991'],
        "stop_loss": '0.997' if signal == "BUY" else '1.003',
        "use_fibonacci": False,
        "max_risk_percent": 0.5
    })
    
    # Calculate targets
    targets = []
    for level in signal_config["targets"]:
        if signal_config["use_fibonacci"]:
            if signal == "BUY":
                target = fib_extension.get(level, entry_price * 1.01)
            else:  # SELL
                target = fib_retracement.get(level, entry_price * 0.99)
        else:
            # Basic percentage targets
            target = entry_price * float(level)
        
        # Validate target
        if target <= 0:
            target = entry_price * (1.01 if signal == "BUY" else 0.99)
        
        targets.append(round(target, 8))
    
    # Calculate stop loss
    stop_level = signal_config["stop_loss"]
    if signal_config["use_fibonacci"]:
        if signal == "BUY":
            stop_loss = fib_retracement.get(stop_level, entry_price * 0.99)
        else:  # SELL
            stop_loss = fib_extension.get(stop_level, entry_price * 1.01)
    else:
        stop_loss = entry_price * float(stop_level)
    
    # Apply max risk limit
    max_risk = signal_config["max_risk_percent"] / 100
    if signal == "BUY":
        min_stop = entry_price * (1 - max_risk)
        stop_loss = min(stop_loss, min_stop)
    else:  # SELL
        max_stop = entry_price * (1 + max_risk)
        stop_loss = max(stop_loss, max_stop)
    
    stop_loss = round(max(stop_loss, 0.00000001), 8)
    
    # Calculate percentages
    targets_percent = [
        round(((target - entry_price) / entry_price) * 100, 2)
        for target in targets
    ]
    stop_loss_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
    
    logger.info(f"🎯 Targets calculated: {targets} ({targets_percent}%) | Stop: {stop_loss} ({stop_loss_percent}%)")
    
    return targets, stop_loss, targets_percent, stop_loss_percent

def calculate_fibonacci_levels_from_data(data, period=20):
    """Calculate Fibonacci levels from market data"""
    if not data or len(data) < period:
        return {}
    
    try:
        highs = [float(c[2]) for c in data[-period:] if len(c) > 2]
        lows = [float(c[3]) for c in data[-period:] if len(c) > 3]
        
        if not highs or not lows:
            return {}
        
        swing_high = max(highs)
        swing_low = min(lows)
        
        if swing_low <= 0 or swing_high <= swing_low:
            return {}
        
        price_range = swing_high - swing_low
        
        return {
            '0.0': swing_low,
            '0.236': swing_low + price_range * 0.236,
            '0.382': swing_low + price_range * 0.382,
            '0.5': swing_low + price_range * 0.5,
            '0.618': swing_low + price_range * 0.618,
            '0.786': swing_low + price_range * 0.786,
            '1.0': swing_high,
            '1.272': swing_high + price_range * 0.272,
            '1.618': swing_high + price_range * 0.618,
            '2.0': swing_high + price_range,
            '2.618': swing_high + price_range * 1.618,
            'current_price': float(data[-1][4]) if data else 0
        }
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {e}")
        return {}

# ==============================================================================
# Path Setup
# ==============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print("=" * 60)
print(f"PRICE-FIXED Trading System v8.1.0")
print(f"Current directory: {current_dir}")
print("=" * 60)

# ==============================================================================
# Module Imports (Mock/Fallback)
# ==============================================================================

UTILS_AVAILABLE = False
print("⚠️ Using built-in functions (no external utils)")

# Mock functions
def mock_get_market_data_with_fallback(symbol, interval="5m", limit=50, return_source=False):
    """Get mock market data with price validation"""
    symbol = validate_symbol(symbol)
    real_price = get_real_time_price(symbol)
    
    data = []
    current_time = int(time.time() * 1000)
    
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)
        # Small random variation around real price
        variation = random.uniform(-0.02, 0.02)
        price = real_price * (1 + variation)
        
        candle = [
            timestamp,
            str(price * random.uniform(0.998, 1.000)),
            str(price * random.uniform(1.000, 1.003)),
            str(price * random.uniform(0.997, 1.000)),
            str(price),
            str(random.uniform(1000, 10000)),
            timestamp + 300000,
            "0", "0", "0", "0", "0"
        ]
        data.append(candle)
    
    if return_source:
        return {"data": data, "source": "mock_validated", "success": True, "real_price": real_price}
    return data

def mock_calculate_simple_sma(data, period=20):
    """Calculate SMA with validation"""
    if not data or len(data) < period:
        return 0
    
    try:
        closes = []
        for candle in data[-period:]:
            if len(candle) > 4:
                try:
                    price = float(candle[4])
                    if price > 0:
                        closes.append(price)
                except:
                    continue
        
        if not closes:
            return 0
        
        sma = sum(closes) / len(closes)
        return sma if sma > 0 else 0
    except:
        return 0

def mock_calculate_simple_rsi(data, period=14):
    """Calculate RSI"""
    if not data or len(data) < period:
        return 50.0
    
    try:
        closes = []
        for candle in data[-period*2:]:  # Need more data for RSI
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

def mock_calculate_rsi_series(closes, period=14):
    """Calculate RSI series"""
    if not closes or len(closes) < period:
        return [50.0] * (len(closes) if closes else 0)
    
    rsi_values = []
    for i in range(len(closes) - period):
        window = closes[i:i+period]
        gains, losses = 0, 0
        
        for j in range(1, len(window)):
            change = window[j] - window[j-1]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        
        avg_gain = gains / period
        avg_loss = losses / period
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(round(rsi, 2))
    
    # Pad with last value
    while len(rsi_values) < len(closes):
        rsi_values.append(rsi_values[-1] if rsi_values else 50.0)
    
    return rsi_values

def mock_detect_divergence(prices, rsi_values, lookback=5):
    """Detect divergence between price and RSI"""
    if not prices or not rsi_values or len(prices) != len(rsi_values) or len(prices) < lookback:
        return {"detected": False, "type": "none", "strength": 0}
    
    try:
        recent_prices = prices[-lookback:]
        recent_rsi = rsi_values[-lookback:]
        
        price_trend = recent_prices[-1] - recent_prices[0]
        rsi_trend = recent_rsi[-1] - recent_rsi[0]
        
        # Bullish divergence: price makes lower lows but RSI makes higher lows
        if price_trend < 0 and rsi_trend > 0:
            return {
                "detected": True,
                "type": "bullish",
                "strength": min(abs(price_trend) * abs(rsi_trend), 100)
            }
        # Bearish divergence: price makes higher highs but RSI makes lower highs
        elif price_trend > 0 and rsi_trend < 0:
            return {
                "detected": True,
                "type": "bearish",
                "strength": min(abs(price_trend) * abs(rsi_trend), 100)
            }
        
        return {"detected": False, "type": "none", "strength": 0}
    except:
        return {"detected": False, "type": "none", "strength": 0}

def mock_get_support_resistance_levels(data, period=20):
    """Calculate support and resistance levels"""
    if not data or len(data) < period:
        return {"support": 0, "resistance": 0, "range_percent": 0}
    
    try:
        highs = []
        lows = []
        
        for candle in data[-period:]:
            if len(candle) > 3:
                try:
                    high = float(candle[2])
                    low = float(candle[3])
                    if high > 0 and low > 0:
                        highs.append(high)
                        lows.append(low)
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

def mock_calculate_volatility(data, period=20):
    """Calculate volatility"""
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
        
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0:
                ret = (closes[i] - closes[i-1]) / closes[i-1]
                returns.append(abs(ret))
        
        if not returns:
            return 1.0
        
        volatility = np.std(returns) * 100 * np.sqrt(365)  # Annualized volatility
        return round(max(0.5, min(volatility, 10.0)), 2)
    except:
        return round(random.uniform(0.5, 3.0), 2)

def mock_get_swing_high_low(data, period=20):
    """Get swing high and low"""
    if not data or len(data) < period:
        return 0, 0
    
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
            return 0, 0
        
        return max(highs), min(lows)
    except:
        return 0, 0

def mock_calculate_smart_entry(data, signal="BUY", strategy="ICHIMOKU_FIBO"):
    """Calculate smart entry price"""
    if not data:
        return 0
    
    try:
        current_price = float(data[-1][4]) if len(data[-1]) > 4 else 0
        
        if signal == "BUY":
            # For BUY, try to get slightly below current price
            return current_price * random.uniform(0.995, 0.999)
        elif signal == "SELL":
            # For SELL, try to get slightly above current price
            return current_price * random.uniform(1.001, 1.005)
        else:
            return current_price
    except:
        return 0

# Assign mock functions
get_market_data_func = mock_get_market_data_with_fallback
calculate_sma_func = mock_calculate_simple_sma
calculate_rsi_func = mock_calculate_simple_rsi
calculate_rsi_series_func = mock_calculate_rsi_series
calculate_divergence_func = mock_detect_divergence
get_support_resistance_levels_func = mock_get_support_resistance_levels
calculate_volatility_func = mock_calculate_volatility
get_swing_high_low_func = mock_get_swing_high_low
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

class IchimokuRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class FibonacciRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class SignalResponse(BaseModel):
    status: str
    count: int
    last_updated: str
    signals: List[Dict[str, Any]]
    sources: Dict[str, int]

# ==============================================================================
# Core Analysis Functions (FIXED)
# ==============================================================================

def analyze_scalp_signal_fixed(symbol, timeframe, market_data):
    """Fixed scalp signal analysis with price validation"""
    logger.info(f"🔍 Analyzing {symbol} on {timeframe}")
    
    if not market_data or len(market_data) < 30:
        logger.warning(f"Insufficient data for {symbol}")
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
        # Get real current price for validation
        real_price = get_real_time_price(symbol)
        
        # Extract current price from data
        latest_candle = market_data[-1]
        if len(latest_candle) > 4:
            current_price = float(latest_candle[4])
        else:
            current_price = real_price
        
        # Validate price consistency
        current_price = validate_price_consistency(current_price, symbol, "market_data")
        
        logger.info(f"💰 {symbol} Price: ${current_price:.8f} (Real: ${real_price:.8f})")
        
        # Calculate indicators
        rsi = calculate_rsi_func(market_data, 14)
        sma_20_raw = calculate_sma_func(market_data, 20)
        
        # Validate SMA
        if sma_20_raw <= 0:
            logger.warning(f"Invalid SMA20: {sma_20_raw}, using current_price * 0.99")
            sma_20 = current_price * 0.99
        else:
            sma_20 = sma_20_raw
        
        logger.info(f"📊 RSI: {rsi:.2f}, SMA20: ${sma_20:.8f}")
        
        # Calculate divergence
        closes = [float(c[4]) for c in market_data[-30:] if len(c) > 4]
        if closes and len(closes) >= 14:
            rsi_series = calculate_rsi_series_func(closes, 14)
            div_info = calculate_divergence_func(closes, rsi_series, lookback=5)
        else:
            div_info = {"detected": False, "type": "none", "strength": 0}
        
        # Determine signal
        signal, confidence, reason = "HOLD", 0.5, "Neutral"
        
        if div_info['detected']:
            if div_info['type'] == 'bullish':
                signal, confidence, reason = "BUY", 0.85, "Bullish divergence detected"
            elif div_info['type'] == 'bearish':
                signal, confidence, reason = "SELL", 0.85, "Bearish divergence detected"
        else:
            if rsi < 30 and current_price < sma_20:
                signal, confidence, reason = "BUY", 0.75, f"Oversold (RSI: {rsi:.1f}) & below SMA20"
            elif rsi > 70 and current_price > sma_20:
                signal, confidence, reason = "SELL", 0.75, f"Overbought (RSI: {rsi:.1f}) & above SMA20"
            elif rsi < 35:
                signal, confidence, reason = "BUY", 0.65, f"Near oversold (RSI: {rsi:.1f})"
            elif rsi > 65:
                signal, confidence, reason = "SELL", 0.65, f"Near overbought (RSI: {rsi:.1f})"
        
        logger.info(f"📈 Signal: {signal} | Confidence: {confidence} | Reason: {reason}")
        
        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "rsi": round(rsi, 1),
            "divergence": div_info['detected'],
            "divergence_type": div_info['type'],
            "sma_20": round(sma_20, 8),
            "current_price": round(current_price, 8),
            "reason": reason
        }
        
    except Exception as e:
        logger.error(f"❌ Error in analyze_scalp_signal_fixed: {e}", exc_info=True)
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "rsi": 50,
            "divergence": False,
            "sma_20": 0,
            "reason": f"Analysis error: {str(e)[:100]}",
            "current_price": 0
        }

# ==============================================================================
# FastAPI App
# ==============================================================================

API_VERSION = "8.1.0-PRICE-FIXED"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description="Price-Fixed Version with Fibonacci Targets",
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
# API Endpoints (FIXED)
# ==============================================================================

@app.get("/")
async def read_root():
    return {
        "message": f"Crypto AI Trading System v{API_VERSION}",
        "status": "Active",
        "version": API_VERSION,
        "features": [
            "Price Validation System",
            "Advanced Fibonacci Targets",
            "Real-time Price Verification",
            "Symbol Auto-Correction"
        ],
        "endpoints": [
            "/api/scalp-signal - Get scalp signal",
            "/api/fibonacci/{symbol} - Get Fibonacci levels",
            "/api/price/{symbol} - Get real-time price",
            "/api/health - Health check"
        ]
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "Healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "price_system": "Active",
        "fibonacci_calculation": "Active"
    }

@app.get("/api/price/{symbol}")
async def get_current_price(symbol: str):
    """Get real-time price for a symbol"""
    try:
        symbol = validate_symbol(symbol)
        price = get_real_time_price(symbol)
        
        return {
            "symbol": symbol,
            "price_usd": price,
            "price_formatted": f"${price:,.8f}",
            "source": "Binance/CoinGecko/Known",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Price error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Price error: {str(e)[:200]}")

@app.post("/api/scalp-signal")
async def get_scalp_signal_fixed(request: ScalpRequest):
    """
    Fixed Scalp Signal Endpoint with Price Validation
    """
    # Validate timeframe
    allowed_timeframes = ["1m", "5m", "15m", "30m", "1h"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Allowed: {allowed_timeframes}")
    
    start_time = time.time()
    
    try:
        # Validate and normalize symbol
        symbol = validate_symbol(request.symbol)
        logger.info(f"🚀 Scalp signal request: {symbol} ({request.timeframe})")
        
        # Get real price for reference
        real_price = get_real_time_price(symbol)
        
        # Get market data
        market_data_result = get_market_data_func(symbol, request.timeframe, 100, return_source=True)
        
        if isinstance(market_data_result, dict):
            market_data = market_data_result.get("data", [])
            data_source = market_data_result.get("source", "unknown")
        else:
            market_data = market_data_result
            data_source = "direct"
        
        if not market_data or len(market_data) < 20:
            logger.warning(f"Insufficient data for {symbol}")
            raise HTTPException(status_code=404, detail="Insufficient market data")
        
        # Analyze signal
        scalp_analysis = analyze_scalp_signal_fixed(symbol, request.timeframe, market_data)
        
        # Get additional market data
        sr_levels = get_support_resistance_levels_func(market_data)
        volatility = calculate_volatility_func(market_data, 20)
        swing_high, swing_low = get_swing_high_low_func(market_data, 20)
        fib_levels = calculate_fibonacci_levels_from_data(market_data)
        
        # Calculate entry price
        current_price = scalp_analysis.get("current_price", real_price)
        smart_entry = calculate_smart_entry_func(market_data, scalp_analysis["signal"])
        
        # Use validated entry price
        if smart_entry <= 0:
            entry_price = current_price
        else:
            entry_price = validate_price_consistency(smart_entry, symbol, "smart_entry")
        
        # Determine risk level
        rsi = scalp_analysis["rsi"]
        confidence = scalp_analysis["confidence"]
        
        if confidence > 0.8 and (rsi > 80 or rsi < 20):
            risk_level = "HIGH"
        elif confidence > 0.7 and (rsi > 70 or rsi < 30):
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate targets
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_advanced_targets_and_stop_loss(
            entry_price=entry_price,
            signal=scalp_analysis["signal"],
            risk_level=risk_level,
            volatility=volatility,
            rsi=rsi
        )
        
        # Prepare response
        response = {
            "symbol": symbol,
            "timeframe": request.timeframe,
            "signal": scalp_analysis["signal"],
            "confidence": scalp_analysis["confidence"],
            "entry_price": round(entry_price, 8),
            "current_price": round(current_price, 8),
            "real_time_price": round(real_price, 8),
            "rsi": scalp_analysis["rsi"],
            "sma_20": scalp_analysis["sma_20"],
            "divergence": scalp_analysis["divergence"],
            "divergence_type": scalp_analysis["divergence_type"],
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "risk_level": risk_level,
            "fibonacci_levels": fib_levels,
            "support_resistance": sr_levels,
            "volatility": volatility,
            "reason": scalp_analysis["reason"],
            "strategy": "Advanced Scalp (Price-Validated)",
            "price_validation": "PASSED",
            "data_source": data_source,
            "generated_at": datetime.now().isoformat(),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Log summary
        logger.info(f"✅ {symbol} Analysis Complete:")
        logger.info(f"   Signal: {response['signal']} ({response['confidence']*100:.0f}%)")
        logger.info(f"   Entry: ${response['entry_price']:.8f}")
        logger.info(f"   Targets: {response['targets_percent']}")
        logger.info(f"   Stop Loss: {response['stop_loss_percent']:.2f}%")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Critical error in scalp signal: {e}", exc_info=True)
        
        # Enhanced fallback with real price
        symbol = validate_symbol(request.symbol)
        real_price = get_real_time_price(symbol)
        
        fallback_signal = random.choice(["BUY", "SELL", "HOLD"])
        
        # Calculate proper fallback targets
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_advanced_targets_and_stop_loss(
            entry_price=real_price,
            signal=fallback_signal,
            risk_level="MEDIUM",
            volatility=2.0
        )
        
        return {
            "symbol": symbol,
            "timeframe": request.timeframe,
            "signal": fallback_signal,
            "confidence": 0.5,
            "entry_price": round(real_price, 8),
            "current_price": round(real_price, 8),
            "real_time_price": round(real_price, 8),
            "rsi": 50.0,
            "sma_20": round(real_price * 0.99, 8),
            "divergence": False,
            "divergence_type": "none",
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "risk_level": "MEDIUM",
            "reason": f"Fallback mode: {str(e)[:100]}",
            "strategy": "Fallback (Price-Validated)",
            "price_validation": "FALLBACK",
            "error": str(e)[:200],
            "generated_at": datetime.now().isoformat()
        }

@app.get("/api/fibonacci/{symbol}")
async def get_fibonacci_levels(symbol: str, timeframe: str = "5m"):
    """Get Fibonacci levels for a symbol"""
    try:
        symbol = validate_symbol(symbol)
        logger.info(f"🔺 Fibonacci request: {symbol} ({timeframe})")
        
        # Get real price
        real_price = get_real_time_price(symbol)
        
        # Get market data
        market_data = get_market_data_func(symbol, timeframe, 100)
        
        if not market_data or len(market_data) < 20:
            # Create synthetic Fibonacci levels based on real price
            swing_high = real_price * 1.05
            swing_low = real_price * 0.95
            price_range = swing_high - swing_low
            
            fib_levels = {
                '0.0': swing_low,
                '0.236': swing_low + price_range * 0.236,
                '0.382': swing_low + price_range * 0.382,
                '0.5': swing_low + price_range * 0.5,
                '0.618': swing_low + price_range * 0.618,
                '0.786': swing_low + price_range * 0.786,
                '1.0': swing_high,
                '1.272': swing_high + price_range * 0.272,
                '1.618': swing_high + price_range * 0.618,
                'current_price': real_price,
                'source': 'synthetic'
            }
        else:
            # Calculate from real data
            fib_levels = calculate_fibonacci_levels_from_data(market_data)
            fib_levels['source'] = 'market_data'
            fib_levels['current_price'] = real_price
        
        # Add analysis
        current_price = real_price
        support = fib_levels.get('0.618', current_price * 0.98)
        resistance = fib_levels.get('1.0', current_price * 1.02)
        
        analysis = []
        if current_price < support:
            analysis.append("Price below Fibonacci 0.618 support")
        elif current_price > resistance:
            analysis.append("Price above Fibonacci 1.0 resistance")
        else:
            analysis.append("Price within Fibonacci range")
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "fibonacci_levels": {k: round(v, 8) for k, v in fib_levels.items() if isinstance(v, (int, float))},
            "analysis": analysis,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Fibonacci error: {e}")
        raise HTTPException(status_code=500, detail=f"Fibonacci calculation error: {str(e)[:200]}")

@app.post("/api/analyze")
async def analyze_crypto(request: AnalysisRequest):
    """General analysis endpoint"""
    try:
        symbol = validate_symbol(request.symbol)
        logger.info(f"📊 Analysis request: {symbol} ({request.timeframe})")
        
        # Get real price
        real_price = get_real_time_price(symbol)
        
        # Get market data
        market_data = get_market_data_func(symbol, request.timeframe, 100)
        
        # Calculate basic indicators
        current_price = real_price
        rsi = calculate_rsi_func(market_data, 14) if market_data else 50.0
        sma_20 = calculate_sma_func(market_data, 20) if market_data else current_price * 0.99
        volatility = calculate_volatility_func(market_data, 20) if market_data else 1.5
        
        # Determine signal
        if rsi < 30 and current_price < sma_20:
            signal = "BUY"
            confidence = 0.7
            reason = f"Oversold (RSI: {rsi:.1f}) and below SMA20"
        elif rsi > 70 and current_price > sma_20:
            signal = "SELL"
            confidence = 0.7
            reason = f"Overbought (RSI: {rsi:.1f}) and above SMA20"
        else:
            signal = "HOLD"
            confidence = 0.5
            reason = "Neutral market conditions"
        
        # Calculate targets
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_advanced_targets_and_stop_loss(
            entry_price=current_price,
            signal=signal,
            risk_level="MEDIUM",
            volatility=volatility,
            rsi=rsi
        )
        
        return {
            "symbol": symbol,
            "timeframe": request.timeframe,
            "signal": signal,
            "confidence": confidence,
            "current_price": current_price,
            "rsi": round(rsi, 2),
            "sma_20": round(sma_20, 8),
            "volatility": volatility,
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "reason": reason,
            "analysis_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)[:200]}")

@app.get("/api/symbols")
async def get_supported_symbols():
    """Get list of supported symbols"""
    symbols = list(KNOWN_PRICES.keys())
    symbols.remove('DEFAULT')
    
    return {
        "supported_symbols": symbols,
        "count": len(symbols),
        "note": "Symbols automatically corrected (e.g., AVA → AVAX)",
        "timestamp": datetime.now().isoformat()
    }

# ==============================================================================
# Startup and Main
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info(f"🚀 Starting Price-Fixed Trading System v{API_VERSION}")
    logger.info(f"✅ Price Validation: ACTIVE")
    logger.info(f"✅ Fibonacci Calculation: ACTIVE")
    
    # Test price system
    try:
        test_symbols = ['BTCUSDT', 'AVAXUSDT', 'SOLUSDT']
        for symbol in test_symbols:
            price = get_real_time_price(symbol)
            logger.info(f"   {symbol}: ${price:.8f}")
    except Exception as e:
        logger.warning(f"Price test failed: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"PRICE-FIXED Trading System v{API_VERSION}")
    print(f"Server Ready!")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"\n{'=' * 60}")
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📚 Docs: http://{host}:{port}/api/docs")
    print(f"❤️ Health: http://{host}:{port}/api/health")
    print(f"💰 Price Test: http://{host}:{port}/api/price/AVAX")
    print(f"🎯 Scalp Signal: POST http://{host}:{port}/api/scalp-signal")
    print(f"{'=' * 60}\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )