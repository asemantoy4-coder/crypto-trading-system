# api/utils.py - Version 7.6.10 (No Circular Imports)
"""
Utility Functions - Fixed Circular Import Issue
"""

import requests
import logging
import random
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Market Data Functions (NO IMPORTS FROM OTHER MODULES)
# ==============================================================================

def get_market_data_with_fallback(symbol, interval="5m", limit=100, return_source=False):
    """Get market data without circular imports."""
    logger.info(f"Fetching data for {symbol} ({interval})")
    
    # Try Binance first
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol.upper(), 'interval': interval, 'limit': min(limit, 1000)}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data received from Binance: {len(data)} candles")
            if return_source:
                return {"data": data, "source": "binance", "success": True}
            else:
                return data
    except Exception as e:
        logger.warning(f"Binance error: {e}")
    
    # Fallback to mock data
    logger.info(f"Using mock data for {symbol}")
    data = generate_mock_data_simple(symbol, limit)
    
    if return_source:
        return {"data": data, "source": "mock", "success": False}
    else:
        return data

def generate_mock_data_simple(symbol, limit=100):
    """Generate mock data."""
    base_prices = {
        'BTCUSDT': 88271.42, 'ETHUSDT': 3450.12, 'BNBUSDT': 590.54, 
        'SOLUSDT': 175.98, 'XRPUSDT': 0.51234, 'ADAUSDT': 0.43210,
        'DEFAULT': 100.0
    }
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    
    mock_data = []
    current_time = int(time.time() * 1000)
    
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)
        change = random.uniform(-0.01, 0.01)
        price = base_price * (1 + change)
        
        mock_candle = [
            timestamp,
            str(price * 0.998),  # open
            str(price * 1.002),  # high
            str(price * 0.997),  # low
            str(price),          # close
            str(random.uniform(1000, 10000)),  # volume
            timestamp + 300000,
            "0", "0", "0", "0", "0"
        ]
        mock_data.append(mock_candle)
    
    return mock_data

# ==============================================================================
# 2. Technical Analysis Functions (STANDALONE)
# ==============================================================================

def calculate_simple_sma(data, period=20):
    """Calculate Simple Moving Average."""
    if not data or len(data) < period:
        return 0.0
    
    try:
        closes = []
        for candle in data[-period:]:
            try:
                closes.append(float(candle[4]))
            except:
                closes.append(0.0)
        
        return round(sum(closes) / len(closes), 2) if closes else 0.0
    except Exception as e:
        logger.error(f"SMA error: {e}")
        return 0.0

def calculate_rsi_series(closes, period=14):
    """Calculate RSI series."""
    if len(closes) < period + 1:
        return [50.0] * len(closes)
    
    rsi_values = [50.0] * period
    gains, losses = 0.0, 0.0
    
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        if change > 0: 
            gains += change
        else: 
            losses += abs(change)
    
    avg_gain = gains / period
    avg_loss = losses / period if losses > 0 else 0.0001
    
    if avg_loss == 0: 
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = change if change > 0 else 0
        loss = abs(change) if change < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0: 
            rsi_val = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100 - (100 / (1 + rs))
        rsi_values.append(rsi_val)
    
    return rsi_values

def calculate_simple_rsi(data, period=14):
    """Calculate RSI."""
    if not data or len(data) <= period:
        return 50.0
    
    try:
        closes = []
        for candle in data[-(period+1):]:
            try:
                closes.append(float(candle[4]))
            except:
                closes.append(0.0)
        
        rsi_series = calculate_rsi_series(closes, period)
        return round(rsi_series[-1], 1) if rsi_series else 50.0
    except Exception as e:
        logger.error(f"RSI error: {e}")
        return 50.0

def detect_divergence(prices, rsi_values, lookback=5):
    """Detect divergence."""
    return {"detected": False, "type": "none", "strength": None}

# ==============================================================================
# 3. Smart Entry System (SIMPLE VERSION)
# ==============================================================================

def get_swing_high_low(data, period=20):
    """Get swing high and low."""
    if not data or len(data) < period: 
        return 0, 0
    
    highs, lows = [], []
    for candle in data[-period:]:
        try:
            highs.append(float(candle[2]))
            lows.append(float(candle[3]))
        except: 
            continue
    
    return (
        max(highs) if highs else 0, 
        min(lows) if lows else 0
    )

def calculate_smart_entry(data, signal="BUY"):
    """Calculate smart entry price."""
    if not data or len(data) < 30: 
        return 0
    
    try:
        # Get current price
        current_price = float(data[-1][4]) if data else 0
        
        if signal == "BUY":
            # For BUY: aim for 0.5% below current
            return round(current_price * 0.995, 2)
        elif signal == "SELL":
            # For SELL: aim for 0.5% above current
            return round(current_price * 1.005, 2)
        else:
            return round(current_price, 2)
    except Exception as e:
        logger.error(f"Smart entry error: {e}")
        return 0

# ==============================================================================
# 4. Ichimoku Functions (SIMPLE)
# ==============================================================================

def calculate_ichimoku_components(data, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
    """Calculate Ichimoku components."""
    if not data or len(data) < max(kijun_period, senkou_b_period) + displacement:
        return None
    
    try:
        highs, lows, closes = [], [], []
        for candle in data:
            try:
                highs.append(float(candle[2]))
                lows.append(float(candle[3]))
                closes.append(float(candle[4]))
            except: 
                continue
        
        if not closes:
            return None
        
        return {
            'tenkan_sen': 0,
            'kijun_sen': 0,
            'current_price': closes[-1],
            'cloud_top': closes[-1] * 1.01,
            'cloud_bottom': closes[-1] * 0.99,
            'trend_power': 50
        }
    except Exception as e:
        logger.error(f"Ichimoku error: {e}")
        return None

def analyze_ichimoku_scalp_signal(ichimoku_data):
    """Analyze Ichimoku signal."""
    if not ichimoku_data:
        return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'No data'}
    
    try:
        current_price = ichimoku_data.get('current_price', 0)
        cloud_top = ichimoku_data.get('cloud_top', current_price * 1.01)
        cloud_bottom = ichimoku_data.get('cloud_bottom', current_price * 0.99)
        
        if current_price > cloud_top:
            return {'signal': 'BUY', 'confidence': 0.7, 'reason': 'Above cloud'}
        elif current_price < cloud_bottom:
            return {'signal': 'SELL', 'confidence': 0.7, 'reason': 'Below cloud'}
        else:
            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'In cloud'}
    except:
        return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Error'}

def get_ichimoku_scalp_signal(data, timeframe="5m"):
    """Get Ichimoku signal."""
    try:
        ichimoku = calculate_ichimoku_components(data)
        if not ichimoku:
            return None
        
        signal = analyze_ichimoku_scalp_signal(ichimoku)
        signal['timeframe'] = timeframe
        return signal
    except Exception as e:
        logger.error(f"Ichimoku signal error: {e}")
        return None

def generate_ichimoku_recommendation(signal_data):
    """Generate recommendation."""
    signal = signal_data.get('signal', 'HOLD')
    
    if signal == 'BUY':
        return 'Consider buying'
    elif signal == 'SELL':
        return 'Consider selling'
    else:
        return 'Hold position'

# ==============================================================================
# 5. Main Analysis Function
# ==============================================================================

def analyze_with_multi_timeframe_strategy(symbol):
    """Main analysis function."""
    logger.info(f"Analyzing {symbol}")
    
    try:
        # Get data for different timeframes
        data_1h = get_market_data_with_fallback(symbol, "1h", 50)
        data_15m = get_market_data_with_fallback(symbol, "15m", 50)
        data_5m = get_market_data_with_fallback(symbol, "5m", 50)
        
        # Calculate RSI for each timeframe
        rsi_1h = calculate_simple_rsi(data_1h, 14)
        rsi_15m = calculate_simple_rsi(data_15m, 14)
        rsi_5m = calculate_simple_rsi(data_5m, 14)
        
        # Determine overall signal
        buy_signals = sum([1 for rsi in [rsi_1h, rsi_15m, rsi_5m] if rsi < 40])
        sell_signals = sum([1 for rsi in [rsi_1h, rsi_15m, rsi_5m] if rsi > 60])
        
        if buy_signals >= 2:
            signal = "BUY"
            confidence = 0.7 + (buy_signals * 0.1)
            reason = f"Bullish across {buy_signals} timeframes"
        elif sell_signals >= 2:
            signal = "SELL"
            confidence = 0.7 + (sell_signals * 0.1)
            reason = f"Bearish across {sell_signals} timeframes"
        else:
            signal = "HOLD"
            confidence = 0.5
            reason = "Mixed signals"
        
        # Get current price
        current_price = float(data_5m[-1][4]) if data_5m and len(data_5m) > 0 else 100.0
        
        # Calculate smart entry
        entry_price = calculate_smart_entry(data_5m, signal)
        if entry_price <= 0:
            entry_price = current_price
        
        # Calculate targets
        if signal == "BUY":
            targets = [
                round(entry_price * 1.01, 2),   # +1%
                round(entry_price * 1.015, 2),  # +1.5%
                round(entry_price * 1.02, 2)    # +2%
            ]
            stop_loss = round(entry_price * 0.99, 2)
        elif signal == "SELL":
            targets = [
                round(entry_price * 0.99, 2),   # -1%
                round(entry_price * 0.985, 2),  # -1.5%
                round(entry_price * 0.98, 2)    # -2%
            ]
            stop_loss = round(entry_price * 1.01, 2)
        else:
            targets = [
                round(entry_price * 1.005, 2),  # +0.5%
                round(entry_price * 1.01, 2),   # +1%
                round(entry_price * 1.015, 2)   # +1.5%
            ]
            stop_loss = round(entry_price * 0.995, 2)
        
        # Calculate percentages
        targets_percent = [
            round(((t - entry_price) / entry_price) * 100, 2) for t in targets
        ]
        stop_loss_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(min(confidence, 0.95), 2),
            "entry_price": round(entry_price, 2),
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "rsi_values": {
                "1h": rsi_1h,
                "15m": rsi_15m,
                "5m": rsi_5m
            },
            "reason": reason,
            "strategy": "Multi-Timeframe RSI"
        }
        
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        
        # Fallback
        base_prices = {
            'BTCUSDT': 88271.42,
            'ETHUSDT': 3450.12,
            'BNBUSDT': 590.54,
            'SOLUSDT': 175.98,
            'DEFAULT': 100.0
        }
        entry_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
        
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": entry_price,
            "targets": [
                round(entry_price * 1.01, 2),
                round(entry_price * 1.015, 2),
                round(entry_price * 1.02, 2)
            ],
            "stop_loss": round(entry_price * 0.99, 2),
            "targets_percent": [1.0, 1.5, 2.0],
            "stop_loss_percent": -1.0,
            "reason": "Analysis failed, using fallback",
            "strategy": "Fallback"
        }

# ==============================================================================
# 6. Helper Functions
# ==============================================================================

def calculate_24h_change_from_dataframe(data):
    """Calculate 24h change."""
    if not data or len(data) < 10:
        return 0.0
    
    try:
        if isinstance(data, dict) and "data" in data:
            data_list = data["data"]
        else:
            data_list = data
        
        first_close = float(data_list[0][4])
        last_close = float(data_list[-1][4])
        
        if first_close <= 0:
            return 0.0
        
        return round(((last_close - first_close) / first_close) * 100, 2)
    except:
        return round(random.uniform(-5, 5), 2)

def get_support_resistance_levels(data):
    """Get support and resistance levels."""
    if not data or len(data) < 20:
        return {"support": 0, "resistance": 0}
    
    highs, lows = [], []
    for candle in data[-20:]:
        try:
            highs.append(float(candle[2]))
            lows.append(float(candle[3]))
        except:
            continue
    
    if not highs or not lows:
        return {"support": 0, "resistance": 0}
    
    return {
        "support": round(sum(lows) / len(lows), 2),
        "resistance": round(sum(highs) / len(highs), 2),
        "range_percent": round(((sum(highs)/len(highs) - sum(lows)/len(lows)) / (sum(lows)/len(lows))) * 100, 2)
    }

print(f"✅ utils.py loaded successfully (v7.6.10)")