"""
Crypto AI Trading System v8.0.0 (ADVANCED FIBONACCI & ICHIMOKU)
Completely Fixed with Advanced Target Calculation
Features:
1. Fibonacci-based target calculation (0.236, 0.382, 0.618, 1.0, 1.618 levels)
2. Ichimoku-based support/resistance levels
3. Dynamic risk adjustment based on volatility
4. Proper error handling and fallback mechanisms
5. Real-time market data validation
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system_advanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. FUNCTIONS FOR ADVANCED TARGET CALCULATION (Fibonacci & Ichimoku)
# ==============================================================================

def calculate_advanced_targets_and_stop_loss(entry_price, signal, risk_level="MEDIUM", 
                                            ichimoku_data=None, support=None, resistance=None,
                                            volatility=1.0, trend_power=50, rsi=50):
    """
    Advanced target calculation using Fibonacci, Ichimoku, and volatility analysis
    
    Fibonacci retracement levels: 0.236, 0.382, 0.5, 0.618, 0.786
    Fibonacci extension levels: 1.0, 1.272, 1.618, 2.0, 2.618
    
    Returns: targets, stop_loss, targets_percent, stop_loss_percent
    """
    if entry_price <= 0:
        return [0, 0, 0], 0, [0, 0, 0], 0
    
    # Extract Ichimoku levels if available
    if ichimoku_data:
        tenkan_sen = ichimoku_data.get('tenkan_sen', entry_price)
        kijun_sen = ichimoku_data.get('kijun_sen', entry_price)
        cloud_top = ichimoku_data.get('cloud_top', entry_price * 1.02)
        cloud_bottom = ichimoku_data.get('cloud_bottom', entry_price * 0.98)
        senkou_span_a = ichimoku_data.get('senkou_span_a', entry_price)
        senkou_span_b = ichimoku_data.get('senkou_span_b', entry_price)
    else:
        tenkan_sen = kijun_sen = cloud_top = cloud_bottom = entry_price
        senkou_span_a = senkou_span_b = entry_price
    
    # Calculate Fibonacci levels based on support/resistance or recent swing
    if support and resistance and support > 0 and resistance > support:
        swing_high = resistance
        swing_low = support
    else:
        # Use default swing levels based on volatility
        swing_high = entry_price * (1 + volatility/100)
        swing_low = entry_price * (1 - volatility/100)
    
    price_range = swing_high - swing_low
    
    # Fibonacci retracement levels (for stop loss and initial targets)
    fib_retracement = {
        '0.236': swing_low + price_range * 0.236,
        '0.382': swing_low + price_range * 0.382,
        '0.5': swing_low + price_range * 0.5,
        '0.618': swing_low + price_range * 0.618,
        '0.786': swing_low + price_range * 0.786
    }
    
    # Fibonacci extension levels (for profit targets)
    fib_extension = {
        '1.0': swing_high,
        '1.272': swing_high + price_range * 0.272,
        '1.618': swing_high + price_range * 0.618,
        '2.0': swing_high + price_range,
        '2.618': swing_high + price_range * 1.618
    }
    
    # Adjust risk parameters based on multiple factors
    volatility_factor = 1 + (volatility / 100)  # Higher volatility = wider stops
    trend_factor = 1 + (trend_power / 100) if trend_power > 50 else 1 - ((50 - trend_power) / 200)
    rsi_factor = 1.2 if rsi > 70 or rsi < 30 else 1.0  # Extreme RSI = adjust targets
    
    # Combined adjustment factor
    adjustment_factor = volatility_factor * trend_factor * rsi_factor
    
    # Risk level configurations
    risk_configs = {
        "HIGH": {
            "BUY": {
                "target_levels": ['1.0', '1.272', '1.618'],  # Fibonacci extension levels
                "stop_level": '0.618',  # Fibonacci retracement level
                "use_ichimoku": True,
                "max_risk_percent": 2.0
            },
            "SELL": {
                "target_levels": ['0.382', '0.236', '0.0'],
                "stop_level": '1.382',
                "use_ichimoku": True,
                "max_risk_percent": 2.0
            },
            "HOLD": {
                "target_levels": ['1.005', '1.01', '1.015'],
                "stop_level": '0.995',
                "use_ichimoku": False,
                "max_risk_percent": 1.0
            }
        },
        "MEDIUM": {
            "BUY": {
                "target_levels": ['0.786', '1.0', '1.272'],
                "stop_level": '0.5',
                "use_ichimoku": True,
                "max_risk_percent": 1.5
            },
            "SELL": {
                "target_levels": ['0.5', '0.382', '0.236'],
                "stop_level": '1.236',
                "use_ichimoku": True,
                "max_risk_percent": 1.5
            },
            "HOLD": {
                "target_levels": ['1.003', '1.006', '1.009'],
                "stop_level": '0.997',
                "use_ichimoku": False,
                "max_risk_percent": 0.5
            }
        },
        "LOW": {
            "BUY": {
                "target_levels": ['0.618', '0.786', '1.0'],
                "stop_level": '0.382',
                "use_ichimoku": False,
                "max_risk_percent": 1.0
            },
            "SELL": {
                "target_levels": ['0.618', '0.5', '0.382'],
                "stop_level": '1.0',
                "use_ichimoku": False,
                "max_risk_percent": 1.0
            },
            "HOLD": {
                "target_levels": ['1.002', '1.004', '1.006'],
                "stop_level": '0.998',
                "use_ichimoku": False,
                "max_risk_percent": 0.3
            }
        }
    }
    
    config = risk_configs.get(risk_level, risk_configs["MEDIUM"])
    signal_config = config.get(signal, config["HOLD"])
    
    # Calculate targets
    targets = []
    for level in signal_config["target_levels"]:
        if signal_config["use_ichimoku"] and ichimoku_data:
            # Use Ichimoku levels for precise targeting
            if signal == "BUY":
                if level == '0.786':
                    target = tenkan_sen
                elif level == '1.0':
                    target = kijun_sen
                elif level == '1.272':
                    target = cloud_top
                else:
                    target = fib_extension.get(level, entry_price * 1.01)
            elif signal == "SELL":
                if level == '0.5':
                    target = kijun_sen
                elif level == '0.382':
                    target = tenkan_sen
                elif level == '0.236':
                    target = cloud_bottom
                else:
                    target = fib_retracement.get(level, entry_price * 0.99)
            else:
                target = entry_price * float(level)
        else:
            # Use Fibonacci levels
            if level in fib_extension:
                target = fib_extension[level]
            elif level in fib_retracement:
                target = fib_retracement[level]
            else:
                target = entry_price * float(level)
        
        # Apply adjustment factor
        if signal == "BUY":
            target = target * adjustment_factor
        elif signal == "SELL":
            target = target / adjustment_factor
        
        targets.append(round(target, 8))
    
    # Calculate stop loss
    stop_level = signal_config["stop_level"]
    if signal_config["use_ichimoku"] and ichimoku_data:
        if signal == "BUY":
            stop_loss = min(cloud_bottom, fib_retracement.get(stop_level, entry_price * 0.99))
        elif signal == "SELL":
            stop_loss = max(cloud_top, fib_extension.get(stop_level, entry_price * 1.01))
        else:
            stop_loss = entry_price * float(stop_level)
    else:
        if stop_level in fib_retracement:
            stop_loss = fib_retracement[stop_level]
        elif stop_level in fib_extension:
            stop_loss = fib_extension[stop_level]
        else:
            stop_loss = entry_price * float(stop_level)
    
    # Ensure stop loss is reasonable
    max_risk = signal_config["max_risk_percent"]
    if signal == "BUY":
        min_stop = entry_price * (1 - max_risk/100)
        stop_loss = min(stop_loss, min_stop)
    elif signal == "SELL":
        max_stop = entry_price * (1 + max_risk/100)
        stop_loss = max(stop_loss, max_stop)
    
    stop_loss = round(stop_loss, 8)
    
    # Calculate percentages
    targets_percent = [
        round(((target - entry_price) / entry_price) * 100, 2)
        for target in targets
    ]
    stop_loss_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
    
    return targets, stop_loss, targets_percent, stop_loss_percent

def calculate_fibonacci_levels(swing_high, swing_low, include_extensions=True):
    """
    Calculate all Fibonacci retracement and extension levels
    """
    if swing_low <= 0 or swing_high <= swing_low:
        return {}
    
    price_range = swing_high - swing_low
    
    fib_levels = {
        '0.0': swing_low,
        '0.236': swing_low + price_range * 0.236,
        '0.382': swing_low + price_range * 0.382,
        '0.5': swing_low + price_range * 0.5,
        '0.618': swing_low + price_range * 0.618,
        '0.786': swing_low + price_range * 0.786,
        '1.0': swing_high
    }
    
    if include_extensions:
        fib_levels.update({
            '1.272': swing_high + price_range * 0.272,
            '1.414': swing_high + price_range * 0.414,
            '1.618': swing_high + price_range * 0.618,
            '2.0': swing_high + price_range,
            '2.618': swing_high + price_range * 1.618
        })
    
    return fib_levels

def calculate_volatility_adaptive_targets(entry_price, signal, volatility, atr=None):
    """
    Calculate targets based on volatility (ATR or standard deviation)
    """
    if volatility <= 0:
        volatility = 1.0
    
    # Base multipliers adjusted for volatility
    if volatility < 0.5:
        # Low volatility - tighter targets
        if signal == "BUY":
            targets = [1.005, 1.01, 1.015]
            stop_loss = 0.99
        elif signal == "SELL":
            targets = [0.995, 0.99, 0.985]
            stop_loss = 1.01
        else:
            targets = [1.002, 1.004, 1.006]
            stop_loss = 0.998
    elif volatility > 2.0:
        # High volatility - wider targets
        if signal == "BUY":
            targets = [1.01, 1.02, 1.03]
            stop_loss = 0.98
        elif signal == "SELL":
            targets = [0.99, 0.98, 0.97]
            stop_loss = 1.02
        else:
            targets = [1.005, 1.01, 1.015]
            stop_loss = 0.995
    else:
        # Medium volatility
        if signal == "BUY":
            targets = [1.008, 1.015, 1.022]
            stop_loss = 0.985
        elif signal == "SELL":
            targets = [0.992, 0.985, 0.978]
            stop_loss = 1.015
        else:
            targets = [1.003, 1.006, 1.009]
            stop_loss = 0.997
    
    # Apply ATR if available
    if atr and atr > 0:
        atr_multiplier = atr / entry_price
        adjusted_targets = []
        for target in targets:
            if signal == "BUY":
                adjusted = entry_price * target + (atr * (target - 1))
            elif signal == "SELL":
                adjusted = entry_price * target - (atr * (1 - target))
            else:
                adjusted = entry_price * target
            adjusted_targets.append(adjusted)
        targets = adjusted_targets
        
        if signal == "BUY":
            stop_loss = entry_price * stop_loss - (atr * 1.5)
        elif signal == "SELL":
            stop_loss = entry_price * stop_loss + (atr * 1.5)
    
    # Calculate final values
    final_targets = [round(entry_price * t if isinstance(t, float) else t, 8) for t in targets]
    final_stop = round(entry_price * stop_loss if isinstance(stop_loss, float) else stop_loss, 8)
    
    targets_percent = [round(((t - entry_price) / entry_price) * 100, 2) for t in final_targets]
    stop_loss_percent = round(((final_stop - entry_price) / entry_price) * 100, 2)
    
    return final_targets, final_stop, targets_percent, stop_loss_percent

# ==============================================================================
# Path Setup
# ==============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print("=" * 60)
print(f"Advanced Trading System Initialization")
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

print("\n[1/3] Importing utils module...")

try:
    # Try different import strategies
    try:
        import api.utils as utils_module
        utils = utils_module
        UTILS_AVAILABLE = True
        print("✅ SUCCESS: Utils imported via api.utils")
    except ImportError:
        try:
            from . import utils
            UTILS_AVAILABLE = True
            print("✅ SUCCESS: Utils imported via relative import")
        except ImportError:
            project_root = os.path.dirname(os.path.dirname(current_dir))
            sys.path.insert(0, project_root)
            import api.utils as utils_module
            utils = utils_module
            UTILS_AVAILABLE = True
            print("✅ SUCCESS: Utils imported via project root")
except Exception as e:
    print(f"❌ Utils import failed: {e}")
    UTILS_AVAILABLE = False

# ==============================================================================
# Import Individual Functions
# ==============================================================================

if UTILS_AVAILABLE:
    print("\n[2/3] Importing individual functions...")
    try:
        from api.utils import (
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
            get_support_resistance_levels,
            calculate_volatility,
            combined_analysis,
            generate_ichimoku_recommendation,
            get_swing_high_low,
            calculate_smart_entry,
            get_fallback_signal
        )
        print("✅ All functions imported successfully")
    except ImportError as e:
        print(f"⚠️ Partial import failed: {e}")
        UTILS_AVAILABLE = False
else:
    print("⚠️ Using mock functions")

# ==============================================================================
# Import Other Modules
# ==============================================================================

print("\n[3/3] Importing other modules...")

try:
    from api.data_collector import get_collected_data
    DATA_COLLECTOR_AVAILABLE = True
    print("✅ data_collector imported")
except ImportError:
    print("⚠️ data_collector not available")
    DATA_COLLECTOR_AVAILABLE = False

try:
    from api.collectors import collect_signals_from_example_site
    COLLECTORS_AVAILABLE = True
    print("✅ collectors imported")
except ImportError:
    print("⚠️ collectors not available")
    COLLECTORS_AVAILABLE = False

# ==============================================================================
# Mock Functions (Fallback)
# ==============================================================================

def mock_get_market_data_with_fallback(symbol, interval="5m", limit=50, return_source=False):
    """Mock data generator"""
    base_prices = {
        'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'BNBUSDT': 590.00,
        'SOLUSDT': 175.00, 'DOGEUSDT': 0.12116, 'ALGOUSDT': 0.1187,
        'DEFAULT': 100.0
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
            timestamp + 300000,
            "0", "0", "0", "0", "0"
        ]
        data.append(candle)
    
    if return_source:
        return {"data": data, "source": "mock", "success": False}
    return data

def mock_calculate_simple_sma(data, period=20):
    if not data or len(data) < period:
        return 0
    closes = [float(c[4]) for c in data[-period:]]
    return sum(closes) / len(closes)

def mock_calculate_simple_rsi(data, period=14):
    return round(random.uniform(30, 70), 2)

def mock_calculate_rsi_series(closes, period=14):
    return [random.uniform(30, 70) for _ in range(len(closes) if closes else 0)]

def mock_detect_divergence(prices, rsi_values, lookback=5):
    return {"detected": False, "type": "none", "strength": None}

def mock_calculate_macd_simple(data, fast=12, slow=26, signal=9):
    return {
        'macd': round(random.uniform(-1, 1), 4),
        'signal': round(random.uniform(-1, 1), 4),
        'histogram': round(random.uniform(-0.5, 0.5), 4)
    }

def mock_analyze_scalp_conditions(data, timeframe):
    return {
        "condition": random.choice(["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE"]),
        "rsi": round(random.uniform(30, 70), 1),
        "sma_20": 0,
        "current_price": float(data[-1][4]) if data else 0,
        "volatility": round(random.uniform(0.5, 3.0), 2),
        "reason": "Mock analysis"
    }

def mock_calculate_ichimoku_components(data, **kwargs):
    try:
        current_price = float(data[-1][4]) if data else 0
        return {
            'tenkan_sen': current_price * random.uniform(0.99, 1.01),
            'kijun_sen': current_price * random.uniform(0.98, 1.02),
            'cloud_top': current_price * random.uniform(1.01, 1.03),
            'cloud_bottom': current_price * random.uniform(0.97, 0.99),
            'current_price': current_price,
            'trend_power': random.uniform(30, 70)
        }
    except:
        return {'current_price': 100}

def mock_analyze_ichimoku_scalp_signal(ichimoku_data):
    signal = random.choice(['BUY', 'SELL', 'HOLD'])
    confidence = random.uniform(0.6, 0.9) if signal != 'HOLD' else random.uniform(0.4, 0.6)
    return {
        'signal': signal,
        'confidence': confidence,
        'reason': 'Mock analysis',
        'levels': {},
        'trend_power': 50
    }

def mock_get_support_resistance_levels(data):
    if not data:
        return {"support": 0, "resistance": 0, "range_percent": 0}
    try:
        highs = [float(c[2]) for c in data[-20:]]
        lows = [float(c[3]) for c in data[-20:]]
        resistance = sum(highs) / len(highs) if highs else 0
        support = sum(lows) / len(lows) if lows else 0
        range_percent = ((resistance - support) / support * 100) if support > 0 else 0
        return {
            "support": round(support, 4),
            "resistance": round(resistance, 4),
            "range_percent": round(range_percent, 2)
        }
    except:
        return {"support": 0, "resistance": 0, "range_percent": 0}

def mock_calculate_volatility(data, period=20):
    return round(random.uniform(0.5, 3.0), 2)

def mock_get_swing_high_low(data, period=20):
    if not data or len(data) < period:
        return 0, 0
    highs = [float(c[2]) for c in data[-period:]]
    lows = [float(c[3]) for c in data[-period:]]
    return max(highs) if highs else 0, min(lows) if lows else 0

def mock_calculate_smart_entry(data, signal="BUY", strategy="ICHIMOKU_FIBO"):
    try:
        base_price = float(data[-1][4]) if data else 0
        if signal == "BUY":
            return base_price * random.uniform(0.99, 1.00)
        elif signal == "SELL":
            return base_price * random.uniform(1.00, 1.01)
        return base_price
    except:
        return 0

# ==============================================================================
# Function Selection
# ==============================================================================

if UTILS_AVAILABLE:
    print("\n" + "=" * 60)
    print("SELECTING: REAL UTILS FUNCTIONS")
    print("=" * 60)
    
    get_market_data_func = get_market_data_with_fallback
    calculate_sma_func = calculate_simple_sma
    calculate_rsi_func = calculate_simple_rsi
    calculate_rsi_series_func = calculate_rsi_series
    calculate_divergence_func = detect_divergence
    calculate_macd_func = calculate_macd_simple
    analyze_scalp_conditions_func = analyze_scalp_conditions
    calculate_ichimoku_func = calculate_ichimoku_components
    analyze_ichimoku_signal_func = analyze_ichimoku_scalp_signal
    get_ichimoku_signal_func = get_ichimoku_scalp_signal
    get_support_resistance_levels_func = get_support_resistance_levels
    calculate_volatility_func = calculate_volatility
    get_swing_high_low_func = get_swing_high_low
    calculate_smart_entry_func = calculate_smart_entry
    combined_analysis_func = combined_analysis
    generate_recommendation_func = generate_ichimoku_recommendation
    
else:
    print("\n" + "=" * 60)
    print("SELECTING: MOCK FUNCTIONS")
    print("=" * 60)
    
    get_market_data_func = mock_get_market_data_with_fallback
    calculate_sma_func = mock_calculate_simple_sma
    calculate_rsi_func = mock_calculate_simple_rsi
    calculate_rsi_series_func = mock_calculate_rsi_series
    calculate_divergence_func = mock_detect_divergence
    calculate_macd_func = mock_calculate_macd_simple
    analyze_scalp_conditions_func = mock_analyze_scalp_conditions
    calculate_ichimoku_func = mock_calculate_ichimoku_components
    analyze_ichimoku_signal_func = mock_analyze_ichimoku_scalp_signal
    get_ichimoku_signal_func = lambda x, y: None
    get_support_resistance_levels_func = mock_get_support_resistance_levels
    calculate_volatility_func = mock_calculate_volatility
    get_swing_high_low_func = mock_get_swing_high_low
    calculate_smart_entry_func = mock_calculate_smart_entry
    combined_analysis_func = lambda x, y: None
    generate_recommendation_func = lambda x: "Mock Recommendation"

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
# Core Analysis Functions
# ==============================================================================

def analyze_scalp_signal(symbol, timeframe, data):
    """Scalp signal analysis"""
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
        
        if sma_20 is None or sma_20 <= 0:
            sma_20 = latest_close * 0.99
        
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
            "sma_20": round(sma_20, 8),
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
            "reason": f"Analysis error: {str(e)[:100]}",
            "current_price": 0
        }

def analyze_ichimoku_scalp(symbol, timeframe, data):
    """Ichimoku scalp analysis with advanced targets"""
    if not data or len(data) < 60:
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "reason": "Insufficient Ichimoku data",
            "entry_price": 0,
            "targets": [0, 0, 0],
            "stop_loss": 0
        }
    
    try:
        ichimoku_data = calculate_ichimoku_func(data)
        if not ichimoku_data:
            return {"signal": "HOLD", "confidence": 0.5, "reason": "Ichimoku calculation failed"}
        
        ichimoku_signal = analyze_ichimoku_signal_func(ichimoku_data)
        current_price = ichimoku_data.get('current_price', float(data[-1][4]) if data else 0)
        
        sr_levels = get_support_resistance_levels_func(data)
        support = sr_levels.get("support", current_price * 0.98)
        resistance = sr_levels.get("resistance", current_price * 1.02)
        
        swing_high, swing_low = get_swing_high_low_func(data, 20)
        volatility = calculate_volatility_func(data, 20)
        rsi = calculate_rsi_func(data, 14)
        trend_power = ichimoku_data.get('trend_power', 50)
        
        # Determine risk level
        confidence = ichimoku_signal['confidence']
        if confidence > 0.8 and trend_power > 70:
            risk_level = "HIGH"
        elif confidence > 0.6 and trend_power > 50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate advanced targets
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_advanced_targets_and_stop_loss(
            entry_price=current_price,
            signal=ichimoku_signal['signal'],
            risk_level=risk_level,
            ichimoku_data=ichimoku_data,
            support=support,
            resistance=resistance,
            volatility=volatility,
            trend_power=trend_power,
            rsi=rsi
        )
        
        # Calculate Fibonacci levels for display
        fib_levels = calculate_fibonacci_levels(swing_high, swing_low)
        
        return {
            "signal": ichimoku_signal['signal'],
            "confidence": ichimoku_signal['confidence'],
            "reason": ichimoku_signal['reason'],
            "entry_price": current_price,
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "risk_level": risk_level,
            "ichimoku": {
                'tenkan_sen': ichimoku_data.get('tenkan_sen'),
                'kijun_sen': ichimoku_data.get('kijun_sen'),
                'cloud_top': ichimoku_data.get('cloud_top'),
                'cloud_bottom': ichimoku_data.get('cloud_bottom')
            },
            "fibonacci": fib_levels,
            "support_resistance": sr_levels,
            "volatility": volatility,
            "trend_power": trend_power,
            "type": "ICHIMOKU_SCALP_ADVANCED"
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_ichimoku_scalp: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "reason": f"Ichimoku error: {str(e)[:100]}"
        }

# ==============================================================================
# FastAPI App
# ==============================================================================

API_VERSION = "8.0.0-ADVANCED"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description="Advanced Fibonacci & Ichimoku Target Calculation",
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
        "features": [
            "Advanced Fibonacci Target Calculation",
            "Ichimoku Cloud Integration",
            "Volatility-Adaptive Stops",
            "Multi-Timeframe Analysis"
        ],
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE
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
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE
        }
    }

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """
    Advanced Scalp Signal with Fibonacci and Ichimoku targets
    """
    allowed_timeframes = ["1m", "5m", "15m"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Allowed: {allowed_timeframes}")
    
    start_time = time.time()
    logger.info(f"Advanced scalp signal request: {request.symbol} ({request.timeframe})")
    
    try:
        # Get market data
        market_data_result = get_market_data_func(request.symbol, request.timeframe, 100, return_source=True)
        
        if isinstance(market_data_result, dict) and "data" in market_data_result:
            market_data = market_data_result["data"]
            data_source = market_data_result.get("source", "unknown")
            data_success = market_data_result.get("success", False)
        else:
            market_data = market_data_result
            data_source = "direct"
            data_success = True
        
        if not market_data or len(market_data) < 20:
            market_data = mock_get_market_data_with_fallback(request.symbol, request.timeframe, 100)
            data_source = "mock_fallback"
            data_success = False
        
        # Analyze scalp signal
        scalp_analysis = analyze_scalp_signal(request.symbol, request.timeframe, market_data)
        
        # Calculate additional indicators
        ichimoku_data = calculate_ichimoku_func(market_data)
        sr_levels = get_support_resistance_levels_func(market_data)
        swing_high, swing_low = get_swing_high_low_func(market_data, 20)
        volatility = calculate_volatility_func(market_data, 20)
        rsi = scalp_analysis["rsi"]
        
        # Calculate smart entry
        try:
            smart_entry = calculate_smart_entry_func(market_data, scalp_analysis["signal"])
            if smart_entry <= 0:
                smart_entry = scalp_analysis.get("current_price", 0)
        except:
            smart_entry = scalp_analysis.get("current_price", 0)
        
        # Determine risk level
        confidence = scalp_analysis["confidence"]
        if confidence > 0.8 and (rsi > 80 or rsi < 20):
            risk_level = "HIGH"
        elif confidence > 0.7 and (rsi > 70 or rsi < 30):
            risk_level = "MEDIUM"
        elif confidence < 0.5:
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"
        
        # Calculate advanced targets
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_advanced_targets_and_stop_loss(
            entry_price=smart_entry,
            signal=scalp_analysis["signal"],
            risk_level=risk_level,
            ichimoku_data=ichimoku_data,
            support=sr_levels.get("support"),
            resistance=sr_levels.get("resistance"),
            volatility=volatility,
            trend_power=50,
            rsi=rsi
        )
        
        # Calculate Fibonacci levels
        fib_levels = calculate_fibonacci_levels(swing_high, swing_low)
        
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": scalp_analysis["signal"],
            "confidence": scalp_analysis["confidence"],
            "entry_price": round(smart_entry, 8),
            "current_price": scalp_analysis.get("current_price", 0),
            "rsi": rsi,
            "divergence": scalp_analysis["divergence"],
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "risk_level": risk_level,
            "strategy": "Advanced Scalp (Fibonacci + Ichimoku)",
            "fibonacci_levels": {k: round(v, 8) for k, v in fib_levels.items()},
            "support_resistance": sr_levels,
            "volatility": volatility,
            "ichimoku_available": ichimoku_data is not None,
            "reason": scalp_analysis["reason"],
            "data_source": data_source,
            "data_success": data_success,
            "generated_at": datetime.now().isoformat(),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        logger.info(f"✅ {response['signal']} signal for {response['symbol']} | "
                   f"Confidence: {response['confidence']} | "
                   f"Targets: {response['targets_percent']}%")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in scalp signal: {e}")
        
        # Fallback with advanced calculation
        fallback_signal = random.choice(["BUY", "SELL", "HOLD"])
        base_prices = {
            'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00,
            'DOGEUSDT': 0.12116, 'ALGOUSDT': 0.1187,
            'DEFAULT': 100.0
        }
        base_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        entry_price = round(base_price * random.uniform(0.99, 1.01), 8)
        
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_advanced_targets_and_stop_loss(
            entry_price=entry_price,
            signal=fallback_signal,
            risk_level="MEDIUM",
            volatility=random.uniform(1.0, 3.0)
        )
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": fallback_signal,
            "confidence": 0.5,
            "entry_price": entry_price,
            "current_price": entry_price,
            "rsi": round(random.uniform(30, 70), 1),
            "divergence": False,
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "risk_level": "MEDIUM",
            "strategy": "Advanced Fallback",
            "reason": f"System error: {str(e)[:100]}",
            "generated_at": datetime.now().isoformat(),
            "error": True
        }

@app.post("/api/ichimoku-scalp")
async def get_ichimoku_scalp_signal(request: IchimokuRequest):
    """
    Advanced Ichimoku Scalp Signal with Fibonacci targets
    """
    allowed_timeframes = ["1m", "5m", "15m", "1h", "4h"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Allowed: {allowed_timeframes}")
    
    try:
        logger.info(f"Ichimoku request: {request.symbol} ({request.timeframe})")
        
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        if not market_data or len(market_data) < 60:
            raise HTTPException(status_code=404, detail="Not enough data for Ichimoku analysis")
        
        ichimoku_analysis = analyze_ichimoku_scalp(request.symbol, request.timeframe, market_data)
        
        # Calculate additional indicators
        rsi = calculate_rsi_func(market_data, 14)
        closes = [float(c[4]) for c in market_data[-30:]] if len(market_data) >= 30 else []
        rsi_series = calculate_rsi_series_func(closes, 14) if closes else []
        div = calculate_divergence_func(closes, rsi_series, lookback=5) if closes else {"detected": False, "type": "none"}
        
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": ichimoku_analysis["signal"],
            "confidence": ichimoku_analysis["confidence"],
            "entry_price": ichimoku_analysis["entry_price"],
            "targets": ichimoku_analysis["targets"],
            "stop_loss": ichimoku_analysis["stop_loss"],
            "targets_percent": ichimoku_analysis.get("targets_percent", [0, 0, 0]),
            "stop_loss_percent": ichimoku_analysis.get("stop_loss_percent", 0),
            "risk_level": ichimoku_analysis.get("risk_level", "MEDIUM"),
            "rsi": round(rsi, 2),
            "divergence": div['detected'],
            "divergence_type": div['type'],
            "reason": ichimoku_analysis["reason"],
            "strategy": f"Advanced Ichimoku Scalp",
            "fibonacci_levels": ichimoku_analysis.get("fibonacci", {}),
            "ichimoku_data": ichimoku_analysis.get("ichimoku", {}),
            "volatility": ichimoku_analysis.get("volatility", 0),
            "trend_power": ichimoku_analysis.get("trend_power", 50),
            "generated_at": datetime.now().isoformat()
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Ichimoku: {e}")
        raise HTTPException(status_code=500, detail=f"Ichimoku analysis error: {str(e)[:200]}")

@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    """Market data endpoint with advanced indicators"""
    try:
        data = get_market_data_func(symbol, timeframe, limit=50)
        if not data:
            raise HTTPException(status_code=404, detail="No market data available")
        
        latest = data[-1] if isinstance(data, list) and len(data) > 0 else []
        if not latest or len(latest) < 6:
            current_price = 100
        else:
            current_price = float(latest[4])
        
        # Calculate indicators
        rsi = calculate_rsi_func(data, 14)
        sma_20 = calculate_sma_func(data, 20)
        volatility = calculate_volatility_func(data, 20)
        swing_high, swing_low = get_swing_high_low_func(data, 20)
        fib_levels = calculate_fibonacci_levels(swing_high, swing_low, include_extensions=False)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "high": float(latest[2]) if len(latest) > 2 else current_price,
            "low": float(latest[3]) if len(latest) > 3 else current_price,
            "rsi_14": round(rsi, 2),
            "sma_20": round(sma_20, 2),
            "volatility": round(volatility, 2),
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "fibonacci_levels": {k: round(v, 2) for k, v in fib_levels.items()},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=f"Market data error: {str(e)[:200]}")

@app.get("/api/fibonacci/{symbol}")
async def get_fibonacci_levels(symbol: str, timeframe: str = "5m"):
    """Get Fibonacci levels for a symbol"""
    try:
        data = get_market_data_func(symbol, timeframe, limit=100)
        if not data or len(data) < 20:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        # Calculate recent high and low
        highs = [float(c[2]) for c in data[-50:]]
        lows = [float(c[3]) for c in data[-50:]]
        recent_high = max(highs) if highs else float(data[-1][4]) * 1.05
        recent_low = min(lows) if lows else float(data[-1][4]) * 0.95
        
        fib_levels = calculate_fibonacci_levels(recent_high, recent_low, include_extensions=True)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "swing_high": recent_high,
            "swing_low": recent_low,
            "range": recent_high - recent_low,
            "current_price": float(data[-1][4]) if data else 0,
            "fibonacci_levels": {k: round(v, 8) for k, v in fib_levels.items()},
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Fibonacci calculation error: {e}")
        raise HTTPException(status_code=500, detail=f"Fibonacci error: {str(e)[:200]}")

@app.post("/api/analyze")
async def analyze_crypto(request: AnalysisRequest):
    """General analysis endpoint"""
    try:
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        if not market_data:
            raise HTTPException(status_code=404, detail="No market data available")
        
        # Basic analysis
        current_price = float(market_data[-1][4]) if market_data else 0
        rsi = calculate_rsi_func(market_data, 14)
        sma_20 = calculate_sma_func(market_data, 20)
        volatility = calculate_volatility_func(market_data, 20)
        
        # Determine signal
        if rsi < 30 and current_price < sma_20:
            signal = "BUY"
            confidence = 0.7
        elif rsi > 70 and current_price > sma_20:
            signal = "SELL"
            confidence = 0.7
        else:
            signal = "HOLD"
            confidence = 0.5
        
        # Calculate targets
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_advanced_targets_and_stop_loss(
            entry_price=current_price,
            signal=signal,
            risk_level="MEDIUM",
            volatility=volatility,
            rsi=rsi
        )
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": signal,
            "confidence": confidence,
            "current_price": current_price,
            "rsi": round(rsi, 2),
            "sma_20": round(sma_20, 2),
            "volatility": round(volatility, 2),
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "analysis_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)[:200]}")

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info(f"🚀 Starting Advanced Crypto AI Trading System v{API_VERSION}")
    logger.info(f"📊 Features: Fibonacci + Ichimoku Target Calculation")
    logger.info(f"🔧 Utils Available: {UTILS_AVAILABLE}")
    
    print(f"\n{'=' * 60}")
    print(f"Advanced Trading System v{API_VERSION}")
    print(f"Fibonacci & Ichimoku Target Calculation")
    print(f"{'=' * 60}\n")

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    print(f"\n{'=' * 60}")
    print(f"Server: http://{host}:{port}")
    print(f"Docs: http://{host}:{port}/api/docs")
    print(f"Health: http://{host}:{port}/api/health")
    print(f"{'=' * 60}\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )