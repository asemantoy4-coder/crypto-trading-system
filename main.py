"""
Crypto AI Trading System v8.0.0 (PRO SCALPER EDITION)
Enhanced with:
1. Professional Scalper Engine with ATR-based risk management
2. AI Confirmation System
3. Multi-timeframe scanning
4. Advanced Ichimoku + RSI Divergence strategies
5. Low latency architecture for high-frequency scalping
6. Optimized target calculations with dynamic volatility
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime, timedelta
import logging
from typing import List, Optional, Dict, Any, Tuple
import random
import sys
import os
import math
import time
import asyncio
import numpy as np

# --- ۱. بررسی کتابخانه‌های اختیاری ---
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

# --- ۲. تنظیمات لاگ (اصلاح شده برای Render) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s : %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # لاگ‌ها در کنسول Render نمایش داده می‌شوند
    ]
)
logger = logging.getLogger("CryptoAIScalper")

# --- ۳. تنظیم مسیرها (Path Setup) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=" * 60)
print(f"PRO SCALPER EDITION v8.5.0")
print(f"Pandas: {'Available' if HAS_PANDAS else 'Not available'}")
print(f"Pandas TA: {'Available' if HAS_PANDAS_TA else 'Not available'}")
print(f"Current directory: {current_dir}")

# --- ۴. وضعیت ماژول‌ها ---
UTILS_AVAILABLE = False
DATA_COLLECTOR_AVAILABLE = False
COLLECTORS_AVAILABLE = False

# --- ۵. وارد کردن هوشمند ماژول Utils (نسخه اصلاح شده) ---
print("\n[1/3] Importing utils module...")

try:
    # چون فایل‌ها در Root هستند، فقط همین یک روش استاندارد کافی است
    import utils as utils_module
    utils = utils_module
    UTILS_AVAILABLE = True
    print(f"✅ SUCCESS: Utils imported via Direct import")
except ImportError as e:
    # تلاش مجدد با متد نسبی در صورت نیاز سرور
    try:
        from . import utils as utils_module
        utils = utils_module
        UTILS_AVAILABLE = True
        print(f"✅ SUCCESS: Utils imported via Relative import")
    except Exception as e2:
        print(f"❌ Critical Error: utils.py not found in {current_dir}")
        UTILS_AVAILABLE = False

# ==============================================================================
# Import Individual Functions
# ==============================================================================

if UTILS_AVAILABLE:
    print("\n[2/3] Importing individual functions...")
    try:
        # Core functions
        get_market_data_with_fallback = utils.get_market_data_with_fallback
        analyze_with_multi_timeframe_strategy = utils.analyze_with_multi_timeframe_strategy
        calculate_24h_change_from_dataframe = utils.calculate_24h_change_from_dataframe
        calculate_simple_sma = utils.calculate_simple_sma
        calculate_simple_rsi = utils.calculate_simple_rsi
        calculate_rsi_series = utils.calculate_rsi_series
        detect_divergence = utils.detect_divergence
        calculate_macd_simple = utils.calculate_macd_simple
        analyze_scalp_conditions = utils.analyze_scalp_conditions
        calculate_ichimoku_components = utils.calculate_ichimoku_components
        analyze_ichimoku_scalp_signal = utils.analyze_ichimoku_scalp_signal
        get_ichimoku_scalp_signal = utils.get_ichimoku_scalp_signal
        get_support_resistance_levels = utils.get_support_resistance_levels
        calculate_volatility = utils.calculate_volatility
        combined_analysis = utils.combined_analysis
        generate_ichimoku_recommendation = utils.generate_ichimoku_recommendation
        get_swing_high_low = utils.get_swing_high_low
        calculate_smart_entry = utils.calculate_smart_entry
        get_fallback_signal = utils.get_fallback_signal
        
        # NEW FUNCTIONS FROM ENHANCEMENT
        try:
            calculate_tdr = utils.calculate_tdr
            get_ichimoku_full = utils.get_ichimoku_full
            calculate_atr = utils.calculate_atr
            HAS_TDR_ATR = True
        except AttributeError:
            HAS_TDR_ATR = False
            print("⚠️ TDR/ATR functions not available in utils")
        
        print("✅ All core functions imported successfully")
    except AttributeError as e:
        print(f"❌ Failed to get functions from utils module: {e}")
        UTILS_AVAILABLE = False
    except Exception as e:
        print(f"❌ Error importing functions: {e}")
        UTILS_AVAILABLE = False

# ==============================================================================
# Import Other Modules
# ==============================================================================

print("\n[3/3] Importing other modules...")

# Data Collector
try:
    from api.data_collector import get_collected_data
    DATA_COLLECTOR_AVAILABLE = True
    print("✅ data_collector imported")
except ImportError as e:
    print(f"⚠️ data_collector not available: {e}")
    DATA_COLLECTOR_AVAILABLE = False

# Collectors
try:
    from api.collectors import collect_signals_from_example_site
    COLLECTORS_AVAILABLE = True
    print("✅ collectors imported")
except ImportError as e:
    print(f"⚠️ collectors not available: {e}")
    COLLECTORS_AVAILABLE = False

# ==============================================================================
# Pydantic Models
# ==============================================================================

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class ScalpRequest(BaseModel):
    # تغییر example به json_schema_extra برای رفع هشدار
    symbol: str = Field(..., json_schema_extra={"example": "BTCUSDT"})
    timeframe: str = Field(default="5m", pattern="^(1m|5m|15m)$")
    use_ai: bool = True

class IchimokuRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class CombinedRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"
    include_ichimoku: bool = True
    include_rsi: bool = True
    include_macd: bool = True

class ScalpConfig(BaseModel):
    # تغییر example به json_schema_extra برای رفع هشدار
    symbol: str = Field(..., json_schema_extra={"example": "BTCUSDT"})
    timeframe: str = Field(default="5m", pattern="^(1m|5m|15m)$")
    risk_reward_ratio: float = Field(default=1.5, ge=1.0)
    leverage: int = Field(default=1, ge=1, le=125)

class SignalDetail(BaseModel):
    symbol: str
    signal: str
    confidence: float
    entry_price: float
    targets: List[float]
    stop_loss: float
    atr_volatility: float
    # استفاده از متد جدید برای زمان
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class SignalResponse(BaseModel):
    status: str
    count: int
    last_updated: str
    signals: List[Dict[str, Any]]
    sources: Dict[str, int]

# ==============================================================================
# PROFESSIONAL SCALPER ENGINE
# ==============================================================================

class ScalperEngine:
    """
    Professional Scalping Engine with ATR-based risk management
    Optimized for low latency and high-frequency trading
    """
    
    @staticmethod
    def calculate_atr_advanced(data: List, period: int = 14) -> Tuple[float, float, float]:
        """
        Calculate Average True Range (ATR) with multiple methods
        Returns: (ATR value, current price, volatility score)
        """
        try:
            if not data or len(data) < period:
                return 0.0, 0.0, 0.0
            
            closes = np.array([float(c[4]) for c in data[-period*2:]])
            highs = np.array([float(c[2]) for c in data[-period*2:]])
            lows = np.array([float(c[3]) for c in data[-period*2:]])
            current_price = closes[-1]
            
            if HAS_PANDAS_TA:
                # Use pandas_ta for professional ATR calculation
                import pandas as pd
                df = pd.DataFrame({
                    'high': highs,
                    'low': lows,
                    'close': closes
                })
                atr_series = ta.atr(df['high'], df['low'], df['close'], length=period)
                atr_value = float(atr_series.iloc[-1])
            else:
                # Manual ATR calculation
                tr = np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(
                        np.abs(highs[1:] - closes[:-1]),
                        np.abs(lows[1:] - closes[:-1])
                    )
                )
                atr_value = float(np.mean(tr[-period:])) if len(tr) >= period else 0.0
            
            # Calculate volatility score (0-100)
            volatility_score = 0.0
            if atr_value > 0 and current_price > 0:
                volatility_percent = (atr_value / current_price) * 100
                volatility_score = min(100.0, volatility_percent * 10)
            
            return atr_value, current_price, volatility_score
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def calculate_tdr_advanced(data: List, lookback_period: int = 20) -> float:
        """
        Calculate Trend Detection Ratio (TDR)
        Returns: 0.0 (ranging) to 1.0 (strong trend)
        """
        try:
            if not data or len(data) < lookback_period:
                return 0.0
            
            closes = np.array([float(c[4]) for c in data[-lookback_period:]])
            
            # Calculate moving averages
            sma_short = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
            sma_long = np.mean(closes) if len(closes) > 0 else closes[-1]
            
            # Calculate price slope
            if len(closes) >= 5:
                x = np.arange(len(closes[-5:]))
                y = closes[-5:]
                slope, _ = np.polyfit(x, y, 1)
                slope_strength = abs(slope) * 1000
            else:
                slope_strength = 0
            
            # Calculate price deviation from mean
            price_mean = np.mean(closes)
            price_std = np.std(closes) if len(closes) > 1 else 0
            deviation_ratio = price_std / price_mean if price_mean > 0 else 0
            
            # Combine factors
            ma_ratio = abs(sma_short - sma_long) / price_mean if price_mean > 0 else 0
            trend_score = min(1.0, (ma_ratio * 5 + min(slope_strength, 0.1) + deviation_ratio * 2) / 3)
            
            return float(trend_score)
            
        except Exception as e:
            logger.error(f"TDR calculation error: {e}")
            return 0.0
    
    @staticmethod
    def generate_pro_scalp_targets(entry_price: float, signal: str, atr_value: float, 
                                   risk_reward: float = 1.5) -> Tuple[List[float], float]:
        """
        Generate professional scalping targets based on ATR volatility
        """
        if entry_price <= 0 or atr_value <= 0:
            # Fallback to percentage-based targets
            if signal == "BUY":
                targets = [
                    round(entry_price * 1.003, 8),
                    round(entry_price * 1.006, 8),
                    round(entry_price * 1.009, 8)
                ]
                stop_loss = round(entry_price * 0.995, 8)
            elif signal == "SELL":
                targets = [
                    round(entry_price * 0.997, 8),
                    round(entry_price * 0.994, 8),
                    round(entry_price * 0.991, 8)
                ]
                stop_loss = round(entry_price * 1.005, 8)
            else:
                targets = [entry_price, entry_price, entry_price]
                stop_loss = entry_price
        else:
            # ATR-based dynamic targets
            if signal == "BUY":
                # Use ATR multiples for targets
                target_multipliers = [1.0, 1.8, 2.5]
                targets = [round(entry_price + (atr_value * mult), 8) for mult in target_multipliers]
                stop_loss = round(entry_price - (atr_value * risk_reward), 8)
            elif signal == "SELL":
                target_multipliers = [1.0, 1.8, 2.5]
                targets = [round(entry_price - (atr_value * mult), 8) for mult in target_multipliers]
                stop_loss = round(entry_price + (atr_value * risk_reward), 8)
            else:
                targets = [entry_price, entry_price, entry_price]
                stop_loss = entry_price
        
        return targets, stop_loss
    
    @staticmethod
    def get_ai_confirmation(symbol: str, signal: str, tdr_score: float, 
                           rsi: float, price: float, use_ai: bool = True) -> str:
        """
        AI Confirmation System - Can be connected to OpenAI/Gemini
        """
        if not use_ai:
            return "AI Analysis: Skipped by user request"
        
        # Internal intelligent decision logic
        if tdr_score < 0.25:
            return "AI Analysis: ⚠️ DO NOT TRADE. Market efficiency is too low (High noise)."
        
        if signal == "BUY":
            if rsi > 72:
                return f"AI Analysis: ⚠️ Risky Buy. RSI {rsi:.1f} shows overbought conditions."
            elif rsi > 65:
                return f"AI Analysis: ⚠️ Caution Buy. RSI {rsi:.1f} approaching overbought."
            else:
                return f"AI Analysis: ✅ Strong Bullish momentum confirmed at ${price:,.2f}."
        
        if signal == "SELL":
            if rsi < 28:
                return f"AI Analysis: ⚠️ Risky Sell. RSI {rsi:.1f} shows oversold conditions."
            elif rsi < 35:
                return f"AI Analysis: ⚠️ Caution Sell. RSI {rsi:.1f} approaching oversold."
            else:
                return f"AI Analysis: ✅ Bearish rejection confirmed at ${price:,.2f}."
        
        return "AI Analysis: ⏸️ Neutral market. Waiting for clear breakout."

# ==============================================================================
# Enhanced Target Calculation Helper
# ==============================================================================

def calculate_enhanced_targets(entry_price: float, signal: str, 
                               strategy: str = "DYNAMIC_ATR", 
                               atr_value: float = None,
                               risk_level: str = "MEDIUM") -> Dict[str, Any]:
    """
    Enhanced target calculation with multiple strategies
    """
    if entry_price <= 0:
        return {
            "targets": [0, 0, 0],
            "stop_loss": 0,
            "targets_percent": [0, 0, 0],
            "stop_loss_percent": 0
        }
    
    strategies = {
        "DYNAMIC_ATR": {
            "BUY": {"targets": [1.003, 1.008, 1.015], "stop_loss": 0.993},
            "SELL": {"targets": [0.997, 0.992, 0.985], "stop_loss": 1.007},
            "HOLD": {"targets": [1.002, 1.004, 1.006], "stop_loss": 0.998}
        },
        "AGGRESSIVE": {
            "BUY": {"targets": [1.008, 1.015, 1.025], "stop_loss": 0.990},
            "SELL": {"targets": [0.992, 0.985, 0.975], "stop_loss": 1.010},
            "HOLD": {"targets": [1.004, 1.008, 1.012], "stop_loss": 0.996}
        },
        "CONSERVATIVE": {
            "BUY": {"targets": [1.002, 1.004, 1.006], "stop_loss": 0.998},
            "SELL": {"targets": [0.998, 0.996, 0.994], "stop_loss": 1.002},
            "HOLD": {"targets": [1.001, 1.002, 1.003], "stop_loss": 0.999}
        }
    }
    
    # Select strategy based on risk level
    if risk_level == "HIGH":
        strategy = "AGGRESSIVE"
    elif risk_level == "LOW":
        strategy = "CONSERVATIVE"
    
    # Get parameters
    params = strategies.get(strategy, strategies["DYNAMIC_ATR"])
    signal_params = params.get(signal, params["HOLD"])
    
    # Apply ATR adjustment if available
    if atr_value and atr_value > 0 and entry_price > 0:
        atr_percent = (atr_value / entry_price)
        adjustment = min(max(atr_percent * 5, 0.5), 3.0)  # Limit adjustment
        
        adjusted_targets = [
            round(entry_price * (1 + (multiplier - 1) * adjustment), 8)
            for multiplier in signal_params["targets"]
        ]
        adjusted_stop = round(entry_price * (1 + (signal_params["stop_loss"] - 1) * adjustment), 8)
        
        targets = adjusted_targets
        stop_loss = adjusted_stop
    else:
        targets = [round(entry_price * multiplier, 8) for multiplier in signal_params["targets"]]
        stop_loss = round(entry_price * signal_params["stop_loss"], 8)
    
    # Calculate percentages
    targets_percent = [
        round(((target - entry_price) / entry_price) * 100, 2)
        for target in targets
    ]
    stop_loss_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
    
    return {
        "targets": targets,
        "stop_loss": stop_loss,
        "targets_percent": targets_percent,
        "stop_loss_percent": stop_loss_percent,
        "strategy_used": strategy
    }

# ==============================================================================
# FastAPI App with Enhanced Features
# ==============================================================================

API_VERSION = "8.0.0-PRO-SCALPER"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description="Professional Scalper Edition - ATR-based Risk Management & AI Confirmation",
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
# ENHANCED API ENDPOINTS
# ==============================================================================

@app.get("/")
async def read_root():
    return {
        "message": f"Crypto AI Trading System v{API_VERSION}",
        "status": "Active",
        "version": API_VERSION,
        "features": [
            "Professional Scalper Engine",
            "ATR-based Risk Management",
            "AI Confirmation System",
            "Multi-timeframe Analysis",
            "Ichimoku + RSI Divergence",
            "Low Latency Architecture"
        ],
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE,
            "pandas_ta": HAS_PANDAS_TA,
            "tdr_atr": HAS_TDR_ATR if UTILS_AVAILABLE else False
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "Healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "performance": {
            "latency": "low",
            "engine": "v8.0.0",
            "memory_usage": "optimized"
        },
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "pandas_available": HAS_PANDAS
        }
    }

@app.post("/api/v1/scalp/analyze", response_model=SignalDetail)
async def get_pro_scalp_signal(config: ScalpConfig):
    """
    Professional Scalp Signal Endpoint
    Combines Ichimoku, RSI Divergence, and ATR Risk Management
    """
    start_time = time.time()
    logger.info(f"🚀 PRO Scalp Request: {config.symbol} [{config.timeframe}] RR: {config.risk_reward_ratio}")
    
    try:
        # Get market data
        from api.utils import get_market_data_with_fallback
        data = get_market_data_with_fallback(config.symbol, config.timeframe, 100)
        
        if not data or len(data) < 50:
            raise HTTPException(status_code=400, detail="Insufficient data for professional scalp analysis")
        
        # Calculate technical indicators
        ichimoku = calculate_ichimoku_components(data)
        rsi_val = calculate_simple_rsi(data, 14)
        
        # Calculate ATR and TDR
        atr_value, current_price, volatility_score = ScalperEngine.calculate_atr_advanced(data)
        tdr_score = ScalperEngine.calculate_tdr_advanced(data)
        
        # Analyze Ichimoku signal
        ichi_signal = analyze_ichimoku_scalp_signal(ichimoku)
        
        # Filter signal with RSI confirmation
        final_signal = "HOLD"
        if ichi_signal['signal'] == 'BUY' and rsi_val < 70:
            final_signal = "BUY"
        elif ichi_signal['signal'] == 'SELL' and rsi_val > 30:
            final_signal = "SELL"
        
        # Calculate smart entry
        try:
            entry_price = calculate_smart_entry(data, final_signal)
            if entry_price <= 0:
                entry_price = current_price
        except:
            entry_price = current_price
        
        # Generate professional targets
        targets, stop_loss = ScalperEngine.generate_pro_scalp_targets(
            entry_price, final_signal, atr_value, config.risk_reward_ratio
        )
        
        # Prepare response
        response = {
            "symbol": config.symbol,
            "signal": final_signal,
            "confidence": min(0.95, ichi_signal['confidence'] * 0.9 + (1 - abs(rsi_val - 50)/50) * 0.1),
            "entry_price": round(entry_price, 8),
            "targets": targets,
            "stop_loss": stop_loss,
            "atr_volatility": round(atr_value, 8),
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "rsi": round(rsi_val, 2),
                "tdr_score": round(tdr_score, 3),
                "volatility_score": round(volatility_score, 1),
                "ichimoku_trend": ichimoku.get('trend_power', 50),
                "market_condition": "Trending" if tdr_score >= 0.25 else "Ranging"
            }
        }
        
        logger.info(f"✅ PRO Signal: {final_signal} for {config.symbol}")
        logger.info(f"   Entry: ${entry_price:.2f}, ATR: {atr_value:.4f}, TDR: {tdr_score:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ PRO Scalp Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Professional analysis error: {str(e)[:200]}")

@app.post("/api/analyze")
async def analyze_market(request: ScalpRequest):
    """
    Enhanced Analysis Endpoint with AI Confirmation
    Combines TDR, Ichimoku, RSI, and ATR-based risk management
    """
    start_time = time.time()
    
    try:
        # 1. Get market data
        data = get_market_data_with_fallback(request.symbol, request.timeframe, 100)
        if not data:
            raise HTTPException(status_code=500, detail="Failed to fetch market data")
        
        # 2. Calculate technical indicators
        tdr_score = ScalperEngine.calculate_tdr_advanced(data)
        
        if UTILS_AVAILABLE and HAS_TDR_ATR:
            ichi = get_ichimoku_full(data)
            atr = calculate_atr(data)
        else:
            ichi = calculate_ichimoku_components(data)
            atr, current_price, _ = ScalperEngine.calculate_atr_advanced(data)
        
        rsi_val = calculate_simple_rsi(data, 14)
        price = ichi.get("current_price", float(data[-1][4]) if data else 0)
        
        # 3. Strategy logic (combine Ichimoku and TDR)
        strategy_signal = "HOLD"
        if tdr_score >= 0.25:  # Only trade if market is trending
            tenkan = ichi.get("tenkan_sen", price)
            kijun = ichi.get("kijun_sen", price)
            cloud_top = ichi.get("cloud_top", price * 1.02)
            cloud_bottom = ichi.get("cloud_bottom", price * 0.98)
            
            if tenkan > kijun and price > cloud_top:
                strategy_signal = "BUY"
            elif tenkan < kijun and price < cloud_bottom:
                strategy_signal = "SELL"
        
        # 4. Dynamic stop loss and targets with ATR
        targets = []
        stop_loss = 0
        
        if strategy_signal == "BUY":
            targets = [
                round(price + (atr * 1.5), 8),
                round(price + (atr * 3.0), 8)
            ]
            stop_loss = round(price - (atr * 2.0), 8)
        elif strategy_signal == "SELL":
            targets = [
                round(price - (atr * 1.5), 8),
                round(price - (atr * 3.0), 8)
            ]
            stop_loss = round(price + (atr * 2.0), 8)
        
        # 5. AI Confirmation
        ai_advice = "AI skipped"
        if request.use_ai:
            ai_advice = ScalperEngine.get_ai_confirmation(
                request.symbol, strategy_signal, tdr_score, rsi_val, price, request.use_ai
            )
        
        # 6. Prepare response
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": strategy_signal,
            "price": round(price, 8),
            "tdr_status": "Trending" if tdr_score >= 0.25 else "Ranging (Avoid)",
            "tdr_value": round(tdr_score, 4),
            "rsi": round(rsi_val, 2),
            "ichimoku": {
                "tenkan": round(ichi.get("tenkan_sen", 0), 4),
                "kijun": round(ichi.get("kijun_sen", 0), 4),
                "cloud_top": round(ichi.get("cloud_top", 0), 4),
                "cloud_bottom": round(ichi.get("cloud_bottom", 0), 4),
                "trend_power": round(ichi.get("trend_power", 50), 1)
            },
            "risk_management": {
                "targets": targets,
                "stop_loss": stop_loss,
                "atr_volatility": round(atr, 8),
                "risk_reward_ratio": round((targets[0] - price) / (price - stop_loss), 2) if strategy_signal != "HOLD" and len(targets) > 0 and stop_loss > 0 else 0
            },
            "ai_confirmation": ai_advice,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION
        }
        
        logger.info(f"✅ Enhanced Analysis: {strategy_signal} for {request.symbol}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Enhanced analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)[:200]}")

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """
    Advanced Scalp Signal: Ichimoku + RSI Divergence + ATR Risk Management
    """
    start_time = time.time()
    logger.info(f"🚀 Advanced Scalp Request: {request.symbol} [{request.timeframe}] AI: {request.use_ai}")
    
    try:
        # 1. Get market data
        market_data_res = get_market_data_with_fallback(request.symbol, request.timeframe, 100, return_source=True)
        data = market_data_res["data"] if isinstance(market_data_res, dict) else market_data_res
        
        if not data or len(data) < 50:
            raise HTTPException(status_code=400, detail="Insufficient data for scalp analysis")
        
        # 2. Combined technical analysis
        ichimoku = calculate_ichimoku_components(data)
        rsi_val = calculate_simple_rsi(data, 14)
        atr, current_price, volatility_score = ScalperEngine.calculate_atr_advanced(data)
        tdr_score = ScalperEngine.calculate_tdr_advanced(data)
        
        # 3. Signal detection with Ichimoku and RSI filter
        ichi_signal = analyze_ichimoku_scalp_signal(ichimoku)
        
        final_signal = "HOLD"
        confidence = 0.5
        
        if ichi_signal['signal'] == 'BUY' and rsi_val < 70:
            if rsi_val < 40:
                final_signal = "BUY"
                confidence = max(0.7, ichi_signal['confidence'])
            elif rsi_val < 60:
                final_signal = "BUY"
                confidence = ichi_signal['confidence'] * 0.9
        elif ichi_signal['signal'] == 'SELL' and rsi_val > 30:
            if rsi_val > 60:
                final_signal = "SELL"
                confidence = max(0.7, ichi_signal['confidence'])
            elif rsi_val > 40:
                final_signal = "SELL"
                confidence = ichi_signal['confidence'] * 0.9
        
        # 4. Calculate smart entry and targets
        try:
            entry_price = calculate_smart_entry(data, final_signal)
            if entry_price <= 0:
                entry_price = current_price
        except:
            entry_price = current_price
        
        # Determine risk level based on volatility
        if volatility_score > 70:
            risk_level = "HIGH"
        elif volatility_score > 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate enhanced targets
        target_data = calculate_enhanced_targets(
            entry_price, final_signal, 
            "DYNAMIC_ATR", atr, risk_level
        )
        
        # 5. AI Confirmation
        ai_advice = ""
        if request.use_ai:
            ai_advice = ScalperEngine.get_ai_confirmation(
                request.symbol, final_signal, tdr_score, rsi_val, entry_price, request.use_ai
            )
        
        # 6. Prepare comprehensive response
        response = {
            "symbol": request.symbol.upper(),
            "timeframe": request.timeframe,
            "signal": final_signal,
            "confidence": round(confidence, 3),
            "entry_price": round(entry_price, 8),
            "current_price": round(current_price, 8),
            "targets": target_data["targets"],
            "stop_loss": target_data["stop_loss"],
            "targets_percent": target_data["targets_percent"],
            "stop_loss_percent": target_data["stop_loss_percent"],
            "analysis": {
                "rsi": round(rsi_val, 2),
                "atr_volatility": round(atr, 8),
                "tdr_score": round(tdr_score, 4),
                "volatility_score": round(volatility_score, 1),
                "ichimoku_trend": round(ichimoku.get('trend_power', 50), 1),
                "market_condition": "Trending" if tdr_score >= 0.25 else "Ranging",
                "risk_level": risk_level,
                "strategy_used": target_data["strategy_used"]
            },
            "ai_confirmation": ai_advice,
            "execution_metrics": {
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "data_points": len(data),
                "signal_strength": "STRONG" if confidence > 0.75 else "MODERATE" if confidence > 0.6 else "WEAK"
            },
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION
        }
        
        logger.info(f"✅ Advanced Scalp: {final_signal} for {request.symbol}")
        logger.info(f"   Confidence: {confidence:.2%}, Risk: {risk_level}, ATR: {atr:.4f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Advanced scalp error: {e}", exc_info=True)
        
        # Comprehensive fallback
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": 0,
            "current_price": 0,
            "targets": [0, 0, 0],
            "stop_loss": 0,
            "analysis": {
                "rsi": 50,
                "atr_volatility": 0,
                "tdr_score": 0,
                "error": str(e)[:100]
            },
            "ai_confirmation": "Analysis failed - using fallback",
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION,
            "status": "FALLBACK"
        }

# Keep existing endpoints with enhancements
@app.post("/api/ichimoku-scalp")
async def get_ichimoku_scalp_signal(request: IchimokuRequest):
    """Enhanced Ichimoku Scalp Endpoint with ATR integration"""
    # ... (existing code with ATR enhancements) ...

@app.get("/api/v1/scan-all/{symbol}")
async def scan_all_timeframes_pro(symbol: str):
    """Professional multi-timeframe scanner"""
    timeframes = ["1m", "5m", "15m", "1h", "4h"]
    results = []
    
    for tf in timeframes:
        try:
            # Use appropriate strategy based on timeframe
            if tf in ["1m", "5m", "15m"]:
                response = await get_scalp_signal(ScalpRequest(symbol=symbol, timeframe=tf, use_ai=False))
            else:
                response = await get_ichimoku_scalp_signal(IchimokuRequest(symbol=symbol, timeframe=tf))
            
            response["timeframe"] = tf
            
            # Add ATR analysis
            data = get_market_data_with_fallback(symbol, tf, 50)
            if data:
                atr, _, vol_score = ScalperEngine.calculate_atr_advanced(data)
                response["atr_analysis"] = {
                    "atr_value": round(atr, 8),
                    "volatility_score": round(vol_score, 1)
                }
            
            results.append(response)
            
        except Exception as e:
            logger.error(f"Error scanning {symbol} on {tf}: {e}")
            results.append({
                "symbol": symbol,
                "timeframe": tf,
                "signal": "ERROR",
                "error": str(e)[:100]
            })
    
    return {
        "symbol": symbol,
        "scanned_at": datetime.now().isoformat(),
        "timeframes_analyzed": len(timeframes),
        "signals_summary": {
            "BUY": len([r for r in results if r.get("signal") == "BUY"]),
            "SELL": len([r for r in results if r.get("signal") == "SELL"]),
            "HOLD": len([r for r in results if r.get("signal") == "HOLD"]),
            "ERROR": len([r for r in results if r.get("signal") == "ERROR"])
        },
        "results": results,
        "recommendation": "Focus on timeframes with consistent signals"
    }

# ==============================================================================
# Performance Monitoring
# ==============================================================================

class PerformanceMonitor:
    """Monitor system performance for low-latency trading"""
    
    def __init__(self):
        self.request_times = []
        self.avg_latency = 0
    
    def record_request(self, processing_time: float):
        self.request_times.append(processing_time)
        if len(self.request_times) > 100:
            self.request_times.pop(0)
        self.avg_latency = np.mean(self.request_times) if self.request_times else 0

performance_monitor = PerformanceMonitor()

@app.middleware("http")
async def monitor_performance(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time
    performance_monitor.record_request(processing_time)
    
    # Add performance header
    response.headers["X-Processing-Time"] = str(round(processing_time * 1000, 2))
    response.headers["X-Avg-Latency"] = str(round(performance_monitor.avg_latency * 1000, 2))
    
    return response

@app.get("/api/v1/performance")
async def get_performance_stats():
    """Get system performance statistics"""
    return {
        "average_latency_ms": round(performance_monitor.avg_latency * 1000, 2),
        "total_requests": len(performance_monitor.request_times),
        "requests_per_second": round(len(performance_monitor.request_times) / 60, 1) if performance_monitor.request_times else 0,
        "performance_grade": "A" if performance_monitor.avg_latency < 0.1 else "B" if performance_monitor.avg_latency < 0.5 else "C",
        "timestamp": datetime.now().isoformat()
    }

# ==============================================================================
# Startup and Main
# ==============================================================================

from contextlib import asynccontextmanager

# ۱. تعریف تابع Lifespan برای مدیریت شروع و پایان برنامه
@asynccontextmanager
async def lifespan(app: FastAPI):
    """مدیریت رویدادهای شروع و پایان برنامه (جایگزین startup_event)"""
    # کدهایی که هنگام شروع (Startup) اجرا می‌شوند:
    logger.info(f"🚀 Starting Crypto AI Trading System v{API_VERSION}")
    logger.info(f"📦 Utils Available: {UTILS_AVAILABLE}")
    logger.info(f"📦 Pandas TA: {HAS_PANDAS_TA}")
    
    # بررسی وجود توابع TDR/ATR در ماژول utils
    has_tdr_atr = hasattr(utils, 'calculate_tdr') if UTILS_AVAILABLE else False
    logger.info(f"📦 TDR/ATR Functions: {has_tdr_atr}")
    logger.info(f"⚡ Performance Mode: Optimized for Scalping")
    
    print(f"\n{'=' * 60}")
    print(f"PRO SCALPER EDITION v{API_VERSION}")
    print(f"{'=' * 60}")
    print("Features:")
    print("  • Professional Scalper Engine")
    print("  • ATR-based Risk Management")
    print("  • AI Confirmation System")
    print("  • Multi-timeframe Analysis")
    print("  • Low Latency Architecture")
    print(f"{'=' * 60}")
    print(f"API Documentation: /docs")  # در FastAPI پورتال داکیومنت معمولاً در /docs است
    print(f"Health Check: /health")
    print(f"{'=' * 60}\n")
    
    yield  # در این نقطه برنامه شروع به کار می‌کند
    
    # کدهایی که هنگام بسته شدن (Shutdown) اجرا می‌شوند:
    logger.info("👋 Shutting down Crypto AI Trading System")

# ۲. معرفی lifespan به اپلیکیشن FastAPI
# این خط را پیدا کن و به این شکل تغییر بده:
app = FastAPI(title="Pro Crypto AI Scalper", lifespan=lifespan)
    
    logger.info("✅ System startup completed successfully!")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Optimized server configuration for scalping
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        workers=2,  # Multi-core processing
        loop="asyncio",  # Async optimization
        timeout_keep_alive=30
    )
