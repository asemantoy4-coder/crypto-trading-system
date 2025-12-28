import os
import sys
import time
import uvicorn
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from typing import List, Optional
import numpy as np

# اضافه کردن مسیر جاری به پایتون برای پیدا کردن فایل‌ها
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# CONFIGURATION
# ==============================================================================

API_VERSION = "8.1.0"
DEBUG_MODE = os.environ.get("DEBUG", "False").lower() == "true"

# Setup logging
logging.basicConfig(
    level=logging.INFO if not DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# MODULE IMPORTS
# ==============================================================================

# بررسی وجود ماژول‌های ضروری
try:
    import pandas as pd
    import pandas_ta as ta # اضافه کردن کلمه as ta
    HAS_PANDAS = True
    HAS_PANDAS_TA = True
    logger.info("✅ Pandas and Pandas_TA integrated successfully")
except ImportError:
    HAS_PANDAS = False
    HAS_PANDAS_TA = False
    logger.warning("⚠️ Critical technical libraries missing")

# بررسی ماژول‌های سفارشی
try:
    # Import from local modules
    from utils import (
        format_binance_price,
        get_enhanced_scalp_signal,
        get_market_data_with_fallback,
        calculate_simple_rsi,
        get_momentum_logic,
        calculate_smart_entry
    )
    UTILS_AVAILABLE = True
    logger.info("✅ Utils module loaded successfully")
except ImportError as e:
    logger.error(f"❌ Error importing utils: {e}")
    UTILS_AVAILABLE = False

# بررسی ماژول ScalperEngine
try:
    from scalper_engine import ScalperEngine
    COLLECTORS_AVAILABLE = True
    logger.info("✅ ScalperEngine loaded successfully")
except ImportError as e:
    logger.error(f"❌ Error importing ScalperEngine: {e}")
    COLLECTORS_AVAILABLE = False
    ScalperEngine = type('ScalperEngine', (), {})

# بررسی ماژول data_collector
try:
    from data_collector import fetch_binance_klines, convert_to_dataframe
    DATA_COLLECTOR_AVAILABLE = True
except ImportError:
    DATA_COLLECTOR_AVAILABLE = False

# بررسی توابع تخصصی
HAS_TDR_ATR = hasattr(ScalperEngine, 'calculate_tdr_advanced') if COLLECTORS_AVAILABLE else False

# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="Crypto AI Trading System",
    description="Professional Scalping & Trading Analysis API",
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# ==============================================================================
# MODELS
# ==============================================================================

from pydantic import BaseModel

class ScalpRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    use_ai: bool = True

class IchimokuRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"

class CombinedRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    strategy: str = "combined"

class SignalDetail(BaseModel):
    symbol: str
    signal: str
    entry_price: float
    stop_loss: float
    targets: List[float]
    momentum_score: float
    user_message: str
    is_risky_for_retail: bool
    execution_type: str
    signal_id: str
    timestamp: str

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
            "Low Latency Architecture",
            "Persian User Interface"
        ],
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE,
            "pandas_ta": HAS_PANDAS_TA,
            "tdr_atr": HAS_TDR_ATR if UTILS_AVAILABLE else False
        },
        "endpoints": {
            "health": "/api/health",
            "analyze_v1": "/api/v1/analyze",
            "analyze_enhanced": "/api/analyze",
            "market_scan": "/api/v1/market-scan",
            "performance": "/api/v1/performance"
        }
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "Healthy",
        "version": API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "performance": {
            "latency": "low",
            "engine": API_VERSION,
            "memory_usage": "optimized"
        },
        "modules": {
            "utils": UTILS_AVAILABLE,
            "scalper_engine": COLLECTORS_AVAILABLE,
            "pandas": HAS_PANDAS,
            "pandas_ta": HAS_PANDAS_TA
        }
    }


@app.post("/api/v1/analyze", response_model=SignalDetail)
async def analyze_pair(request: ScalpRequest):
    """
    Advanced Analysis Endpoint with Momentum Logic and Tight Stop Loss
    Uses enhanced utils module for professional analysis
    """
    start_time = time.time()
    logger.info(f"🚀 Advanced Analysis Request: {request.symbol} [{request.timeframe}]")
    
    try:
        if not UTILS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="ماژول تحلیلی در دسترس نیست. لطفا بعداً تلاش کنید."
            )
        
        # دریافت داده‌های بازار
        data = get_market_data_with_fallback(request.symbol, request.timeframe, 100)
        if not data or len(data) < 20:
            raise HTTPException(
                status_code=400,
                detail="دیتای کافی برای تحلیل وجود ندارد"
            )
        
        # استفاده از تابع پیشرفته تحلیل از utils
        analysis_result = get_enhanced_scalp_signal(data, request.symbol, request.timeframe)
        
        # فرمت‌بندی قیمت‌ها برای بایننس
        current_price = float(data[-1][4])
        clean_entry = format_binance_price(current_price, request.symbol)
        clean_sl = format_binance_price(analysis_result.get("stop_loss", current_price * 0.9985), request.symbol)
        clean_targets = [
            format_binance_price(target, request.symbol)
            for target in analysis_result.get("targets", [])
        ]
        
        # محاسبه شتاب (Momentum)
        close_prices = [float(c[4]) for c in data[-10:]]
        final_signal = analysis_result.get("signal", "HOLD")
        roc, persian_msg, risky = get_momentum_logic(close_prices, final_signal)
        
        # AI تایید (در صورت درخواست)
        ai_advice = ""
        if request.use_ai and COLLECTORS_AVAILABLE:
            try:
                rsi_val = calculate_simple_rsi(data, 14)
                tdr_score = ScalperEngine.calculate_tdr_advanced(data) if HAS_TDR_ATR else 0.5
                ai_advice = ScalperEngine.get_ai_confirmation(
                    request.symbol, final_signal, tdr_score, rsi_val, current_price, True
                )
                persian_msg = f"{persian_msg} | {ai_advice}"
            except:
                pass
        
        # خروجی نهایی
        return {
            "symbol": request.symbol,
            "signal": final_signal,
            "entry_price": clean_entry,
            "stop_loss": clean_sl,
            "targets": clean_targets,
            "momentum_score": roc,
            "user_message": persian_msg,
            "is_risky_for_retail": risky,
            "execution_type": "MARKET" if risky else "LIMIT",
            "signal_id": f"{request.symbol}_{int(time.time())}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in analyze_pair: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"خطا در تحلیل دیتای صرافی: {str(e)[:100]}"
        )


@app.post("/api/analyze")
async def analyze_market(request: ScalpRequest):
    """
    Enhanced Analysis Endpoint using utils module
    """
    start_time = time.time()
    logger.info(f"📊 Enhanced Analysis Request: {request.symbol} [{request.timeframe}]")
    
    try:
        if not UTILS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Utils module not available. System maintenance in progress."
            )
        
        # دریافت داده‌های بازار
        data = get_market_data_with_fallback(request.symbol, request.timeframe, 100)
        if not data or len(data) < 20:
            raise HTTPException(
                status_code=400,
                detail="Insufficient market data for analysis"
            )
        
        # استفاده از تابع پیشرفته تحلیل از utils
        result = get_enhanced_scalp_signal(data, request.symbol, request.timeframe)
        
        # محاسبه شاخص‌های اضافی
        rsi_val = calculate_simple_rsi(data, 14)
        current_price = float(data[-1][4])
        
        # محاسبه ATR (در صورت موجود بودن)
        atr_value = 0
        volatility_score = 50
        if COLLECTORS_AVAILABLE:
            try:
                atr_value, _, volatility_score = ScalperEngine.calculate_atr_advanced(data)
            except:
                pass
        
        # AI تایید (در صورت درخواست)
        ai_advice = "AI analysis skipped"
        if request.use_ai and COLLECTORS_AVAILABLE:
            try:
                tdr_score = ScalperEngine.calculate_tdr_advanced(data) if HAS_TDR_ATR else 0.5
                ai_advice = ScalperEngine.get_ai_confirmation(
                    request.symbol, 
                    result.get("signal", "HOLD"), 
                    tdr_score, 
                    rsi_val, 
                    current_price, 
                    request.use_ai
                )
            except Exception as e:
                ai_advice = f"AI error: {str(e)[:50]}"
        
        # محاسبه نسبت ریسک به ریوارد
        stop_loss = result.get("stop_loss", current_price)
        targets = result.get("targets", [])
        risk_reward = 0
        if len(targets) > 0 and stop_loss > 0 and current_price > 0:
            if result.get("signal") == "BUY":
                risk = current_price - stop_loss
                reward = targets[0] - current_price
                if risk > 0:
                    risk_reward = round(reward / risk, 2)
            elif result.get("signal") == "SELL":
                risk = stop_loss - current_price
                reward = current_price - targets[0]
                if risk > 0:
                    risk_reward = round(reward / risk, 2)
        
        # آماده‌سازی پاسخ
        response = {
            "symbol": request.symbol.upper(),
            "timeframe": request.timeframe,
            "signal": result.get("signal", "HOLD"),
            "confidence": result.get("confidence", 0.5),
            "price": current_price,
            "entry_price": result.get("entry_price", current_price),
            "tdr_status": "Trending" if result.get("confidence", 0) > 0.6 else "Ranging",
            "tdr_value": round(result.get("confidence", 0.5) * 100, 2),
            "rsi": round(rsi_val, 2),
            "momentum": result.get("momentum_message", ""),
            "risk_management": {
                "targets": targets,
                "stop_loss": stop_loss,
                "atr_volatility": round(atr_value, 8),
                "volatility_score": round(volatility_score, 1),
                "risk_reward_ratio": risk_reward,
                "max_risk_percent": result.get("stop_loss_percent", 1.5)
            },
            "ai_confirmation": ai_advice,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": API_VERSION,
            "source": "Enhanced Utils Analysis"
        }
        
        logger.info(f"✅ Enhanced Analysis: {result.get('signal', 'HOLD')} for {request.symbol}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Enhanced analysis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis error: {str(e)[:200]}"
        )


# ==============================================================================
# MARKET SCANNER
# ==============================================================================

@app.get("/api/v1/market-scan")
async def market_scanner():
    """
    اسکنر بازار برای پیشنهادات لحظه‌ای
    """
    try:
        if not UTILS_AVAILABLE:
            return {
                "status": "warning",
                "message": "ماژول تحلیل در دسترس نیست",
                "data": [],
                "server_time": datetime.now(timezone.utc).isoformat()
            }
        
        # نمادهای محبوب برای اسکن
        popular_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        top_picks = []
        
        for symbol in popular_symbols[:3]:  # فقط ۳ نماد اول برای سرعت
            try:
                data = get_market_data_with_fallback(symbol, "1h", 50)
                if data and len(data) >= 20:
                    result = get_enhanced_scalp_signal(data, symbol, "1h")
                    
                    # طبقه‌بندی وضعیت
                    status = "NEUTRAL"
                    if result.get("signal") == "BUY":
                        status = "HOT" if result.get("confidence", 0) > 0.7 else "WATCH"
                    elif result.get("signal") == "SELL":
                        status = "COLD" if result.get("confidence", 0) > 0.7 else "CAUTION"
                    
                    top_picks.append({
                        "symbol": symbol,
                        "trend_1h": "صعودی" if result.get("signal") == "BUY" else "نزولی" if result.get("signal") == "SELL" else "رنج",
                        "signal": result.get("signal", "HOLD"),
                        "suggestion": result.get("momentum_message", "تحلیل در دسترس نیست"),
                        "ai_score": round(result.get("confidence", 0.5) * 100),
                        "status": status,
                        "confidence": round(result.get("confidence", 0.5), 2)
                    })
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        return {
            "status": "success",
            "data": top_picks,
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "total_scanned": len(popular_symbols),
            "successful_scans": len(top_picks)
        }
        
    except Exception as e:
        logger.error(f"Market scanner error: {e}")
        return {
            "status": "error",
            "message": f"خطا در اسکن بازار: {str(e)[:100]}",
            "data": [],
            "server_time": datetime.now(timezone.utc).isoformat()
        }


@app.get("/api/v1/scan-all/{symbol}")
async def scan_all_timeframes_pro(symbol: str):
    """Professional multi-timeframe scanner using utils module"""
    timeframes = ["1m", "5m", "15m", "1h", "4h"]
    results = []
    
    for tf in timeframes:
        try:
            data = get_market_data_with_fallback(symbol, tf, 50)
            if data and len(data) >= 20:
                result = get_enhanced_scalp_signal(data, symbol, tf)
                
                # محاسبه ATR
                atr_value = 0
                volatility_score = 50
                if COLLECTORS_AVAILABLE:
                    try:
                        atr_value, _, volatility_score = ScalperEngine.calculate_atr_advanced(data)
                    except:
                        pass
                
                # اضافه کردن تحلیل ATR
                result["atr_analysis"] = {
                    "atr_value": round(atr_value, 8),
                    "volatility_score": round(volatility_score, 1)
                }
                result["timeframe"] = tf
                
                results.append(result)
            else:
                results.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "signal": "ERROR",
                    "error": "دیتای کافی وجود ندارد"
                })
                
        except Exception as e:
            logger.error(f"Error scanning {symbol} on {tf}: {e}")
            results.append({
                "symbol": symbol,
                "timeframe": tf,
                "signal": "ERROR",
                "error": str(e)[:100]
            })
    
    # خلاصه سیگنال‌ها
    signals_summary = {
        "BUY": len([r for r in results if r.get("signal") == "BUY"]),
        "SELL": len([r for r in results if r.get("signal") == "SELL"]),
        "HOLD": len([r for r in results if r.get("signal") == "HOLD"]),
        "ERROR": len([r for r in results if r.get("signal") == "ERROR"])
    }
    
    # پیشنهاد کلی
    recommendation = "HOLD"
    if signals_summary["BUY"] >= 3:
        recommendation = "STRONG_BUY"
    elif signals_summary["SELL"] >= 3:
        recommendation = "STRONG_SELL"
    elif signals_summary["BUY"] >= 2:
        recommendation = "BUY"
    elif signals_summary["SELL"] >= 2:
        recommendation = "SELL"
    
    return {
        "symbol": symbol,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "timeframes_analyzed": len(timeframes),
        "signals_summary": signals_summary,
        "overall_recommendation": recommendation,
        "recommendation_logic": f"بر اساس تطابق {signals_summary['BUY']} تایم‌فرم خرید و {signals_summary['SELL']} تایم‌فرم فروش",
        "results": results
    }


# ==============================================================================
# PERFORMANCE MONITORING
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
    
    # Add performance headers
    response.headers["X-Processing-Time"] = str(round(processing_time * 1000, 2))
    response.headers["X-Avg-Latency"] = str(round(performance_monitor.avg_latency * 1000, 2))
    response.headers["X-API-Version"] = API_VERSION
    
    return response


@app.get("/api/v1/performance")
async def get_performance_stats():
    """Get system performance statistics"""
    latency_ms = round(performance_monitor.avg_latency * 1000, 2)
    
    # تعیین درجه عملکرد
    if latency_ms < 50:
        grade = "A+"
    elif latency_ms < 100:
        grade = "A"
    elif latency_ms < 200:
        grade = "B"
    elif latency_ms < 500:
        grade = "C"
    else:
        grade = "D"
    
    return {
        "average_latency_ms": latency_ms,
        "total_requests": len(performance_monitor.request_times),
        "requests_per_minute": round(len(performance_monitor.request_times) / 5, 1) if len(performance_monitor.request_times) > 0 else 0,
        "performance_grade": grade,
        "system_status": "Optimal" if grade in ["A+", "A"] else "Good" if grade == "B" else "Needs Attention",
        "modules": {
            "utils": UTILS_AVAILABLE,
            "scalper_engine": COLLECTORS_AVAILABLE,
            "pandas_ta": HAS_PANDAS_TA
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ==============================================================================
# COMPATIBILITY ENDPOINTS (برای نسخه‌های قدیمی)
# ==============================================================================

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """Legacy endpoint for backward compatibility"""
    return await analyze_market(request)


@app.post("/api/ichimoku-scalp")
async def get_ichimoku_scalp_signal(request: IchimokuRequest):
    """Legacy Ichimoku endpoint"""
    return {
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "signal": "HOLD",
        "message": "این endpoint به /api/analyze منتقل شده است",
        "redirect": "/api/analyze",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ==============================================================================
# STARTUP AND MAIN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info(f"🚀 Starting Crypto AI Trading System v{API_VERSION}")
    logger.info(f"📦 Utils Available: {UTILS_AVAILABLE}")
    logger.info(f"📦 Pandas TA: {HAS_PANDAS_TA}")
    logger.info(f"📦 ScalperEngine: {COLLECTORS_AVAILABLE}")
    logger.info(f"⚡ Performance Mode: Optimized for Scalping")
    
    print(f"\n{'=' * 60}")
    print(f"PRO SCALPER EDITION v{API_VERSION}")
    print(f"{'=' * 60}")
    print("Features:")
    print("  • Professional Scalper Engine")
    print("  • ATR-based Risk Management")
    print("  • Enhanced Utils Integration")
    print("  • Multi-timeframe Analysis")
    print("  • Low Latency Architecture")
    print("  • Persian User Interface")
    print(f"{'=' * 60}")
    print(f"API Documentation: /api/docs")
    print(f"Health Check: /api/health")
    print(f"Performance Stats: /api/v1/performance")
    print(f"Market Scanner: /api/v1/market-scan")
    print(f"{'=' * 60}\n")
    
    logger.info("✅ System startup completed successfully!")


# ==============================================================================
# EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # تنظیم پورت برای Render و داکر
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info" if DEBUG_MODE else "warning",
        access_log=True
    )
