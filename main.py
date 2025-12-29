import os
import sys
import time
import uvicorn
import logging
import random
import pandas as pd
import requests
import yfinance as yf
import pandas_ta as ta
import numpy as np
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Final, Dict, Any
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

API_VERSION: Final[str] = "9.1.0"
DEBUG_MODE: bool = os.environ.get("DEBUG", "false").lower() == "true"

# تنظیمات لاگینگ
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CryptoAIScalperPro")

# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="Crypto AI Scalper Pro",
    description="Advanced Technical Analysis & Trading Signals API",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    debug=DEBUG_MODE
)

# تنظیمات CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# MODELS
# ==============================================================================

class AnalysisRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    client_id: str = "guest"

class AdminRequest(BaseModel):
    admin_key: str
    action: str
    target_client_id: str = None

# ==============================================================================
# DATABASES
# ==============================================================================

active_users: Dict[str, Dict[str, Any]] = {}
blacklisted_ips: set = set()
ADMIN_SECRET: str = os.environ.get("ADMIN_SECRET", "SECRET_ADMIN_123")
REQUEST_LIMIT_PER_MINUTE: int = 60

# کش برای کاهش درخواست‌های تکراری
analysis_cache: Dict[str, Dict[str, Any]] = {}
CACHE_DURATION = 30  # ثانیه

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def clear_old_cache():
    """پاک کردن کش‌های قدیمی"""
    current_time = time.time()
    keys_to_delete = []
    
    for key, value in analysis_cache.items():
        if current_time - value.get('timestamp', 0) > CACHE_DURATION:
            keys_to_delete.append(key)
    
    for key in keys_to_delete:
        del analysis_cache[key]

def check_rate_limit(client_id: str) -> bool:
    """بررسی محدودیت نرخ درخواست"""
    current_time = time.time()
    minute_ago = current_time - 60
    
    if client_id in active_users:
        user_data = active_users[client_id]
        request_times = user_data.get('request_times', [])
        
        # حذف درخواست‌های قدیمی
        request_times = [t for t in request_times if t > minute_ago]
        
        if len(request_times) >= REQUEST_LIMIT_PER_MINUTE:
            return False
        
        request_times.append(current_time)
        active_users[client_id]['request_times'] = request_times
    
    return True

def get_binance_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """دریافت داده از بایننس"""
    try:
        # مپ کردن timeframe به فرمت بایننس
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1h", "4h": "4h", "1d": "1d"
        }
        
        binance_interval = interval_map.get(timeframe, "5m")
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={binance_interval}&limit=100"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # تبدیل به عدد
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Binance data for {symbol}: {e}")
        raise

def get_yahoo_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """دریافت داده از Yahoo Finance"""
    try:
        # تنظیم period بر اساس timeframe
        period_map = {
            "1m": "1d", "5m": "5d", "15m": "5d",
            "1h": "1mo", "4h": "3mo", "1d": "6mo"
        }
        
        period = period_map.get(timeframe, "5d")
        
        # دانلود داده
        df = yf.download(
            symbol,
            period=period,
            interval=timeframe,
            progress=False,
            auto_adjust=True
        )
        
        if df.empty:
            raise ValueError("No data received from Yahoo Finance")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
        raise

# ==============================================================================
# CORE ANALYSIS ENGINE
# ==============================================================================

def get_on_demand_analysis(symbol: str, timeframe: str) -> Dict[str, Any]:
    """موتور تحلیل تکنیکال پیشرفته"""
    
    # بررسی کش
    cache_key = f"{symbol}_{timeframe}"
    if cache_key in analysis_cache:
        cache_data = analysis_cache[cache_key]
        if time.time() - cache_data.get('timestamp', 0) < CACHE_DURATION:
            return cache_data['data']
    
    try:
        logger.info(f"Analyzing {symbol} on {timeframe} timeframe")
        
        # ۱. دریافت داده‌ها
        is_forex = any(x in symbol.upper() for x in ["EURUSD", "GBPUSD", "XAU", "GOLD", "US30", "NAS100"])
        
        if is_forex:
            # نمادهای فارکس
            yf_symbol = "GC=F" if any(x in symbol.upper() for x in ["XAU", "GOLD"]) else f"{symbol}=X"
            df = get_yahoo_data(yf_symbol, timeframe)
        else:
            # نمادهای کریپتو
            df = get_binance_data(symbol, timeframe)
        
        if df.empty or len(df) < 50:
            return {"error": "Insufficient data for analysis", "symbol": symbol}
        
        # ۲. محاسبه اندیکاتورهای پیشرفته
        # RSI
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # EMA ها
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        df['EMA_200'] = ta.ema(df['close'], length=200)
        
        # استوکاستیک
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        df = pd.concat([df, stoch], axis=1)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        df = pd.concat([df, bb], axis=1)
        
        # ATR برای نوسان
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # ۳. تحلیل روند
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # وضعیت RSI
        rsi_val = last_row['RSI']
        rsi_status = "خرید اشباع" if rsi_val > 70 else "فروش اشباع" if rsi_val < 30 else "خنثی"
        
        # وضعیت استوکاستیک
        k_val = last_row.get('STOCHk_14_3_3', 50)
        d_val = last_row.get('STOCHd_14_3_3', 50)
        stoch_cross = "صعودی" if k_val > d_val and prev_row.get('STOCHk_14_3_3', 0) <= prev_row.get('STOCHd_14_3_3', 0) else \
                     "نزولی" if k_val < d_val and prev_row.get('STOCHk_14_3_3', 0) >= prev_row.get('STOCHd_14_3_3', 0) else "خنثی"
        
        # وضعیت MACD
        macd_val = last_row.get('MACD_12_26_9', 0)
        macd_signal = last_row.get('MACDs_12_26_9', 0)
        macd_trend = "صعودی" if macd_val > macd_signal else "نزولی"
        
        # وضعیت بولینگر
        bb_upper = last_row.get('BBU_20_2.0', last_row['close'])
        bb_lower = last_row.get('BBL_20_2.0', last_row['close'])
        bb_position = "بالای باند" if last_row['close'] > bb_upper else "پایین باند" if last_row['close'] < bb_lower else "درون باند"
        
        # ۴. سیستم امتیازدهی
        score = 50  # امتیاز پایه
        
        # امتیاز RSI
        if rsi_val < 40: score += 15
        elif rsi_val > 60: score -= 15
        
        # امتیاز روند EMA
        if last_row['EMA_20'] > last_row['EMA_50'] > last_row['EMA_200']: score += 20
        elif last_row['EMA_20'] < last_row['EMA_50'] < last_row['EMA_200']: score -= 20
        
        # امتیاز استوکاستیک
        if k_val < 20 and d_val < 20: score += 10  # اشباع فروش
        elif k_val > 80 and d_val > 80: score -= 10  # اشباع خرید
        
        # ۵. تصمیم نهایی
        if score >= 70:
            decision = "خرید قوی (Strong Buy) 🚀"
            color = "green"
            confidence = "high"
        elif score >= 60:
            decision = "خرید (Buy) 📈"
            color = "lightgreen"
            confidence = "medium"
        elif score <= 30:
            decision = "فروش قوی (Strong Sell) 📉"
            color = "red"
            confidence = "high"
        elif score <= 40:
            decision = "فروش (Sell) 🔻"
            color = "orange"
            confidence = "medium"
        else:
            decision = "صبر (Neutral) ⏸️"
            color = "gray"
            confidence = "low"
        
        # ۶. آماده‌سازی نتیجه
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": f"{last_row['close']:,.5f}" if last_row['close'] < 1 else f"{last_row['close']:,.2f}",
            "price_raw": float(last_row['close']),
            "indicators": {
                "rsi": round(rsi_val, 2),
                "rsi_status": rsi_status,
                "stoch_k": round(k_val, 2),
                "stoch_d": round(d_val, 2),
                "stoch_cross": stoch_cross,
                "ema_trend": "صعودی" if last_row['EMA_20'] > last_row['EMA_50'] else "نزولی",
                "macd_trend": macd_trend,
                "bb_position": bb_position,
                "atr": round(float(last_row['ATR']), 4) if pd.notna(last_row['ATR']) else 0,
                "volume": int(last_row['volume']) if pd.notna(last_row['volume']) else 0
            },
            "analysis": {
                "score": score,
                "final_decision": decision,
                "color": color,
                "confidence": confidence,
                "trend_strength": "قوی" if abs(score - 50) > 25 else "متوسط" if abs(score - 50) > 15 else "ضعیف",
                "risk_level": "پایین" if confidence == "high" and abs(score - 50) > 30 else "متوسط" if confidence == "medium" else "بالا"
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cache_hit": False
        }
        
        # ذخیره در کش
        analysis_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {str(e)}", exc_info=True)
        return {
            "error": f"Analysis failed: {str(e)[:100]}",
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    """صفحه اصلی"""
    return {
        "status": "online",
        "service": "Crypto AI Scalper Pro",
        "version": API_VERSION,
        "endpoints": {
            "health": "/health-check",
            "analyze": "POST /analyze",
            "admin": "/admin/dashboard?admin_key=YOUR_SECRET",
            "scan": "/market-scan"
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/health-check")
async def health_check():
    """بررسی سلامت سرویس"""
    return {
        "status": "healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uptime": time.time() - app.startup_time if hasattr(app, 'startup_time') else 0,
        "cache_size": len(analysis_cache),
        "active_users": len(active_users)
    }

@app.post("/analyze")
async def analyze(request: Request):
    """آنالیز اصلی نماد"""
    try:
        data = await request.json()
        client_id = data.get("client_id", "guest")
        symbol = data.get("symbol", "BTCUSDT").upper()
        timeframe = data.get("timeframe", "5m")
        
        # اعتبارسنجی
        if not symbol or len(symbol) < 3:
            raise HTTPException(status_code=400, detail="Invalid symbol")
        
        # بررسی لیست سیاه
        if client_id in blacklisted_ips:
            raise HTTPException(
                status_code=403,
                detail="Access denied. Your client ID has been blocked."
            )
        
        # بررسی محدودیت نرخ
        if not check_rate_limit(client_id):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait a minute."
            )
        
        # ثبت کاربر
        active_users[client_id] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "ip": request.client.host if request.client else "unknown",
            "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "request_times": active_users.get(client_id, {}).get('request_times', []) + [time.time()]
        }
        
        # تحلیل
        result = get_on_demand_analysis(symbol, timeframe)
        
        # اضافه کردن metadata
        result["metadata"] = {
            "client_id": client_id,
            "processing_time": time.time(),
            "cache_cleared": len(analysis_cache)
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)[:100]}")

@app.get("/market-scan")
async def market_scan():
    """اسکن چند نماد محبوب"""
    try:
        popular_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", 
            "XAUUSD", "EURUSD", "GBPUSD",
            "SOLUSDT", "ADAUSDT", "DOTUSDT"
        ]
        
        results = []
        for symbol in popular_symbols[:6]:  # محدودیت برای جلوگیری از overload
            try:
                analysis = get_on_demand_analysis(symbol, "1h")
                if "error" not in analysis:
                    results.append({
                        "symbol": symbol,
                        "signal": analysis.get("analysis", {}).get("final_decision", "Neutral"),
                        "score": analysis.get("analysis", {}).get("score", 50),
                        "price": analysis.get("price", "N/A"),
                        "trend": analysis.get("indicators", {}).get("ema_trend", "Unknown")
                    })
            except Exception as e:
                logger.warning(f"Failed to scan {symbol}: {e}")
                continue
        
        return {
            "status": "success",
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_scanned": len(popular_symbols[:6]),
            "successful": len(results),
            "results": sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        }
        
    except Exception as e:
        logger.error(f"Market scan error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Market scan failed")

# ==============================================================================
# ADMIN ENDPOINTS
# ==============================================================================

@app.get("/admin/dashboard")
async def admin_dashboard(admin_key: str):
    """داشبورد ادمین"""
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    
    # پاک کردن کاربران غیرفعال (بیش از 10 دقیقه)
    current_time = time.time()
    inactive_users = []
    
    for client_id, user_data in list(active_users.items()):
        last_seen_str = user_data.get('last_seen', '')
        try:
            last_seen = datetime.strptime(last_seen_str, "%Y-%m-%d %H:%M:%S").timestamp()
            if current_time - last_seen > 600:  # 10 دقیقه
                inactive_users.append(client_id)
        except:
            pass
    
    for client_id in inactive_users:
        del active_users[client_id]
    
    return {
        "status": "admin_dashboard",
        "active_users_count": len(active_users),
        "blacklisted_count": len(blacklisted_ips),
        "cache_size": len(analysis_cache),
        "active_users": active_users,
        "blacklisted_ips": list(blacklisted_ips)[:20],  # فقط 20 آیتم اول
        "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/admin/manage-user")
async def manage_user(request: Request):
    """مدیریت کاربران توسط ادمین"""
    try:
        data = await request.json()
        admin_key = data.get("admin_key")
        action = data.get("action")
        target_id = data.get("target_client_id")
        
        if admin_key != ADMIN_SECRET:
            raise HTTPException(status_code=401, detail="Invalid admin key")
        
        if not target_id:
            raise HTTPException(status_code=400, detail="Target client ID required")
        
        if action == "block":
            blacklisted_ips.add(target_id)
            if target_id in active_users:
                del active_users[target_id]
            return {"status": "success", "message": f"User {target_id} blocked"}
        
        elif action == "unblock":
            if target_id in blacklisted_ips:
                blacklisted_ips.remove(target_id)
            return {"status": "success", "message": f"User {target_id} unblocked"}
        
        elif action == "clear_cache":
            analysis_cache.clear()
            return {"status": "success", "message": "Cache cleared"}
        
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin management error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Admin operation failed")

# ==============================================================================
# MIDDLEWARE & EVENTS
# ==============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """میان‌افزار برای اندازه‌گیری زمان پردازش"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    response.headers["X-API-Version"] = API_VERSION
    
    # پاک کردن کش قدیمی
    if random.random() < 0.1:  # 10% chance on each request
        clear_old_cache()
    
    return response

@app.on_event("startup")
async def startup_event():
    """رویداد شروع برنامه"""
    app.startup_time = time.time()
    logger.info(f"""
    ============================================
    🚀 CRYPTO AI SCALPER PRO v{API_VERSION}
    ============================================
    Status: ✅ ONLINE
    Debug Mode: {'✅ ON' if DEBUG_MODE else '❌ OFF'}
    Admin Secret: {ADMIN_SECRET[:5]}...
    ============================================
    Endpoints:
    • GET  /              - Main page
    • GET  /health-check  - Health status
    • POST /analyze       - Analyze symbol
    • GET  /market-scan   - Market scanner
    • GET  /admin/dashboard?admin_key=... - Admin
    ============================================
    """)

@app.on_event("shutdown")
async def shutdown_event():
    """رویداد خاموشی برنامه"""
    logger.info("Shutting down Crypto AI Scalper Pro...")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import random  # برای پاک کردن تصادفی کش
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=DEBUG_MODE,
        log_level="debug" if DEBUG_MODE else "info",
        access_log=True,
        timeout_keep_alive=30
    )