import os
import sys
import time
import uvicorn
import logging
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
from typing import List, Final, Dict, Any, Optional
import random

# ==============================================================================
# CONFIGURATION
# ==============================================================================

API_VERSION: Final[str] = "9.2.1"
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

class DirectAnalysisRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"

class AdminRequest(BaseModel):
    admin_key: str
    action: str
    target_client_id: Optional[str] = None

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

def convert_to_yahoo_symbol(symbol: str) -> str:
    """تبدیل نماد به فرمت Yahoo Finance"""
    symbol = symbol.upper().strip()
    
    # نمادهای فارکس
    if symbol in ["XAUUSD", "GOLD"]:
        return "GC=F"
    elif symbol == "XAGUSD":
        return "SI=F"
    elif symbol in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]:
        return f"{symbol}=X"
    
    # نمادهای کریپتو
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}-USD"
    elif symbol.endswith("USD"):
        return symbol.replace("USD", "-USD")
    
    return symbol

def get_binance_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """دریافت داده از بایننس"""
    try:
        # مپ کردن timeframe به فرمت بایننس
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "30m": "30m", "1h": "1h", "4h": "4h", 
            "1d": "1d", "1w": "1w", "1M": "1M"
        }
        
        binance_interval = interval_map.get(timeframe, "5m")
        
        # ساخت URL درخواست
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={binance_interval}&limit=100"
        
        # ارسال درخواست با timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # پردازش پاسخ
        data = response.json()
        
        # بررسی اینکه آیا داده معتبر است
        if not data or len(data) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # تبدیل به عدد
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # تبدیل timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # حذف ردیف‌های با داده‌های NaN
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        logger.info(f"Received {len(df)} rows from Binance for {symbol}")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Binance API request failed for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing Binance data for {symbol}: {e}")
        return pd.DataFrame()

def get_yahoo_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """دریافت داده از Yahoo Finance"""
    try:
        # تنظیم period بر اساس timeframe
        period_map = {
            "1m": "1d", "5m": "5d", "15m": "5d",
            "30m": "5d", "1h": "1mo", "4h": "3mo",
            "1d": "6mo", "1w": "1y", "1M": "2y"
        }
        
        period = period_map.get(timeframe, "5d")
        
        # دانلود داده
        df = yf.download(
            symbol,
            period=period,
            interval=timeframe,
            progress=False,
            auto_adjust=True,
            threads=True
        )
        
        if df.empty:
            logger.warning(f"No data received from Yahoo Finance for {symbol}")
            return pd.DataFrame()
        
        # تغییر نام ستون‌ها برای سازگاری
        df.columns = [col[0].lower() for col in df.columns]
        
        logger.info(f"Received {len(df)} rows from Yahoo for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
        return pd.DataFrame()

def get_cached_analysis(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    """بررسی کش برای تحلیل‌های اخیر"""
    cache_key = f"{symbol}_{timeframe}"
    if cache_key in analysis_cache:
        cache_data = analysis_cache[cache_key]
        # بررسی زمان انقضای کش
        if time.time() - cache_data.get('timestamp', 0) < CACHE_DURATION:
            result = cache_data['data'].copy()
            result['metadata']['cached'] = True
            return result
    return None

def set_cached_analysis(symbol: str, timeframe: str, data: Dict[str, Any]):
    """ذخیره تحلیل در کش"""
    cache_key = f"{symbol}_{timeframe}"
    analysis_cache[cache_key] = {
        'data': data,
        'timestamp': time.time()
    }

# ==============================================================================
# CORE ANALYSIS ENGINE
# ==============================================================================

def perform_technical_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """انجام تحلیل تکنیکال روی داده‌ها"""
    try:
        # بررسی ساختار DataFrame
        if df.empty:
            return None
            
        # اطمینان از وجود ستون‌های ضروری
        required_columns = ['close', 'high', 'low', 'open', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return None
        
        # ۱. محاسبه اندیکاتورهای اصلی
        # RSI
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # استوکاستیک
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        
        # EMA ها
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        
        # حجم (Volume)
        df['VOLUME_SMA'] = ta.sma(df['volume'], length=20)
        
        # اطمینان از ادغام صحیح DataFrame ها
        if stoch is not None and not stoch.empty:
            df = pd.concat([df, stoch], axis=1)
        if macd is not None and not macd.empty:
            df = pd.concat([df, macd], axis=1)
        if bb is not None and not bb.empty:
            df = pd.concat([df, bb], axis=1)
        
        # ۲. مقادیر آخرین کندل
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        
        # ۳. استخراج مقادیر اندیکاتورها
        # RSI
        rsi_val = last_row['RSI'] if pd.notna(last_row['RSI']) else 50
        rsi_status = "خرید اشباع" if rsi_val > 70 else "فروش اشباع" if rsi_val < 30 else "خنثی"
        
        # استوکاستیک
        stoch_k_col = next((c for c in df.columns if 'stochk' in c.lower()), None)
        stoch_d_col = next((c for c in df.columns if 'stochd' in c.lower()), None)
        
        k_val = last_row[stoch_k_col] if stoch_k_col and stoch_k_col in df.columns and pd.notna(last_row[stoch_k_col]) else 50
        d_val = last_row[stoch_d_col] if stoch_d_col and stoch_d_col in df.columns and pd.notna(last_row[stoch_d_col]) else 50
        
        stoch_status = "خرید اشباع" if k_val < 20 else "فروش اشباع" if k_val > 80 else "خنثی"
        
        # EMA Cross
        ema_cross = "صعودی" if last_row['EMA_9'] > last_row['EMA_20'] and prev_row['EMA_9'] <= prev_row['EMA_20'] else \
                   "نزولی" if last_row['EMA_9'] < last_row['EMA_20'] and prev_row['EMA_9'] >= prev_row['EMA_20'] else "خنثی"
        
        # MACD
        macd_val = last_row.get('MACD_12_26_9', 0)
        macd_signal = last_row.get('MACDs_12_26_9', 0)
        macd_hist = last_row.get('MACDh_12_26_9', 0)
        macd_trend = "صعودی" if macd_hist > 0 else "نزولی"
        
        # Bollinger Bands
        bb_upper = last_row.get('BBU_20_2.0', last_row['close'])
        bb_lower = last_row.get('BBL_20_2.0', last_row['close'])
        bb_middle = last_row.get('BBM_20_2.0', last_row['close'])
        
        bb_position = "بالای باند" if last_row['close'] > bb_upper else "پایین باند" if last_row['close'] < bb_lower else "درون باند"
        
        # حجم
        volume_ratio = (last_row['volume'] / last_row['VOLUME_SMA']) if last_row['VOLUME_SMA'] > 0 else 1
        volume_status = "بالا" if volume_ratio > 1.5 else "پایین" if volume_ratio < 0.5 else "معمولی"
        
        # ۴. سیستم امتیازدهی
        score = 50  # امتیاز پایه
        reasons = []
        
        # امتیاز RSI
        if rsi_val < 35: 
            score += 15
            reasons.append("RSI در منطقه خرید اشباع")
        elif rsi_val > 65: 
            score -= 15
            reasons.append("RSI در منطقه فروش اشباع")
        
        # امتیاز روند EMA
        if last_row['EMA_9'] > last_row['EMA_20'] > last_row['EMA_50']: 
            score += 25
            reasons.append("روند صعودی قوی")
        elif last_row['EMA_9'] < last_row['EMA_20'] < last_row['EMA_50']: 
            score -= 25
            reasons.append("روند نزولی قوی")
        
        # امتیاز استوکاستیک
        if k_val < 20 and d_val < 20: 
            score += 10
            reasons.append("استوکاستیک در منطقه خرید اشباع")
        elif k_val > 80 and d_val > 80: 
            score -= 10
            reasons.append("استوکاستیک در منطقه فروش اشباع")
        
        # امتیاز MACD
        if macd_hist > 0: 
            score += 5
        elif macd_hist < 0: 
            score -= 5
        
        # امتیاز حجم
        if volume_ratio > 2: 
            score += 10
            reasons.append("حجم معاملات بسیار بالا")
        elif volume_ratio < 0.3: 
            score -= 5
            reasons.append("حجم معاملات پایین")
        
        # محدود کردن امتیاز
        score = max(0, min(100, score))
        
        # ۵. تصمیم نهایی
        if score >= 75:
            decision = "خرید قوی (Strong Buy) 🚀"
            color = "green"
            confidence = "بالا"
            action = "BUY"
        elif score >= 60:
            decision = "خرید (Buy) 📈"
            color = "lightgreen"
            confidence = "متوسط"
            action = "BUY"
        elif score <= 25:
            decision = "فروش قوی (Strong Sell) 📉"
            color = "red"
            confidence = "بالا"
            action = "SELL"
        elif score <= 40:
            decision = "فروش (Sell) 🔻"
            color = "orange"
            confidence = "متوسط"
            action = "SELL"
        else:
            decision = "صبر (Neutral) ⏸️"
            color = "gray"
            confidence = "پایین"
            action = "HOLD"
        
        # ۶. محاسبه اهداف و حد ضرر
        current_price = float(last_row['close'])
        atr_val = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1] if len(df) >= 14 else current_price * 0.02
        
        # محاسبات بر اساس نوع سیگنال
        if action == "BUY":
            stop_loss = current_price - (atr_val * 1.5)
            target1 = current_price + (atr_val * 1)
            target2 = current_price + (atr_val * 2)
            target3 = current_price + (atr_val * 3)
        elif action == "SELL":
            stop_loss = current_price + (atr_val * 1.5)
            target1 = current_price - (atr_val * 1)
            target2 = current_price - (atr_val * 2)
            target3 = current_price - (atr_val * 3)
        else:
            stop_loss = current_price * 0.95
            target1 = current_price * 1.05
            target2 = current_price * 1.10
            target3 = current_price * 1.15
        
        return {
            "price": current_price,
            "price_change": float(((current_price - float(prev_row['close'])) / float(prev_row['close'])) * 100) if 'close' in prev_row else 0,
            "indicators": {
                "rsi": round(rsi_val, 2),
                "rsi_status": rsi_status,
                "stoch_k": round(k_val, 2),
                "stoch_d": round(d_val, 2),
                "stoch_status": stoch_status,
                "ema_9": round(float(last_row['EMA_9']), 4),
                "ema_20": round(float(last_row['EMA_20']), 4),
                "ema_50": round(float(last_row['EMA_50']), 4),
                "ema_cross": ema_cross,
                "macd": round(macd_val, 4),
                "macd_signal": round(macd_signal, 4),
                "macd_hist": round(macd_hist, 4),
                "macd_trend": macd_trend,
                "bb_position": bb_position,
                "bb_upper": round(float(bb_upper), 4),
                "bb_lower": round(float(bb_lower), 4),
                "bb_middle": round(float(bb_middle), 4),
                "volume_ratio": round(volume_ratio, 2),
                "volume_status": volume_status
            },
            "analysis": {
                "score": score,
                "final_decision": decision,
                "action": action,
                "color": color,
                "confidence": confidence,
                "reasons": reasons,
                "trend_strength": "قوی" if abs(score - 50) > 30 else "متوسط" if abs(score - 50) > 15 else "ضعیف",
                "risk_level": "پایین" if confidence == "بالا" and abs(score - 50) > 30 else "متوسط" if confidence == "متوسط" else "بالا"
            },
            "trading": {
                "stop_loss": round(stop_loss, 4),
                "targets": [
                    round(target1, 4),
                    round(target2, 4),
                    round(target3, 4)
                ],
                "risk_reward": round(abs(target1 - current_price) / abs(stop_loss - current_price), 2) if stop_loss != current_price else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Technical analysis error: {str(e)}", exc_info=True)
        return None

def get_on_demand_analysis(symbol: str, timeframe: str) -> Dict[str, Any]:
    """موتور تحلیل تکنیکال پیشرفته"""
    
    # بررسی کش
    cached_result = get_cached_analysis(symbol, timeframe)
    if cached_result:
        return cached_result
    
    try:
        logger.info(f"Analyzing {symbol} on {timeframe} timeframe")
        
        # ۱. ابتدا سعی کنید از Binance دیتا بگیرید (پایدارتر برای کریپتو)
        symbol_clean = symbol.upper().replace("-USD", "USDT").replace("=X", "")
        df = get_binance_data(symbol_clean, timeframe)
        
        # ۲. اگر بایننس دیتا نداد (مثلاً برای طلا یا فارکس)، از یاهو استفاده کن
        if df.empty:
            yf_symbol = convert_to_yahoo_symbol(symbol)
            df = get_yahoo_data(yf_symbol, timeframe)
        
        if df.empty or len(df) < 20:
            return {
                "error": True,
                "message": "عدم دسترسی به دیتای بازار (لطفاً نماد را چک کنید)",
                "symbol": symbol
            }
        
        # ادامه تحلیل...
        analysis_result = perform_technical_analysis(df)
        
        if analysis_result is None:
            return {
                "error": True,
                "message": "خطا در تحلیل تکنیکال",
                "symbol": symbol
            }
        
        # ۳. آماده‌سازی نتیجه
        is_forex = any(x in symbol.upper() for x in ["EURUSD", "GBPUSD", "XAU", "GOLD", "US30", "NAS100", "XAG"])
        
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": analysis_result['price'],
            "price_change": analysis_result['price_change'],
            "price_formatted": f"{analysis_result['price']:,.5f}" if analysis_result['price'] < 1 else f"{analysis_result['price']:,.2f}",
            "indicators": analysis_result['indicators'],
            "analysis": analysis_result['analysis'],
            "trading": analysis_result['trading'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "cached": False,
                "data_points": len(df),
                "market": "Forex/Commodity" if is_forex else "Crypto",
                "data_source": "Yahoo Finance" if df.empty and not df.empty else "Binance",
                "success": True
            }
        }
        
        # ذخیره در کش
        set_cached_analysis(symbol, timeframe, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {str(e)}", exc_info=True)
        return {
            "error": True,
            "message": f"خطای تحلیل: {str(e)[:100]}",
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "success": False,
                "error_details": str(e)
            }
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
            "analyze_raw": "POST /analyze-raw",
            "admin": "/admin/dashboard?admin_key=YOUR_SECRET",
            "scan": "/market-scan",
            "performance": "/performance"
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
        "active_users": len(active_users),
        "blacklisted_ips": len(blacklisted_ips)
    }

@app.post("/analyze")
async def analyze(item: AnalysisRequest, request: Request):
    """آنالیز اصلی با استفاده از مدل Pydantic"""
    start_time = time.time()
    
    try:
        symbol = item.symbol.upper()
        timeframe = item.timeframe
        client_id = item.client_id
        
        # اعتبارسنجی
        if not symbol or len(symbol) < 3:
            raise HTTPException(status_code=400, detail="نماد نامعتبر است")
        
        # بررسی لیست سیاه
        if client_id in blacklisted_ips:
            raise HTTPException(
                status_code=403,
                detail="دسترسی مسدود شده است. شناسه کاربری شما در لیست سیاه قرار دارد."
            )
        
        # بررسی محدودیت نرخ
        if not check_rate_limit(client_id):
            raise HTTPException(
                status_code=429,
                detail="محدودیت نرخ درخواست. لطفاً یک دقیقه صبر کنید."
            )
        
        # ثبت کاربر
        active_users[client_id] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "ip": request.client.host if request.client else "unknown",
            "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "request_count": active_users.get(client_id, {}).get('request_count', 0) + 1
        }
        
        # تحلیل
        result = get_on_demand_analysis(symbol, timeframe)
        
        # اضافه کردن metadata
        if "metadata" in result:
            result["metadata"]["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
            result["metadata"]["client_id"] = client_id
        else:
            result["metadata"] = {
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "client_id": client_id
            }
        
        # اگر خطا وجود دارد، HTTPException برگردان
        if result.get("error", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "خطا در تحلیل")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطای داخلی سرور: {str(e)[:100]}")

@app.post("/analyze-raw")
async def analyze_raw(request: DirectAnalysisRequest):
    """آنالیز مستقیم بدون نیاز به client_id (برای تست)"""
    start_time = time.time()
    
    try:
        symbol = request.symbol.upper()
        timeframe = request.timeframe
        
        if not symbol or len(symbol) < 3:
            return {
                "error": True,
                "message": "نماد نامعتبر است",
                "symbol": request.symbol
            }
        
        # تحلیل
        result = get_on_demand_analysis(symbol, timeframe)
        
        # اضافه کردن زمان پردازش
        if "metadata" in result:
            result["metadata"]["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        else:
            result["metadata"] = {
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Raw analysis error: {str(e)}", exc_info=True)
        return {
            "error": True,
            "message": str(e)[:100],
            "symbol": request.symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

@app.get("/market-scan")
async def market_scan():
    """اسکن چند نماد محبوب"""
    start_time = time.time()
    
    try:
        popular_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", 
            "XAUUSD", "EURUSD", "GBPUSD",
            "SOLUSDT", "ADAUSDT", "XRPUSDT",
            "DOTUSDT", "DOGEUSDT", "MATICUSDT"
        ]
        
        results = []
        for symbol in popular_symbols[:8]:  # محدودیت برای جلوگیری از overload
            try:
                analysis = get_on_demand_analysis(symbol, "1h")
                if not analysis.get("error", False):
                    results.append({
                        "symbol": symbol,
                        "signal": analysis.get("analysis", {}).get("final_decision", "صبر (Neutral) ⏸️"),
                        "action": analysis.get("analysis", {}).get("action", "HOLD"),
                        "score": analysis.get("analysis", {}).get("score", 50),
                        "price": analysis.get("price_formatted", "N/A"),
                        "price_change": round(analysis.get("price_change", 0), 2),
                        "trend": analysis.get("analysis", {}).get("trend_strength", "ضعیف"),
                        "rsi": analysis.get("indicators", {}).get("rsi", 50),
                        "confidence": analysis.get("analysis", {}).get("confidence", "پایین")
                    })
            except Exception as e:
                logger.warning(f"Failed to scan {symbol}: {e}")
                continue
        
        return {
            "status": "success",
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_scanned": len(popular_symbols[:8]),
            "successful": len(results),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "results": sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        }
        
    except Exception as e:
        logger.error(f"Market scan error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"خطا در اسکن بازار: {str(e)[:100]}",
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": []
        }

@app.get("/performance")
async def get_performance_stats():
    """آمار عملکرد سیستم"""
    if analysis_cache:
        cache_hits = sum(1 for key in analysis_cache if analysis_cache[key]['data'].get('metadata', {}).get('cached', False))
        cache_hit_rate = (cache_hits / len(analysis_cache)) * 100 if analysis_cache else 0
    else:
        cache_hit_rate = 0
    
    # محاسبه زمان فعالیت
    uptime_seconds = time.time() - app.startup_time if hasattr(app, 'startup_time') else 0
    uptime_str = str(timedelta(seconds=int(uptime_seconds)))
    
    # محاسبه میانگین زمان پردازش
    processing_times = []
    for cache_data in analysis_cache.values():
        if 'metadata' in cache_data['data']:
            processing_times.append(cache_data['data']['metadata'].get('processing_time_ms', 0))
    
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    return {
        "status": "healthy",
        "uptime": uptime_str,
        "cache_size": len(analysis_cache),
        "cache_hit_rate": f"{cache_hit_rate:.1f}%",
        "active_users": len(active_users),
        "blacklisted_ips": len(blacklisted_ips),
        "total_requests": sum(user.get('request_count', 0) for user in active_users.values()),
        "avg_processing_time_ms": round(avg_processing_time, 2),
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "memory_usage_mb": round(os.getpid().memory_info().rss / 1024 / 1024, 2) if hasattr(os, 'getpid') else 0
    }

# ==============================================================================
# ADMIN ENDPOINTS
# ==============================================================================

@app.get("/admin/dashboard")
async def admin_dashboard(admin_key: str):
    """داشبورد ادمین"""
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="کلید ادمین نامعتبر است")
    
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
    
    # پاک کردن کش قدیمی
    clear_old_cache()
    
    # آمار نمادهای درخواستی
    symbol_stats = {}
    for user_data in active_users.values():
        symbol = user_data.get('symbol', 'unknown')
        symbol_stats[symbol] = symbol_stats.get(symbol, 0) + 1
    
    return {
        "status": "admin_dashboard",
        "active_users_count": len(active_users),
        "blacklisted_count": len(blacklisted_ips),
        "cache_size": len(analysis_cache),
        "popular_symbols": sorted(symbol_stats.items(), key=lambda x: x[1], reverse=True)[:10],
        "active_users": {k: v for k, v in list(active_users.items())[:20]},  # فقط 20 کاربر اول
        "blacklisted_ips": list(blacklisted_ips)[:20],  # فقط 20 آیتم اول
        "cache_keys": list(analysis_cache.keys())[:20],  # فقط 20 کلید اول
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
            raise HTTPException(status_code=401, detail="کلید ادمین نامعتبر است")
        
        if not target_id and action not in ["clear_cache", "clear_inactive", "clear_all_cache"]:
            raise HTTPException(status_code=400, detail="شناسه کاربر مورد نیاز است")
        
        if action == "block":
            blacklisted_ips.add(target_id)
            if target_id in active_users:
                del active_users[target_id]
            return {
                "status": "success", 
                "message": f"کاربر {target_id} مسدود شد"
            }
        
        elif action == "unblock":
            if target_id in blacklisted_ips:
                blacklisted_ips.remove(target_id)
            return {
                "status": "success", 
                "message": f"کاربر {target_id} از حالت مسدود خارج شد"
            }
        
        elif action == "clear_cache":
            analysis_cache.clear()
            return {
                "status": "success", 
                "message": "کش پاک شد"
            }
        
        elif action == "clear_inactive":
            # پاک کردن کاربران غیرفعال
            inactive_count = 0
            current_time = time.time()
            
            for client_id, user_data in list(active_users.items()):
                last_seen_str = user_data.get('last_seen', '')
                try:
                    last_seen = datetime.strptime(last_seen_str, "%Y-%m-%d %H:%M:%S").timestamp()
                    if current_time - last_seen > 600:  # 10 دقیقه
                        del active_users[client_id]
                        inactive_count += 1
                except:
                    pass
            
            return {
                "status": "success", 
                "message": f"{inactive_count} کاربر غیرفعال پاک شدند"
            }
        
        elif action == "clear_all_cache":
            analysis_cache.clear()
            active_users.clear()
            return {
                "status": "success", 
                "message": "تمامی کش‌ها و کاربران پاک شدند"
            }
        
        else:
            raise HTTPException(status_code=400, detail="عملیات نامعتبر")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin management error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="عملیات ادمین ناموفق بود")

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
    
    # پاک کردن کش قدیمی (با احتمال 10%)
    if random.random() < 0.1:
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
    • GET  /              - صفحه اصلی
    • GET  /health-check  - وضعیت سلامت
    • POST /analyze       - تحلیل نماد
    • POST /analyze-raw   - تحلیل ساده (بدون client_id)
    • GET  /market-scan   - اسکنر بازار
    • GET  /performance   - آمار عملکرد
    • GET  /admin/dashboard?admin_key=... - داشبورد ادمین
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