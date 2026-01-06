import os
import time
import threading
import schedule
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import json
import requests
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی
load_dotenv()

# ۱. راه‌اندازی اپلیکیشن Flask
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# ۲. ایمپورت مشروط ماژول‌های خاص
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    print("⚠️ ماژول ccxt یافت نشد - حالت شبیه‌سازی فعال می‌شود")
    CCXT_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    print("⚠️ ماژول yfinance یافت نشد")
    YFINANCE_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    print("⚠️ ماژول ta یافت نشد")
    TA_AVAILABLE = False

# ۳. تنظیمات سیستم - همه تنظیمات در اینجا
WATCHLIST = os.environ.get("WATCHLIST", "BTC/USDT,ETH/USDT").split(",")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# تنظیمات سیستم
class SystemConfig:
    CHECK_INTERVAL = 30  # ثانیه
    MIN_SCORE = 3  # حداقل امتیاز برای سیگنال
    TRADING_HOURS = (0, 23)  # فعالیت شبانه‌روزی
    MAX_HISTORY = 100  # حداکثر تاریخچه ذخیره‌شده
    RISK_FREE_ENABLED = True  # فعال‌سازی حالت ریسک‌فری
    MULTI_STRATEGY_SCAN_INTERVAL = 7200  # ثانیه (2 ساعت)
    TOP_COINS_LIMIT = 50  # تعداد ارزهای برتر برای اسکن
    USE_MULTI_STRATEGY = True  # فعال/غیرفعال کردن استراتژی ترکیبی

# ۴. متغیرهای گلوبال
ACTIVE_SIGNALS: Dict[str, Dict] = {}
SIGNAL_HISTORY: List[Dict] = []
SYSTEM_START_TIME = datetime.now(pytz.timezone('Asia/Tehran'))

# ۵. کلاس‌های شبیه‌سازی برای زمانی که ماژول اصلی موجود نیست
class ExchangeSimulator:
    """شبیه‌ساز صرافی برای زمانی که ccxt موجود نیست"""
    
    def __init__(self):
        self.exchange_name = "Binance Simulator"
        self.markets = {
            'BTC/USDT': {'symbol': 'BTC/USDT', 'base': 'BTC', 'quote': 'USDT'},
            'ETH/USDT': {'symbol': 'ETH/USDT', 'base': 'ETH', 'quote': 'USDT'},
            'BNB/USDT': {'symbol': 'BNB/USDT', 'base': 'BNB', 'quote': 'USDT'}
        }
    
    def fetch_ohlcv(self, symbol, timeframe='5m', limit=100):
        """شبیه‌سازی دریافت داده OHLCV"""
        try:
            # داده‌های ساختگی تولید می‌کنیم
            base_price = {
                'BTC/USDT': 50000,
                'ETH/USDT': 3000,
                'BNB/USDT': 400
            }.get(symbol, 100)
            
            ohlcv = []
            current_time = int(time.time() * 1000)
            
            for i in range(limit):
                timestamp = current_time - (i * 300000)  # هر 5 دقیقه
                open_price = base_price * (1 + np.sin(i/10) * 0.01)
                high_price = open_price * (1 + abs(np.sin(i/5)) * 0.02)
                low_price = open_price * (1 - abs(np.cos(i/5)) * 0.02)
                close_price = base_price * (1 + np.sin((i+1)/10) * 0.01)
                volume = 1000 + np.sin(i/3) * 500
                
                ohlcv.append([
                    timestamp,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume
                ])
            
            return list(reversed(ohlcv))  # قدیمی به جدید
        except:
            return None
    
    def fetch_ticker(self, symbol):
        """شبیه‌سازی دریافت تیکر"""
        try:
            base_price = {
                'BTC/USDT': 50000,
                'ETH/USDT': 3000,
                'BNB/USDT': 400
            }.get(symbol, 100)
            
            change = np.sin(time.time() / 1000) * 0.01
            current_price = base_price * (1 + change)
            
            return {
                'symbol': symbol,
                'last': current_price,
                'high': current_price * 1.01,
                'low': current_price * 0.99,
                'volume': 1000000
            }
        except:
            return None
    
    def fetch_tickers(self):
        """شبیه‌سازی دریافت همه تیکرها"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        tickers = {}
        
        for symbol in symbols:
            ticker = self.fetch_ticker(symbol)
            if ticker:
                tickers[symbol.replace('/', '')] = ticker
        
        return tickers

# ۶. توابع کمکی
def get_iran_time() -> datetime:
    """محاسبه زمان فعلی تهران"""
    return datetime.now(pytz.timezone('Asia/Tehran'))

def send_telegram_message(text: str) -> bool:
    """ارسال پیام به تلگرام"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"📤 تلگرام شبیه‌سازی: {text[:100]}...")
        return True
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': text,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ خطا در ارسال تلگرام: {e}")
        return False

def load_signal_history():
    """بارگذاری تاریخچه سیگنال‌ها از فایل"""
    global SIGNAL_HISTORY
    try:
        if os.path.exists('signal_history.json'):
            with open('signal_history.json', 'r') as f:
                SIGNAL_HISTORY = json.load(f)
                print(f"✅ تاریخچه {len(SIGNAL_HISTORY)} سیگنال بارگذاری شد")
    except Exception as e:
        print(f"❌ خطا در بارگذاری تاریخچه: {e}")

def save_signal_history():
    """ذخیره تاریخچه سیگنال‌ها در فایل"""
    try:
        with open('signal_history.json', 'w') as f:
            json.dump(SIGNAL_HISTORY[-SystemConfig.MAX_HISTORY:], f, indent=2)
    except Exception as e:
        print(f"❌ خطا در ذخیره تاریخچه: {e}")

# ۷. توابع تحلیل تکنیکال
def calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """محاسبه اندیکاتورهای تکنیکال"""
    try:
        # استفاده از pandas و numpy برای محاسبات پایه
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # محاسبات ساده
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        
        # RSI ساده
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD ساده
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        # بولینگر باندز
        bb_ma = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_ma + (bb_std * 2)
        bb_lower = bb_ma - (bb_std * 2)
        
        # محاسبه امتیاز (ساده شده)
        score = 0
        
        # سیگنال روند
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            score += 2
        
        # سیگنال RSI
        if rsi.iloc[-1] < 30:
            score += 2  # اشباع فروش
        elif rsi.iloc[-1] > 70:
            score -= 2  # اشباع خرید
        
        # سیگنال MACD
        if macd.iloc[-1] > signal_line.iloc[-1]:
            score += 1
        
        # موقعیت نسبت به بولینگر
        current_price = close.iloc[-1]
        if current_price < bb_lower.iloc[-1]:
            score += 2  # نزدیک به باند پایین
        elif current_price > bb_upper.iloc[-1]:
            score -= 2  # نزدیک به باند بالا
        
        return {
            'score': score,
            'price': current_price,
            'rsi': rsi.iloc[-1],
            'macd': macd.iloc[-1],
            'signal': signal_line.iloc[-1],
            'sma_20': sma_20.iloc[-1],
            'sma_50': sma_50.iloc[-1]
        }
        
    except Exception as e:
        print(f"❌ خطا در محاسبه اندیکاتورها: {e}")
        return {'score': 0, 'price': df['close'].iloc[-1] if len(df) > 0 else 0}

# ۸. تحلیل استراتژی ترکیبی
def calculate_multi_strategy_signals(df: pd.DataFrame) -> tuple:
    """محاسبه سیگنال‌های استراتژی ترکیبی"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        
        # میانگین متحرک ساده
        ma_50 = close.rolling(window=50).mean()
        ma_200 = close.rolling(window=200).mean()
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # بررسی روند صعودی
        is_bullish = (
            close.iloc[-1] > ma_50.iloc[-1] and 
            ma_50.iloc[-1] > ma_200.iloc[-1]
        )
        
        # بررسی FVG ساده (الگوی گپ)
        has_fvg = False
        if len(df) >= 3:
            # الگوی ساده برای FVG
            prev_low = low.iloc[-2]
            current_high = high.iloc[-1]
            if current_high > prev_low * 1.005:  # گپ 0.5% رو به بالا
                has_fvg = True
        
        current_price = close.iloc[-1]
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else current_price * 0.02
        
        return is_bullish, current_price, current_atr, has_fvg
        
    except Exception as e:
        print(f"❌ خطا در استراتژی ترکیبی: {e}")
        return False, df['close'].iloc[-1] if len(df) > 0 else 0, 0, False

# ۹. بدنه اصلی تحلیل و ارسال پیام
def analyze_and_broadcast(symbol: str, force: bool = False) -> Dict[str, Any]:
    """تحلیل نماد و ارسال سیگنال در صورت وجود شرایط"""
    try:
        # بررسی زمان معاملاتی
        iran_time = get_iran_time()
        if not force and not (SystemConfig.TRADING_HOURS[0] <= iran_time.hour <= SystemConfig.TRADING_HOURS[1]):
            print(f"⏰ خارج از ساعت معاملاتی ({iran_time.hour}:{iran_time.minute})")
            return {"status": "outside_trading_hours"}
        
        # تنظیم نماد
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        exchange_symbol = symbol
        
        # دریافت داده
        ohlcv_data = None
        
        if CCXT_AVAILABLE:
            try:
                exchange = ccxt.binance()
                ohlcv_data = exchange.fetch_ohlcv(exchange_symbol, '5m', limit=100)
            except:
                pass
        
        if ohlcv_data is None:
            # استفاده از شبیه‌ساز
            exchange_sim = ExchangeSimulator()
            ohlcv_data = exchange_sim.fetch_ohlcv(exchange_symbol, '5m', limit=100)
        
        if not ohlcv_data:
            print(f"⚠️ داده‌ای برای {symbol} دریافت نشد.")
            return {"status": "no_data", "symbol": symbol}
        
        # تبدیل به DataFrame
        df = pd.DataFrame(
            ohlcv_data, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # تحلیل تکنیکال
        analysis = calculate_indicators(df)
        score = analysis.get('score', 0)
        current_price = analysis.get('price', 0)
        
        print(f"📊 تحلیل {symbol}: امتیاز={score}, قیمت={current_price}")
        
        # بررسی شرایط سیگنال
        if abs(score) >= SystemConfig.MIN_SCORE or force:
            side = "BUY" if score >= 0 else "SELL"
            
            # محاسبه حد ضرر و تارگت‌ها
            if side == "BUY":
                sl = current_price * 0.995
                risk = current_price - sl
                tp1 = current_price + (risk * 1.5)
                tp2 = current_price + (risk * 3)
            else:  # SELL
                sl = current_price * 1.005
                risk = sl - current_price
                tp1 = current_price - (risk * 1.5)
                tp2 = current_price - (risk * 3)
            
            # ذخیره اطلاعات سیگنال
            signal_data = {
                'symbol': clean_symbol,
                'side': side,
                'entry': current_price,
                'score': abs(score),
                'exit_levels': {
                    'tp1': tp1,
                    'tp2': tp2,
                    'stop_loss': sl,
                    'direction': side
                },
                'timestamp': iran_time.isoformat(),
                'status': 'ACTIVE',
                'force': force,
                'strategy': 'SCALP'
            }
            
            # بررسی وجود سیگنال فعال برای این نماد
            if clean_symbol in ACTIVE_SIGNALS:
                old_status = ACTIVE_SIGNALS[clean_symbol].get('status', 'UNKNOWN')
                print(f"⚠️ سیگنال فعال قبلی برای {clean_symbol} با وضعیت {old_status}")
                
                if old_status == 'ACTIVE':
                    return {
                        "status": "active_signal_exists",
                        "symbol": clean_symbol,
                        "message": "سیگنال فعال قبلی هنوز باز است"
                    }
            
            # ذخیره در حافظه فعال
            ACTIVE_SIGNALS[clean_symbol] = signal_data
            
            # اضافه به تاریخچه
            SIGNAL_HISTORY.append(signal_data.copy())
            if len(SIGNAL_HISTORY) > SystemConfig.MAX_HISTORY:
                SIGNAL_HISTORY.pop(0)
            
            # ساخت پیام تلگرام
            emoji = "🟢" if side == "BUY" else "🔴"
            signal_type = "🔧 FORCE" if force else "🚀 AUTO"
            
            msg = (
                f"{signal_type} *SIGNAL: {clean_symbol}* {emoji}\n"
                f"📶 Direction: {side}\n"
                f"📊 Score: {abs(score)}/10\n"
                f"💵 Entry Price: {current_price:.4f}\n"
                f"🎯 Take Profit 1: {tp1:.4f}\n"
                f"🎯 Take Profit 2: {tp2:.4f}\n"
                f"🛑 Stop Loss: {sl:.4f}\n"
                f"📈 Risk/Reward: 1:3\n"
                f"⏰ Time: {iran_time.strftime('%H:%M:%S')}\n"
                f"#{clean_symbol.replace('USDT', '')} #{side}"
            )
            
            # ارسال به تلگرام
            success = send_telegram_message(msg)
            
            if success:
                print(f"✅ سیگنال {clean_symbol} ارسال شد. وضعیت: ACTIVE")
                return {
                    "status": "success",
                    "symbol": clean_symbol,
                    "side": side,
                    "entry": current_price,
                    "tp1": tp1,
                    "tp2": tp2,
                    "sl": sl,
                    "strategy": "SCALP"
                }
            else:
                print(f"❌ ارسال سیگنال {clean_symbol} ناموفق بود")
                if clean_symbol in ACTIVE_SIGNALS:
                    del ACTIVE_SIGNALS[clean_symbol]
                return {"status": "telegram_error", "symbol": clean_symbol}
        
        else:
            print(f"ℹ️ امتیاز {clean_symbol}: {score} (کمتر از حد نصاب {SystemConfig.MIN_SCORE})")
            return {
                "status": "low_score",
                "symbol": clean_symbol,
                "score": score,
                "min_required": SystemConfig.MIN_SCORE
            }
            
    except Exception as e:
        error_msg = f"❌ خطا در تحلیل {symbol}: {str(e)}"
        print(error_msg)
        return {"status": "error", "symbol": symbol, "error": str(e)}

# ۱۰. تحلیل با استراتژی ترکیبی
def analyze_with_multi_strategy(symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
    """تحلیل با استراتژی ترکیبی"""
    try:
        exchange_symbol = symbol
        
        # دریافت داده
        ohlcv_data = None
        
        if CCXT_AVAILABLE:
            try:
                exchange = ccxt.binance()
                ohlcv_data = exchange.fetch_ohlcv(exchange_symbol, timeframe, limit=100)
            except:
                pass
        
        if ohlcv_data is None:
            exchange_sim = ExchangeSimulator()
            ohlcv_data = exchange_sim.fetch_ohlcv(exchange_symbol, timeframe, limit=100)
        
        if not ohlcv_data:
            print(f"⚠️ داده‌ای برای {symbol} دریافت نشد.")
            return {"status": "no_data", "symbol": symbol}
        
        # تبدیل به DataFrame
        df = pd.DataFrame(
            ohlcv_data, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        bars_df = pd.DataFrame({
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume']
        })
        
        # فراخوانی استراتژی ترکیبی
        is_bull, price, atr, has_fvg = calculate_multi_strategy_signals(bars_df)
        
        if is_bull:
            current_price = df['close'].iloc[-1]
            sl = current_price - (atr * 1.5)
            tp = current_price + (atr * 2.5)
            
            signal_data = {
                'symbol': symbol.replace("/", ""),
                'side': 'BUY',
                'entry': current_price,
                'exit_levels': {
                    'tp1': tp,
                    'tp2': tp * 1.5,
                    'stop_loss': sl,
                    'direction': 'BUY',
                    'atr': atr
                },
                'timestamp': get_iran_time().isoformat(),
                'status': 'ACTIVE',
                'strategy': 'MULTI',
                'has_fvg': has_fvg,
                'timeframe': timeframe
            }
            
            # ساخت پیام تلگرام
            msg = (
                f"🚀 **سیگنال ترکیبی پیشرفته**\n"
                f"━━━━━━━━━━━━━━━\n"
                f"📊 نماد: #{symbol.replace('/', '')}\n"
                f"📈 تایم‌فریم: {timeframe}\n"
                f"🟢 ورود: `{current_price:.4f}`\n"
                f"🔴 استاپ داینامیک: `{sl:.4f}` \n"
                f"🎯 تارگت اول: `{tp:.4f}` \n"
                f"🧱 تاییدیه FVG: {'✅' if has_fvg else '❌'}\n"
                f"📊 ATR: `{atr:.4f}`\n"
                f"⏰ زمان: {get_iran_time().strftime('%H:%M:%S')}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"🏷️ #MultiStrategy"
            )
            
            success = send_telegram_message(msg)
            
            if success:
                ACTIVE_SIGNALS[symbol.replace("/", "")] = signal_data
                SIGNAL_HISTORY.append(signal_data.copy())
                
                print(f"✅ سیگنال ترکیبی {symbol} ارسال شد")
                return {
                    "status": "success",
                    "symbol": symbol,
                    "strategy": "MULTI",
                    "entry": current_price,
                    "tp": tp,
                    "sl": sl,
                    "has_fvg": has_fvg
                }
        
        return {"status": "no_signal", "symbol": symbol}
        
    except Exception as e:
        error_msg = f"❌ خطا در تحلیل ترکیبی {symbol}: {str(e)}"
        print(error_msg)
        return {"status": "error", "symbol": symbol, "error": str(e)}

# ۱۱. مسیرهای وب (Routes)
@app.route('/')
def home():
    """صفحه اصلی"""
    return jsonify({
        "status": "online",
        "name": "Crypto Trading Bot",
        "version": "3.0",
        "iran_time": get_iran_time().strftime('%Y-%m-%d %H:%M:%S'),  # اصلاح شد: %Y-%m-%d
        "active_signals": len(ACTIVE_SIGNALS),
        "strategies": {
            "scalp": "فعال",
            "multi_strategy": "فعال" if SystemConfig.USE_MULTI_STRATEGY else "غیرفعال"
        },
        "trading_hours": f"{SystemConfig.TRADING_HOURS[0]}:00 - {SystemConfig.TRADING_HOURS[1]}:00"
    })

@app.route('/signals')
def signals_status():
    """نمایش وضعیت سیگنال‌های فعال"""
    active_signals = []
    
    for symbol, data in ACTIVE_SIGNALS.items():
        # دریافت قیمت لحظه‌ای
        current_price = data['entry']  # در نسخه ساده از قیمت ورودی استفاده می‌کنیم
        
        active_signals.append({
            'symbol': symbol,
            'side': data['side'],
            'entry': data['entry'],
            'current_price': current_price,
            'tp1': data['exit_levels']['tp1'],
            'tp2': data['exit_levels']['tp2'],
            'sl': data['exit_levels']['stop_loss'],
            'status': data['status'],
            'strategy': data.get('strategy', 'SCALP'),
            'score': data.get('score', 0),
            'timestamp': data['timestamp']
        })
    
    return jsonify({
        "active_signals": active_signals,
        "active_count": len(active_signals),
        "total_history": len(SIGNAL_HISTORY),
        "system_time": get_iran_time().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/analyze/<symbol>')
def analyze_symbol(symbol: str):
    """تحلیل دستی یک نماد"""
    force = request.args.get('force', 'false').lower() == 'true'
    result = analyze_and_broadcast(symbol, force=force)
    return jsonify(result)

@app.route('/multi_analyze/<symbol>')
def multi_analyze_symbol(symbol: str):
    """تحلیل دستی با استراتژی ترکیبی"""
    timeframe = request.args.get('timeframe', '1h')
    result = analyze_with_multi_strategy(symbol, timeframe)
    return jsonify(result)

@app.route('/force_analyze')
def force_analyze():
    """تحلیل اجباری کل واچ‌لیست"""
    results = []
    
    print(f"🚀 شروع تحلیل اجباری {len(WATCHLIST)} نماد")
    
    for symbol in WATCHLIST:
        try:
            result = analyze_and_broadcast(symbol, force=True)
            results.append(result)
            time.sleep(1)
            
        except Exception as e:
            results.append({
                "symbol": symbol,
                "status": "error",
                "error": str(e)
            })
    
    return jsonify({
        "status": "completed",
        "total": len(WATCHLIST),
        "successful": len([r for r in results if r.get('status') == 'success']),
        "results": results
    })

@app.route('/stats')
def system_stats():
    """آمار سیستم"""
    total_signals = len(SIGNAL_HISTORY)
    scalp_signals = len([s for s in SIGNAL_HISTORY if s.get('strategy') == 'SCALP'])
    multi_signals = len([s for s in SIGNAL_HISTORY if s.get('strategy') == 'MULTI'])
    
    successful_signals = len([s for s in SIGNAL_HISTORY if s.get('status', '').startswith('CLOSED_TP')])
    stop_loss_signals = len([s for s in SIGNAL_HISTORY if s.get('status') == 'CLOSED_SL'])
    active_signals = len(ACTIVE_SIGNALS)
    
    return jsonify({
        "system": {
            "start_time": SYSTEM_START_TIME.strftime('%Y-%m-%d %H:%M:%S'),
            "uptime": str(datetime.now(pytz.timezone('Asia/Tehran')) - SYSTEM_START_TIME),
            "iran_time": get_iran_time().strftime('%Y-%m-%d %H:%M:%S')
        },
        "performance": {
            "total_signals": total_signals,
            "scalp_signals": scalp_signals,
            "multi_strategy_signals": multi_signals,
            "active_signals": active_signals,
            "successful_closed": successful_signals,
            "stop_loss_closed": stop_loss_signals,
            "win_rate": f"{(successful_signals/(successful_signals+stop_loss_signals)*100 if (successful_signals+stop_loss_signals) > 0 else 0):.1f}%"
        },
        "config": {
            "trading_hours": SystemConfig.TRADING_HOURS,
            "min_score": SystemConfig.MIN_SCORE,
            "use_multi_strategy": SystemConfig.USE_MULTI_STRATEGY
        },
        "watchlist": WATCHLIST,
        "modules": {
            "ccxt": CCXT_AVAILABLE,
            "ta": TA_AVAILABLE,
            "yfinance": YFINANCE_AVAILABLE
        }
    })

@app.route('/webhook', methods=['POST'])
def tradingview_webhook():
    """دریافت سیگنال از تریدینگ‌ویو"""
    try:
        data = request.json
        if not data:
            return jsonify({"status": "empty_data"}), 400
        
        symbol = data.get('symbol', 'Unknown')
        side = data.get('side', 'N/A')
        price = data.get('price', 0)
        sl = data.get('sl', 0)
        tp = data.get('tp', 0)

        emoji = "🟢" if side == "BUY" else "🔴"
        
        msg = (
            f"🚀 *NEW SIGNAL FROM TRADINGVIEW* {emoji}\n"
            f"📊 Symbol: {symbol}\n"
            f"📶 Direction: {side}\n"
            f"💵 Entry: {price}\n"
            f"🎯 Target: {tp}\n"
            f"🛑 Stop Loss: {sl}\n"
            f"⏰ Time: {get_iran_time().strftime('%H:%M:%S')}"
        )
        
        success = send_telegram_message(msg)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "سیگنال دریافت و ارسال شد",
                "data": data
            })
        else:
            return jsonify({
                "status": "telegram_error",
                "message": "خطا در ارسال به تلگرام"
            }), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ۱۲. توابع زمان‌بندی
def hourly_job():
    """تحلیل ساعتی"""
    now = get_iran_time()
    
    if SystemConfig.TRADING_HOURS[0] <= now.hour <= SystemConfig.TRADING_HOURS[1]:
        print(f"⏰ شروع تحلیل ساعتی ساعت {now.hour}:{now.minute:02d}")
        
        for symbol in WATCHLIST:
            analyze_and_broadcast(symbol, force=False)
            time.sleep(2)
    
    else:
        print(f"⏰ خارج از ساعت معاملاتی ({now.hour}:{now.minute:02d})")

def multi_strategy_job():
    """اسکنر استراتژی ترکیبی"""
    print(f"🚀 شروع اسکنر استراتژی ترکیبی - {get_iran_time().strftime('%H:%M:%S')}")
    
    if not SystemConfig.USE_MULTI_STRATEGY:
        print("ℹ️ استراتژی ترکیبی غیرفعال است")
        return
    
    try:
        # در این نسخه ساده شده، فقط واچ‌لیست را تحلیل می‌کنیم
        for symbol in WATCHLIST:
            try:
                analyze_with_multi_strategy(symbol, '1h')
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ خطا در تحلیل {symbol}: {e}")
                continue
        
        print(f"📈 اسکن کامل شد.")
        
    except Exception as e:
        print(f"❌ خطا در اسکن: {e}")

def run_scheduler():
    """اجرای زمان‌بند"""
    # اجرای هر ساعت در دقیقه ۰
    schedule.every().hour.at(":00").do(hourly_job)
    
    # اجرای اسکنر استراتژی ترکیبی هر ۲ ساعت
    schedule.every(SystemConfig.MULTI_STRATEGY_SCAN_INTERVAL).seconds.do(multi_strategy_job)
    
    print("⏰ زمان‌بند راه‌اندازی شد")
    
    while True:
        schedule.run_pending()
        time.sleep(30)

# ۱۳. نقطه شروع اجرای برنامه
if __name__ == "__main__":
    # بارگذاری تاریخچه
    load_signal_history()
    
    print(f"🚀 ربات ترید با موفقیت در پورت {port} راه‌اندازی شد")
    print(f"⏰ زمان شروع سیستم (تهران): {SYSTEM_START_TIME.strftime('%H:%M:%S')}")
    
    print("\n" + "="*60)
    print("🚀 Crypto Trading Bot v3.0")
    print("="*60)
    print(f"📅 تاریخ: {get_iran_time().strftime('%Y-%m-%d')}")
    print(f"⏰ ساعت: {get_iran_time().strftime('%H:%M:%S')}")
    print(f"📊 واچ‌لیست: {', '.join(WATCHLIST)}")
    print(f"⚙️ ساعت معاملاتی: {SystemConfig.TRADING_HOURS[0]}:00 - {SystemConfig.TRADING_HOURS[1]}:00")
    print(f"📈 حداقل امتیاز سیگنال: {SystemConfig.MIN_SCORE}")
    print("="*60)
    
    # ذخیره خودکار تاریخچه هنگام خروج
    import atexit
    atexit.register(save_signal_history)
    
    # راه‌اندازی زمان‌بند در یک thread جداگانه
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    print(f"🌐 سرور در حال راه‌اندازی روی پورت {port}...")
    print("="*60 + "\n")
    
    # اجرای سرور Flask
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
