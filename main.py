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

# ==================== CONFIGURATION ====================
load_dotenv()

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8396237816:AAFBwYRj319UI1FxTG_EjdoLsgfRDsWMImY")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7037205717")

# System Configuration
class SystemConfig:
    # Trading Settings
    CHECK_INTERVAL = 20  # seconds
    MIN_SCORE = 3  # minimum score for signal
    TRADING_HOURS = (0, 23)  # 24-hour trading
    MAX_HISTORY = 100
    
    # Risk Management
    RISK_FREE_ENABLED = True
    STOP_LOSS_PERCENT = 1.0  # 1%
    TAKE_PROFIT_1_PERCENT = 1.5  # 1.5%
    TAKE_PROFIT_2_PERCENT = 3.0  # 3.0%
    
    # Multi Strategy
    USE_MULTI_STRATEGY = True
    MULTI_STRATEGY_INTERVAL = 7200  # 2 hours
    TOP_COINS_LIMIT = 20

# Watchlist
WATCHLIST = os.environ.get("WATCHLIST", "BTC/USDT,ETH/USDT,BNB/USDT,ADA/USDT,SOL/USDT").split(",")

# Data Storage
ACTIVE_SIGNALS: Dict[str, Dict] = {}
SIGNAL_HISTORY: List[Dict] = []
SYSTEM_START_TIME = datetime.now(pytz.timezone('Asia/Tehran'))

# ==================== HELPER FUNCTIONS ====================

def get_iran_time() -> datetime:
    """Get current Tehran time"""
    return datetime.now(pytz.timezone('Asia/Tehran'))

def log_message(message: str, level: str = "INFO"):
    """Log message with timestamp"""
    timestamp = get_iran_time().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def send_telegram_message(text: str) -> bool:
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log_message("⚠️ تلگرام تنظیم نشده", "WARNING")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': text,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            log_message("✅ پیام تلگرام ارسال شد")
            return True
        else:
            log_message(f"❌ خطای تلگرام: {response.status_code}", "ERROR")
            return False
            
    except Exception as e:
        log_message(f"❌ خطا در ارسال تلگرام: {e}", "ERROR")
        return False

def load_signal_history():
    """Load signal history from file"""
    global SIGNAL_HISTORY
    try:
        if os.path.exists('signal_history.json'):
            with open('signal_history.json', 'r') as f:
                SIGNAL_HISTORY = json.load(f)
                log_message(f"✅ تاریخچه {len(SIGNAL_HISTORY)} سیگنال بارگذاری شد")
    except Exception as e:
        log_message(f"❌ خطا در بارگذاری تاریخچه: {e}", "ERROR")

def save_signal_history():
    """Save signal history to file"""
    try:
        with open('signal_history.json', 'w') as f:
            json.dump(SIGNAL_HISTORY[-SystemConfig.MAX_HISTORY:], f, indent=2)
    except Exception as e:
        log_message(f"❌ خطا در ذخیره تاریخچه: {e}", "ERROR")

# ==================== TECHNICAL ANALYSIS ====================

class TechnicalAnalyzer:
    """Technical analysis with multiple indicators"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2.0):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        
        upper_band = sma + (rolling_std * std)
        lower_band = sma - (rolling_std * std)
        
        return {
            'sma': sma,
            'upper': upper_band,
            'lower': lower_band
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Complete technical analysis of price data"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Calculate indicators
        rsi = TechnicalAnalyzer.calculate_rsi(close)
        macd_data = TechnicalAnalyzer.calculate_macd(close)
        bb_data = TechnicalAnalyzer.calculate_bollinger_bands(close)
        atr = TechnicalAnalyzer.calculate_atr(high, low, close)
        
        # Moving averages
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        # Volume analysis
        volume_sma = volume.rolling(window=20).mean()
        volume_ratio = volume / volume_sma
        
        # Calculate score
        score = 0
        current_price = close.iloc[-1]
        
        # 1. Trend (25%)
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            score += 2.5
        
        # 2. RSI (25%)
        current_rsi = rsi.iloc[-1]
        if 30 <= current_rsi <= 70:
            score += 1.0
        elif current_rsi < 30:  # Oversold
            score += 2.5
        elif current_rsi > 70:  # Overbought
            score -= 2.5
        
        # 3. MACD (20%)
        if macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1]:
            score += 2.0
        
        # 4. Bollinger Bands (15%)
        if current_price <= bb_data['lower'].iloc[-1]:
            score += 1.5  # Near lower band
        elif current_price >= bb_data['upper'].iloc[-1]:
            score -= 1.5  # Near upper band
        
        # 5. Volume (15%)
        if volume_ratio.iloc[-1] > 1.5:
            score += 1.5  # High volume
        
        return {
            'score': round(score, 2),
            'price': current_price,
            'indicators': {
                'rsi': round(current_rsi, 2),
                'macd': round(macd_data['macd'].iloc[-1], 4),
                'signal': round(macd_data['signal'].iloc[-1], 4),
                'bb_upper': round(bb_data['upper'].iloc[-1], 2),
                'bb_lower': round(bb_data['lower'].iloc[-1], 2),
                'bb_middle': round(bb_data['sma'].iloc[-1], 2),
                'atr': round(atr.iloc[-1], 4),
                'sma_20': round(sma_20.iloc[-1], 2),
                'sma_50': round(sma_50.iloc[-1], 2)
            }
        }

# ==================== EXCHANGE DATA ====================

class ExchangeData:
    """Fetch data from exchange (simulated for now)"""
    
    @staticmethod
    def fetch_ohlcv(symbol: str, timeframe: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data"""
        try:
            # Simulated data (in production, connect to real exchange)
            np.random.seed(int(time.time()))
            
            # Base price based on symbol
            base_prices = {
                'BTC/USDT': 50000,
                'ETH/USDT': 3000,
                'BNB/USDT': 400,
                'ADA/USDT': 0.5,
                'SOL/USDT': 100
            }
            
            base_price = base_prices.get(symbol.upper(), 100)
            
            # Generate synthetic data
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
            
            # Random walk for prices
            returns = np.random.randn(limit) * 0.002
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Create OHLCV data
            open_prices = prices * (1 + np.random.randn(limit) * 0.001)
            high_prices = np.maximum(open_prices, prices) * (1 + np.random.rand(limit) * 0.005)
            low_prices = np.minimum(open_prices, prices) * (1 - np.random.rand(limit) * 0.005)
            close_prices = prices
            volumes = np.random.rand(limit) * 1000 + 500
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            })
            
            df.set_index('timestamp', inplace=True)
            
            log_message(f"📊 داده‌های {symbol} دریافت شد: {len(df)} کندل")
            return df
            
        except Exception as e:
            log_message(f"❌ خطا در دریافت داده {symbol}: {e}", "ERROR")
            return None
    
    @staticmethod
    def fetch_current_price(symbol: str) -> Optional[float]:
        """Fetch current price"""
        try:
            df = ExchangeData.fetch_ohlcv(symbol, '1m', 2)
            if df is not None and not df.empty:
                return float(df['close'].iloc[-1])
        except:
            pass
        return None

# ==================== TRADING ENGINE ====================

def analyze_symbol(symbol: str, force: bool = False) -> Dict[str, Any]:
    """Analyze symbol and generate signal if conditions met"""
    try:
        # Check trading hours
        iran_time = get_iran_time()
        if not force and not (SystemConfig.TRADING_HOURS[0] <= iran_time.hour <= SystemConfig.TRADING_HOURS[1]):
            log_message(f"⏰ خارج از ساعت معاملاتی ({iran_time.hour}:{iran_time.minute})")
            return {"status": "outside_trading_hours"}
        
        # Format symbol
        if '/' not in symbol and 'USDT' in symbol:
            symbol = symbol.replace('USDT', '/USDT')
        
        clean_symbol = symbol.replace("/", "").upper()
        
        # Fetch data
        df = ExchangeData.fetch_ohlcv(symbol, '5m', 100)
        if df is None or df.empty:
            log_message(f"⚠️ داده‌ای برای {symbol} دریافت نشد")
            return {"status": "no_data", "symbol": symbol}
        
        # Technical analysis
        analysis = TechnicalAnalyzer.analyze_dataframe(df)
        score = analysis['score']
        current_price = analysis['price']
        
        log_message(f"📊 تحلیل {symbol}: امتیاز={score}, قیمت={current_price:.2f}")
        
        # Check signal conditions
        if abs(score) >= SystemConfig.MIN_SCORE or force:
            side = "BUY" if score >= 0 else "SELL"
            
            # Calculate exit levels
            if side == "BUY":
                sl_price = current_price * (1 - SystemConfig.STOP_LOSS_PERCENT / 100)
                tp1_price = current_price * (1 + SystemConfig.TAKE_PROFIT_1_PERCENT / 100)
                tp2_price = current_price * (1 + SystemConfig.TAKE_PROFIT_2_PERCENT / 100)
            else:  # SELL
                sl_price = current_price * (1 + SystemConfig.STOP_LOSS_PERCENT / 100)
                tp1_price = current_price * (1 - SystemConfig.TAKE_PROFIT_1_PERCENT / 100)
                tp2_price = current_price * (1 - SystemConfig.TAKE_PROFIT_2_PERCENT / 100)
            
            # Prepare signal data
            signal_data = {
                'symbol': clean_symbol,
                'side': side,
                'entry': round(current_price, 4),
                'score': round(abs(score), 2),
                'exit_levels': {
                    'tp1': round(tp1_price, 4),
                    'tp2': round(tp2_price, 4),
                    'stop_loss': round(sl_price, 4),
                    'direction': side
                },
                'indicators': analysis['indicators'],
                'timestamp': iran_time.isoformat(),
                'status': 'ACTIVE',
                'notifications_sent': {
                    'tp1': False,
                    'tp2': False,
                    'sl': False
                },
                'force': force,
                'strategy': 'MAIN'
            }
            
            # Check for existing active signal
            if clean_symbol in ACTIVE_SIGNALS:
                old_signal = ACTIVE_SIGNALS[clean_symbol]
                if old_signal.get('status') == 'ACTIVE':
                    log_message(f"⚠️ سیگنال فعال قبلی برای {clean_symbol} وجود دارد")
                    return {
                        "status": "active_signal_exists",
                        "symbol": clean_symbol
                    }
            
            # Store in memory
            ACTIVE_SIGNALS[clean_symbol] = signal_data
            
            # Add to history
            SIGNAL_HISTORY.append(signal_data.copy())
            if len(SIGNAL_HISTORY) > SystemConfig.MAX_HISTORY:
                SIGNAL_HISTORY.pop(0)
            
            # Prepare Telegram message
            emoji = "🟢" if side == "BUY" else "🔴"
            signal_type = "🔧 FORCE" if force else "🚀 AUTO"
            
            telegram_msg = (
                f"{signal_type} *SIGNAL: {clean_symbol}* {emoji}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"📶 جهت: {side}\n"
                f"⭐ امتیاز: {abs(score):.1f}/10\n"
                f"💰 قیمت ورود: `{current_price:.2f}`\n"
                f"📊 RSI: `{analysis['indicators']['rsi']:.1f}`\n"
                f"🎯 تارگت ۱: `{tp1_price:.2f}`\n"
                f"🎯 تارگت ۲: `{tp2_price:.2f}`\n"
                f"🛑 استاپ‌لاس: `{sl_price:.2f}`\n"
                f"📈 نسبت R/R: ۱:۳\n"
                f"⏰ زمان: {iran_time.strftime('%H:%M:%S')}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"#{clean_symbol.replace('USDT', '')} #{side}"
            )
            
            # Send to Telegram
            telegram_sent = send_telegram_message(telegram_msg)
            
            if telegram_sent:
                log_message(f"✅ سیگنال {clean_symbol} ارسال شد")
                return {
                    "status": "success",
                    "symbol": clean_symbol,
                    "side": side,
                    "entry": current_price,
                    "score": abs(score),
                    "tp1": tp1_price,
                    "tp2": tp2_price,
                    "sl": sl_price,
                    "telegram_sent": True
                }
            else:
                log_message(f"⚠️ سیگنال ثبت شد اما تلگرام ارسال نشد: {clean_symbol}")
                return {
                    "status": "registered_no_telegram",
                    "symbol": clean_symbol,
                    "side": side,
                    "entry": current_price,
                    "message": "سیگنال ثبت شد اما به تلگرام ارسال نشد"
                }
        
        else:
            log_message(f"ℹ️ امتیاز ناکافی {symbol}: {score:.1f} (حداقل: {SystemConfig.MIN_SCORE})")
            return {
                "status": "low_score",
                "symbol": symbol,
                "score": round(score, 2),
                "min_required": SystemConfig.MIN_SCORE
            }
            
    except Exception as e:
        log_message(f"❌ خطا در تحلیل {symbol}: {e}", "ERROR")
        return {"status": "error", "symbol": symbol, "error": str(e)}

# ==================== SCHEDULER FUNCTIONS ====================

def hourly_scan():
    """Hourly scan of watchlist"""
    log_message("⏰ شروع اسکن ساعتی")
    
    for symbol in WATCHLIST:
        try:
            analyze_symbol(symbol, force=False)
            time.sleep(1)  # Delay between symbols
        except Exception as e:
            log_message(f"❌ خطا در اسکن {symbol}: {e}", "ERROR")
            continue
    
    log_message("✅ اسکن ساعتی کامل شد")

def multi_strategy_scan():
    """Multi-strategy scan"""
    if not SystemConfig.USE_MULTI_STRATEGY:
        return
    
    log_message("🚀 شروع اسکن استراتژی ترکیبی")
    
    # Scan top coins
    for i, symbol in enumerate(WATCHLIST[:SystemConfig.TOP_COINS_LIMIT]):
        try:
            # Analyze with different timeframes
            result = analyze_symbol(symbol, force=False)
            
            if result.get('status') == 'success':
                log_message(f"✅ سیگنال پیدا شد: {symbol}")
            
            time.sleep(0.5)  # Delay
            
        except Exception as e:
            log_message(f"⚠️ خطا در تحلیل {symbol}: {e}", "WARNING")
            continue
    
    log_message("📈 اسکن استراتژی ترکیبی کامل شد")

def monitor_active_signals():
    """Monitor active signals for exit conditions"""
    while True:
        try:
            symbols_to_check = list(ACTIVE_SIGNALS.keys())
            
            if not symbols_to_check:
                time.sleep(SystemConfig.CHECK_INTERVAL)
                continue
            
            for symbol in symbols_to_check:
                if symbol not in ACTIVE_SIGNALS:
                    continue
                
                signal_data = ACTIVE_SIGNALS[symbol]
                current_price = ExchangeData.fetch_current_price(symbol)
                
                if current_price is None:
                    continue
                
                # Check exit conditions (simplified)
                levels = signal_data['exit_levels']
                side = signal_data['side']
                
                if side == 'BUY':
                    if current_price >= levels['tp2']:
                        # TP2 hit
                        close_signal(symbol, current_price, "TP2")
                    elif current_price >= levels['tp1'] and not signal_data['notifications_sent']['tp1']:
                        # TP1 hit
                        notify_target_hit(symbol, current_price, signal_data, "TP1")
                        signal_data['notifications_sent']['tp1'] = True
                    elif current_price <= levels['stop_loss']:
                        # Stop loss hit
                        close_signal(symbol, current_price, "SL")
                
                elif side == 'SELL':
                    if current_price <= levels['tp2']:
                        close_signal(symbol, current_price, "TP2")
                    elif current_price <= levels['tp1'] and not signal_data['notifications_sent']['tp1']:
                        notify_target_hit(symbol, current_price, signal_data, "TP1")
                        signal_data['notifications_sent']['tp1'] = True
                    elif current_price >= levels['stop_loss']:
                        close_signal(symbol, current_price, "SL")
            
            time.sleep(SystemConfig.CHECK_INTERVAL)
            
        except Exception as e:
            log_message(f"❌ خطا در مانیتورینگ: {e}", "ERROR")
            time.sleep(30)

def close_signal(symbol: str, close_price: float, reason: str):
    """Close a signal"""
    if symbol not in ACTIVE_SIGNALS:
        return
    
    signal_data = ACTIVE_SIGNALS[symbol]
    
    # Calculate P&L
    entry = signal_data['entry']
    if signal_data['side'] == 'BUY':
        pnl_percent = ((close_price - entry) / entry) * 100
    else:
        pnl_percent = ((entry - close_price) / entry) * 100
    
    # Update signal data
    signal_data['closed_at'] = close_price
    signal_data['closed_time'] = get_iran_time().isoformat()
    signal_data['close_reason'] = reason
    signal_data['pnl_percent'] = round(pnl_percent, 2)
    signal_data['status'] = f"CLOSED_{reason}"
    
    # Send notification
    emoji = "💰" if "TP" in reason else "🛑"
    title = "تارگت نهایی" if reason == "TP2" else "استاپ لاس" if reason == "SL" else "تارگت اول"
    
    telegram_msg = (
        f"{emoji} *{symbol} - {title}*\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📊 نماد: {symbol}\n"
        f"💰 قیمت بسته شدن: `{close_price:.2f}`\n"
        f"📈 قیمت ورود: `{entry:.2f}`\n"
        f"📊 سود/ضرر: `{pnl_percent:.2f}%`\n"
        f"🎯 دلیل: {reason}\n"
        f"⏰ زمان: {get_iran_time().strftime('%H:%M:%S')}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"✅ پوزیشن بسته شد"
    )
    
    send_telegram_message(telegram_msg)
    
    # Remove from active signals
    del ACTIVE_SIGNALS[symbol]
    
    log_message(f"📋 سیگنال {symbol} بسته شد: {reason} ({pnl_percent:.2f}%)")

def notify_target_hit(symbol: str, price: float, signal_data: Dict, target: str):
    """Notify when target is hit"""
    telegram_msg = (
        f"✅ *{symbol} - {target} Hit*\n"
        f"🎯 {target}: `{signal_data['exit_levels'][target.lower()]:.2f}`\n"
        f"💰 قیمت فعلی: `{price:.2f}`\n"
        f"🔄 پوزیشن همچنان باز است\n"
        f"⏰ {get_iran_time().strftime('%H:%M:%S')}"
    )
    
    send_telegram_message(telegram_msg)

def run_scheduler():
    """Run the job scheduler"""
    # Schedule hourly scan at minute 0
    schedule.every().hour.at(":00").do(hourly_scan)
    
    # Schedule multi-strategy scan every 2 hours
    schedule.every(SystemConfig.MULTI_STRATEGY_INTERVAL).seconds.do(multi_strategy_scan)
    
    # Initial scan
    threading.Thread(target=hourly_scan, daemon=True).start()
    
    log_message("⏰ زمان‌بند راه‌اندازی شد")
    
    while True:
        schedule.run_pending()
        time.sleep(30)

# ==================== API ROUTES ====================

@app.route('/')
def home():
    """Home page"""
    return jsonify({
        "status": "online",
        "name": "Crypto Trading System",
        "version": "4.0",
        "telegram": f"@{TELEGRAM_BOT_TOKEN.split(':')[0] if ':' in TELEGRAM_BOT_TOKEN else 'N/A'}",
        "iran_time": get_iran_time().strftime('%Y-%m-%d %H:%M:%S'),
        "active_signals": len(ACTIVE_SIGNALS),
        "config": {
            "min_score": SystemConfig.MIN_SCORE,
            "trading_hours": SystemConfig.TRADING_HOURS,
            "watchlist": WATCHLIST
        }
    })

@app.route('/analyze/<symbol>')
def api_analyze(symbol: str):
    """Analyze a symbol"""
    force = request.args.get('force', 'false').lower() == 'true'
    result = analyze_symbol(symbol, force)
    return jsonify(result)

@app.route('/scan')
def api_scan():
    """Scan all symbols"""
    results = []
    
    for symbol in WATCHLIST:
        try:
            result = analyze_symbol(symbol, force=True)
            results.append(result)
            time.sleep(0.5)
        except Exception as e:
            results.append({
                "symbol": symbol,
                "status": "error",
                "error": str(e)
            })
    
    return jsonify({
        "status": "completed",
        "total": len(WATCHLIST),
        "results": results
    })

@app.route('/signals')
def api_signals():
    """Get active signals"""
    return jsonify({
        "active": list(ACTIVE_SIGNALS.values()),
        "count": len(ACTIVE_SIGNALS),
        "total_history": len(SIGNAL_HISTORY)
    })

@app.route('/stats')
def api_stats():
    """System statistics"""
    successful = len([s for s in SIGNAL_HISTORY if s.get('status', '').startswith('CLOSED_TP')])
    stopped = len([s for s in SIGNAL_HISTORY if s.get('status') == 'CLOSED_SL'])
    
    return jsonify({
        "system": {
            "uptime": str(get_iran_time() - SYSTEM_START_TIME),
            "start_time": SYSTEM_START_TIME.strftime('%Y-%m-%d %H:%M:%S'),
            "current_time": get_iran_time().strftime('%Y-%m-%d %H:%M:%S')
        },
        "performance": {
            "total_signals": len(SIGNAL_HISTORY),
            "active_signals": len(ACTIVE_SIGNALS),
            "successful": successful,
            "stopped": stopped,
            "win_rate": f"{(successful/(successful+stopped)*100):.1f}%" if (successful+stopped) > 0 else "0%"
        },
        "telegram": {
            "connected": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "chat_id": TELEGRAM_CHAT_ID
        }
    })

@app.route('/test')
def test_telegram():
    """Test Telegram connection"""
    test_msg = (
        "🤖 *تست سیستم ترید*\n"
        "━━━━━━━━━━━━━━━\n"
        "✅ اتصال تلگرام موفق\n"
        "🚀 سیستم: نسخه ۴.۰\n"
        "📊 امکانات: تحلیل تکنیکال کامل\n"
        "⏰ زمان: " + get_iran_time().strftime('%H:%M:%S') + "\n"
        "━━━━━━━━━━━━━━━\n"
        "📈 آماده ارسال سیگنال!"
    )
    
    success = send_telegram_message(test_msg)
    
    return jsonify({
        "status": "success" if success else "error",
        "message": "پیام تست ارسال شد" if success else "خطا در ارسال",
        "timestamp": get_iran_time().isoformat()
    })

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Load history
    load_signal_history()
    
    # Print startup banner
    print("\n" + "="*70)
    print("🚀 CRYPTO TRADING SYSTEM v4.0")
    print("="*70)
    print(f"📅 تاریخ: {get_iran_time().strftime('%Y-%m-%d')}")
    print(f"⏰ ساعت: {get_iran_time().strftime('%H:%M:%S')}")
    print(f"🤖 تلگرام: {'✅ فعال' if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else '❌ غیرفعال'}")
    print(f"📊 واچ‌لیست: {len(WATCHLIST)} نماد")
    print(f"⚙️ حداقل امتیاز: {SystemConfig.MIN_SCORE}")
    print("="*70)
    
    # Send startup message
    startup_msg = (
        "🚀 *سیستم ترید راه‌اندازی شد*\n"
        "━━━━━━━━━━━━━━━\n"
        f"📅 تاریخ: {get_iran_time().strftime('%Y-%m-%d')}\n"
        f"⏰ ساعت: {get_iran_time().strftime('%H:%M:%S')}\n"
        f"📊 واچ‌لیست: {len(WATCHLIST)} نماد\n"
        f"🤖 ربات: @CryptoAseman122_bot\n"
        "━━━━━━━━━━━━━━━\n"
        "✅ سیستم آماده ارسال سیگنال!"
    )
    send_telegram_message(startup_msg)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_active_signals, daemon=True)
    monitor_thread.start()
    
    # Start scheduler thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Register exit handler
    import atexit
    atexit.register(save_signal_history)
    atexit.register(lambda: send_telegram_message("🔄 سیستم در حال خاموش شدن..."))
    
    print(f"🌐 سرور در حال راه‌اندازی روی پورت {port}...")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
