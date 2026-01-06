import os
import time
import threading
import schedule
from flask import Flask, jsonify
from datetime import datetime
import pytz
import pandas as pd
import ccxt
import requests
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی
load_dotenv()

# ۱. راه‌اندازی اپلیکیشن Flask
app = Flask(__name__)
port = int(os.getenv("PORT", 5000))

# ۲. تنظیمات
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EXCHANGES = ["bybit", "mexc"]  # اولویت صرافی‌ها

# تنظیمات سیستم
class Config:
    CHECK_INTERVAL = 7200  # ثانیه (۲ ساعت)
    TOP_COINS_LIMIT = 100  # تعداد ارزهای برتر
    MIN_VOLUME = 1000000  # حداقل حجم ۲۴ ساعته (دلار)

# ۳. متغیرهای گلوبال
exchange_instance = None
system_start_time = datetime.now(pytz.timezone('Asia/Tehran'))

# ۴. توابع کمکی
def get_iran_time():
    """دریافت زمان تهران"""
    return datetime.now(pytz.timezone('Asia/Tehran'))

def send_telegram_message(text):
    """ارسال پیام به تلگرام"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"پیام تلگرام: {text[:100]}...")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': text,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"خطا در ارسال تلگرام: {e}")
        return False

def initialize_exchange():
    """راه‌اندازی اتصال به صرافی"""
    global exchange_instance
    
    for exchange_name in EXCHANGES:
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange_instance = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            print(f"✅ اتصال به صرافی {exchange_name.upper()} برقرار شد")
            return True
        except Exception as e:
            print(f"❌ خطا در اتصال به {exchange_name}: {e}")
            continue
    
    print("❌ نتوانستیم به هیچ صرافی متصل شویم")
    return False

def fetch_top_coins(limit=Config.TOP_COINS_LIMIT):
    """دریافت ارزهای برتر بر اساس حجم"""
    if not exchange_instance:
        print("❌ صرافی راه‌اندازی نشده")
        return []
    
    try:
        print("🔍 در حال دریافت ارزهای برتر...")
        tickers = exchange_instance.fetch_tickers()
        
        # فیلتر جفت‌های USDT
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT'):
                volume = ticker.get('quoteVolume', 0)
                if volume >= Config.MIN_VOLUME:
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': volume,
                        'price': ticker.get('last', 0)
                    })
        
        # مرتب‌سازی بر اساس حجم
        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        
        # انتخاب برترین‌ها
        top_coins = [pair['symbol'] for pair in usdt_pairs[:limit]]
        
        print(f"✅ {len(top_coins)} ارز برتر دریافت شد")
        return top_coins
        
    except Exception as e:
        print(f"❌ خطا در دریافت ارزها: {e}")
        return []

def analyze_coin(symbol):
    """تحلیل ساده یک ارز"""
    try:
        # دریافت داده قیمت
        ohlcv = exchange_instance.fetch_ohlcv(symbol, '1h', limit=50)
        
        if not ohlcv:
            return None
        
        # تبدیل به DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # محاسبات ساده
        close_prices = df['close']
        current_price = close_prices.iloc[-1]
        price_24h_ago = close_prices.iloc[0] if len(close_prices) >= 24 else close_prices.iloc[0]
        
        # تغییرات درصدی
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        
        # تحلیل ساده
        score = 0
        reasons = []
        
        if change_24h > 5:
            score += 2
            reasons.append("📈 رشد ۲۴ ساعته قوی")
        elif change_24h < -5:
            score -= 2
            reasons.append("📉 افت ۲۴ ساعته")
        
        # حجم فعلی
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].mean()
        
        if current_volume > avg_volume * 1.5:
            score += 1
            reasons.append("🔥 افزایش حجم معاملات")
        
        return {
            'symbol': symbol,
            'price': current_price,
            'change_24h': change_24h,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
            'score': score,
            'reasons': reasons
        }
        
    except Exception as e:
        print(f"❌ خطا در تحلیل {symbol}: {e}")
        return None

def scan_coins():
    """اسکن ارزهای برتر"""
    print(f"\n{'='*50}")
    print(f"🚀 شروع اسکن - {get_iran_time().strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    
    # دریافت ارزهای برتر
    top_coins = fetch_top_coins(50)  # فقط ۵۰ ارز برای شروع
    
    if not top_coins:
        print("❌ هیچ ارزی دریافت نشد")
        return
    
    print(f"🔍 تحلیل {len(top_coins)} ارز...")
    
    candidates = []
    
    for symbol in top_coins[:20]:  # فقط ۲۰ ارز اول برای تست
        try:
            analysis = analyze_coin(symbol)
            if analysis and analysis['score'] > 0:
                candidates.append(analysis)
                print(f"✅ کاندید: {symbol} - امتیاز: {analysis['score']}")
            
            time.sleep(0.5)  # جلوگیری از Rate Limit
            
        except Exception as e:
            print(f"⚠️ خطا در {symbol}: {e}")
            continue
    
    # ارسال نتایج
    if candidates:
        send_results(candidates)
    else:
        print("ℹ️ هیچ کاندید مناسبی یافت نشد")
    
    print(f"{'='*50}")
    print(f"✅ اسکن کامل شد. {len(candidates)} کاندید یافت شد.")
    print(f"{'='*50}\n")

def send_results(candidates):
    """ارسال نتایج به تلگرام"""
    # مرتب‌سازی بر اساس امتیاز
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # ساخت پیام
    msg = f"<b>🚨 اسکن ارزهای برتر</b>\n"
    msg += f"⏰ زمان: {get_iran_time().strftime('%H:%M:%S')}\n"
    msg += f"📊 تعداد کاندیدها: {len(candidates)}\n\n"
    
    for i, coin in enumerate(candidates[:5], 1):  # فقط ۵ تا برتر
        symbol_clean = coin['symbol'].replace('/', '')
        msg += f"<b>{i}. {symbol_clean}</b>\n"
        msg += f"💰 قیمت: ${coin['price']:.4f}\n"
        msg += f"📈 تغییر ۲۴h: {coin['change_24h']:+.2f}%\n"
        msg += f"🎯 امتیاز: {coin['score']}/3\n"
        
        for reason in coin['reasons']:
            msg += f"• {reason}\n"
        
        msg += "\n"
    
    msg += f"\n<i>توجه: این تحلیل آموزشی است و توصیه مالی نیست.</i>"
    
    # ارسال پیام
    send_telegram_message(msg)
    print(f"📤 نتایج به تلگرام ارسال شد")

# ۵. مسیرهای Flask
@app.route('/')
def home():
    """صفحه اصلی"""
    return jsonify({
        "status": "online",
        "name": "Crypto Pump Scanner",
        "version": "1.0",
        "iran_time": get_iran_time().strftime('%Y-%m-%d %H:%M:%S'),
        "uptime": str(datetime.now(pytz.timezone('Asia/Tehran')) - system_start_time),
        "exchange": exchange_instance.name if exchange_instance else "Not connected"
    })

@app.route('/scan')
def scan_now():
    """اجرای دستی اسکن"""
    scan_coins()
    return jsonify({
        "status": "scan_started",
        "time": get_iran_time().strftime('%H:%M:%S')
    })

@app.route('/health')
def health_check():
    """بررسی سلامت سیستم"""
    exchange_status = "connected" if exchange_instance else "disconnected"
    
    return jsonify({
        "status": "healthy",
        "exchange": exchange_status,
        "timestamp": get_iran_time().isoformat()
    })

# ۶. زمان‌بندی
def run_scheduler():
    """اجرای زمان‌بند"""
    # اسکن هر ۲ ساعت
    schedule.every(Config.CHECK_INTERVAL).seconds.do(scan_coins)
    
    # همچنین یک بار در روز نیمه‌شب
    schedule.every().day.at("00:00").do(scan_coins)
    
    print("⏰ زمان‌بند راه‌اندازی شد")
    print(f"📅 اسکن هر {Config.CHECK_INTERVAL//3600} ساعت اجرا می‌شود")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# ۷. راه‌اندازی اصلی
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("🚀 راه‌اندازی ربات اسکنر پامپ ارزهای دیجیتال")
    print(f"{'='*60}")
    
    # راه‌اندازی اتصال به صرافی
    if not initialize_exchange():
        print("❌ لطفا اتصال اینترنت و API صرافی‌ها را بررسی کنید")
        exit(1)
    
    # اطلاعات اولیه
    print(f"📅 تاریخ: {get_iran_time().strftime('%Y-%m-%d')}")
    print(f"⏰ ساعت: {get_iran_time().strftime('%H:%M:%S')}")
    print(f"🏦 صرافی: {exchange_instance.name if exchange_instance else 'N/A'}")
    print(f"📊 تعداد ارزها: {Config.TOP_COINS_LIMIT}")
    print(f"⏰ بازه اسکن: هر {Config.CHECK_INTERVAL//3600} ساعت")
    print(f"{'='*60}\n")
    
    # راه‌اندازی زمان‌بند در Thread جداگانه
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # تست اولیه
    print("🔍 تست اولیه اتصال...")
    test_coins = fetch_top_coins(5)
    if test_coins:
        print(f"✅ تست موفق: {len(test_coins)} ارز دریافت شد")
    else:
        print("⚠️ تست اولیه ناموفق بود")
    
    print(f"\n🌐 سرور در حال راه‌اندازی روی پورت {port}...")
    print(f"📌 آدرس‌های قابل دسترسی:")
    print(f"   • http://localhost:{port}/")
    print(f"   • http://localhost:{port}/scan")
    print(f"   • http://localhost:{port}/health")
    print(f"{'='*60}\n")
    
    # اجرای سرور Flask
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
