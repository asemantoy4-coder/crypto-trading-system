import os
import time
import threading
import schedule
from flask import Flask, jsonify, request
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import json
import requests
from typing import Dict, List
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی
load_dotenv()

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# ==================== CONFIGURATION ====================
# استفاده از متغیرهای محیطی Render
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# اگر در محیط توسعه هستیم و متغیرها تنظیم نشده، از مقادیر تستی استفاده کن
if not TELEGRAM_BOT_TOKEN:
    TELEGRAM_BOT_TOKEN = "8396237816:AAFBwYRj319UI1FxTG_EjdoLsgfRDsWMImY"
if not TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_ID = "7037205717"

WATCHLIST = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]

print("="*60)
print("🚀 سیستم ترید ارز دیجیتال")
print("="*60)
print(f"🤖 ربات تلگرام: {'✅ تنظیم شده' if TELEGRAM_BOT_TOKEN else '❌ تنظیم نشده'}")
print(f"👤 Chat ID: {TELEGRAM_CHAT_ID}")
print(f"📊 واچ‌لیست: {len(WATCHLIST)} نماد")
print("="*60)

# ==================== GLOBAL VARIABLES ====================
ACTIVE_SIGNALS: Dict[str, Dict] = {}
SIGNAL_HISTORY: List[Dict] = []

# ==================== HELPER FUNCTIONS ====================

def get_iran_time():
    return datetime.now(pytz.timezone('Asia/Tehran'))

def send_telegram_message(text: str) -> bool:
    """ارسال پیام به تلگرام"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ تلگرام تنظیم نشده")
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
            print(f"✅ پیام تلگرام ارسال شد به {TELEGRAM_CHAT_ID}")
            return True
        else:
            print(f"❌ خطای تلگرام: {response.status_code}")
            print(f"📝 پاسخ: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ خطا در ارسال تلگرام: {e}")
        return False

# ==================== TRADING FUNCTIONS ====================

def analyze_symbol(symbol: str, force: bool = False) -> Dict:
    """تحلیل یک نماد و ارسال سیگنال"""
    try:
        # تنظیم نماد
        if '/' not in symbol and 'USDT' in symbol:
            symbol = symbol.replace('USDT', '/USDT')
        
        clean_symbol = symbol.replace("/", "").upper()
        
        # اگر سیگنال فعال داریم
        if clean_symbol in ACTIVE_SIGNALS and not force:
            return {
                "status": "active_signal_exists",
                "symbol": clean_symbol,
                "message": "سیگنال فعال قبلی وجود دارد"
            }
        
        # داده‌های نمونه
        prices = {
            'BTC/USDT': 51234.56,
            'ETH/USDT': 3123.45,
            'BNB/USDT': 423.67,
            'ADA/USDT': 0.56,
            'SOL/USDT': 112.34
        }
        
        current_price = prices.get(symbol.upper(), 100.0)
        score = 8
        
        # ساخت پیام
        msg = (
            f"🚀 *سیگنال معاملاتی جدید*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 نماد: {symbol}\n"
            f"🟢 جهت: BUY\n"
            f"⭐ امتیاز: {score}/10\n"
            f"💰 قیمت ورود: `{current_price:.2f}`\n"
            f"🎯 تارگت ۱: `{current_price * 1.02:.2f}`\n"
            f"🎯 تارگت ۲: `{current_price * 1.04:.2f}`\n"
            f"🛑 استاپ‌لاس: `{current_price * 0.98:.2f}`\n"
            f"📊 نسبت سود/ضرر: ۱:۲\n"
            f"⏰ زمان: {get_iran_time().strftime('%H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🏷️ #{clean_symbol.replace('USDT', '')} #BUY"
        )
        
        # ارسال به تلگرام
        telegram_sent = send_telegram_message(msg)
        
        if telegram_sent:
            # ذخیره سیگنال
            signal_data = {
                'symbol': clean_symbol,
                'side': 'BUY',
                'entry': current_price,
                'score': score,
                'tp1': current_price * 1.02,
                'tp2': current_price * 1.04,
                'sl': current_price * 0.98,
                'timestamp': get_iran_time().isoformat(),
                'status': 'ACTIVE'
            }
            
            ACTIVE_SIGNALS[clean_symbol] = signal_data
            SIGNAL_HISTORY.append(signal_data)
            
            return {
                "status": "success",
                "symbol": symbol,
                "telegram_sent": True,
                "signal": signal_data
            }
        else:
            return {
                "status": "telegram_error",
                "symbol": symbol,
                "message": "خطا در ارسال تلگرام"
            }
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==================== API ROUTES ====================

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "name": "Crypto Trading System",
        "telegram_bot": "@CryptoAseman122_bot",
        "telegram_chat_id": TELEGRAM_CHAT_ID,
        "settings_from": "Render Environment Variables"
    })

@app.route('/test')
def test():
    """تست تلگرام"""
    test_msg = (
        "✅ *تست اتصال تلگرام*\n"
        f"🤖 ربات: @CryptoAseman122_bot\n"
        f"👤 Chat ID: {TELEGRAM_CHAT_ID}\n"
        f"⏰ زمان: {get_iran_time().strftime('%H:%M:%S')}\n"
        f"🌐 سرور: Render\n"
        "━━━━━━━━━━━━━━━\n"
        "🚀 سیستم کامل کار می‌کند!"
    )
    
    success = send_telegram_message(test_msg)
    
    return jsonify({
        "status": "success" if success else "error",
        "message": "پیام تست ارسال شد" if success else "خطا در ارسال",
        "telegram_chat_id": TELEGRAM_CHAT_ID
    })

@app.route('/analyze/<symbol>')
def analyze_endpoint(symbol: str):
    """تحلیل نماد"""
    force = request.args.get('force', 'false').lower() == 'true'
    return jsonify(analyze_symbol(symbol, force))

@app.route('/scan')
def scan_all():
    """اسکن همه نمادها"""
    results = []
    
    for symbol in WATCHLIST:
        try:
            result = analyze_symbol(symbol, force=True)
            results.append(result)
            time.sleep(0.3)
        except Exception as e:
            results.append({
                "symbol": symbol,
                "status": "error",
                "error": str(e)
            })
    
    return jsonify({
        "status": "completed",
        "results": results
    })

@app.route('/clear')
def clear_signals():
    """پاک کردن سیگنال‌های فعال"""
    global ACTIVE_SIGNALS
    count = len(ACTIVE_SIGNALS)
    ACTIVE_SIGNALS.clear()
    
    return jsonify({
        "status": "success",
        "message": f"{count} سیگنال پاک شد"
    })

@app.route('/signals')
def list_signals():
    """لیست سیگنال‌ها"""
    return jsonify({
        "active": list(ACTIVE_SIGNALS.values()),
        "count": len(ACTIVE_SIGNALS)
    })

@app.route('/stats')
def stats():
    """آمار"""
    return jsonify({
        "system": {
            "telegram_chat_id": TELEGRAM_CHAT_ID,
            "telegram_connected": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "watchlist_count": len(WATCHLIST)
        },
        "signals": {
            "active": len(ACTIVE_SIGNALS),
            "total": len(SIGNAL_HISTORY)
        }
    })

# ==================== MAIN ====================

if __name__ == "__main__":
    # ارسال پیام شروع
    start_msg = (
        "🚀 *سیستم ترید راه‌اندازی شد*\n"
        f"⏰ {get_iran_time().strftime('%H:%M:%S')}\n"
        f"📊 {len(WATCHLIST)} نماد در واچ‌لیست\n"
        f"👤 Chat ID: {TELEGRAM_CHAT_ID}\n"
        "✅ آماده ارسال سیگنال!"
    )
    send_telegram_message(start_msg)
    
    print(f"🌐 سرور روی پورت {port}")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=False)
