import os, time, threading, schedule, json, pytz
from flask import Flask, jsonify, request
from datetime import datetime
import pandas as pd
import exchange_handler, utils, config
from strategies import calculate_master_signals
from typing import Dict, List, Any

# ۱. تنظیمات پایه و اپلیکیشن
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# متغیرهای حافظه سیستم
WATCHLIST = getattr(config, 'WATCHLIST', ["BTCUSDT", "ETHUSDT"])
ACTIVE_SIGNALS: Dict[str, Dict] = {}
SIGNAL_HISTORY: List[Dict] = []
SYSTEM_START_TIME = datetime.now(pytz.timezone('Asia/Tehran'))

class SystemConfig:
    CHECK_INTERVAL = 20  # ثانیه
    MIN_SCORE = 3
    TRADING_HOURS = (0, 23)
    MAX_HISTORY = 100
    RISK_FREE_ENABLED = True
    MULTI_STRATEGY_INTERVAL = 7200 # ۲ ساعت

# ==================== توابع کمکی و لاگ ====================
def get_now(): 
    return datetime.now(pytz.timezone('Asia/Tehran'))

def log(lvl, sym, msg):
    icons = {'info': '📊', 'success': '✅', 'error': '❌', 'warning': '⚠️', 'signal': '🚀'}
    print(f"[{get_now().strftime('%H:%M:%S')}] {icons.get(lvl, '📝')} {sym}: {msg}")

def load_history():
    global SIGNAL_HISTORY
    if os.path.exists('signal_history.json'):
        try:
            with open('signal_history.json', 'r') as f:
                SIGNAL_HISTORY = json.load(f)
        except: SIGNAL_HISTORY = []

def save_history():
    try:
        with open('signal_history.json', 'w') as f:
            json.dump(SIGNAL_HISTORY[-SystemConfig.MAX_HISTORY:], f, indent=2)
    except: pass

# ==================== استراتژی ترکیبی (Multi-Strategy) ====================
def analyze_with_multi_strategy(symbol, timeframe='1h'):
    try:
        clean_sym = symbol.replace("/", "").upper()
        df = exchange_handler.DataHandler.fetch_data(clean_sym, timeframe, limit=100)
        if df is None or df.empty: return {"status": "no_data"}
        
        # استراتژی اصلی: ZLMA + RSI + FVG
        is_bull, price, atr, has_fvg = calculate_master_signals(df)
        
        if is_bull:
            current_price = df['close'].iloc[-1]
            sl = current_price - (atr * 1.5)
            tp = current_price + (atr * 2.5)
            
            sig = {
                'symbol': clean_sym, 'side': 'BUY', 'entry': current_price,
                'exit_levels': {'tp1': tp, 'stop_loss': sl, 'direction': 'BUY'},
                'timestamp': get_now().isoformat(), 'status': 'ACTIVE',
                'strategy': 'MULTI', 'has_fvg': has_fvg,
                'notifications_sent': {'tp1': False, 'sl': False}
            }
            
            if clean_sym not in ACTIVE_SIGNALS:
                ACTIVE_SIGNALS[clean_sym] = sig
                SIGNAL_HISTORY.append(sig.copy())
                msg = (f"🚀 *MULTI-STRATEGY SIGNAL*\n"
                       f"📊 Symbol: #{clean_sym}\n"
                       f"🟢 Entry: `{current_price:.4f}`\n"
                       f"🎯 Target: `{tp:.4f}`\n"
                       f"🛑 Stop: `{sl:.4f}`\n"
                       f"🧱 FVG Confirm: {'✅' if has_fvg else '❌'}")
                utils.send_telegram_notification(msg, 'BUY')
                log('signal', clean_sym, "Multi-Strategy Signal Sent")
                return {"status": "success"}
        return {"status": "no_signal"}
    except Exception as e:
        log('error', symbol, f"Multi-Strat Error: {e}")
        return {"status": "error"}

# ==================== استراتژی اسکالپ (Scalp) ====================
def analyze_and_broadcast(symbol, force=False):
    try:
        clean_sym = symbol.replace("/", "").upper()
        df = exchange_handler.DataHandler.fetch_data(clean_sym, '5m', limit=100)
        if df is None or df.empty: return
        
        res = utils.generate_scalp_signals(df)
        score, price = res.get('score', 0), res.get('price', 0)
        
        if abs(score) >= SystemConfig.MIN_SCORE or force:
            side = "BUY" if score >= 0 else "SELL"
            sl = price * 0.995 if side == "BUY" else price * 1.005
            risk = abs(price - sl)
            tp1 = price + (risk * 1.5) if side == "BUY" else price - (risk * 1.5)
            
            sig = {
                'symbol': clean_sym, 'side': side, 'entry': price,
                'exit_levels': {'tp1': tp1, 'stop_loss': sl, 'direction': side},
                'timestamp': get_now().isoformat(), 'status': 'ACTIVE',
                'strategy': 'SCALP', 'score': abs(score),
                'notifications_sent': {'tp1': False, 'sl': False}
            }
            
            if clean_sym not in ACTIVE_SIGNALS:
                ACTIVE_SIGNALS[clean_sym] = sig
                SIGNAL_HISTORY.append(sig.copy())
                msg = f"🔥 *SCALP SIGNAL*\n{clean_sym}: {side}\nEntry: {price:.4f}\nTP1: {tp1:.4f}\nScore: {abs(score)}"
                utils.send_telegram_notification(msg, side)
                log('signal', clean_sym, f"Scalp {side} Sent")
    except Exception as e:
        log('error', symbol, f"Scalp Error: {e}")

# ==================== مانیتورینگ و زمان‌بندی (ضد هنگ) ====================
def check_targets():
    log('info', 'SYSTEM', "Price Monitor Started")
    while True:
        try:
            if not ACTIVE_SIGNALS:
                time.sleep(15)
                continue
            
            for sym in list(ACTIVE_SIGNALS.keys()):
                sig = ACTIVE_SIGNALS[sym]
                ticker = exchange_handler.DataHandler.fetch_ticker(sym)
                if not ticker: continue
                
                p = ticker.get('last', 0)
                lv = sig['exit_levels']
                side = lv['direction']
                
                # بررسی Target
                if (side == 'BUY' and p >= lv['tp1']) or (side == 'SELL' and p <= lv['tp1']):
                    utils.send_telegram_notification(f"✅ {sym} Target Hit! Price: {p}", "INFO")
                    ACTIVE_SIGNALS.pop(sym, None)
                    save_history()
                
                # بررسی Stop Loss
                elif (side == 'BUY' and p <= lv['stop_loss']) or (side == 'SELL' and p >= lv['stop_loss']):
                    utils.send_telegram_notification(f"🛑 {sym} Stop Loss Hit! Price: {p}", "STOP")
                    ACTIVE_SIGNALS.pop(sym, None)
                    save_history()
            
            time.sleep(SystemConfig.CHECK_INTERVAL)
        except Exception as e:
            log('error', 'MONITOR', str(e))
            time.sleep(10)

def run_scheduler():
    log('info', 'SYSTEM', "Scheduler Thread Started")
    # ۱. اسکن ساعتی اسکالپ
    schedule.every().hour.at(":00").do(lambda: [analyze_and_broadcast(s) for s in WATCHLIST])
    # ۲. اسکن ۲ ساعته استراتژی ترکیبی
    schedule.every(2).hours.do(lambda: [analyze_with_multi_strategy(s) for s in WATCHLIST])
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(5)
        except Exception as e:
            log('error', 'SCHEDULER', str(e))
            time.sleep(10)

# ==================== وب‌سرویس Flask ====================
@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "active_signals": len(ACTIVE_SIGNALS),
        "uptime": str(get_now() - SYSTEM_START_TIME)
    })

@app.route('/analyze/<symbol>')
def manual_analyze(symbol):
    threading.Thread(target=analyze_and_broadcast, args=(symbol, True)).start()
    return jsonify({"message": f"Analysis started for {symbol}"})

@app.route('/multi/<symbol>')
def manual_multi(symbol):
    res = analyze_with_multi_strategy(symbol)
    return jsonify(res)

# ==================== اجرای نهایی ====================
if __name__ == "__main__":
    load_history()
    
    # اجرای بخش‌های پس‌زمینه در ترد جداگانه
    t1 = threading.Thread(target=check_targets, daemon=True)
    t2 = threading.Thread(target=run_scheduler, daemon=True)
    t1.start()
    t2.start()
    
    log('success', 'SYSTEM', f"Bot Fully Active on Port {port}")
    
    # اجرای وب‌سرور برای پایداری در Render
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
