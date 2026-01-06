import os, time, threading, schedule, json, pytz
from flask import Flask, jsonify, request
from datetime import datetime
import pandas as pd
import exchange_handler, utils, config
from strategies import calculate_master_signals
from typing import Dict, List, Any

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# تنظیمات حافظه و سیستم
WATCHLIST = getattr(config, 'WATCHLIST', ["BTCUSDT", "ETHUSDT"])
ACTIVE_SIGNALS: Dict[str, Dict] = {}
SIGNAL_HISTORY: List[Dict] = []

class SystemConfig:
    CHECK_INTERVAL = 20
    MIN_SCORE = 3
    TRADING_HOURS = (0, 23)
    MAX_HISTORY = 100
    RISK_FREE_ENABLED = True
    USE_MULTI_STRATEGY = True
    TOP_COINS_LIMIT = 50

def get_now(): return datetime.now(pytz.timezone('Asia/Tehran'))

def log(lvl, sym, msg):
    icons = {'info': '📊', 'success': '✅', 'error': '❌', 'scan': '🔍'}
    print(f"[{get_now().strftime('%H:%M:%S')}] {icons.get(lvl, '📝')} {sym}: {msg}")

# ==================== استراتژی ترکیبی (Multi-Strategy) ====================
def analyze_with_multi_strategy(symbol, timeframe='1h'):
    try:
        clean_sym = symbol.replace("/", "").upper()
        df = exchange_handler.DataHandler.fetch_data(clean_sym, timeframe, limit=100)
        if df is None or df.empty: return {"status": "no_data"}
        
        # فراخوانی استراتژی اصلی تو از فایل strategies
        is_bull, price, atr, has_fvg = calculate_master_signals(df)
        
        if is_bull:
            current_price = df['close'].iloc[-1]
            sl = current_price - (atr * 1.5)
            tp = current_price + (atr * 2.5)
            
            sig = {
                'symbol': clean_sym, 'side': 'BUY', 'entry': current_price,
                'exit_levels': {'tp1': tp, 'stop_loss': sl, 'direction': 'BUY'},
                'timestamp': get_now().isoformat(), 'status': 'ACTIVE',
                'strategy': 'MULTI', 'has_fvg': has_fvg
            }
            
            if clean_sym not in ACTIVE_SIGNALS:
                ACTIVE_SIGNALS[clean_sym] = sig
                SIGNAL_HISTORY.append(sig.copy())
                msg = (f"🚀 *MULTI-STRATEGY SIGNAL*\nSymbol: #{clean_sym}\nEntry: {current_price:.4f}\n"
                       f"TP: {tp:.4f}\nSL: {sl:.4f}\nFVG: {'✅' if has_fvg else '❌'}")
                utils.send_telegram_notification(msg, 'BUY')
                return {"status": "success"}
        return {"status": "no_signal"}
    except Exception as e:
        log('error', symbol, f"Multi-Strat Error: {e}")
        return {"status": "error"}

# ==================== استراتژی اسکالپ و مانیتورینگ ====================
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
            tp1 = price + (abs(price-sl) * 1.5) if side == "BUY" else price - (abs(price-sl) * 1.5)
            
            sig = {
                'symbol': clean_sym, 'side': side, 'entry': price,
                'exit_levels': {'tp1': tp1, 'stop_loss': sl, 'direction': side},
                'timestamp': get_now().isoformat(), 'status': 'ACTIVE',
                'strategy': 'SCALP'
            }
            ACTIVE_SIGNALS[clean_sym] = sig
            SIGNAL_HISTORY.append(sig.copy())
            utils.send_telegram_notification(f"🔥 *SCALP SIGNAL*\n{clean_sym}: {side}\nEntry: {price}", side)
    except Exception as e: log('error', symbol, str(e))

def check_targets():
    while True:
        try:
            for sym in list(ACTIVE_SIGNALS.keys()):
                sig = ACTIVE_SIGNALS[sym]
                ticker = exchange_handler.DataHandler.fetch_ticker(sym)
                if not ticker: continue
                p, lv = ticker.get('last', 0), sig['exit_levels']
                
                # خروج در سود (TP)
                if (lv['direction'] == 'BUY' and p >= lv['tp1']) or (lv['direction'] == 'SELL' and p <= lv['tp1']):
                    utils.send_telegram_notification(f"✅ {sym} Target Hit!", "INFO")
                    del ACTIVE_SIGNALS[sym]
                
                # خروج در ضرر (SL)
                elif (lv['direction'] == 'BUY' and p <= lv['stop_loss']) or (lv['direction'] == 'SELL' and p >= lv['stop_loss']):
                    utils.send_telegram_notification(f"🛑 {sym} Stop Loss Hit!", "STOP")
                    del ACTIVE_SIGNALS[sym]
            time.sleep(SystemConfig.CHECK_INTERVAL)
        except: time.sleep(10)

def run_scheduler():
    # اسکن ساعتی واچ‌لیست (Scalp)
    schedule.every().hour.at(":00").do(lambda: [analyze_and_broadcast(s) for s in WATCHLIST])
    # اسکن ارزهای برتر (Multi-Strategy) هر ۲ ساعت
    schedule.every(2).hours.do(lambda: [analyze_with_multi_strategy(s) for s in WATCHLIST[:10]])
    while True:
        schedule.run_pending()
        time.sleep(5)

# ==================== مسیرهای Flask ====================
@app.route('/')
def home(): return jsonify({"status": "running", "active": len(ACTIVE_SIGNALS)})

@app.route('/multi/<symbol>')
def force_multi(symbol): return jsonify(analyze_with_multi_strategy(symbol))

if __name__ == "__main__":
    threading.Thread(target=check_targets, daemon=True).start()
    threading.Thread(target=run_scheduler, daemon=True).start()
    log('info', 'SYSTEM', "All Strategies Active. Bot Started.")
    app.run(host='0.0.0.0', port=port)
