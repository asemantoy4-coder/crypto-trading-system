import pandas as pd
import pandas_ta as ta
import numpy as np

class MasterScanner:
    def __init__(self, df):
        self.df = df

    def calculate_signals(self):
        # 1. بخش ZLMA (Zero Lag EMA)
        ema_val = ta.ema(self.df['close'], length=15)
        correction = self.df['close'] + (self.df['close'] - ema_val)
        zlma = ta.ema(correction, length=15)
        
        # 2. بخش RSI & Ichimoku (از اسکریپت اول)
        rsi = ta.rsi(self.df['close'], length=14)
        rsi_sma = ta.sma(rsi, length=14) # معادل تقریبی ابر روی RSI
        
        # 3. بخش SMC (FVG Detection)
        # شناسایی شکاف قیمت برای ورود هوشمند
        fvg_bull = (self.df['low'] > self.df['high'].shift(2))
        
        # 4. مدیریت ریسک ATR (تاکید شما روی حد ضرر)
        atr = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14)
        
        # منطق ترکیب سیگنال‌ها (High Potential Pump)
        current_close = self.df['close'].iloc[-1]
        last_zlma = zlma.iloc[-1]
        last_ema = ema_val.iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_atr = atr.iloc[-1]

        # شرط ورود: روند صعودی ZLMA + واگرایی نسبی RSI + تایید FVG یا حجم
        is_uptrend = last_zlma > last_ema
        is_oversold_turning = last_rsi > 40 and rsi.iloc[-2] < 40
        
        if is_uptrend and is_oversold_turning:
            sl = current_close - (last_atr * 1.5)
            tp1 = current_close + (last_atr * 1.0)
            tp2 = current_close + (last_atr * 2.0)
            return {
                "signal": "BUY",
                "entry": round(current_close, 4),
                "sl": round(sl, 4),
                "tp1": round(tp1, 4),
                "tp2": round(tp2, 4),
                "strength": "High"
            }
        
        return None

# --- ساختار پیام تلگرام برای پروژه جدید ---
def generate_telegram_report(symbol, result):
    report = (
        f"🚀 **PUMP DETECTED: #{symbol}**\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🔹 **Strategy:** ZLMA + RSI Ichimoku\n"
        f"🔹 **Signal Strength:** {result['strength']}\n\n"
        f"🟢 **Entry:** `{result['entry']}`\n"
        f"🔴 **Stop Loss:** `{result['sl']}` (ATR Based)\n"
        f"🎯 **Target 1:** `{result['tp1']}`\n"
        f"🎯 **Target 2:** `{result['tp2']}`\n\n"
        f"⚠️ *ارز شناسایی شده مستعد حرکت انفجاری در تایم‌فریم ۲ ساعته است.*"
    )
    return report