# api/collectors.py - نسخه اصلاح شده بدون app
"""
Collectors module - Lightweight version
"""

import random
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def collect_signals_from_example_site(symbols=None, timeframe="5m", include_analysis=False):
    """
    جمع‌آوری سیگنال‌های نمونه - نسخه سازگار
    """
    logger.info(f"📡 جمع‌آوری سیگنال‌ها ({timeframe})")
    
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    signals = []
    
    for symbol in symbols:
        # داده‌های پایه
        base_prices = {
            'BTCUSDT': 88271.00,
            'ETHUSDT': 3450.00,
            'BNBUSDT': 590.00,
            'SOLUSDT': 175.00,
            'XRPUSDT': 0.62
        }
        
        base_price = base_prices.get(symbol.upper(), 100.00)
        current_price = base_price * random.uniform(0.99, 1.01)
        
        # سیگنال تصادفی
        signal_type = random.choices(["BUY", "SELL", "HOLD"], weights=[0.35, 0.35, 0.30])[0]
        
        if signal_type == "HOLD":
            confidence = round(random.uniform(0.5, 0.7), 2)
        else:
            confidence = round(random.uniform(0.65, 0.85), 2)
        
        # محاسبه اهداف
        if signal_type == "BUY":
            targets = [
                round(current_price * 1.01, 2),
                round(current_price * 1.02, 2),
                round(current_price * 1.03, 2)
            ]
            stop_loss = round(current_price * 0.99, 2)
        elif signal_type == "SELL":
            targets = [
                round(current_price * 0.99, 2),
                round(current_price * 0.98, 2),
                round(current_price * 0.97, 2)
            ]
            stop_loss = round(current_price * 1.01, 2)
        else:
            targets = []
            stop_loss = current_price
        
        signal = {
            "symbol": symbol.upper(),
            "signal": signal_type,
            "confidence": confidence,
            "entry_price": round(current_price, 2),
            "targets": targets,
            "stop_loss": stop_loss,
            "timeframe": timeframe,
            "source": "Example Collector",
            "reason": f"سیگنال {signal_type} بر اساس تحلیل نمونه",
            "timestamp": datetime.now().isoformat(),
            "type": "COLLECTED"
        }
        
        # اگر تحلیل درخواست شده
        if include_analysis:
            signal["analysis"] = {
                "rsi": round(30 + random.random() * 40, 1),
                "sma_20": round(current_price * random.uniform(0.99, 1.01), 2),
                "volume": round(random.uniform(1000, 10000), 2)
            }
        
        signals.append(signal)
    
    logger.info(f"✅ جمع‌آوری {len(signals)} سیگنال")
    return signals

# تابع کمکی برای main.py
def get_scalp_signals(timeframe="5m"):
    """سیگنال‌های اسکالپ"""
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    return collect_signals_from_example_site(
        symbols=symbols,
        timeframe=timeframe,
        include_analysis=True
    )

print(f"✅ collectors.py loaded - Simple version")