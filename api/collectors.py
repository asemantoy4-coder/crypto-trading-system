# api/collectors.py
"""
Collectors module - Lightweight version
"""

import random
from datetime import datetime

def collect_signals_from_example_site():
    """جمع‌آوری سیگنال‌های نمونه"""
    signals = []
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    for symbol in symbols:
        signals.append({
            "symbol": symbol,
            "signal": random.choice(["BUY", "SELL", "HOLD"]),
            "confidence": round(random.uniform(0.6, 0.9), 2),
            "source": "Example Site",
            "timestamp": datetime.now().isoformat()
        })
    
    return signals