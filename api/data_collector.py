# api/data_collector.py
"""
Data Collector - Lightweight version
"""

from datetime import datetime
import random

def get_collected_data(symbols=None, timeframe="5m", limit=50):
    """دریافت داده جمع‌آوری شده"""
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT"]
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "price_data": {},
        "market_metrics": {
            "total_market_cap": 1.8e12,
            "btc_dominance": 52.5,
            "volume_24h": 75.3e9
        },
        "summary": {
            "symbols_collected": len(symbols),
            "total_data_points": 100
        }
    }