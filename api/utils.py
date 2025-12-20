# api/utils.py
"""
Utility functions - Lightweight version for Render
"""

import requests
import random
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def get_market_data_with_fallback(symbol, timeframe="5m", limit=50):
    """
    دریافت داده بازار (نسخه سبک)
    """
    try:
        # تلاش برای Binance
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': timeframe,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # بازگشت داده mock
    return generate_mock_data(symbol, limit)

def analyze_with_multi_timeframe_strategy(symbol):
    """
    تحلیل چندزمانی (نسخه شبیه‌سازی)
    """
    signals = ["BUY", "SELL", "HOLD"]
    return {
        "symbol": symbol,
        "signal": random.choice(signals),
        "confidence": round(random.uniform(0.6, 0.9), 2),
        "entry_price": round(random.uniform(50000, 51000), 2),
        "targets": [
            round(random.uniform(52000, 53000), 2),
            round(random.uniform(54000, 55000), 2)
        ],
        "stop_loss": round(random.uniform(48000, 49000), 2),
        "strategy": "Multi-Timeframe Simulation"
    }

def calculate_24h_change_from_dataframe(data):
    """محاسبه تغییرات ۲۴ ساعته"""
    if isinstance(data, list) and len(data) > 10:
        old_price = float(data[0][4])  # close price اولین کندل
        current_price = float(data[-1][4])  # close price آخرین کندل
        return ((current_price - old_price) / old_price) * 100
    return round(random.uniform(-5, 5), 2)

def generate_mock_data(symbol, limit):
    """تولید داده آزمایشی"""
    base_price = 50000 if "BTC" in symbol else 3000
    data = []
    
    for i in range(limit):
        timestamp = int((datetime.now() - timedelta(minutes=i*5)).timestamp() * 1000)
        price = base_price * (1 + random.uniform(-0.02, 0.02))
        
        data.append([
            timestamp,  # open time
            str(price * 0.998),  # open
            str(price * 1.005),  # high
            str(price * 0.995),  # low
            str(price),  # close
            str(random.uniform(1000, 10000)),  # volume
            timestamp + 300000,  # close time
            "0",  # quote asset volume
            "0",  # number of trades
            "0",  # taker buy base asset volume
            "0",  # taker buy quote asset volume
            "0"   # ignore
        ])
    
    return data