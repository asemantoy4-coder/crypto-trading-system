"""
Configuration for Crypto Trading System
"""

API_VERSION = "1.0.0"
API_TITLE = "Crypto Trading API"
API_DESCRIPTION = "Real-time cryptocurrency signals and analysis"

# LBank API
LBANK_API_BASE = "https://api.lbkex.com/v2"

# Symbol Mapping
SYMBOL_MAPPING = {
    'BTCUSDT': 'btc_usdt',
    'ETHUSDT': 'eth_usdt',
    'BNBUSDT': 'bnb_usdt',
    'SOLUSDT': 'sol_usdt'
}

def get_version():
    return API_VERSION

def get_all_config():
    return {
        "version": API_VERSION,
        "title": API_TITLE,
        "description": API_DESCRIPTION
    }