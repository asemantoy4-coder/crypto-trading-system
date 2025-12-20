# api/config.py
"""
Configuration for Crypto Trading System
"""

def get_version():
    return "7.0.0"

def get_all_config():
    return {
        "version": "7.0.0",
        "title": "Crypto AI Trading System v7.0",
        "description": "Multi-source signal API",
        "environment": "render"
    }