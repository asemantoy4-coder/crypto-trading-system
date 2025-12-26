"""
Crypto AI Trading System v7.7.0 - SIMPLE DEPLOYMENT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import logging
from typing import List, Optional, Dict, Any
import random
import sys
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. تابع محاسبه تارگت
# ==============================================================================

def calculate_targets_and_stop_loss(entry_price, signal, risk_level="MEDIUM"):
    """تابع مرکزی برای محاسبه تارگت‌ها و استاپ لاس"""
    if entry_price <= 0:
        return [0, 0, 0], 0, [0, 0, 0], 0
    
    risk_params = {
        "HIGH": {
            "BUY": {"targets": [1.015, 1.025, 1.035], "stop_loss": 0.985},
            "SELL": {"targets": [0.985, 0.975, 0.965], "stop_loss": 1.015},
            "HOLD": {"targets": [1.005, 1.010, 1.015], "stop_loss": 0.995}
        },
        "MEDIUM": {
            "BUY": {"targets": [1.005, 1.010, 1.015], "stop_loss": 0.995},
            "SELL": {"targets": [0.995, 0.990, 0.985], "stop_loss": 1.005},
            "HOLD": {"targets": [1.003, 1.006, 1.009], "stop_loss": 0.997}
        },
        "LOW": {
            "BUY": {"targets": [1.003, 1.006, 1.009], "stop_loss": 0.997},
            "SELL": {"targets": [0.997, 0.994, 0.991], "stop_loss": 1.003},
            "HOLD": {"targets": [1.002, 1.004, 1.006], "stop_loss": 0.998}
        }
    }
    
    params = risk_params.get(risk_level, risk_params["MEDIUM"])
    signal_params = params.get(signal, params["HOLD"])
    
    targets = [round(entry_price * multiplier, 8) for multiplier in signal_params["targets"]]
    stop_loss = round(entry_price * signal_params["stop_loss"], 8)
    
    targets_percent = [
        round(((target - entry_price) / entry_price) * 100, 2)
        for target in targets
    ]
    stop_loss_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
    
    return targets, stop_loss, targets_percent, stop_loss_percent

# ==============================================================================
# Pydantic Models
# ==============================================================================

class ScalpRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class IchimokuRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

# ==============================================================================
# FastAPI App
# ==============================================================================

app = FastAPI(
    title="Crypto AI Trading System v7.7.0",
    description="Simple Deployment Version",
    version="7.7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# API Endpoints (ساده شده)
# ==============================================================================

@app.get("/")
async def read_root():
    return {
        "message": "Crypto AI Trading System v7.7.0",
        "status": "Active",
        "endpoints": {
            "/api/health": "Health check",
            "/api/scalp": "Scalp signal (POST)",
            "/api/ichimoku": "Ichimoku signal (POST)"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "Healthy",
        "version": "7.7.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/scalp")
async def get_scalp_signal(request: ScalpRequest):
    """Scalp Signal Endpoint - ساده شده"""
    try:
        logger.info(f"Scalp request: {request.symbol} ({request.timeframe})")
        
        # Generate mock signal
        signal = random.choice(["BUY", "SELL", "HOLD"])
        weights = {"BUY": 0.4, "SELL": 0.4, "HOLD": 0.2}
        signal = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        
        # Mock price
        base_prices = {
            'BTCUSDT': 88271.00,
            'ETHUSDT': 3450.00,
            'DOGEUSDT': 0.12116,
            'ALGOUSDT': 0.1187,
            'DEFAULT': 100.0
        }
        base_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        entry_price = round(base_price * random.uniform(0.99, 1.01), 8)
        
        # Calculate targets
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_targets_and_stop_loss(
            entry_price, signal, "MEDIUM"
        )
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": signal,
            "confidence": round(random.uniform(0.6, 0.9), 2),
            "entry_price": entry_price,
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "rsi": round(random.uniform(30, 70), 1),
            "strategy": "Simple Scalp",
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])

@app.post("/api/ichimoku")
async def get_ichimoku_signal(request: IchimokuRequest):
    """Ichimoku Signal Endpoint - ساده شده"""
    try:
        logger.info(f"Ichimoku request: {request.symbol} ({request.timeframe})")
        
        # Generate mock signal
        signal = random.choice(["BUY", "SELL", "HOLD"])
        
        # Mock price
        base_prices = {
            'BTCUSDT': 88271.00,
            'ETHUSDT': 3450.00,
            'DEFAULT': 100.0
        }
        base_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        entry_price = round(base_price * random.uniform(0.99, 1.01), 8)
        
        # Calculate targets
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_targets_and_stop_loss(
            entry_price, signal, "MEDIUM"
        )
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": signal,
            "confidence": round(random.uniform(0.65, 0.95), 2),
            "entry_price": entry_price,
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "strategy": "Simple Ichimoku",
            "trend_power": round(random.uniform(40, 80), 1),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    print(f"\n{'=' * 60}")
    print(f"🚀 Crypto AI Trading System v7.7.0")
    print(f"📡 Server: http://{host}:{port}")
    print(f"📚 Docs: http://{host}:{port}/docs")
    print(f"{'=' * 60}\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )