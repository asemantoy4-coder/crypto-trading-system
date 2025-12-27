# app.py - Ultra Simple FastAPI App
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

print("=" * 60)
print("STARTING CRYPTO TRADING API - SIMPLE VERSION")
print("=" * 60)

# Display current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"📁 Current directory: {current_dir}")
print(f"📄 Files: {os.listdir(current_dir)}")

# Try to import utils
try:
    import utils
    print("✅ Utils imported successfully")
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Utils not available: {e}")
    UTILS_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="Crypto Trading API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple endpoints
@app.get("/")
async def root():
    return {
        "message": "Crypto Trading API v1.0.0",
        "status": "Running",
        "utils_available": UTILS_AVAILABLE,
        "directory": current_dir
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": "2025-12-27T14:00:00Z"}

@app.get("/test")
async def test():
    return {"test": "success", "message": "API is working"}

# Scalp signal endpoint (simple version)
@app.post("/api/scalp-signal")
async def scalp_signal(symbol: str = "BTCUSDT", timeframe: str = "5m"):
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": "BUY",
        "confidence": 0.75,
        "message": "Simple signal response"
    }

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 Starting server on port {port}")
    print(f"📚 Documentation: http://0.0.0.0:{port}/docs")
    print(f"❤️  Health check: http://0.0.0.0:{port}/health")
    print("=" * 60)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        log_level="info"
    )