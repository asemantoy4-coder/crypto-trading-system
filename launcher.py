# launcher.py
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    import uvicorn
    from api.main import app
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 Starting server on {host}:{port}")
    print(f"📁 Project root: {project_root}")
    print(f"📦 Python path: {sys.path}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )