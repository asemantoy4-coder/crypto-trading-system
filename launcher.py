# launcher.py
import sys
import os

print("🚀 Crypto Trading API Launcher")
print(f"Current directory: {os.getcwd()}")

# Add multiple possible paths
possible_paths = [
    os.path.dirname(os.path.abspath(__file__)),  # Current directory
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api'),  # api directory
    '/opt/render/project/src',  # Render default
    '/opt/render/project/src/api',  # Render api directory
]

for path in possible_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added to path: {path}")

print(f"Python path: {sys.path}")

try:
    print("Trying to import from api.main...")
    from api.main import app
    print("✅ Success: Imported from api.main")
except ImportError as e:
    print(f"❌ api.main import failed: {e}")
    try:
        print("Trying to import from main...")
        from main import app
        print("✅ Success: Imported from main")
    except ImportError as e2:
        print(f"❌ main import failed: {e2}")
        print("Creating new app...")
        from fastapi import FastAPI
        app = FastAPI(title="New App")
        print("✅ Created new FastAPI app")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"\n🌐 Starting server on {host}:{port}")
    print(f"📡 App title: {app.title}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
