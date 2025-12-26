# main_direct.py (در root پروژه)
"""
This is a direct version that runs without api. prefix
"""

import sys
import os

# First, try to run the original main.py from api directory
try:
    # Change to api directory
    api_dir = os.path.join(os.path.dirname(__file__), 'api')
    if os.path.exists(api_dir):
        os.chdir(api_dir)
        sys.path.insert(0, api_dir)
        
        print(f"Changed to directory: {api_dir}")
        
        # Now run the main.py from api directory
        import main
        app = main.app
        print("✅ Successfully loaded app from api/main.py")
    else:
        raise Exception("api directory not found")
        
except Exception as e:
    print(f"❌ Failed to load from api directory: {e}")
    
    # Fallback: Create a simple app
    from fastapi import FastAPI
    app = FastAPI(title="Direct API")
    
    @app.get("/")
    async def root():
        return {"message": "Direct API running"}
    
    @app.get("/api/health")
    async def health():
        return {"status": "healthy", "mode": "direct"}

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"🚀 Starting Direct API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)