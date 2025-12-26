# wsgi.py
import sys
import os

# Setup path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from api.main import app

# For Gunicorn
application = app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 Starting WSGI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)