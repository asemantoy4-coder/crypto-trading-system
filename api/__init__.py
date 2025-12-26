# api/__init__.py
print("📦 Initializing API package...")

try:
    # First try direct import
    from .main import app
    __all__ = ['app']
    print("✅ API module initialized successfully from .main")
except ImportError as e:
    print(f"⚠️ Relative import failed: {e}")
    try:
        # Try absolute import
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from main import app
        __all__ = ['app']
        print("✅ API module initialized successfully from main")
    except ImportError as e2:
        print(f"❌ Absolute import failed: {e2}")
        # Create a fallback app
        from fastapi import FastAPI
        app = FastAPI(title="Fallback API")
        __all__ = ['app']
        print("⚠️ Created fallback FastAPI app")