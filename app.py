# app.py - LAUNCHER FILE (در ریشه پروژه)
import os
import sys

print("=" * 60)
print("🚀 LAUNCHING CRYPTO TRADING API")
print("=" * 60)

# اضافه کردن مسیر api به sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
api_dir = os.path.join(current_dir, "api")

print(f"📁 Root directory: {current_dir}")
print(f"📁 API directory: {api_dir}")

# چک کردن وجود فایل‌ها
print("\n📄 Checking files:")
for f in ["api/__init__.py", "api/main.py", "api/utils.py"]:
    path = os.path.join(current_dir, f)
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"  {exists} {f}")

# اضافه کردن مسیرها
sys.path.insert(0, current_dir)
sys.path.insert(0, api_dir)

print(f"\n📦 Python path: {sys.path}")

# حالا api.main را import می‌کنیم
try:
    from api.main import app
    print("✅ SUCCESS: Imported app from api.main")
    
    # برای اجرای مستقیم
    if __name__ == "__main__":
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        print(f"\n🌐 Starting server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
        
except ImportError as e:
    print(f"❌ ERROR: Could not import api.main: {e}")
    
    # ایجاد یک app ساده به عنوان fallback
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/")
    async def root():
        return {"error": "Could not load main module", "details": str(e)}
    
    if __name__ == "__main__":
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
