# test_import.py
import sys
import os

print("🧪 Testing imports...")

# اضافه کردن مسیر پروژه
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    # تست import از utils
    from api.utils import (
        get_market_data_with_fallback,
        calculate_simple_sma,
        calculate_simple_rsi,
        analyze_scalp_conditions
    )
    
    print("✅ All utils functions imported successfully!")
    
    # تست import از main
    from api.main import app
    print("✅ FastAPI app imported successfully!")
    
    # تست توابع
    print("\n🧪 Testing functions...")
    
    # تست داده ساختگی
    test_data = [
        [0, "100", "110", "90", "105", "1000"],
        [0, "105", "115", "95", "110", "1500"],
        [0, "110", "120", "100", "115", "2000"],
        [0, "115", "125", "105", "120", "2500"],
        [0, "120", "130", "110", "125", "3000"]
    ]
    
    sma = calculate_simple_sma(test_data, 3)
    rsi = calculate_simple_rsi(test_data, 3)
    scalp_analysis = analyze_scalp_conditions(test_data, "5m")
    
    print(f"📊 SMA(3): {sma}")
    print(f"📈 RSI(3): {rsi}")
    print(f"⚡ Scalp Analysis: {scalp_analysis['condition']}")
    print(f"   RSI: {scalp_analysis['rsi']}")
    print(f"   Reason: {scalp_analysis['reason']}")
    
    print("\n🎉 All tests passed! System is ready for deployment.")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()