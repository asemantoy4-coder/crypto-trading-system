"""
Test script for Crypto AI Trading System API
For structure where main.py is inside api/ folder
"""

import requests
import json
import time
import sys
import os

# Add parent directory to path for local imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
BASE_URL = "http://localhost:8000"  # For local testing
# For production testing, use your Render URL:
# BASE_URL = "https://your-app.onrender.com"

def test_health():
    """Test health endpoint."""
    print("🧪 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS")
            print(f"   Version: {data.get('version')}")
            print(f"   Utils Available: {data.get('modules', {}).get('utils')}")
            return True
        else:
            print(f"   ❌ FAILED: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ FAILED: Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

def test_scalp_signal(symbol="DOGEUSDT", timeframe="5m"):
    """Test scalp signal endpoint."""
    print(f"\n🧪 Testing scalp signal for {symbol} ({timeframe})...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/scalp-signal",
            json={"symbol": symbol, "timeframe": timeframe},
            timeout=15
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS - Processing time: {processing_time:.2f}ms")
            print(f"   Signal: {data.get('signal')}")
            print(f"   Confidence: {data.get('confidence')}")
            print(f"   Entry Price: ${data.get('entry_price'):.8f}")
            print(f"   RSI: {data.get('rsi')}")
            print(f"   Targets: {[f'${t:.8f}' for t in data.get('targets', [])]}")
            print(f"   Stop Loss: ${data.get('stop_loss'):.8f}")
            print(f"   Risk Level: {data.get('risk_level')}")
            print(f"   Reason: {data.get('reason', 'N/A')[:50]}...")
            
            # Validate targets calculation
            entry = data.get('entry_price', 0)
            targets = data.get('targets', [])
            if entry > 0 and len(targets) >= 3:
                print(f"   📊 Target Percentages: ", end="")
                for i, target in enumerate(targets[:3], 1):
                    percent = ((target - entry) / entry) * 100
                    print(f"TP{i}: {percent:+.2f}%  ", end="")
                print()
            
            return True
        else:
            print(f"   ❌ FAILED - Status: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ❌ FAILED: Request timeout")
        return False
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

def test_ichimoku_signal(symbol="BTCUSDT", timeframe="1h"):
    """Test ichimoku signal endpoint."""
    print(f"\n🧪 Testing ichimoku signal for {symbol} ({timeframe})...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/ichimoku-scalp",
            json={"symbol": symbol, "timeframe": timeframe},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS")
            print(f"   Signal: {data.get('signal')}")
            print(f"   Confidence: {data.get('confidence')}")
            print(f"   Recommendation: {data.get('recommendation')}")
            print(f"   Trend Power: {data.get('trend_power')}/100")
            print(f"   Risk Level: {data.get('risk_level')}")
            return True
        else:
            print(f"   ❌ FAILED - Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

def test_market_data(symbol="ETHUSDT"):
    """Test market data endpoint."""
    print(f"\n🧪 Testing market data for {symbol}...")
    
    try:
        response = requests.get(
            f"{BASE_URL}/market/{symbol}?timeframe=5m",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS")
            print(f"   Current Price: ${data.get('current_price'):.2f}")
            print(f"   RSI: {data.get('rsi_14')}")
            print(f"   Change 24h: {data.get('change_24h')}%")
            print(f"   Source: {data.get('source')}")
            print(f"   Divergence: {data.get('divergence')}")
            return True
        else:
            print(f"   ❌ FAILED - Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

def test_combined_analysis(symbol="BTCUSDT"):
    """Test combined analysis endpoint."""
    print(f"\n🧪 Testing combined analysis for {symbol}...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/combined-analysis",
            json={
                "symbol": symbol,
                "timeframe": "5m",
                "include_ichimoku": True,
                "include_rsi": True,
                "include_macd": True
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS")
            print(f"   Signal: {data.get('signal')}")
            print(f"   Confidence: {data.get('confidence')}")
            print(f"   Price: ${data.get('price'):.2f}")
            return True
        else:
            print(f"   ❌ FAILED - Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("🚀 CRYPTO AI TRADING SYSTEM - COMPREHENSIVE API TESTS")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print()
    
    results = {
        "health": test_health(),
        "market_data": test_market_data("DOGEUSDT"),
        "scalp_signals": [],
        "ichimoku": test_ichimoku_signal("BTCUSDT", "1h"),
        "combined": test_combined_analysis("ETHUSDT")
    }
    
    # Test scalp signals for different symbols
    symbols = ["DOGEUSDT", "BTCUSDT", "ETHUSDT", "ALGOUSDT"]
    for symbol in symbols:
        success = test_scalp_signal(symbol, "5m")
        results["scalp_signals"].append(success)
        time.sleep(1)  # Avoid rate limiting
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, result in results.items():
        if isinstance(result, list):
            passed = sum(result)
            total = len(result)
            status = f"{passed}/{total} passed"
            total_tests += total
            passed_tests += passed
        else:
            status = "✅ PASSED" if result else "❌ FAILED"
            total_tests += 1
            passed_tests += 1 if result else 0
        
        print(f"   {test_name:20} {status}")
    
    print(f"\n   Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Check logs above.")
    
    print("=" * 70)

def quick_test():
    """Quick test for basic functionality."""
    print("🔍 Quick Test - Checking basic functionality...")
    
    if test_health():
        print("\n✅ Server is running. Testing a single scalp signal...")
        test_scalp_signal("DOGEUSDT", "5m")
    else:
        print("\n❌ Server is not responding. Please start the server first.")
        print("   Run: python api/main.py")

if __name__ == "__main__":
    # Check if we should run quick test or full test
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        # Wait a moment for server to start if just launched
        time.sleep(2)
        run_all_tests()