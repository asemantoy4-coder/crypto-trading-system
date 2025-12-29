"""
ماژول ابزارهای کمکی برای ربات اسکالپر ارز دیجیتال
نسخه 8.5.0 - استراتژی اسکالپ شتابی با استاپ‌لاس تنگ
نویسنده: تیم توسعه کریپتو AI
"""

import logging
from datetime import datetime

# تنظیمات لاگر
logger = logging.getLogger("CryptoAIScalper")

# ==============================================================================
# 1. توابع عمومی کمکی
# ==============================================================================

def format_binance_price(price, symbol):
    """
    رند کردن قیمت طبق استانداردهای صرافی برای جلوگیری از خطای بخش جاوا
    """
    try:
        price = float(price)
        symbol_upper = symbol.upper()
        
        # تعیین دقت بر اساس نماد
        if "BTC" in symbol_upper or "ETH" in symbol_upper:
            return round(price, 2)
        elif "SHIB" in symbol_upper:
            return round(price, 8)
        elif "DOGE" in symbol_upper:
            return round(price, 6)
        elif "XRP" in symbol_upper or "ADA" in symbol_upper:
            return round(price, 5)
        elif "ALGO" in symbol_upper:
            return round(price, 4)
        else:
            return round(price, 4)
    except Exception as e:
        logger.error(f"❌ Error in format_binance_price for {symbol}: {e}")
        return price

def get_market_data_with_fallback(symbol, timeframe, limit):
    """
    تابع کمکی برای دریافت دیتا (باید در main یا اینجا باشد)
    این تابع معمولاً با استفاده از کتابخانه binance دیتا می‌گیرد
    """
    logger.debug(f"دریافت داده بازار برای {symbol} - تایم‌فریم: {timeframe}")
    # اینجا باید کد اتصال به API Binance قرار گیرد
    # فعلاً آرایه خالی برمی‌گرداند
    return []

# ==============================================================================
# 2. توابع تحلیل تکنیکال
# ==============================================================================

def calculate_atr(data, period=14):
    """محاسبه میانگین محدوده واقعی برای تعیین نوسان بازار"""
    try:
        if len(data) < period + 1: 
            return 0
        
        highs = [float(c[2]) for c in data[-period:]]
        lows = [float(c[3]) for c in data[-period:]]
        closes = [float(c[4]) for c in data[-period-1:-1]]
        
        tr_list = []
        for i in range(len(highs)):
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i]), abs(lows[i]-closes[i]))
            tr_list.append(tr)
        return sum(tr_list) / period
    except Exception as e:
        logger.error(f"❌ Error in calculate_atr: {e}")
        return 0

def calculate_tdr(data, period=14):
    """محاسبه بازده کل (Total Daily Return) یا کارایی بازار"""
    try:
        if len(data) < period: 
            return 0.2
        
        start_price = float(data[-period][1])
        end_price = float(data[-1][4])
        return abs((end_price - start_price) / start_price)
    except Exception as e:
        logger.error(f"❌ Error in calculate_tdr: {e}")
        return 0.2

def get_ichimoku_scalp_signal(data, timeframe):
    """تحلیل سریع ایچیموکو برای تایید روند"""
    # یک پیاده‌سازی ساده برای جلوگیری از کراش
    try:
        if not data or len(data) < 52:
            return {"signal": "HOLD", "confidence": 0.5, "details": "داده ناکافی"}
        
        # استخراج قیمت‌های کلیدی
        current_price = float(data[-1][4])
        high_9 = max([float(c[2]) for c in data[-9:]])
        low_9 = min([float(c[3]) for c in data[-9:]])
        
        # محاسبه Tenkan-sen (خط تبدیل)
        tenkan_sen = (high_9 + low_9) / 2
        
        # محاسبه Kijun-sen (خط پایه)
        high_26 = max([float(c[2]) for c in data[-26:]])
        low_26 = min([float(c[3]) for c in data[-26:]])
        kijun_sen = (high_26 + low_26) / 2
        
        # تحلیل سیگنال
        signal = "HOLD"
        confidence = 0.5
        
        if current_price > tenkan_sen > kijun_sen:
            signal = "BUY"
            confidence = 0.7
        elif current_price < tenkan_sen < kijun_sen:
            signal = "SELL"
            confidence = 0.7
        elif current_price > kijun_sen:
            signal = "BUY"
            confidence = 0.6
        elif current_price < kijun_sen:
            signal = "SELL"
            confidence = 0.6
        
        return {
            "signal": signal,
            "confidence": confidence,
            "details": {
                "tenkan_sen": round(tenkan_sen, 4),
                "kijun_sen": round(kijun_sen, 4),
                "current_price": round(current_price, 4)
            }
        }
    except Exception as e:
        logger.error(f"❌ Error in get_ichimoku_scalp_signal: {e}")
        return {"signal": "HOLD", "confidence": 0.5, "details": "خطا در محاسبات"}

# ==============================================================================
# 3. توابع محاسبه شتاب (Momentum)
# ==============================================================================

def calculate_momentum_roc(data, period=5):
    """
    محاسبه نرخ تغییرات سریع (Rate of Change)
    مبنای اصلی استراتژی شتابی شما
    """
    if not data or len(data) < period + 1:
        logger.warning(f"Insufficient data for momentum ROC: {len(data) if data else 0}")
        return 0.0
    
    try:
        # استخراج قیمت‌های بسته شدن
        closes = []
        for candle in data[-period-1:]:
            if len(candle) > 4:
                try:
                    closes.append(float(candle[4]))
                except (ValueError, TypeError):
                    continue
        
        if len(closes) < period + 1:
            return 0.0
        
        # محاسبه ROC
        current_price = closes[-1]
        past_price = closes[-period-1]
        
        if past_price == 0:
            return 0.0
        
        roc = ((current_price - past_price) / past_price) * 100
        
        logger.debug(f"Momentum ROC ({period} period): {roc:.3f}% (From {past_price:.4f} to {current_price:.4f})")
        return round(roc, 3)
        
    except Exception as e:
        logger.error(f"❌ Error in calculate_momentum_roc: {e}")
        return 0.0

def get_momentum_persian_msg(roc, signal):
    """
    تولید پیام فارسی اختصاصی برای رابط کاربری HTML
    """
    is_risky = False
    msg = ""
    
    # تشخیص وضعیت شتاب
    roc_abs = abs(roc)
    
    if roc_abs > 1.0:  # شتاب بسیار شدید (پرریسک)
        is_risky = True
        msg = "🚨 هشدار شدید: شتاب قیمت بسیار بالاست (انفجاری). احتمال لغزش قیمت (Slippage) زیاد است. توصیه: ورود با حجم کم."
    elif roc_abs > 0.8:  # شتاب بالا
        is_risky = True
        msg = "⚠️ هشدار: شتاب قیمت بالاست. مراقب نوسانات ناگهانی باشید. احتمال اصلاح سریع وجود دارد."
    elif roc_abs > 0.5:  # شتاب متوسط
        if signal == "BUY" and roc > 0:
            msg = "📈 شتاب صعودی متوسط. شرایط نسبتاً امن برای اسکالپ با استاپ‌لاس تنگ."
        elif signal == "SELL" and roc < 0:
            msg = "📉 شتاب نزولی متوسط. فشار فروش قابل توجه. نقطه خروج مناسب."
        else:
            msg = "⚡ شتاب قابل توجه اما با جهت نامشخص. منتظر تایید جهت حرکت باشید."
    elif roc_abs > 0.1:  # شتاب خفیف
        if signal == "BUY" and roc > 0:
            msg = "✅ شتاب صعودی تایید شد. نقطه ورود مناسب برای اسکالپ شتابی با ریسک کنترل‌شده."
        elif signal == "SELL" and roc < 0:
            msg = "✅ شتاب نزولی تایید شد. قدرت فروشندگان در حال افزایش است. فرصت فروش کوتاه‌مدت."
        else:
            msg = "↔️ شتاب خفیف در بازار. منتظر شکست روند باشید."
    else:
        if signal == "BUY":
            msg = "⏳ شتاب صعودی ضعیف است. منتظر تاییدیه حرکت یا ورود در پولبک باشید."
        elif signal == "SELL":
            msg = "⏳ شتاب نزولی ضعیف است. احتمالاً بازار در حال تجمیع برای حرکت بعدی است."
        else:
            msg = "⏸️ وضعیت شتاب خنثی است. بازار در تعادل. منتظر شکست باشید."
    
    # اضافه کردن اطلاعات عددی
    msg += f" (ROC: {roc:.2f}%)"
    
    return msg, is_risky

# ==============================================================================
# 4. توابع استاپ‌لاس تنگ (روش شخصی شما)
# ==============================================================================

def calculate_tight_scalp_levels(price, signal, atr_value, symbol=None):
    """
    محاسبه حد ضرر فوق‌تنگ و تارگت‌های سریع
    در روش شما استاپ‌لاس نباید از 0.2% فراتر برود
    """
    try:
        price = float(price)
        atr_value = float(atr_value)
        
        if price <= 0 or atr_value <= 0:
            logger.warning(f"Invalid price or ATR for tight levels: price={price}, atr={atr_value}")
            return [], 0
        
        # تنظیم حد ضرر حداکثر 0.2% (روش شخصی شما)
        max_sl_percent = 0.002  # 0.2%
        min_sl_percent = 0.001  # حداقل 0.1% برای امنیت
        
        # محاسبه بر اساس سیگنال
        if signal == "BUY":
            # محاسبه استاپ‌لاس: کمترین مقدار بین 0.2% و 1.2 برابر ATR
            sl_by_percent = price * (1 - max_sl_percent)
            sl_by_atr = price - (atr_value * 1.2)
            
            # انتخاب امن‌ترین گزینه (کمتر کاهش دهد)
            stop_loss = max(sl_by_percent, sl_by_atr, price * (1 - min_sl_percent))
            
            # محاسبه تارگت‌های سریع
            target_1 = price + (atr_value * 0.8)  # تارگت اول محافظه‌کارانه
            target_2 = price + (atr_value * 1.5)  # تارگت دوم
            target_3 = price + (atr_value * 2.0)  # تارگت سوم برای شرایط قوی
            
            targets = [target_1, target_2, target_3]
            
            # اعمال فرمت‌بندی قیمت برای صرافی
            if symbol:
                stop_loss = format_binance_price(stop_loss, symbol)
                targets = [format_binance_price(t, symbol) for t in targets]
            
            logger.debug(f"Tight BUY levels for {symbol if symbol else 'unknown'}: Entry={price:.4f}, SL={stop_loss:.4f} ({((price-stop_loss)/price*100):.2f}%), T1={targets[0]:.4f}, T2={targets[1]:.4f}, T3={targets[2]:.4f}")
            
        elif signal == "SELL":
            # محاسبه استاپ‌لاس: کمترین مقدار بین 0.2% و 1.2 برابر ATR
            sl_by_percent = price * (1 + max_sl_percent)
            sl_by_atr = price + (atr_value * 1.2)
            
            # انتخاب امن‌ترین گزینه (کمتر افزایش دهد)
            stop_loss = min(sl_by_percent, sl_by_atr, price * (1 + min_sl_percent))
            
            # محاسبه تارگت‌های سریع
            target_1 = price - (atr_value * 0.8)
            target_2 = price - (atr_value * 1.5)
            target_3 = price - (atr_value * 2.0)
            
            targets = [target_1, target_2, target_3]
            
            # اعمال فرمت‌بندی قیمت برای صرافی
            if symbol:
                stop_loss = format_binance_price(stop_loss, symbol)
                targets = [format_binance_price(t, symbol) for t in targets]
            
            logger.debug(f"Tight SELL levels for {symbol if symbol else 'unknown'}: Entry={price:.4f}, SL={stop_loss:.4f} ({((stop_loss-price)/price*100):.2f}%), T1={targets[0]:.4f}, T2={targets[1]:.4f}, T3={targets[2]:.4f}")
            
        else:
            logger.debug("HOLD signal - no tight levels calculated")
            return [], 0
        
        # بررسی منطقی بودن مقادیر
        valid_targets = []
        if signal == "BUY":
            for target in targets:
                if target > price:
                    valid_targets.append(round(target, 8))
        elif signal == "SELL":
            for target in targets:
                if target < price:
                    valid_targets.append(round(target, 8))
        
        # اگر هیچ تارگت معتبری نداشتیم، از روش قبلی استفاده می‌کنیم
        if not valid_targets:
            logger.warning(f"No valid targets for {signal} signal, using fallback")
            if signal == "BUY":
                valid_targets = [round(price * 1.005, 8), round(price * 1.01, 8)]
            elif signal == "SELL":
                valid_targets = [round(price * 0.995, 8), round(price * 0.99, 8)]
        
        # بررسی استاپ‌لاس
        stop_loss = round(stop_loss, 8)
        
        # محاسبه نسبت ریسک به ریوارد
        if len(valid_targets) > 0:
            if signal == "BUY":
                risk = price - stop_loss
                reward = valid_targets[0] - price
            elif signal == "SELL":
                risk = stop_loss - price
                reward = price - valid_targets[0]
            
            if risk > 0:
                risk_reward = round(reward / risk, 2)
                logger.debug(f"Risk/Reward Ratio: {risk_reward}:1")
            else:
                risk_reward = 0
        else:
            risk_reward = 0
        
        return valid_targets, stop_loss
        
    except Exception as e:
        logger.error(f"❌ Error in calculate_tight_scalp_levels: {e}")
        return [], 0

# ==============================================================================
# 5. توابع سیگنال اصلی
# ==============================================================================

def get_enhanced_scalp_signal(data, symbol, timeframe="5m"):
    """
    سیگنال اسکالپ پیشرفته با ترکیب تمام اندیکاتورهای شما
    """
    try:
        if not data or len(data) < 30:
            logger.warning(f"Insufficient data for enhanced scalp signal: {symbol}")
            return None
        
        # 1. تحلیل ایچیموکو
        ichimoku_signal = get_ichimoku_scalp_signal(data, timeframe)
        
        # 2. محاسبه ATR
        atr_value = calculate_atr(data, 14)
        
        # 3. محاسبه شتاب (Momentum)
        momentum_roc = calculate_momentum_roc(data, 5)
        
        # 4. محاسبه TDR (کارایی بازار)
        tdr_value = calculate_tdr(data, 14)
        
        # 5. قیمت جاری
        current_price = float(data[-1][4]) if len(data[-1]) > 4 else 0
        
        if current_price <= 0:
            return None
        
        # 6. تعیین سیگنال نهایی با وزن‌دهی
        signal_weights = {
            "BUY": 0.0,
            "SELL": 0.0,
            "HOLD": 0.0
        }
        
        # وزن‌دهی ایچیموکو (40%)
        if ichimoku_signal:
            ich_signal = ichimoku_signal.get('signal', 'HOLD')
            ich_confidence = ichimoku_signal.get('confidence', 0.5)
            signal_weights[ich_signal] += ich_confidence * 40
        
        # وزن‌دهی شتاب (30%)
        if momentum_roc > 0.1:  # شتاب صعودی
            signal_weights["BUY"] += 30
        elif momentum_roc < -0.1:  # شتاب نزولی
            signal_weights["SELL"] += 30
        
        # وزن‌دهی TDR (20%)
        if tdr_value > 0.25:  # بازار رونددار
            # اگر روند صعودی است
            if signal_weights["BUY"] > signal_weights["SELL"]:
                signal_weights["BUY"] += 20
            else:
                signal_weights["SELL"] += 20
        else:  # بازار رنج
            signal_weights["HOLD"] += 20
        
        # وزن‌دهی قیمت نسبت به ATR (10%)
        if atr_value > 0:
            volatility_ratio = (atr_value / current_price) * 100
            if volatility_ratio < 0.3:  # نوسان کم - مناسب اسکالپ
                if signal_weights["BUY"] > signal_weights["SELL"]:
                    signal_weights["BUY"] += 10
                else:
                    signal_weights["SELL"] += 10
            else:  # نوسان بالا - ریسک بیشتر
                signal_weights["HOLD"] += 10
        
        # تعیین سیگنال ناشی
        final_signal = max(signal_weights, key=signal_weights.get)
        total_weight = sum(signal_weights.values())
        
        if total_weight > 0:
            confidence = signal_weights[final_signal] / total_weight
        else:
            confidence = 0.5
        
        # 7. محاسبه نقاط ورود و خروج با استاپ‌لاس تنگ
        entry_price = format_binance_price(current_price, symbol)
        targets, stop_loss = calculate_tight_scalp_levels(entry_price, final_signal, atr_value, symbol)
        
        # 8. تولید پیام فارسی
        momentum_msg, is_risky = get_momentum_persian_msg(momentum_roc, final_signal)
        
        # 9. آماده‌سازی نتیجه
        result = {
            "symbol": symbol,
            "signal": final_signal,
            "confidence": round(confidence, 3),
            "entry_price": entry_price,
            "targets": targets,
            "stop_loss": stop_loss,
            "momentum_roc": momentum_roc,
            "momentum_message": momentum_msg,
            "is_risky": is_risky,
            "atr_value": round(atr_value, 6),
            "tdr_value": round(tdr_value, 3),
            "timeframe": timeframe,
            "strategy": "Enhanced Scalp Pro v8.5.0",
            "analysis_details": {
                "ichimoku_signal": ichimoku_signal.get('signal', 'N/A') if ichimoku_signal else 'N/A',
                "ichimoku_confidence": ichimoku_signal.get('confidence', 0) if ichimoku_signal else 0,
                "market_efficiency": "TRENDING" if tdr_value > 0.25 else "RANGING",
                "volatility_level": "LOW" if (atr_value/current_price*100) < 0.3 else "HIGH",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"🎯 Enhanced Scalp Signal for {symbol}: {final_signal} (Confidence: {confidence:.2f}, ROC: {momentum_roc:.2f}%)")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in get_enhanced_scalp_signal for {symbol}: {e}")
        return None

# ==============================================================================
# 6. توابع کمکی اضافی
# ==============================================================================

def validate_market_data(data, symbol):
    """
    اعتبارسنجی داده‌های دریافتی از بازار
    """
    if not data:
        logger.error(f"❌ No data received for {symbol}")
        return False
    
    if len(data) < 20:
        logger.warning(f"⚠️ Insufficient data points for {symbol}: {len(data)}")
        return False
    
    # بررسی ساختار هر کندل
    for i, candle in enumerate(data[-10:]):  # بررسی 10 کندل آخر
        if len(candle) < 5:
            logger.error(f"❌ Invalid candle structure at position {i}: {candle}")
            return False
        
        # بررسی مقادیر عددی
        try:
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            
            if any(x <= 0 for x in [open_price, high_price, low_price, close_price]):
                logger.error(f"❌ Invalid price values in candle {i}: {candle}")
                return False
            
            # بررسی منطقی بودن قیمت‌ها
            if low_price > high_price:
                logger.error(f"❌ Low > High in candle {i}: Low={low_price}, High={high_price}")
                return False
            
            if not (low_price <= open_price <= high_price):
                logger.warning(f"⚠️ Open price out of range in candle {i}")
            
            if not (low_price <= close_price <= high_price):
                logger.warning(f"⚠️ Close price out of range in candle {i}")
                
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Error parsing candle data at position {i}: {e}")
            return False
    
    logger.debug(f"✅ Market data validated successfully for {symbol} ({len(data)} candles)")
    return True

def calculate_position_size(balance, risk_percentage, entry_price, stop_loss):
    """
    محاسبه حجم پوزیشن بر اساس میزان ریسک
    """
    try:
        if entry_price <= 0 or stop_loss <= 0:
            return 0
        
        # محاسبه مقدار ریسک بر اساس درصد بالانس
        risk_amount = balance * (risk_percentage / 100)
        
        # محاسبه ریسک به ازای هر واحد
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        # محاسبه حجم پوزیشن
        position_size = risk_amount / risk_per_unit
        
        logger.debug(f"Position size calculation: Balance={balance:.2f}, Risk%={risk_percentage}%, Risk/Unit={risk_per_unit:.8f}, Size={position_size:.8f}")
        
        return round(position_size, 8)
        
    except Exception as e:
        logger.error(f"❌ Error in calculate_position_size: {e}")
        return 0

def get_timestamp_string():
    """دریافت رشته زمانی برای لاگ‌ها"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==============================================================================
# 7. تابع اصلی برای تست
# ==============================================================================

if __name__ == "__main__":
    # تست ساده توابع
    print("🧪 Testing utils.py functions...")
    
    # تست فرمت‌بندی قیمت
    test_cases = [
        ("BTCUSDT", 45000.123456),
        ("ETHUSDT", 3000.987654),
        ("SHIBUSDT", 0.000012345678),
        ("DOGEUSDT", 0.123456),
        ("XRPUSDT", 0.567890)
    ]
    
    for symbol, price in test_cases:
        formatted = format_binance_price(price, symbol)
        print(f"{symbol}: {price} -> {formatted}")
    
    print("\n✅ All functions are ready for use!")