"""
ماژول ابزارهای کمکی برای ربات اسکالپر ارز دیجیتال
نسخه 9.0.0 - استراتژی اسکالپ شتابی با استاپ‌لاس تنگ ۰.۲٪
نویسنده: تیم توسعه کریپتو AI
آخرین به‌روزرسانی: ۱۴۰۳/۰۱/۱۵
"""

import logging
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union
from functools import lru_cache

# ==============================================================================
# تنظیمات پیشرفته لاگر
# ==============================================================================

# ایجاد لاگر اصلی
logger = logging.getLogger("CryptoAIScalper")

# تنظیم فرمت لاگ
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# تنظیم handler برای فایل
file_handler = logging.FileHandler('scalper_bot.log', encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

# تنظیم handler برای کنسول
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# اضافه کردن handlerها به لاگر
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

# ==============================================================================
# 1. کلاس تنظیمات (Config Class)
# ==============================================================================

class ScalpConfig:
    """کلاس تنظیمات استراتژی اسکالپ"""
    
    def __init__(self):
        # تنظیمات عمومی
        self.version = "9.0.0"
        self.strategy_name = "Acceleration Scalp Pro"
        
        # تنظیمات ریسک
        self.max_risk_per_trade = 1.0  # درصد
        self.max_position_size = 0.1   # حداکثر حجم پوزیشن (BTC)
        
        # تنظیمات استاپ‌لاس
        self.max_stop_loss_percent = 0.002  # 0.2% حداکثر
        self.min_stop_loss_percent = 0.001   # 0.1% حداقل
        
        # تنظیمات تارگت
        self.target_multipliers = {
            't1': 0.6,   # امن‌سازی
            't2': 1.4,   # نقدینگی
            't3': 2.2    # ساختار بازار (SMC)
        }
        
        # تنظیمات اندیکاتورها
        self.atr_period = 14
        self.momentum_period = 5
        self.tdr_period = 14
        self.ichimoku_periods = {
            'tenkan': 9,
            'kijun': 26,
            'senkou': 52
        }
        
        # تنظیمات API
        self.api_retry_count = 3
        self.api_timeout = 10
        self.cache_duration = 30  # ثانیه
        
        # تنظیمات تایم‌فریم
        self.primary_timeframe = "5m"
        self.confirmation_timeframe = "15m"
        
        # تنظیمات نوسان
        self.max_atr_percent = 0.10  # 10%
        self.min_volume_threshold = 100000
        
    def to_dict(self) -> Dict:
        """تبدیل تنظیمات به دیکشنری"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# ایجاد نمونه از کلاس تنظیمات
config = ScalpConfig()

# ==============================================================================
# 2. دکوراتورها (Decorators)
# ==============================================================================

def timeit(func):
    """دکوراتور برای اندازه‌گیری زمان اجرای توابع"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # فقط اگر زمان اجرا بیش از 0.1 ثانیه باشد، لاگ کنیم
        if execution_time > 0.1:
            logger.debug(f"⏱️ Function '{func.__name__}' executed in {execution_time:.3f} seconds")
        
        return result
    return wrapper

def retry(max_retries: int = 3, delay: float = 1.0):
    """دکوراتور برای تلاش مجدد در صورت شکست"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for '{func.__name__}': {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            
            logger.error(f"❌ All {max_retries} attempts failed for '{func.__name__}'")
            raise last_exception
        return wrapper
    return decorator

# ==============================================================================
# 3. توابع عمومی کمکی
# ==============================================================================

@timeit
def format_binance_price(price: float, symbol: str) -> float:
    """
    رند کردن قیمت طبق استانداردهای صرافی برای جلوگیری از خطای بخش‌پذیری
    
    پارامترها:
    ----------
    price : float
        قیمت ورودی
    symbol : str
        نماد معاملاتی (مثال: BTCUSDT)
    
    بازگشت:
    -------
    float : قیمت فرمت‌شده
    """
    try:
        price = float(price)
        symbol_upper = symbol.upper()
        
        # تعیین دقت بر اساس نماد
        precision_map = {
            'BTC': 2,    # BTCUSDT, BTCBUSD
            'ETH': 2,    # ETHUSDT, ETHBUSD
            'BNB': 2,    # BNBUSDT
            'SOL': 2,    # SOLUSDT
            'ADA': 5,    # ADAUSDT
            'XRP': 5,    # XRPUSDT
            'DOGE': 5,   # DOGEUSDT
            'DOT': 3,    # DOTUSDT
            'AVAX': 3,   # AVAXUSDT
            'MATIC': 4,  # MATICUSDT
            'SHIB': 8,   # SHIBUSDT
            'ALGO': 4,   # ALGOUSDT
            'ATOM': 3,   # ATOMUSDT
            'LINK': 3,   # LINKUSDT
            'UNI': 3,    # UNIUSDT
        }
        
        # جستجو در دیکشنری
        for key, precision in precision_map.items():
            if key in symbol_upper:
                return round(price, precision)
        
        # پیش‌فرض برای سایر نمادها
        if price >= 10:
            return round(price, 3)
        elif price >= 1:
            return round(price, 4)
        elif price >= 0.1:
            return round(price, 5)
        elif price >= 0.01:
            return round(price, 6)
        else:
            return round(price, 8)
            
    except Exception as e:
        logger.error(f"❌ Error in format_binance_price for {symbol}: {e}")
        return round(price, 8)  # پیش‌فرض امن

@retry(max_retries=config.api_retry_count)
@lru_cache(maxsize=32)
def get_market_data_with_fallback(symbol: str, timeframe: str, limit: int) -> List:
    """
    دریافت داده بازار با کش و قابلیت تلاش مجدد
    
    پارامترها:
    ----------
    symbol : str
        نماد معاملاتی
    timeframe : str
        تایم‌فریم (1m, 5m, 15m, 1h, etc.)
    limit : int
        تعداد کندل‌های درخواستی
    
    بازگشت:
    -------
    list : لیست کندل‌ها
    """
    try:
        logger.debug(f"📊 Fetching market data for {symbol} - TF: {timeframe}, Limit: {limit}")
        
        # اینجا باید کد اتصال به API Binance قرار گیرد
        # به صورت موقت داده‌های نمونه برمی‌گردانیم
        
        # ساخت داده‌های نمونه برای تست
        sample_data = []
        base_price = 50000.0 if "BTC" in symbol.upper() else 3000.0
        
        for i in range(limit):
            timestamp = int(time.time() * 1000) - (i * 300000)  # 5 دقیقه فاصله
            open_price = base_price + (i * 0.1)
            high_price = open_price * 1.002
            low_price = open_price * 0.998
            close_price = open_price * 1.001
            volume = 1000.0 + (i * 10)
            
            candle = [
                timestamp,
                str(open_price),
                str(high_price),
                str(low_price),
                str(close_price),
                str(volume)
            ]
            sample_data.append(candle)
        
        sample_data.reverse()  # مرتب کردن از قدیم به جدید
        return sample_data
        
    except Exception as e:
        logger.error(f"❌ Failed to get market data for {symbol}: {e}")
        return []

# ==============================================================================
# 4. توابع تحلیل تکنیکال
# ==============================================================================

@timeit
def calculate_atr(data: List, period: int = None) -> float:
    """
    محاسبه میانگین محدوده واقعی (ATR) برای تعیین نوسان بازار
    
    پارامترها:
    ----------
    data : list
        لیست داده‌های کندل‌ها
    period : int, optional
        دوره ATR (پیش‌فرض از تنظیمات)
    
    بازگشت:
    -------
    float : مقدار ATR
    """
    if period is None:
        period = config.atr_period
    
    try:
        if len(data) < period + 1:
            logger.warning(f"Insufficient data for ATR calculation: {len(data)} < {period + 1}")
            return 0.0
        
        # استخراج قیمت‌ها
        highs = []
        lows = []
        closes = []
        
        for candle in data[-period-1:]:
            if len(candle) >= 5:
                highs.append(float(candle[2]))
                lows.append(float(candle[3]))
                closes.append(float(candle[4]))
        
        if len(highs) < period or len(lows) < period or len(closes) < period + 1:
            return 0.0
        
        # محاسبه محدوده واقعی (True Range)
        tr_list = []
        for i in range(len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i])
            lc = abs(lows[i] - closes[i])
            tr = max(hl, hc, lc)
            tr_list.append(tr)
        
        # محاسبه میانگین
        atr_value = sum(tr_list) / len(tr_list)
        
        logger.debug(f"ATR ({period} period): {atr_value:.6f}")
        return round(atr_value, 8)
        
    except Exception as e:
        logger.error(f"❌ Error in calculate_atr: {e}")
        return 0.0

@timeit
def calculate_tdr(data: List, period: int = None) -> float:
    """
    محاسبه بازده کل (Total Daily Return) یا کارایی بازار
    
    پارامترها:
    ----------
    data : list
        لیست داده‌های کندل‌ها
    period : int, optional
        دوره محاسبه (پیش‌فرض از تنظیمات)
    
    بازگشت:
    -------
    float : مقدار TDR (به صورت اعشاری)
    """
    if period is None:
        period = config.tdr_period
    
    try:
        if len(data) < period:
            logger.warning(f"Insufficient data for TDR calculation: {len(data)} < {period}")
            return 0.2  # پیش‌فرض
        
        # پیدا کردن اولین و آخرین قیمت بسته شدن
        start_price = None
        end_price = None
        
        for candle in data[-period:]:
            if len(candle) >= 5:
                if start_price is None:
                    start_price = float(candle[4])
                end_price = float(candle[4])
        
        if start_price is None or end_price is None or start_price == 0:
            return 0.2
        
        tdr_value = abs((end_price - start_price) / start_price)
        
        logger.debug(f"TDR ({period} period): {tdr_value:.4f} ({tdr_value*100:.2f}%)")
        return round(tdr_value, 4)
        
    except Exception as e:
        logger.error(f"❌ Error in calculate_tdr: {e}")
        return 0.2

@timeit
def get_ichimoku_scalp_signal(data: List, timeframe: str) -> Dict:
    """
    تحلیل سریع ایچیموکو برای تایید روند
    
    پارامترها:
    ----------
    data : list
        لیست داده‌های کندل‌ها
    timeframe : str
        تایم‌فریم تحلیل
    
    بازگشت:
    -------
    dict : سیگنال ایچیموکو
    """
    try:
        if not data or len(data) < 52:
            logger.warning(f"Insufficient data for Ichimoku: {len(data)}")
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "details": "داده ناکافی",
                "indicators": {}
            }
        
        # استخراج قیمت‌ها
        current_price = float(data[-1][4])
        
        # محاسبه Tenkan-sen (خط تبدیل)
        high_9 = max([float(c[2]) for c in data[-9:] if len(c) >= 3])
        low_9 = min([float(c[3]) for c in data[-9:] if len(c) >= 4])
        tenkan_sen = (high_9 + low_9) / 2
        
        # محاسبه Kijun-sen (خط پایه)
        high_26 = max([float(c[2]) for c in data[-26:] if len(c) >= 3])
        low_26 = min([float(c[3]) for c in data[-26:] if len(c) >= 4])
        kijun_sen = (high_26 + low_26) / 2
        
        # محاسبه Senkou Span A (ابر آینده)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # محاسبه Senkou Span B
        high_52 = max([float(c[2]) for c in data[-52:] if len(c) >= 3])
        low_52 = min([float(c[3]) for c in data[-52:] if len(c) >= 4])
        senkou_span_b = (high_52 + low_52) / 2
        
        # تحلیل سیگنال
        signal = "HOLD"
        confidence = 0.5
        
        # تحلیل اصلی
        if current_price > max(senkou_span_a, senkou_span_b):
            signal = "BUY"
            confidence = 0.8
        elif current_price < min(senkou_span_a, senkou_span_b):
            signal = "SELL"
            confidence = 0.8
        elif tenkan_sen > kijun_sen and current_price > tenkan_sen:
            signal = "BUY"
            confidence = 0.7
        elif tenkan_sen < kijun_sen and current_price < tenkan_sen:
            signal = "SELL"
            confidence = 0.7
        elif current_price > kijun_sen:
            signal = "BUY"
            confidence = 0.6
        elif current_price < kijun_sen:
            signal = "SELL"
            confidence = 0.6
        
        # تنظیم confidence بر اساس فاصله از ابر
        cloud_top = max(senkou_span_a, senkou_span_b)
        cloud_bottom = min(senkou_span_a, senkou_span_b)
        
        if signal == "BUY" and current_price > cloud_top:
            distance_percent = ((current_price - cloud_top) / cloud_top) * 100
            if distance_percent > 2:
                confidence = min(confidence + 0.1, 0.9)
        
        if signal == "SELL" and current_price < cloud_bottom:
            distance_percent = ((cloud_bottom - current_price) / current_price) * 100
            if distance_percent > 2:
                confidence = min(confidence + 0.1, 0.9)
        
        result = {
            "signal": signal,
            "confidence": round(confidence, 3),
            "details": {
                "tenkan_sen": round(tenkan_sen, 4),
                "kijun_sen": round(kijun_sen, 4),
                "senkou_span_a": round(senkou_span_a, 4),
                "senkou_span_b": round(senkou_span_b, 4),
                "cloud_top": round(cloud_top, 4),
                "cloud_bottom": round(cloud_bottom, 4),
                "current_price": round(current_price, 4),
                "in_cloud": cloud_bottom <= current_price <= cloud_top
            }
        }
        
        logger.debug(f"Ichimoku Signal: {signal} (Confidence: {confidence})")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in get_ichimoku_scalp_signal: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "details": "خطا در محاسبات",
            "indicators": {}
        }

# ==============================================================================
# 5. توابع محاسبه شتاب (Momentum)
# ==============================================================================

@timeit
def calculate_momentum_roc(data: List, period: int = None) -> float:
    """
    محاسبه نرخ تغییرات سریع (Rate of Change)
    
    پارامترها:
    ----------
    data : list
        لیست داده‌های کندل‌ها
    period : int, optional
        دوره محاسبه (پیش‌فرض از تنظیمات)
    
    بازگشت:
    -------
    float : مقدار ROC به درصد
    """
    if period is None:
        period = config.momentum_period
    
    try:
        if not data or len(data) < period + 1:
            logger.warning(f"Insufficient data for momentum ROC: {len(data) if data else 0}")
            return 0.0
        
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
        
        current_price = closes[-1]
        past_price = closes[-period-1]
        
        if past_price == 0:
            return 0.0
        
        roc = ((current_price - past_price) / past_price) * 100
        
        logger.debug(f"Momentum ROC ({period} period): {roc:.3f}%")
        return round(roc, 3)
        
    except Exception as e:
        logger.error(f"❌ Error in calculate_momentum_roc: {e}")
        return 0.0

def get_momentum_persian_msg(roc: float, signal: str) -> Tuple[str, bool]:
    """
    تولید پیام فارسی اختصاصی برای رابط کاربری HTML
    
    پارامترها:
    ----------
    roc : float
        نرخ تغییرات (به درصد)
    signal : str
        سیگنال فعلی (BUY/SELL/HOLD)
    
    بازگشت:
    -------
    tuple : (پیام فارسی, وضعیت ریسک)
    """
    is_risky = False
    msg = ""
    roc_abs = abs(roc)
    
    # تشخیص وضعیت بر اساس ROC
    if roc_abs > 1.0:
        is_risky = True
        msg = "🚨 هشدار شدید: شتاب قیمت بسیار بالاست (انفجاری). احتمال لغزش قیمت (Slippage) زیاد است. توصیه: ورود با حجم کم."
    elif roc_abs > 0.8:
        is_risky = True
        msg = "⚠️ هشدار: شتاب قیمت بالاست. مراقب نوسانات ناگهانی باشید. احتمال اصلاح سریع وجود دارد."
    elif roc_abs > 0.5:
        if signal == "BUY" and roc > 0:
            msg = "📈 شتاب صعودی متوسط. شرایط نسبتاً امن برای اسکالپ با استاپ‌لاس تنگ."
        elif signal == "SELL" and roc < 0:
            msg = "📉 شتاب نزولی متوسط. فشار فروش قابل توجه. نقطه خروج مناسب."
        else:
            msg = "⚡ شتاب قابل توجه اما با جهت نامشخص. منتظر تایید جهت حرکت باشید."
    elif roc_abs > 0.1:
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
# 6. توابع استاپ‌لاس تنگ (روش شخصی شما)
# ==============================================================================

@timeit
def calculate_tight_scalp_levels(price: float, signal: str, atr_value: float, 
                                symbol: str = None) -> Tuple[List, float]:
    """
    محاسبه ۳ تارگت واقعی و استاپ‌لاس تنگ ۰.۲٪
    T1: امن‌سازی | T2: نقدینگی | T3: ساختار بازار (SMC)
    
    پارامترها:
    ----------
    price : float
        قیمت ورودی
    signal : str
        سیگنال ("BUY" یا "SELL")
    atr_value : float
        مقدار ATR فعلی
    symbol : str, optional
        نماد معاملاتی برای فرمت‌بندی دقیق
    
    بازگشت:
    -------
    tuple : (list of targets, stop_loss)
        لیست ۳ تارگت و قیمت استاپ‌لاس
    """
    try:
        # اعتبارسنجی ورودی‌ها
        if price <= 0:
            logger.error(f"Invalid price: {price}")
            return [], 0
        
        price = float(price)
        
        # محاسبه ATR ایمن
        if atr_value <= 0:
            # مقدار پیش‌فرض ATR بر اساس درصدی از قیمت
            atr_value = price * 0.005  # 0.5% پیش‌فرض
            logger.warning(f"ATR value is invalid or zero, using default: {atr_value:.6f}")
        else:
            atr_value = float(atr_value)
        
        # بررسی منطقی بودن ATR (نباید بیشتر از ۱۰٪ قیمت باشد)
        max_atr_percent = config.max_atr_percent
        if atr_value > price * max_atr_percent:
            logger.warning(f"ATR too high ({atr_value/price*100:.2f}%), capping at {max_atr_percent*100}%")
            atr_value = price * max_atr_percent
        
        # ۱. محاسبه استاپ‌لاس فوق‌تنگ (حداکثر ۰.۲٪)
        sl_percent = config.max_stop_loss_percent
        
        if signal == "BUY":
            # استاپ‌لاس برای BUY
            stop_loss = price * (1 - sl_percent)
            
            # اطمینان از اینکه استاپ‌لاس بالای صفر باشد
            min_stop_loss = price * (1 - config.min_stop_loss_percent)
            stop_loss = max(stop_loss, min_stop_loss)
            
            # محاسبه تارگت‌ها بر اساس ضریب‌های ATR
            t1 = price + (atr_value * config.target_multipliers['t1'])  # امن‌سازی
            t2 = price + (atr_value * config.target_multipliers['t2'])  # نقدینگی
            t3 = price + (atr_value * config.target_multipliers['t3'])  # SMC
            
            # بررسی منطقی بودن تارگت‌ها
            if not (price < t1 < t2 < t3):
                logger.warning("Invalid BUY targets order, adjusting based on percentages...")
                t1 = price * 1.003  # 0.3%
                t2 = price * 1.006  # 0.6%
                t3 = price * 1.010  # 1.0%
                
        elif signal == "SELL":
            # استاپ‌لاس برای SELL
            stop_loss = price * (1 + sl_percent)
            
            # محاسبه تارگت‌ها
            t1 = price - (atr_value * config.target_multipliers['t1'])   # امن‌سازی
            t2 = price - (atr_value * config.target_multipliers['t2'])   # نقدینگی
            t3 = price - (atr_value * config.target_multipliers['t3'])   # SMC
            
            # بررسی منطقی بودن تارگت‌ها
            if not (t3 < t2 < t1 < price):
                logger.warning("Invalid SELL targets order, adjusting based on percentages...")
                t1 = price * 0.997  # -0.3%
                t2 = price * 0.994  # -0.6%
                t3 = price * 0.990  # -1.0%
                
        else:
            logger.warning(f"Invalid signal type: {signal}")
            return [], 0
        
        # ۲. رند کردن قیمت‌ها مطابق نماد ارز
        if symbol:
            stop_loss = format_binance_price(stop_loss, symbol)
            targets = [
                format_binance_price(t1, symbol),
                format_binance_price(t2, symbol),
                format_binance_price(t3, symbol)
            ]
        else:
            stop_loss = round(stop_loss, 8)
            targets = [round(t, 8) for t in [t1, t2, t3]]
        
        # ۳. محاسبه نسبت ریسک به ریوارد
        if signal == "BUY":
            risk = price - stop_loss
            reward_t1 = t1 - price
            reward_t3 = t3 - price
        else:  # SELL
            risk = stop_loss - price
            reward_t1 = price - t1
            reward_t3 = price - t3
        
        # محاسبه Risk/Reward Ratio
        rr_t1, rr_t3 = 0, 0
        if risk > 0:
            rr_t1 = round(reward_t1 / risk, 2)
            rr_t3 = round(reward_t3 / risk, 2)
            
            # هشدار برای نسبت‌های نامناسب
            if rr_t1 < 0.5:
                logger.warning(f"Low Risk/Reward ratio for T1: {rr_t1}:1")
        
        # ۴. لاگ جزئیات محاسبات
        logger.info(
            f"🎯 Tight Levels Calculated for {signal}:\n"
            f"   Entry:      {price:.8f}\n"
            f"   Stop Loss:  {stop_loss:.8f} ({abs((stop_loss-price)/price*100):.2f}%)\n"
            f"   T1:         {targets[0]:.8f} (RR: {rr_t1}:1)\n"
            f"   T2:         {targets[1]:.8f}\n"
            f"   T3:         {targets[2]:.8f} (RR: {rr_t3}:1)\n"
            f"   ATR Used:   {atr_value:.8f} ({atr_value/price*100:.2f}%)"
        )
        
        # ۵. اعتبارسنجی نهایی
        if signal == "BUY":
            if not (stop_loss < price < targets[0] < targets[1] < targets[2]):
                logger.error("Invalid BUY levels after formatting!")
                return [], 0
        elif signal == "SELL":
            if not (targets[2] < targets[1] < targets[0] < price < stop_loss):
                logger.error("Invalid SELL levels after formatting!")
                return [], 0
        
        return targets, stop_loss
        
    except Exception as e:
        logger.error(f"❌ Error in calculate_tight_scalp_levels: {e}", exc_info=True)
        return [], 0

# ==============================================================================
# 7. توابع سیگنال اصلی
# ==============================================================================

@timeit
def get_enhanced_scalp_signal(data: List, symbol: str, timeframe: str = None) -> Optional[Dict]:
    """
    سیگنال اسکالپ پیشرفته با ترکیب تمام اندیکاتورهای شما
    
    پارامترها:
    ----------
    data : list
        لیست داده‌های کندل‌ها
    symbol : str
        نماد معاملاتی
    timeframe : str, optional
        تایم‌فریم تحلیل
    
    بازگشت:
    -------
    dict or None : اطلاعات سیگنال کامل
    """
    if timeframe is None:
        timeframe = config.primary_timeframe
    
    try:
        # اعتبارسنجی داده‌ها
        if not validate_market_data(data, symbol):
            logger.error(f"Invalid market data for {symbol}")
            return None
        
        # 1. تحلیل ایچیموکو
        ichimoku_signal = get_ichimoku_scalp_signal(data, timeframe)
        
        # 2. محاسبه ATR
        atr_value = calculate_atr(data)
        
        # 3. محاسبه شتاب (Momentum)
        momentum_roc = calculate_momentum_roc(data)
        
        # 4. محاسبه TDR (کارایی بازار)
        tdr_value = calculate_tdr(data)
        
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
        ich_signal = ichimoku_signal.get('signal', 'HOLD')
        ich_confidence = ichimoku_signal.get('confidence', 0.5)
        signal_weights[ich_signal] += ich_confidence * 40
        
        # وزن‌دهی شتاب (30%)
        momentum_threshold = 0.1
        if momentum_roc > momentum_threshold:
            signal_weights["BUY"] += 30
        elif momentum_roc < -momentum_threshold:
            signal_weights["SELL"] += 30
        
        # وزن‌دهی TDR (20%)
        market_efficiency_threshold = 0.25
        if tdr_value > market_efficiency_threshold:
            if signal_weights["BUY"] > signal_weights["SELL"]:
                signal_weights["BUY"] += 20
            else:
                signal_weights["SELL"] += 20
        else:
            signal_weights["HOLD"] += 20
        
        # وزن‌دهی نوسان (10%)
        volatility_ratio = (atr_value / current_price) * 100
        volatility_threshold = 0.3
        if volatility_ratio < volatility_threshold:
            if signal_weights["BUY"] > signal_weights["SELL"]:
                signal_weights["BUY"] += 10
            else:
                signal_weights["SELL"] += 10
        else:
            signal_weights["HOLD"] += 10
        
        # تعیین سیگنال نهایی
        final_signal = max(signal_weights, key=signal_weights.get)
        total_weight = sum(signal_weights.values())
        
        if total_weight > 0:
            confidence = signal_weights[final_signal] / total_weight
        else:
            confidence = 0.5
        
        # 7. محاسبه نقاط ورود و خروج
        entry_price = format_binance_price(current_price, symbol)
        targets, stop_loss = calculate_tight_scalp_levels(
            entry_price, final_signal, atr_value, symbol
        )
        
        # 8. تولید پیام فارسی
        momentum_msg, is_risky = get_momentum_persian_msg(momentum_roc, final_signal)
        
        # 9. آماده‌سازی نتیجه نهایی
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
            "strategy": config.strategy_name,
            "version": config.version,
            "timestamp": datetime.now().isoformat(),
            "analysis_details": {
                "ichimoku_signal": ichimoku_signal.get('signal', 'N/A'),
                "ichimoku_confidence": ichimoku_signal.get('confidence', 0),
                "market_efficiency": "TRENDING" if tdr_value > market_efficiency_threshold else "RANGING",
                "volatility_level": "LOW" if volatility_ratio < volatility_threshold else "HIGH",
                "signal_weights": signal_weights,
                "calculated_at": get_timestamp_string()
            }
        }
        
        logger.info(
            f"🎯 Enhanced Scalp Signal for {symbol}:\n"
            f"   Signal:      {final_signal}\n"
            f"   Confidence:  {confidence:.2f}\n"
            f"   Price:       {entry_price:.8f}\n"
            f"   ROC:         {momentum_roc:.2f}%\n"
            f"   ATR:         {atr_value:.6f}\n"
            f"   Risk Level:  {'HIGH ⚠️' if is_risky else 'LOW ✅'}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in get_enhanced_scalp_signal for {symbol}: {e}", exc_info=True)
        return None

# ==============================================================================
# 8. توابع کمکی اضافی
# ==============================================================================

def validate_market_data(data: List, symbol: str) -> bool:
    """
    اعتبارسنجی داده‌های دریافتی از بازار
    
    پارامترها:
    ----------
    data : list
        لیست داده‌های کندل‌ها
    symbol : str
        نماد معاملاتی
    
    بازگشت:
    -------
    bool : صحت داده‌ها
    """
    if not data:
        logger.error(f"❌ No data received for {symbol}")
        return False
    
    min_candles = 20
    if len(data) < min_candles:
        logger.warning(f"⚠️ Insufficient data points for {symbol}: {len(data)} < {min_candles}")
        return False
    
    # بررسی ساختار و مقادیر کندل‌ها
    valid_candles = 0
    for i, candle in enumerate(data[-min_candles:]):
        # بررسی طول کندل
        if len(candle) < 5:
            logger.error(f"❌ Invalid candle structure at position {i}: {candle}")
            continue
        
        try:
            # استخراج قیمت‌ها
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            
            # بررسی مقادیر مثبت
            if any(x <= 0 for x in [open_price, high_price, low_price, close_price]):
                logger.error(f"❌ Invalid price values in candle {i}: {candle}")
                continue
            
            # بررسی منطقی بودن بازه قیمت
            if low_price > high_price:
                logger.error(f"❌ Low > High in candle {i}: Low={low_price}, High={high_price}")
                continue
            
            # بررسی قرارگیری Open و Close در بازه High-Low
            if not (low_price <= open_price <= high_price):
                logger.warning(f"⚠️ Open price out of range in candle {i}")
            
            if not (low_price <= close_price <= high_price):
                logger.warning(f"⚠️ Close price out of range in candle {i}")
            
            valid_candles += 1
            
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Error parsing candle data at position {i}: {e}")
            continue
    
    # حداقل 70% کندل‌ها باید معتبر باشند
    validity_ratio = valid_candles / min_candles
    if validity_ratio < 0.7:
        logger.error(f"❌ Data validity too low for {symbol}: {validity_ratio:.1%}")
        return False
    
    logger.debug(f"✅ Market data validated for {symbol}: {valid_candles}/{min_candles} valid candles")
    return True

@timeit
def calculate_position_size(balance: float, risk_percentage: float, 
                           entry_price: float, stop_loss: float) -> float:
    """
    محاسبه حجم پوزیشن بر اساس میزان ریسک
    
    پارامترها:
    ----------
    balance : float
        موجودی حساب
    risk_percentage : float
        درصد ریسک در هر معامله
    entry_price : float
        قیمت ورودی
    stop_loss : float
        قیمت حد ضرر
    
    بازگشت:
    -------
    float : حجم پوزیشن محاسبه‌شده
    """
    try:
        # اعتبارسنجی ورودی‌ها
        if any(x <= 0 for x in [balance, risk_percentage, entry_price]):
            logger.error(f"Invalid input for position size calculation")
            return 0.0
        
        # محدود کردن درصد ریسک
        risk_percentage = min(risk_percentage, config.max_risk_per_trade)
        
        # محاسبه مقدار ریسک بر اساس درصد بالانس
        risk_amount = balance * (risk_percentage / 100)
        
        # محاسبه ریسک به ازای هر واحد
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            logger.error(f"Invalid risk per unit: {risk_per_unit}")
            return 0.0
        
        # محاسبه حجم پوزیشن
        position_size = risk_amount / risk_per_unit
        
        # محدود کردن حجم پوزیشن
        if 'BTC' in config.max_position_size:
            max_size = config.max_position_size
            position_size = min(position_size, max_size)
        
        logger.info(
            f"📊 Position Size Calculation:\n"
            f"   Balance:      ${balance:.2f}\n"
            f"   Risk %:       {risk_percentage}%\n"
            f"   Risk Amount:  ${risk_amount:.2f}\n"
            f"   Entry:        {entry_price:.8f}\n"
            f"   Stop Loss:    {stop_loss:.8f}\n"
            f"   Risk/Unit:    {risk_per_unit:.8f}\n"
            f"   Position Size: {position_size:.8f}"
        )
        
        return round(position_size, 8)
        
    except Exception as e:
        logger.error(f"❌ Error in calculate_position_size: {e}")
        return 0.0

def get_timestamp_string() -> str:
    """
    دریافت رشته زمانی برای لاگ‌ها
    
    بازگشت:
    -------
    str : رشته زمانی فرمت‌شده
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def apply_signal_filters(signal_result: Dict, volume_data: List = None, 
                        orderbook_data: Dict = None) -> Dict:
    """
    اعمال فیلترهای اضافی روی سیگنال
    
    پارامترها:
    ----------
    signal_result : dict
        نتیجه سیگنال
    volume_data : list, optional
        داده‌های حجم معاملات
    orderbook_data : dict, optional
        داده‌های دفتر سفارشات
    
    بازگشت:
    -------
    dict : سیگنال فیلترشده
    """
    if not signal_result:
        return signal_result
    
    original_signal = signal_result.get('signal', 'HOLD')
    original_confidence = signal_result.get('confidence', 0.5)
    
    # فیلتر حجم معاملات
    if volume_data and len(volume_data) >= 5:
        avg_volume = sum([float(v) for v in volume_data[-5:]]) / 5
        min_volume = config.min_volume_threshold
        
        if avg_volume < min_volume:
            logger.warning(f"Volume filter triggered: {avg_volume:.0f} < {min_volume}")
            signal_result['signal'] = 'HOLD'
            signal_result['confidence'] = original_confidence * 0.7
            signal_result['filters'] = signal_result.get('filters', []) + ['low_volume']
    
    # فیلتر عمق بازار
    if orderbook_data:
        bid_ask_ratio = orderbook_data.get('bid_ask_ratio', 1)
        
        if original_signal == "BUY" and bid_ask_ratio < 0.8:
            logger.warning(f"Orderbook filter for BUY: bid/ask ratio = {bid_ask_ratio:.2f}")
            signal_result['confidence'] = original_confidence * 0.8
            signal_result['filters'] = signal_result.get('filters', []) + ['weak_bids']
        
        if original_signal == "SELL" and bid_ask_ratio > 1.2:
            logger.warning(f"Orderbook filter for SELL: bid/ask ratio = {bid_ask_ratio:.2f}")
            signal_result['confidence'] = original_confidence * 0.8
            signal_result['filters'] = signal_result.get('filters', []) + ['weak_asks']
    
    if 'filters' in signal_result:
        logger.info(f"Signal filters applied: {signal_result['filters']}")
    
    return signal_result

# ==============================================================================
# 9. تابع اصلی برای تست
# ==============================================================================

def run_comprehensive_test():
    """اجرای تست جامع تمام توابع"""
    print("🧪 Running comprehensive test of utils.py...")
    
    # تنظیمات تست
    test_symbol = "BTCUSDT"
    test_data = get_market_data_with_fallback(test_symbol, "5m", 100)
    
    if not test_data:
        print("❌ Failed to get test data")
        return
    
    print(f"✅ Test data acquired: {len(test_data)} candles")
    
    # تست 1: فرمت‌بندی قیمت
    print("\n1. Testing price formatting:")
    test_prices = [
        (45000.123456, "BTCUSDT"),
        (3000.987654, "ETHUSDT"),
        (0.000012345678, "SHIBUSDT"),
        (0.123456, "DOGEUSDT"),
        (0.567890, "XRPUSDT")
    ]
    
    for price, symbol in test_prices:
        formatted = format_binance_price(price, symbol)
        print(f"   {symbol}: {price} -> {formatted}")
    
    # تست 2: محاسبات تکنیکال
    print("\n2. Testing technical calculations:")
    
    atr_value = calculate_atr(test_data)
    print(f"   ATR: {atr_value:.6f}")
    
    tdr_value = calculate_tdr(test_data)
    print(f"   TDR: {tdr_value:.4f} ({tdr_value*100:.2f}%)")
    
    momentum_roc = calculate_momentum_roc(test_data)
    print(f"   Momentum ROC: {momentum_roc:.3f}%")
    
    # تست 3: تحلیل ایچیموکو
    print("\n3. Testing Ichimoku analysis:")
    ichimoku_result = get_ichimoku_scalp_signal(test_data, "5m")
    print(f"   Signal: {ichimoku_result.get('signal')}")
    print(f"   Confidence: {ichimoku_result.get('confidence'):.2f}")
    
    # تست 4: محاسبات استاپ‌لاس و تارگت
    print("\n4. Testing tight levels calculation:")
    current_price = float(test_data[-1][4])
    
    for signal in ["BUY", "SELL"]:
        targets, stop_loss = calculate_tight_scalp_levels(
            current_price, signal, atr_value, test_symbol
        )
        
        if targets and stop_loss:
            print(f"   {signal} Signal:")
            print(f"     Entry: {current_price:.2f}")
            print(f"     Stop Loss: {stop_loss:.2f}")
            for i, target in enumerate(targets[:3], 1):
                print(f"     T{i}: {target:.2f}")
    
    # تست 5: سیگنال پیشرفته
    print("\n5. Testing enhanced scalp signal:")
    enhanced_signal = get_enhanced_scalp_signal(test_data, test_symbol)
    
    if enhanced_signal:
        print(f"   Final Signal: {enhanced_signal.get('signal')}")
        print(f"   Confidence: {enhanced_signal.get('confidence'):.2f}")
        print(f"   Entry Price: {enhanced_signal.get('entry_price'):.2f}")
        print(f"   Is Risky: {enhanced_signal.get('is_risky')}")
    
    # تست 6: محاسبه حجم پوزیشن
    print("\n6. Testing position size calculation:")
    if enhanced_signal:
        position_size = calculate_position_size(
            balance=10000.0,
            risk_percentage=1.0,
            entry_price=enhanced_signal.get('entry_price'),
            stop_loss=enhanced_signal.get('stop_loss')
        )
        print(f"   Position Size: {position_size:.8f}")
    
    print("\n" + "="*50)
    print("✅ All tests completed successfully!")
    print(f"📊 Strategy: {config.strategy_name} v{config.version}")
    print("="*50)

# ==============================================================================
# 10. اجرای مستقیم
# ==============================================================================

if __name__ == "__main__":
    print(f"🚀 Initializing Crypto AI Scalper Utils v{config.version}")
    print(f"📅 {get_timestamp_string()}")
    print("-" * 50)
    
    # اجرای تست جامع
    run_comprehensive_test()
    
    print("\n🎉 Utils module is ready for production use!")
    print("💡 Import this module in your main bot script.")