import os
from dotenv import load_dotenv

# بارگذاری متغیرها (در سیستم محلی از .env و در سرور از Environment Variables)
load_dotenv()

# --- تنظیمات تلگرام (ایمن شده) ---
# توکن ربات از Environment Variable خوانده می‌شود (امنیت بالا)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# آیدی کانال شما
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@AsemanSignals")

# --- تنظیمات واچ‌لیست (ترکیب ارزهای جدید و قدیمی پامپی) ---
# اگر در رندر متغیری ست نشود، این لیست پیش‌فرض اسکن می‌شود
DEFAULT_WATCH = "BTCUSDT,ETHUSDT,ENAUSDT,1INCHUSDT,UNIUSDT,XRPUSDT,ADAUSDT,SOLUSDT,DOTUSDT,MATICUSDT"
WATCHLIST_STR = os.getenv("WATCHLIST", DEFAULT_WATCH)
WATCHLIST = [s.strip() for s in WATCHLIST_STR.split(",")]

# --- تنظیمات استراتژی و فیلترها (بهینه شده برای ارزهای قدیمی) ---
# امتیاز 2 برای تایید پامپ‌های میان‌مدت عالی است
MIN_SIGNAL_QUALITY = float(os.getenv("MIN_SIGNAL_QUALITY", "2.0"))

# فیلترهای سخت‌گیرانه برای افزایش تعداد سیگنال در ارزهای قدیمی غیرفعال است
ENABLE_MULTI_TF_FILTER = False
ENABLE_MARKET_REGIME_FILTER = False

# --- تنظیمات زمانی ---
TIMEZONE = "Asia/Tehran"

# --- کلیدهای صرافی (اختیاری) ---
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
