import os
from dotenv import load_dotenv

load_dotenv()

# --- تنظیمات تلگرام (ایمن شده) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") # مقدار را فقط در Render وارد کنید
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@AsemanSignals")

# --- تنظیمات واچ‌لیست ---
# اضافه کردن چند ارز قدیمی و پامپی طبق استراتژی دیشب
DEFAULT_WATCHLIST = "BTCUSDT,ETHUSDT,ENAUSDT,1INCHUSDT,UNIUSDT,XRPUSDT,ADAUSDT,SOLUSDT"
WATCHLIST_STR = os.getenv("WATCHLIST", DEFAULT_WATCHLIST)
WATCHLIST = [s.strip() for s in WATCHLIST_STR.split(",")]

# --- تنظیمات استراتژی ---
# امتیاز 2 برای شروع خوب است تا سیگنال‌های پامپی قدیمی را از دست ندهید
MIN_SIGNAL_QUALITY = float(os.getenv("MIN_SIGNAL_QUALITY", "2.0"))

ENABLE_MULTI_TF_FILTER = False 
ENABLE_MARKET_REGIME_FILTER = False

# --- تنظیمات زمانی ---
TIMEZONE = "Asia/Tehran"
