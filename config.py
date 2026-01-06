import os
from dotenv import load_dotenv

load_dotenv()

# اطلاعات استخراج شده از تصویر BotFather
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8396237816:AAFBwYRj319UI1FxTG_EjdoLsgfRDsWMImY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@AsemanSignals")

# تنظیمات واچ‌لیست
DEFAULT_WATCH = "BTCUSDT,ETHUSDT,ENAUSDT,1INCHUSDT,UNIUSDT,XRPUSDT,ADAUSDT,SOLUSDT,DOTUSDT,MATICUSDT"
WATCHLIST_STR = os.getenv("WATCHLIST", DEFAULT_WATCH)
WATCHLIST = [s.strip() for s in WATCHLIST_STR.split(",")]

# پارامترهای فنی
MIN_SIGNAL_QUALITY = float(os.getenv("MIN_SIGNAL_QUALITY", "2.0"))
MULTI_STRATEGY_SCAN_INTERVAL_MINUTES = 120
TIMEZONE = "Asia/Tehran"
