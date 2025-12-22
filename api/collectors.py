# در main.py می‌توانید از توابع جدید استفاده کنید:

@app.get("/api/scalp-collection")
async def get_scalp_collection(timeframe: str = "5m"):
    """دریافت سیگنال‌های اسکالپ جمع‌آوری شده"""
    signals = collect_scalp_signals(timeframe=timeframe)
    return {"signals": signals, "count": len(signals)}

@app.get("/api/signal-summary")
async def get_signal_summary_endpoint(timeframe: str = "5m"):
    """دریافت خلاصه سیگنال‌ها"""
    summary = get_signal_summary(timeframe=timeframe)
    return summary