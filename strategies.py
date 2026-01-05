import pandas_ta as ta

def calculate_master_signals(df):
    # 1. ZLMA Trend (Zero Lag MA)
    ema_val = ta.ema(df['close'], length=15)
    correction = df['close'] + (df['close'] - ema_val)
    zlma = ta.ema(correction, length=15)
    
    # 2. RSI & Ichimoku Confirmation
    rsi = ta.rsi(df['close'], length=14)
    rsi_sma = ta.sma(rsi, length=14)
    
    # 3. ATR for Dynamic Stop Loss (خیلی مهم)
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # 4. FVG Detection (SMC)
    fvg_bull = (df['low'] > df['high'].shift(2))
    
    curr = -1
    # شرط نهایی: روند ZLMA صعودی + RSI بالای خط تایید + تاییدیه FVG
    is_bullish = (zlma.iloc[curr] > ema_val.iloc[curr]) and (rsi.iloc[curr] > rsi_sma.iloc[curr])
    
    return is_bullish, zlma.iloc[curr], atr.iloc[curr], fvg_bull.iloc[curr]