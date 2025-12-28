# ==============================================================================
# ENHANCED API ENDPOINTS
# ==============================================================================

@app.get("/")
async def read_root():
    return {
        "message": f"Crypto AI Trading System v{API_VERSION}",
        "status": "Active",
        "version": API_VERSION,
        "features": [
            "Professional Scalper Engine",
            "ATR-based Risk Management",
            "AI Confirmation System",
            "Multi-timeframe Analysis",
            "Ichimoku + RSI Divergence",
            "Low Latency Architecture",
            "Persian User Interface"
        ],
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE,
            "pandas_ta": HAS_PANDAS_TA,
            "tdr_atr": HAS_TDR_ATR if UTILS_AVAILABLE else False
        }
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "Healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "performance": {
            "latency": "low",
            "engine": "v8.0.0",
            "memory_usage": "optimized"
        },
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "pandas_available": HAS_PANDAS
        }
    }


@app.post("/api/v1/analyze", response_model=SignalDetail)
async def analyze_pair(request: ScalpRequest):
    """
    Advanced Analysis Endpoint with Momentum Logic and Tight Stop Loss
    Combines Ichimoku, RSI Divergence, ATR Risk Management, and Momentum Scoring
    """
    start_time = time.time()
    logger.info(f"🚀 Advanced Analysis Request: {request.symbol} [{request.timeframe}]")
    
    try:
        # --- بخش دریافت دیتا ---
        data = get_market_data_with_fallback(request.symbol, request.timeframe, 100)
        if not data or len(data) < 20:
            raise HTTPException(
                status_code=400,
                detail="دیتای کافی برای تحلیل وجود ندارد"
            )
        
        # --- استخراج قیمت‌ها برای تحلیل شتابی ---
        price = float(data[-1][4])  # قیمت Close آخرین کندل
        close_prices = [float(c[4]) for c in data[-10:]]  # ۱۰ قیمت آخر برای محاسبه شتاب
        
        # --- ۱. استراتژی اصلی (ایچیموکو + RSI) ---
        ichimoku = calculate_ichimoku_components(data)
        rsi_val = calculate_simple_rsi(data, 14)
        
        # تحلیل سیگنال ایچیموکو
        ichi_signal = analyze_ichimoku_scalp_signal(ichimoku)
        
        # فیلتر کردن سیگنال با RSI
        final_signal = "HOLD"
        if ichi_signal['signal'] == 'BUY' and rsi_val < 70:
            final_signal = "BUY"
        elif ichi_signal['signal'] == 'SELL' and rsi_val > 30:
            final_signal = "SELL"
        
        # --- ۲. استراتژی شتابی (Momentum) و محاسبه استاپ‌لاس فوق‌تنگ ---
        # محاسبه شتاب و پیام فارسی
        roc, persian_msg, risky = get_momentum_logic(close_prices, final_signal)
        
        # محاسبه استاپ‌لاس 0.15 درصدی برای روش شخصی شما
        tight_stop_loss = price * 0.9985
        
        # --- ۳. محاسبه اهداف با استفاده از ATR ---
        atr_value, _, volatility_score = ScalperEngine.calculate_atr_advanced(data)
        
        if atr_value > 0 and final_signal != "HOLD":
            targets, stop_loss = ScalperEngine.generate_pro_scalp_targets(
                price, final_signal, atr_value, 1.5
            )
        else:
            # Fallback targets for HOLD or no ATR
            if final_signal == "BUY":
                targets = [price * 1.005, price * 1.01]
                stop_loss = tight_stop_loss
            elif final_signal == "SELL":
                targets = [price * 0.995, price * 0.99]
                stop_loss = price * 1.0015
            else:
                targets = [price, price]
                stop_loss = price
        
        # --- ۴. رند کردن اعداد برای جلوگیری از خطای بخش جاوا (Binance Precision) ---
        clean_entry = format_binance_price(price, request.symbol)
        clean_sl = format_binance_price(stop_loss, request.symbol)
        clean_targets = [
            format_binance_price(target, request.symbol)
            for target in targets
        ]
        
        # --- ۵. AI Confirmation (اختیاری) ---
        tdr_score = ScalperEngine.calculate_tdr_advanced(data)
        ai_advice = ""
        if request.use_ai:
            ai_advice = ScalperEngine.get_ai_confirmation(
                request.symbol, final_signal, tdr_score, rsi_val, price, request.use_ai
            )
            persian_msg = f"{persian_msg} | {ai_advice}"
        
        # --- ۶. خروجی نهایی یکپارچه ---
        return {
            "symbol": request.symbol,
            "signal": final_signal,
            "entry_price": clean_entry,
            "stop_loss": clean_sl,
            "targets": clean_targets,
            "momentum_score": roc,
            "user_message": persian_msg,
            "is_risky_for_retail": risky,
            "execution_type": "MARKET" if risky else "LIMIT",  # اضافه شده برای بخش جاوا
            "signal_id": f"{request.symbol}_{int(time.time())}",  # شناسه یکتا
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in analyze_pair: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="خطا در تحلیل دیتای صرافی"
        )


@app.post("/api/analyze")
async def analyze_market(request: ScalpRequest):
    """
    Enhanced Analysis Endpoint with AI Confirmation
    Combines TDR, Ichimoku, RSI, and ATR-based risk management
    """
    start_time = time.time()
    
    try:
        # 1. Get market data
        data = get_market_data_with_fallback(request.symbol, request.timeframe, 100)
        if not data:
            raise HTTPException(
                status_code=500, 
                detail="Failed to fetch market data"
            )
        
        # 2. Calculate technical indicators
        tdr_score = ScalperEngine.calculate_tdr_advanced(data)
        
        if UTILS_AVAILABLE and HAS_TDR_ATR:
            ichi = get_ichimoku_full(data)
            atr = calculate_atr(data)
        else:
            ichi = calculate_ichimoku_components(data)
            atr, current_price, _ = ScalperEngine.calculate_atr_advanced(data)
        
        rsi_val = calculate_simple_rsi(data, 14)
        price = ichi.get("current_price", float(data[-1][4]) if data else 0)
        
        # 3. Strategy logic (combine Ichimoku and TDR)
        strategy_signal = "HOLD"
        if tdr_score >= 0.25:  # Only trade if market is trending
            tenkan = ichi.get("tenkan_sen", price)
            kijun = ichi.get("kijun_sen", price)
            cloud_top = ichi.get("cloud_top", price * 1.02)
            cloud_bottom = ichi.get("cloud_bottom", price * 0.98)
            
            if tenkan > kijun and price > cloud_top:
                strategy_signal = "BUY"
            elif tenkan < kijun and price < cloud_bottom:
                strategy_signal = "SELL"
        
        # 4. Dynamic stop loss and targets with ATR
        targets = []
        stop_loss = 0
        
        if strategy_signal == "BUY":
            targets = [
                round(price + (atr * 1.5), 8),
                round(price + (atr * 3.0), 8)
            ]
            stop_loss = round(price - (atr * 2.0), 8)
        elif strategy_signal == "SELL":
            targets = [
                round(price - (atr * 1.5), 8),
                round(price - (atr * 3.0), 8)
            ]
            stop_loss = round(price + (atr * 2.0), 8)
        
        # 5. AI Confirmation
        ai_advice = "AI skipped"
        if request.use_ai:
            ai_advice = ScalperEngine.get_ai_confirmation(
                request.symbol, strategy_signal, tdr_score, rsi_val, price, request.use_ai
            )
        
        # 6. Prepare response
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": strategy_signal,
            "price": round(price, 8),
            "tdr_status": "Trending" if tdr_score >= 0.25 else "Ranging (Avoid)",
            "tdr_value": round(tdr_score, 4),
            "rsi": round(rsi_val, 2),
            "ichimoku": {
                "tenkan": round(ichi.get("tenkan_sen", 0), 4),
                "kijun": round(ichi.get("kijun_sen", 0), 4),
                "cloud_top": round(ichi.get("cloud_top", 0), 4),
                "cloud_bottom": round(ichi.get("cloud_bottom", 0), 4),
                "trend_power": round(ichi.get("trend_power", 50), 1)
            },
            "risk_management": {
                "targets": targets,
                "stop_loss": stop_loss,
                "atr_volatility": round(atr, 8),
                "risk_reward_ratio": round(
                    (targets[0] - price) / (price - stop_loss), 2
                ) if strategy_signal != "HOLD" and len(targets) > 0 and stop_loss > 0 else 0
            },
            "ai_confirmation": ai_advice,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION
        }
        
        logger.info(f"✅ Enhanced Analysis: {strategy_signal} for {request.symbol}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Enhanced analysis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis error: {str(e)[:200]}"
        )


# ==============================================================================
# بخش اسکنر بازار (Market Scanner) - اصلاح شده
# ==============================================================================

@app.get("/api/v1/market-scan")
async def market_scanner():
    """
    اسکنر ۱ ساعته برای پیشنهادات AI در HTML
    """
    # در نسخه واقعی، اینجا باید تابعی برای فیلتر کردن ارزهای با حجم بالا فراخوانی شود
    top_picks = [
        {
            "symbol": "SOLUSDT",
            "trend_1h": "صعودی قوی (Bullish)",
            "suggestion": "مناسب برای اسکلپ شتابی در تایم‌فریم پایین",
            "ai_score": 92,
            "status": "HOT"
        },
        {
            "symbol": "ETHUSDT",
            "trend_1h": "رنج (Neutral)",
            "suggestion": "صبر کنید تا قیمت از محدوده تراکم خارج شود",
            "ai_score": 58,
            "status": "WATCH"
        }
    ]
    return {
        "status": "success", 
        "data": top_picks,
        "server_time": datetime.now(timezone.utc).isoformat()
    }


@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """
    Advanced Scalp Signal: Ichimoku + RSI Divergence + ATR Risk Management
    """
    start_time = time.time()
    logger.info(f"🚀 Advanced Scalp Request: {request.symbol} [{request.timeframe}] AI: {request.use_ai}")
    
    try:
        # 1. Get market data
        market_data_res = get_market_data_with_fallback(
            request.symbol, request.timeframe, 100, return_source=True
        )
        data = market_data_res["data"] if isinstance(market_data_res, dict) else market_data_res
        
        if not data or len(data) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient data for scalp analysis"
            )
        
        # 2. Combined technical analysis
        ichimoku = calculate_ichimoku_components(data)
        rsi_val = calculate_simple_rsi(data, 14)
        atr, current_price, volatility_score = ScalperEngine.calculate_atr_advanced(data)
        tdr_score = ScalperEngine.calculate_tdr_advanced(data)
        
        # 3. Signal detection with Ichimoku and RSI filter
        ichi_signal = analyze_ichimoku_scalp_signal(ichimoku)
        
        final_signal = "HOLD"
        confidence = 0.5
        
        if ichi_signal['signal'] == 'BUY' and rsi_val < 70:
            if rsi_val < 40:
                final_signal = "BUY"
                confidence = max(0.7, ichi_signal['confidence'])
            elif rsi_val < 60:
                final_signal = "BUY"
                confidence = ichi_signal['confidence'] * 0.9
        elif ichi_signal['signal'] == 'SELL' and rsi_val > 30:
            if rsi_val > 60:
                final_signal = "SELL"
                confidence = max(0.7, ichi_signal['confidence'])
            elif rsi_val > 40:
                final_signal = "SELL"
                confidence = ichi_signal['confidence'] * 0.9
        
        # 4. Calculate smart entry and targets
        try:
            entry_price = calculate_smart_entry(data, final_signal)
            if entry_price <= 0:
                entry_price = current_price
        except:
            entry_price = current_price
        
        # Determine risk level based on volatility
        if volatility_score > 70:
            risk_level = "HIGH"
        elif volatility_score > 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate enhanced targets
        target_data = calculate_enhanced_targets(
            entry_price, final_signal, "DYNAMIC_ATR", atr, risk_level
        )
        
        # 5. AI Confirmation
        ai_advice = ""
        if request.use_ai:
            ai_advice = ScalperEngine.get_ai_confirmation(
                request.symbol, final_signal, tdr_score, rsi_val, entry_price, request.use_ai
            )
        
        # 6. Prepare comprehensive response
        response = {
            "symbol": request.symbol.upper(),
            "timeframe": request.timeframe,
            "signal": final_signal,
            "confidence": round(confidence, 3),
            "entry_price": round(entry_price, 8),
            "current_price": round(current_price, 8),
            "targets": target_data["targets"],
            "stop_loss": target_data["stop_loss"],
            "targets_percent": target_data["targets_percent"],
            "stop_loss_percent": target_data["stop_loss_percent"],
            "analysis": {
                "rsi": round(rsi_val, 2),
                "atr_volatility": round(atr, 8),
                "tdr_score": round(tdr_score, 4),
                "volatility_score": round(volatility_score, 1),
                "ichimoku_trend": round(ichimoku.get('trend_power', 50), 1),
                "market_condition": "Trending" if tdr_score >= 0.25 else "Ranging",
                "risk_level": risk_level,
                "strategy_used": target_data["strategy_used"]
            },
            "ai_confirmation": ai_advice,
            "execution_metrics": {
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "data_points": len(data),
                "signal_strength": "STRONG" if confidence > 0.75 else "MODERATE" if confidence > 0.6 else "WEAK"
            },
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION
        }
        
        logger.info(f"✅ Advanced Scalp: {final_signal} for {request.symbol}")
        logger.info(f"   Confidence: {confidence:.2%}, Risk: {risk_level}, ATR: {atr:.4f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Advanced scalp error: {e}", exc_info=True)
        
        # Comprehensive fallback
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": "HOLD",
            "confidence": 0.5,
            "entry_price": 0,
            "current_price": 0,
            "targets": [0, 0, 0],
            "stop_loss": 0,
            "analysis": {
                "rsi": 50,
                "atr_volatility": 0,
                "tdr_score": 0,
                "error": str(e)[:100]
            },
            "ai_confirmation": "Analysis failed - using fallback",
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION,
            "status": "FALLBACK"
        }


@app.post("/api/ichimoku-scalp")
async def get_ichimoku_scalp_signal(request: IchimokuRequest):
    """Enhanced Ichimoku Scalp Endpoint with ATR integration"""
    allowed_timeframes = ["1m", "5m", "15m", "1h", "4h"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid timeframe. Allowed: {allowed_timeframes}"
        )
    
    try:
        logger.info(f"🌌 Ichimoku request: {request.symbol} ({request.timeframe})")
        
        market_data = get_market_data_with_fallback(request.symbol, request.timeframe, 100)
        if not market_data or len(market_data) < 60:
            raise HTTPException(
                status_code=404, 
                detail="Not enough data for Ichimoku analysis"
            )
        
        ichimoku_analysis = analyze_ichimoku_scalp_signal(calculate_ichimoku_components(market_data))
        
        # Calculate additional indicators
        rsi = calculate_simple_rsi(market_data, 14)
        closes = [float(c[4]) for c in market_data]
        rsi_series = calculate_rsi_series(closes, 14)
        div = detect_divergence(closes, rsi_series, lookback=5)
        recommendation = generate_ichimoku_recommendation(ichimoku_analysis)
        
        # Calculate percentages
        entry_price = ichimoku_analysis.get("entry_price", 0)
        targets = ichimoku_analysis.get("targets", [0, 0, 0])
        stop_loss = ichimoku_analysis.get("stop_loss", 0)
        
        tp_percents = [
            round(((target - entry_price) / entry_price) * 100, 2)
            for target in targets[:3]
        ] if len(targets) >= 3 else [0, 0, 0]
        
        sl_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2) if entry_price > 0 else 0
        
        # Determine risk level
        confidence = ichimoku_analysis.get("confidence", 0.5)
        risk_level = "LOW"
        if confidence > 0.8:
            risk_level = "HIGH"
        elif confidence > 0.6:
            risk_level = "MEDIUM"
        
        # Calculate ATR for volatility
        atr_value, _, _ = ScalperEngine.calculate_atr_advanced(market_data)
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": ichimoku_analysis.get("signal", "HOLD"),
            "confidence": confidence,
            "entry_price": entry_price,
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": tp_percents,
            "stop_loss_percent": sl_percent,
            "rsi": round(rsi, 2),
            "divergence": div['detected'],
            "divergence_type": div['type'],
            "reason": ichimoku_analysis.get("reason", ""),
            "strategy": f"Ichimoku Scalp ({request.timeframe})",
            "recommendation": recommendation,
            "trend_power": ichimoku_analysis.get("trend_analysis", {}).get("power", 50),
            "risk_level": risk_level,
            "atr_volatility": round(atr_value, 8),
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Ichimoku: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Ichimoku analysis error: {str(e)[:200]}"
        )


@app.post("/api/combined-analysis")
async def get_combined_analysis(request: CombinedRequest):
    """Combined Analysis Endpoint."""
    try:
        market_data = get_market_data_with_fallback(request.symbol, request.timeframe, 100)
        if not market_data:
            raise HTTPException(status_code=404, detail="No market data available")
        
        results = combined_analysis(market_data, request.timeframe)
        
        if results:
            return results
        else:
            # Fallback analysis
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'details': {'note': 'Combined analysis returned None'},
                'price': float(market_data[-1][4]) if market_data else 0,
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error in combined analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Combined analysis error: {str(e)[:200]}"
        )


@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    """Market data endpoint."""
    try:
        data = get_market_data_with_fallback(symbol, timeframe, limit=50)
        if not data:
            raise HTTPException(status_code=404, detail="No market data available")
        
        latest = data[-1] if isinstance(data, list) and len(data) > 0 else []
        if not latest or len(latest) < 6:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "Mock",
                "current_price": 100,
                "change_24h": 0
            }
        
        change_24h = calculate_24h_change_from_dataframe(data)
        rsi = calculate_simple_rsi(data, 14)
        sma_20 = calculate_simple_sma(data, 20)
        closes = [float(c[4]) for c in data]
        rsi_series = calculate_rsi_series(closes, 14)
        div = detect_divergence(closes, rsi_series, lookback=5)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": "Binance" if UTILS_AVAILABLE else "Mock",
            "current_price": float(latest[4]),
            "high": float(latest[2]),
            "low": float(latest[3]),
            "change_24h": change_24h,
            "rsi_14": round(rsi, 2),
            "sma_20": round(sma_20, 2),
            "divergence": div['detected'],
            "divergence_type": div['type'],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Market data error: {str(e)[:200]}"
        )


@app.get("/api/v1/scan-all/{symbol}")
async def scan_all_timeframes_pro(symbol: str):
    """Professional multi-timeframe scanner"""
    timeframes = ["1m", "5m", "15m", "1h", "4h"]
    results = []
    
    for tf in timeframes:
        try:
            # Use appropriate strategy based on timeframe
            if tf in ["1m", "5m", "15m"]:
                response = await get_scalp_signal(ScalpRequest(symbol=symbol, timeframe=tf, use_ai=False))
            else:
                response = await get_ichimoku_scalp_signal(IchimokuRequest(symbol=symbol, timeframe=tf))
            
            response["timeframe"] = tf
            
            # Add ATR analysis
            data = get_market_data_with_fallback(symbol, tf, 50)
            if data:
                atr, _, vol_score = ScalperEngine.calculate_atr_advanced(data)
                response["atr_analysis"] = {
                    "atr_value": round(atr, 8),
                    "volatility_score": round(vol_score, 1)
                }
            
            results.append(response)
            
        except Exception as e:
            logger.error(f"Error scanning {symbol} on {tf}: {e}")
            results.append({
                "symbol": symbol,
                "timeframe": tf,
                "signal": "ERROR",
                "error": str(e)[:100]
            })
    
    return {
        "symbol": symbol,
        "scanned_at": datetime.now().isoformat(),
        "timeframes_analyzed": len(timeframes),
        "signals_summary": {
            "BUY": len([r for r in results if r.get("signal") == "BUY"]),
            "SELL": len([r for r in results if r.get("signal") == "SELL"]),
            "HOLD": len([r for r in results if r.get("signal") == "HOLD"]),
            "ERROR": len([r for r in results if r.get("signal") == "ERROR"])
        },
        "results": results,
        "recommendation": "Focus on timeframes with consistent signals"
    }

# ==============================================================================
# Performance Monitoring
# ==============================================================================

class PerformanceMonitor:
    """Monitor system performance for low-latency trading"""
    
    def __init__(self):
        self.request_times = []
        self.avg_latency = 0
    
    def record_request(self, processing_time: float):
        self.request_times.append(processing_time)
        if len(self.request_times) > 100:
            self.request_times.pop(0)
        self.avg_latency = np.mean(self.request_times) if self.request_times else 0


performance_monitor = PerformanceMonitor()


@app.middleware("http")
async def monitor_performance(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time
    performance_monitor.record_request(processing_time)
    
    # Add performance header
    response.headers["X-Processing-Time"] = str(round(processing_time * 1000, 2))
    response.headers["X-Avg-Latency"] = str(round(performance_monitor.avg_latency * 1000, 2))
    
    return response


@app.get("/api/v1/performance")
async def get_performance_stats():
    """Get system performance statistics"""
    return {
        "average_latency_ms": round(performance_monitor.avg_latency * 1000, 2),
        "total_requests": len(performance_monitor.request_times),
        "requests_per_second": round(len(performance_monitor.request_times) / 60, 1) if performance_monitor.request_times else 0,
        "performance_grade": "A" if performance_monitor.avg_latency < 0.1 else "B" if performance_monitor.avg_latency < 0.5 else "C",
        "timestamp": datetime.now().isoformat()
    }

# ==============================================================================
# Startup and Main
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info(f"🚀 Starting Crypto AI Trading System v{API_VERSION}")
    logger.info(f"📦 Utils Available: {UTILS_AVAILABLE}")
    logger.info(f"📦 Pandas TA: {HAS_PANDAS_TA}")
    logger.info(f"📦 TDR/ATR Functions: {HAS_TDR_ATR if UTILS_AVAILABLE else False}")
    logger.info(f"⚡ Performance Mode: Optimized for Scalping")
    
    print(f"\n{'=' * 60}")
    print(f"PRO SCALPER EDITION v{API_VERSION}")
    print(f"{'=' * 60}")
    print("Features:")
    print("  • Professional Scalper Engine")
    print("  • ATR-based Risk Management")
    print("  • AI Confirmation System")
    print("  • Multi-timeframe Analysis")
    print("  • Low Latency Architecture")
    print(f"{'=' * 60}")
    print(f"API Documentation: /api/docs")
    print(f"Health Check: /api/health")
    print(f"Performance Stats: /api/v1/performance")
    print(f"{'=' * 60}\n")
    
    logger.info("✅ System startup completed successfully!")


# ==============================================================================
# Execution Entry Point
# ==============================================================================

if __name__ == "__main__":
    # تنظیم پورت برای Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)