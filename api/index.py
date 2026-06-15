import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))

app = FastAPI()
_bot_app = None
_model = None
_angel = None
_config = None

def get_config():
    global _config
    if _config:
        return _config
    from config import Config
    _config = Config()
    return _config

def get_model():
    global _model
    if _model is not None:
        return _model
    from model import load_model
    _model = load_model(get_config().MODEL_PATH)
    return _model

def get_angel():
    global _angel
    if _angel is not None:
        return _angel
    from angel import AngelClient
    _angel = AngelClient(get_config())
    _angel.login()
    return _angel

async def get_bot_app():
    global _bot_app
    if _bot_app is not None:
        return _bot_app
    from telegram.ext import Application
    _bot_app = Application.builder().token(TOKEN).build()
    from bot_handlers import setup_handlers
    setup_handlers(_bot_app)
    await _bot_app.initialize()
    await _bot_app.start()
    logger.info("Bot app initialized")
    return _bot_app

def gen_signal():
    cfg = get_config()
    model = get_model()
    angel = get_angel()
    if model is None:
        return None
    try:
        from features import compute_features, features_to_array
        from signals import compute_indicator_signals, compute_bias_score, combine_ml_and_indicators
        from model import predict_next

        df = angel.get_nifty_candles(100)
        if df is None or len(df) < 55:
            import yfinance as yf, pandas as pd
            yf_df = yf.download(cfg.NIFTY_YAHOO, period="15d", interval="15m", progress=False)
            if yf_df.empty:
                return None
            yf_df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in yf_df.columns]
            yf_df = yf_df.reset_index()
            yf_df = yf_df.rename(columns={"Datetime": "timestamp",
                f"Open_{cfg.NIFTY_YAHOO}": "open", f"High_{cfg.NIFTY_YAHOO}": "high",
                f"Low_{cfg.NIFTY_YAHOO}": "low", f"Close_{cfg.NIFTY_YAHOO}": "close",
                f"Volume_{cfg.NIFTY_YAHOO}": "volume"})
            yf_df["timestamp"] = pd.to_datetime(yf_df["timestamp"])
            df = yf_df.dropna().reset_index(drop=True)
        if df is None or len(df) < 55:
            return None

        feat = compute_features(df)
        arr = features_to_array(feat)
        ml_dir, ml_conf = predict_next(model, arr)
        ind = compute_indicator_signals(df)
        bias = compute_bias_score(ind)
        from signals import combine_ml_and_indicators as combine
        final_dir, final_conf = combine(ml_dir, ml_conf, bias)
        ltp = angel.get_ltp() if angel.logged_in else None
        if ltp is None and df is not None:
            ltp = round(float(df["close"].iloc[-1]), 2)

        now = datetime.now(IST)
        labels = {0: ("DOWN", "\u2b07", "\U0001f534"), 1: ("UP", "\U0001f53c", "\U0001f7e2")}
        dl, arrw, emj = labels.get(final_dir, ("NEUTRAL", "\u27a1\ufe0f", "\u26aa"))
        return {
            "direction": dl, "direction_code": final_dir,
            "confidence": round(final_conf, 1), "ltp": ltp,
            "time": now.strftime("%I:%M %p"), "arrow": arrw, "emoji": emj
        }
    except Exception as e:
        logger.error(f"Signal: {e}")
        return None

@app.post("/api/webhook")
async def webhook(request: Request):
    data = await request.json()
    logger.info(f"Webhook received: {data.get('message', {}).get('text', '')}")
    bot_app = await get_bot_app()
    from telegram import Update
    update = Update.de_json(data, bot_app.bot)
    await bot_app.process_update(update)
    return {"ok": True}

@app.get("/api/set_webhook")
async def set_webhook():
    cfg = get_config()
    import requests
    vercel_url = os.environ.get("VERCEL_URL", "")
    url = f"https://{vercel_url}/api/webhook" if vercel_url else ""
    if not url:
        return {"error": "VERCEL_URL not set"}
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/setWebhook", json={"url": url})
    return r.json()

@app.get("/api/health")
async def health():
    model = get_model()
    return {
        "status": "ok",
        "model": "loaded" if model else "not loaded",
        "time": datetime.now(IST).strftime("%I:%M %p")
    }

@app.get("/api/scheduled")
async def scheduled():
    if not CHAT_ID:
        return {"error": "CHAT_ID not set"}
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return {"status": "weekend"}
    minutes = now.hour * 60 + now.minute
    if not (555 <= minutes <= 931):
        return {"status": "market closed"}
    result = gen_signal()
    if result and result["confidence"] >= 60:
        signal = "BUY" if result["direction"] == "UP" else "SELL"
        msg = (
            f"{result['arrow']} *Nifty 30min Signal*\n\n"
            f"Action: *{signal}*\n"
            f"Confidence: {result['confidence']:.0f}%\n"
            f"Nifty: {result['ltp']}\n"
            f"Time: {result['time']} IST"
        )
        import requests
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", json={
            "chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"
        })
        return {"status": "signal_sent", "signal": signal, "confidence": result["confidence"]}
    return {"status": "no_signal", "result": result}

@app.get("/api/next")
async def next_signal():
    result = gen_signal()
    if result and result["confidence"] >= 60:
        signal = "BUY" if result["direction"] == "UP" else "SELL"
        return {"signal": signal, "confidence": result["confidence"], "ltp": result["ltp"], "time": result["time"]}
    return {"signal": None, "confidence": 0}
