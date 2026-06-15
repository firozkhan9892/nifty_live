import asyncio
import json
import os
import logging
from datetime import datetime, timezone, timedelta

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import Config
from angel import AngelClient
from features import compute_features, features_to_array
from signals import compute_indicator_signals, compute_bias_score, combine_ml_and_indicators
from model import load_model, predict_next, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

config = Config()
angel = AngelClient(config)
model = None

def is_market_hours():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 555 <= minutes <= 931

def generate_signal():
    global model
    if model is None:
        model = load_model(config.MODEL_PATH)
        if model is None:
            return None
    try:
        df = angel.get_nifty_candles(config.HISTORICAL_CANDLES)
        if df is None or len(df) < 55:
            import yfinance as yf
            import pandas as pd
            yf_df = yf.download(config.NIFTY_YAHOO, period="15d", interval="15m", progress=False)
            if yf_df.empty:
                return None
            yf_df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in yf_df.columns]
            yf_df = yf_df.reset_index()
            yf_df = yf_df.rename(columns={"Datetime": "timestamp",
                f"Open_{config.NIFTY_YAHOO}": "open", f"High_{config.NIFTY_YAHOO}": "high",
                f"Low_{config.NIFTY_YAHOO}": "low", f"Close_{config.NIFTY_YAHOO}": "close",
                f"Volume_{config.NIFTY_YAHOO}": "volume"})
            yf_df["timestamp"] = pd.to_datetime(yf_df["timestamp"])
            df = yf_df.dropna().reset_index(drop=True)
        if df is None or len(df) < 55:
            return None

        feat = compute_features(df)
        arr = features_to_array(feat)
        ml_dir, ml_conf = predict_next(model, arr)
        ind = compute_indicator_signals(df)
        bias = compute_bias_score(ind)
        final_dir, final_conf = combine_ml_and_indicators(ml_dir, ml_conf, bias)
        ltp = angel.get_ltp() if angel.logged_in else None
        if ltp is None and df is not None:
            ltp = round(float(df["close"].iloc[-1]), 2)

        now = datetime.now(IST)
        labels = {0: ("DOWN", "\u2b07", "\U0001f534"), 1: ("UP", "\U0001f53c", "\U0001f7e2")}
        dl, arrw, emj = labels.get(final_dir, ("NEUTRAL", "\u27a1\ufe0f", "\u26aa"))

        result = {
            "direction": dl, "direction_code": final_dir,
            "confidence": round(final_conf, 1), "ltp": ltp,
            "ml_direction": "UP" if ml_dir else "DOWN",
            "ml_confidence": round(ml_conf, 1), "bias_score": round(bias, 3),
            "time": now.strftime("%I:%M %p"), "arrow": arrw, "emoji": emj
        }
        _save_prediction(result)
        logger.info(f"Signal: {dl} ({final_conf:.1f}%)")
        return result
    except Exception as e:
        logger.error(f"Signal: {e}")
        return None

def _save_prediction(result):
    path = config.ACCURACY_DB
    stats = {"total": 0, "correct": 0, "history": []}
    if os.path.exists(path):
        with open(path) as f:
            stats = json.load(f)
    stats["history"].append({"time": result["time"], "direction": result["direction"],
        "confidence": result["confidence"], "ltp": result["ltp"], "result": "waiting"})
    if len(stats["history"]) > 500:
        stats["history"] = stats["history"][-500:]
    stats["total"] = len(stats["history"])
    stats["correct"] = sum(1 for h in stats["history"] if h.get("result") == "correct")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

def _load_stats():
    path = config.ACCURACY_DB
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"total": 0, "correct": 0, "history": []}

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Nifty Buy/Sell Signal Bot*\n\n"
        "Har 15 min Nifty ka agla 30 min direction predict karta hoon\n"
        "BUY ya SELL signal bhejta hoon\n"
        f"Model: 57% accuracy at confidence >= 50%\n\n"
        "/next - Agla signal abhi\n"
        "/accuracy - Performance\n"
        "/last - Last 10 signals\n"
        "/status - Bot status"
    )

async def next_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating signal...")
    result = await asyncio.to_thread(generate_signal)
    if result and result["confidence"] >= 60:
        signal = "BUY" if result["direction"] == "UP" else "SELL"
        msg = (
            f"{result['arrow']} *Nifty 30min Signal*\n\n"
            f"Action: *{signal}*\n"
            f"Confidence: {result['confidence']:.0f}%\n"
            f"Nifty: {result['ltp']}\n"
            f"Time: {result['time']} IST"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    elif result:
        await update.message.reply_text(f"Signal weak (confidence: {result['confidence']:.0f}%). Isse skip karein.")
    else:
        await update.message.reply_text("Signal generate nahi ho paaya")

async def accuracy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = _load_stats()
    if stats["total"] == 0:
        await update.message.reply_text("Abhi tak koi record nahi")
        return
    recent = [h for h in stats["history"] if h.get("result") != "waiting"][-50:]
    correct_recent = sum(1 for h in recent if h.get("result") == "correct")
    total_recent = len(recent)
    recent_acc = correct_recent / total_recent * 100 if total_recent > 0 else 0
    overall_acc = stats["correct"] / stats["total"] * 100
    await update.message.reply_text(
        f"Total: {stats['total']} | Correct: {stats['correct']}\n"
        f"Overall: {overall_acc:.1f}%\n"
        f"Last 50: {recent_acc:.1f}%"
    )

async def last_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = _load_stats()
    if not stats["history"]:
        await update.message.reply_text("Koi prediction nahi")
        return
    lines = []
    for p in stats["history"][-10:]:
        icon = {"correct": "+", "wrong": "-", "waiting": "?"}.get(p.get("result"), "?")
        lines.append(f"{icon} {p['time']} {p['direction']} ({p['confidence']:.0f}%)")
    await update.message.reply_text("Last 10:\n" + "\n".join(lines))

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = "Loaded" if model else "Not loaded"
    now = datetime.now(IST)
    market = "Open" if is_market_hours() else "Closed"
    await update.message.reply_text(f"Model: {m}\nMarket: {market}\nTime: {now.strftime('%I:%M %p')} IST")

async def scheduler(app):
    await asyncio.sleep(15)
    while True:
        try:
            if not is_market_hours():
                await asyncio.sleep(120)
                continue
            now = datetime.now(IST)
            minutes = now.hour * 60 + now.minute
            r = minutes % 15
            target = minutes - r + 14
            if target <= minutes:
                target += 15
            sleep_sec = (target - minutes) * 60 - now.second
            if sleep_sec > 300:
                sleep_sec = 30
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)
            if not is_market_hours():
                continue
            result = await asyncio.to_thread(generate_signal)
            if result and result["direction_code"] != 2 and result["confidence"] >= 60:
                signal = "BUY" if result["direction"] == "UP" else "SELL"
                msg = (
                    f"{result['arrow']} *Nifty 30min Signal*\n\n"
                    f"Action: *{signal}*\n"
                    f"Confidence: {result['confidence']:.0f}%\n"
                    f"Nifty: {result['ltp']}\n"
                    f"Time: {result['time']} IST"
                )
                await app.bot.send_message(
                    chat_id=config.TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown"
                )
            elif result and result["confidence"] < 60:
                logger.info(f"Signal skipped (low confidence: {result['confidence']:.1f}%)")
        except Exception as e:
            logger.error(f"Scheduler: {e}")
            await asyncio.sleep(30)

async def main():
    global model
    logger.info("Starting Nifty Predictor Bot...")

    model = load_model(config.MODEL_PATH)
    if model is None:
        logger.warning("Model not found - run python train.py")

    ok = angel.login()
    if not ok:
        logger.warning("Angel One failed - using yfinance")

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("next", next_cmd))
    app.add_handler(CommandHandler("accuracy", accuracy_cmd))
    app.add_handler(CommandHandler("last", last_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    asyncio.create_task(scheduler(app))

    logger.info("Bot started!")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped")
    except Exception as e:
        logger.error(f"Fatal: {e}")
        import traceback
        traceback.print_exc()
