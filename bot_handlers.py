import json
import os
import logging
from datetime import datetime, timezone, timedelta
from telegram import Update
from telegram.ext import CommandHandler, ContextTypes

logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))

def _load_stats():
    path = "accuracy.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"total": 0, "correct": 0, "history": []}

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Nifty Buy/Sell Signal Bot*\n\n"
        "Har 15 min Nifty ka agla 30 min direction predict karta hoon\n"
        "BUY ya SELL signal bhejta hoon\n"
        "Model: 57% accuracy\n\n"
        "/next - Agla signal abhi\n"
        "/accuracy - Performance\n"
        "/last - Last 10 signals\n"
        "/status - Bot status"
    )

async def next_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating signal...")
    import requests
    vercel_url = os.environ.get("VERCEL_URL", "")
    if not vercel_url:
        await update.message.reply_text("Not deployed on Vercel")
        return
    r = requests.get(f"https://{vercel_url}/api/next")
    data = r.json()
    if data.get("signal"):
        arrow = "\U0001f53c" if data["signal"] == "BUY" else "\u2b07"
        msg = (
            f"{arrow} *Nifty 30min Signal*\n\n"
            f"Action: *{data['signal']}*\n"
            f"Confidence: {data['confidence']:.0f}%\n"
            f"Nifty: {data['ltp']}\n"
            f"Time: {data['time']} IST"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
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
    import requests
    vercel_url = os.environ.get("VERCEL_URL", "")
    model = "Not loaded"
    if vercel_url:
        r = requests.get(f"https://{vercel_url}/api/health")
        model = r.json().get("model", "unknown")
    now = datetime.now(IST)
    market = "Open" if 555 <= now.hour * 60 + now.minute <= 931 and now.weekday() < 5 else "Closed"
    await update.message.reply_text(f"Model: {model}\nMarket: {market}\nTime: {now.strftime('%I:%M %p')} IST")

def setup_handlers(app):
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("next", next_cmd))
    app.add_handler(CommandHandler("accuracy", accuracy_cmd))
    app.add_handler(CommandHandler("last", last_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
