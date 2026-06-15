import json
import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))

class TelegramBot:
    def __init__(self, config, signal_fn=None):
        self.config = config
        self.signal_fn = signal_fn
        self.app = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "📊 *Nifty 15-min Direction Predictor*\n\n"
            "Har 15 min pe Nifty ka agla direction predict karta hoon\n"
            "ML + Technical Indicators ka hybrid system\n\n"
            "*Commands:*\n"
            "/next — Agla prediction abhi\n"
            "/accuracy — Bot ki performance\n"
            "/last — Last 10 predictions\n"
            "/status — Bot health"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def next_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("⏳ Generating signal, please wait...")
        try:
            if self.signal_fn:
                result = await asyncio.to_thread(self.signal_fn)
            else:
                result = None
            if result:
                msg = self._format_prediction(result)
            else:
                msg = "⏳ Signal generate nahi ho paaya"
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Signal error: {e}")
            await update.message.reply_text("❌ Error generating signal")

    async def accuracy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = self._load_accuracy()
        if not stats or stats["total"] == 0:
            await update.message.reply_text("📉 Abhi tak koi prediction record nahi hai")
            return
        recent = [h for h in stats["history"] if h.get("result") != "waiting"][-50:]
        correct_recent = sum(1 for h in recent if h.get("result") == "correct")
        total_recent = len(recent)
        recent_acc = correct_recent / total_recent * 100 if total_recent > 0 else 0
        overall_acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        msg = (
            f"📈 *Accuracy Report*\n\n"
            f"Total: {stats['total']}\n"
            f"✅ Correct: {stats['correct']}\n"
            f"❌ Wrong: {stats['total'] - stats['correct']}\n"
            f"🎯 Overall: `{overall_acc:.1f}%`\n"
            f"📊 Last 50: `{recent_acc:.1f}%`"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def last_predictions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = self._load_accuracy()
        if not stats or "history" not in stats or len(stats["history"]) == 0:
            await update.message.reply_text("📉 Abhi tak koi prediction nahi hai")
            return
        history = stats["history"][-10:]
        lines = ["📋 *Last 10 Predictions*\n"]
        for p in reversed(history):
            status = p.get("result", "waiting")
            icon = "✅" if status == "correct" else ("❌" if status == "wrong" else "⏳")
            lines.append(
                f"{icon} {p['time']} | {p['direction']} ({p['confidence']:.0f}%)"
                f" @ {p.get('ltp', '?')}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from model import load_model
        model = load_model(self.config.MODEL_PATH)
        model_ok = "✅ Loaded" if model else "❌ Not loaded"
        now = datetime.now(IST)
        market = "🟢 Open" if self._is_market_open() else "🔴 Closed"
        await update.message.reply_text(
            f"🤖 *Bot Status*\n\n"
            f"Model: {model_ok}\n"
            f"Market: {market}\n"
            f"Time: {now.strftime('%I:%M %p')} IST\n"
            f"Interval: 15-min\n"
            f"System: ML + Indicators",
            parse_mode="Markdown"
        )

    def _is_market_open(self):
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return False
        minutes = now.hour * 60 + now.minute
        return 555 <= minutes <= 931

    def _format_prediction(self, result):
        return (
            f"{result['emoji']} *Nifty 15-min Signal*\n\n"
            f"Direction: *{result['direction']}*\n"
            f"Confidence: {result['confidence']:.0f}%\n"
            f"Current: {result['ltp']}\n"
            f"Time: {result['time']} IST\n\n"
            f"ML: {result['ml_direction']} ({result['ml_confidence']:.0f}%)"
        )

    def _load_accuracy(self):
        path = self.config.ACCURACY_DB
        if not os.path.exists(path):
            return {"total": 0, "correct": 0, "history": []}
        with open(path) as f:
            return json.load(f)

    async def send_alert(self, text):
        if not self.app:
            return
        try:
            await self.app.bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Send alert failed: {e}")

    def setup(self, application: Application):
        self.app = application
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("next", self.next_signal))
        application.add_handler(CommandHandler("accuracy", self.accuracy))
        application.add_handler(CommandHandler("last", self.last_predictions))
        application.add_handler(CommandHandler("status", self.status))
