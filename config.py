import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))

    ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
    ANGEL_API_KEY = os.getenv("ANGEL_API_KEY")
    ANGEL_SECRET_KEY = os.getenv("ANGEL_SECRET_KEY")
    ANGEL_PASSWORD = os.getenv("ANGEL_PASSWORD")
    ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")

    MODEL_PATH = "model.pkl"
    ACCURACY_DB = "accuracy.json"
    NIFTY_SYMBOL = "NIFTY"
    EXCHANGE = "NSE"
    INTERVAL = "FIFTEEN_MINUTE"
    TRAIN_PERIOD_DAYS = 365
    HISTORICAL_CANDLES = 100
    NIFTY_YAHOO = "^NSEI"
