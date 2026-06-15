import pyotp
import logging
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect

logger = logging.getLogger(__name__)

class AngelClient:
    def __init__(self, config):
        self.config = config
        self.obj = None
        self.feed_token = None
        self.logged_in = False

    def login(self):
        try:
            totp = pyotp.TOTP(self.config.ANGEL_TOTP_SECRET).now()
            self.obj = SmartConnect(api_key=self.config.ANGEL_API_KEY)
            data = self.obj.generateSession(
                clientCode=self.config.ANGEL_CLIENT_ID,
                password=self.config.ANGEL_PASSWORD,
                totp=totp
            )
            if data.get("status"):
                self.feed_token = self.obj.getfeedToken()
                self.logged_in = True
                logger.info("Angel One login successful")
                return True
            else:
                logger.error(f"Angel One login failed: {data}")
                return False
        except Exception as e:
            logger.error(f"Angel One login error: {e}")
            return False

    def get_historical_data(self, symboltoken, interval, from_date, to_date):
        if not self.logged_in:
            logger.warning("Not logged in to Angel One")
            return None
        try:
            params = {
                "exchange": self.config.EXCHANGE,
                "symboltoken": symboltoken,
                "interval": interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }
            response = self.obj.getCandleData(params)
            if not response or not response.get("status"):
                logger.error(f"API error: {response}")
                return None
            candles = response.get("data", [])
            if not candles or len(candles) == 0:
                return None

            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return None

    def get_nifty_candles(self, count=100):
        now = datetime.now()
        to_date = now
        from_date = now - timedelta(days=15)
        return self.get_historical_data(
            symboltoken="99926000",
            interval=self.config.INTERVAL,
            from_date=from_date,
            to_date=to_date
        )

    def get_ltp(self):
        if not self.logged_in:
            return None
        try:
            data = self.obj.ltpData("NSE", "NIFTY", "99926000")
            if data and data.get("data"):
                return data["data"].get("ltp")
            return None
        except Exception as e:
            logger.error(f"Failed to get LTP: {e}")
            return None
