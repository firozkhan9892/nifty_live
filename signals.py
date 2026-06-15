import numpy as np
import logging
from features import compute_rsi, compute_ema, compute_macd, compute_bollinger, compute_atr

logger = logging.getLogger(__name__)

def compute_indicator_signals(df):
    signals = {}
    close = df["close"]

    rsi14 = compute_rsi(close, 14)
    rsi7 = compute_rsi(close, 7)
    ema9 = compute_ema(close, 9)
    ema21 = compute_ema(close, 21)
    ema50 = compute_ema(close, 50) if len(df) >= 50 else close
    macd_l, macd_s, macd_h = compute_macd(close)
    bb = compute_bollinger(close)
    atr = compute_atr(df)

    rv = rsi14.iloc[-1]
    rv_prev = rsi14.iloc[-2] if len(rsi14) >= 2 else 50
    cp = close.iloc[-1]
    cp_prev = close.iloc[-2]

    signals["rsi_oversold"] = rv < 35
    signals["rsi_overbought"] = rv > 65
    signals["rsi_mild_oversold"] = 35 <= rv < 45
    signals["rsi_mild_overbought"] = 55 < rv <= 65

    signals["rsi_bullish_div"] = bool(rv > rv_prev and cp < cp_prev)
    signals["rsi_bearish_div"] = bool(rv < rv_prev and cp > cp_prev)
    signals["rsi_rising"] = bool(rv > rv_prev)
    signals["rsi_7_oversold"] = bool(rsi7.iloc[-1] < 30) if len(rsi7) >= 1 else False
    signals["rsi_7_overbought"] = bool(rsi7.iloc[-1] > 70) if len(rsi7) >= 1 else False

    signals["ema_bullish"] = bool(ema9.iloc[-1] > ema21.iloc[-1])
    signals["ema_bearish"] = bool(ema9.iloc[-1] < ema21.iloc[-1])
    signals["ema_cross_up"] = bool(len(ema9) >= 2 and ema9.iloc[-2] <= ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1])
    signals["ema_cross_down"] = bool(len(ema9) >= 2 and ema9.iloc[-2] >= ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1])
    signals["price_above_ema21"] = bool(cp > ema21.iloc[-1])
    signals["price_above_ema50"] = bool(len(ema50) > 1 and cp > ema50.iloc[-1])
    signals["ema_alignment_bull"] = bool(ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]) if len(ema50) > 1 else False
    signals["ema_alignment_bear"] = bool(ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]) if len(ema50) > 1 else False

    signals["macd_bullish"] = bool(macd_l.iloc[-1] > macd_s.iloc[-1])
    signals["macd_bearish"] = bool(macd_l.iloc[-1] < macd_s.iloc[-1])
    signals["macd_cross_up"] = bool(len(macd_l) >= 2 and macd_l.iloc[-2] <= macd_s.iloc[-2] and macd_l.iloc[-1] > macd_s.iloc[-1])
    signals["macd_cross_down"] = bool(len(macd_l) >= 2 and macd_l.iloc[-2] >= macd_s.iloc[-2] and macd_l.iloc[-1] < macd_s.iloc[-1])
    signals["macd_hist_rising"] = bool(len(macd_h) >= 2 and macd_h.iloc[-1] > macd_h.iloc[-2])
    signals["macd_hist_falling"] = bool(len(macd_h) >= 2 and macd_h.iloc[-1] < macd_h.iloc[-2])
    signals["macd_hist_positive"] = bool(macd_h.iloc[-1] > 0)
    signals["macd_hist_turning_up"] = bool(len(macd_h) >= 3 and macd_h.iloc[-2] <= macd_h.iloc[-3] and macd_h.iloc[-1] > macd_h.iloc[-2])

    bb_val = bb.iloc[-1]
    signals["bb_lower"] = bool(bb_val < 0.2)
    signals["bb_upper"] = bool(bb_val > 0.8)
    signals["bb_mid"] = bool(0.4 <= bb_val <= 0.6)

    vol = df["volume"]
    vol_sma = vol.rolling(20).mean()
    signals["volume_spike"] = bool(len(vol_sma) > 0 and vol.iloc[-1] > vol_sma.iloc[-1] * 1.5)
    signals["volume_drop"] = bool(len(vol_sma) > 0 and vol.iloc[-1] < vol_sma.iloc[-1] * 0.5)

    body = abs(close.iloc[-1] - df["open"].iloc[-1])
    candle_range = df["high"].iloc[-1] - df["low"].iloc[-1]
    body_ratio = body / candle_range if candle_range > 0 else 0
    signals["doji"] = bool(body_ratio < 0.1)
    prev_body = abs(close.iloc[-2] - df["open"].iloc[-2])
    signals["bullish_engulf"] = bool(
        close.iloc[-2] < df["open"].iloc[-2] and
        close.iloc[-1] > df["open"].iloc[-1] and
        df["open"].iloc[-1] < close.iloc[-2] and
        close.iloc[-1] > df["open"].iloc[-2]
    )
    signals["bearish_engulf"] = bool(
        close.iloc[-2] > df["open"].iloc[-2] and
        close.iloc[-1] < df["open"].iloc[-1] and
        df["open"].iloc[-1] > close.iloc[-2] and
        close.iloc[-1] < df["open"].iloc[-2]
    )
    signals["close_near_high"] = bool((df["high"].iloc[-1] - close.iloc[-1]) / (df["high"].iloc[-1] - df["low"].iloc[-1]) < 0.2 if candle_range > 0 else False)
    signals["close_near_low"] = bool((close.iloc[-1] - df["low"].iloc[-1]) / (df["high"].iloc[-1] - df["low"].iloc[-1]) < 0.2 if candle_range > 0 else False)

    ret_5 = close.pct_change(5).iloc[-1]
    signals["positive_5c"] = bool(not np.isnan(ret_5) and ret_5 > 0)
    signals["negative_5c"] = bool(not np.isnan(ret_5) and ret_5 < 0)

    atr_val = atr.iloc[-1] if len(atr) > 0 else 0
    atr_avg = atr.iloc[-10:].mean() if len(atr) >= 10 else atr_val
    signals["volatility_high"] = bool(atr_val > atr_avg * 1.2 if atr_avg > 0 else False)
    signals["volatility_low"] = bool(atr_val < atr_avg * 0.8 if atr_avg > 0 else False)

    return signals

def compute_bias_score(signals):
    score = 0
    weights = {
        "rsi_oversold": 2.5, "rsi_7_oversold": 2, "rsi_mild_oversold": 1,
        "rsi_bullish_div": 3, "rsi_rising": 1,
        "ema_bullish": 2, "ema_cross_up": 3.5, "price_above_ema21": 1.5,
        "price_above_ema50": 1.5, "ema_alignment_bull": 3,
        "macd_bullish": 2, "macd_cross_up": 3, "macd_hist_rising": 1.5,
        "macd_hist_turning_up": 2.5, "macd_hist_positive": 1,
        "bb_lower": 1.5, "volume_spike": 1,
        "bullish_engulf": 2.5, "close_near_high": 1,
        "positive_5c": 1.5, "volatility_low": 0.5,

        "rsi_overbought": -2.5, "rsi_7_overbought": -2, "rsi_mild_overbought": -1,
        "rsi_bearish_div": -3, "rsi_rising": -1,
        "ema_bearish": -2, "ema_cross_down": -3.5, "price_above_ema21": -1.5,
        "price_above_ema50": 1.5,
        "ema_alignment_bear": -3,
        "macd_bearish": -2, "macd_cross_down": -3, "macd_hist_falling": -1.5,
        "bb_upper": -1.5, "volume_drop": -0.5,
        "bearish_engulf": -2.5, "close_near_low": -1,
        "negative_5c": -1.5, "volatility_high": -0.5,
    }

    for sig, weight in weights.items():
        if signals.get(sig):
            score += weight

    return np.clip(score / 20, -1, 1)

def combine_ml_and_indicators(ml_direction, ml_confidence, indicator_score):
    ml_signal = 1 if ml_direction == 1 else -1
    ml_weight = ml_confidence / 100.0

    combined = ml_signal * ml_weight * 0.55 + indicator_score * 0.45

    direction = 1 if combined > 0.05 else (0 if combined < -0.05 else 2)
    confidence = min(abs(combined) * 100 + 50, 95)

    return direction, confidence
