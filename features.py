import numpy as np
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

_DAILY_DF = None
_VIX_DF = None

def _load_daily_data():
    global _DAILY_DF
    try:
        df = yf.download("^NSEI", period="6mo", interval="1d", progress=False)
        if not df.empty:
            df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
            _DAILY_DF = df.rename(columns={
                f"Open_^NSEI": "open", f"High_^NSEI": "high",
                f"Low_^NSEI": "low", f"Close_^NSEI": "close",
                f"Volume_^NSEI": "volume"
            })
    except Exception as e:
        logger.warning(f"Daily data: {e}")

def _load_vix_data():
    global _VIX_DF
    try:
        df = yf.download("^INDIAVIX", period="6mo", interval="1d", progress=False)
        if not df.empty:
            df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
            _VIX_DF = df.rename(columns={
                f"Open_^INDIAVIX": "open", f"High_^INDIAVIX": "high",
                f"Low_^INDIAVIX": "low", f"Close_^INDIAVIX": "close"
            })
    except Exception as e:
        logger.warning(f"VIX data: {e}")

_load_daily_data()
_load_vix_data()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = compute_ema(macd_line, signal)
    return macd_line, macd_signal, macd_line - macd_signal

def compute_bollinger(series, period=20, std_dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - (sma - std_dev * std)) / ((sma + std_dev * std) - (sma - std_dev * std)).replace(0, np.nan)

def compute_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"] - df["close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_candle_pattern(row, prev):
    body = abs(row["close"] - row["open"])
    range_c = row["high"] - row["low"]
    body_ratio = body / range_c if range_c > 0 else 0
    upper_wick = (row["high"] - max(row["open"], row["close"])) / range_c if range_c > 0 else 0
    lower_wick = (min(row["open"], row["close"]) - row["low"]) / range_c if range_c > 0 else 0
    prev_body = abs(prev["close"] - prev["open"])
    prev_range = prev["high"] - prev["low"]

    is_doji = body_ratio < 0.1
    is_hammer = lower_wick > 2 * body and upper_wick < body and body_ratio > 0.1 if range_c > 0 else False
    is_bullish_engulfing = prev["close"] < prev["open"] and row["close"] > row["open"] and row["open"] < prev["close"] and row["close"] > prev["open"]
    is_bearish_engulfing = prev["close"] > prev["open"] and row["close"] < row["open"] and row["open"] > prev["close"] and row["close"] < prev["open"]
    is_marubozu = body_ratio > 0.95
    return is_doji, is_hammer, is_bullish_engulfing, is_bearish_engulfing, is_marubozu

def compute_features(df):
    if df is None or len(df) < 55:
        return None

    df = df.copy()
    close = df["close"].values
    close_s = df["close"]
    last = df.iloc[-1]
    prev = df.iloc[-2]

    features = {}
    cr = last["high"] - last["low"]
    body = abs(last["close"] - last["open"])
    features["body_ratio"] = body / cr if cr > 0 else 0
    features["upper_wick"] = (last["high"] - max(last["open"], last["close"])) / cr if cr > 0 else 0
    features["lower_wick"] = (min(last["open"], last["close"]) - last["low"]) / cr if cr > 0 else 0
    features["return_pct"] = (last["close"] - last["open"]) / last["open"] * 100
    features["range_pct"] = (last["high"] - last["low"]) / last["open"] * 100
    features["prev_close_ratio"] = last["close"] / prev["close"] - 1
    features["gap_pct"] = (last["open"] - prev["close"]) / prev["close"] * 100
    features["close_ma20_ratio"] = last["close"] / close_s.iloc[-20:].mean() - 1 if len(df) >= 20 else 0
    features["close_ma50_ratio"] = last["close"] / close_s.iloc[-50:].mean() - 1 if len(df) >= 50 else 0

    rsi14 = compute_rsi(close_s, 14)
    rsi7 = compute_rsi(close_s, 7)
    rsi5 = compute_rsi(close_s, 5)
    features["rsi_14"] = rsi14.iloc[-1]
    features["rsi_7"] = rsi7.iloc[-1]
    features["rsi_5"] = rsi5.iloc[-1]
    features["rsi_14_change"] = rsi14.iloc[-1] - rsi14.iloc[-2]
    features["rsi_oversold"] = 1 if rsi14.iloc[-1] < 35 else 0
    features["rsi_overbought"] = 1 if rsi14.iloc[-1] > 65 else 0

    features["rsi_divergence"] = 0
    if len(rsi14) >= 5 and len(close_s) >= 5:
        if close_s.iloc[-1] < close_s.iloc[-5] and rsi14.iloc[-1] > rsi14.iloc[-5]:
            features["rsi_divergence"] = 1
        elif close_s.iloc[-1] > close_s.iloc[-5] and rsi14.iloc[-1] < rsi14.iloc[-5]:
            features["rsi_divergence"] = -1

    ema9 = compute_ema(close_s, 9)
    ema21 = compute_ema(close_s, 21)
    ema50 = compute_ema(close_s, 50) if len(df) >= 50 else close_s
    ema200 = compute_ema(close_s, 200) if len(df) >= 200 else close_s
    features["ema_ratio_9_21"] = ema9.iloc[-1] / ema21.iloc[-1] if ema21.iloc[-1] != 0 else 1
    features["ema_ratio_21_50"] = ema21.iloc[-1] / ema50.iloc[-1] if ema50.iloc[-1] != 0 else 1
    features["price_vs_ema9"] = last["close"] / ema9.iloc[-1] - 1
    features["price_vs_ema21"] = last["close"] / ema21.iloc[-1] - 1
    features["price_vs_ema200"] = last["close"] / ema200.iloc[-1] - 1
    features["ema_9_slope"] = (ema9.iloc[-1] - ema9.iloc[-3]) / ema9.iloc[-3] if len(ema9) >= 3 else 0
    features["ema_21_slope"] = (ema21.iloc[-1] - ema21.iloc[-3]) / ema21.iloc[-3] if len(ema21) >= 3 else 0
    features["ema_bullish"] = 1 if ema9.iloc[-1] > ema21.iloc[-1] else 0
    features["ema_cross_up"] = 1 if (len(ema9) >= 2 and ema9.iloc[-2] <= ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1]) else 0
    features["ema_cross_down"] = 1 if (len(ema9) >= 2 and ema9.iloc[-2] >= ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1]) else 0

    macd_l, macd_s, macd_h = compute_macd(close_s)
    features["macd"] = macd_l.iloc[-1]
    features["macd_signal"] = macd_s.iloc[-1]
    features["macd_hist"] = macd_h.iloc[-1]
    features["macd_hist_change"] = macd_h.iloc[-1] - macd_h.iloc[-2] if len(macd_h) >= 2 else 0
    features["macd_bullish"] = 1 if macd_l.iloc[-1] > macd_s.iloc[-1] else 0
    features["macd_cross_up"] = 1 if (len(macd_l) >= 2 and macd_l.iloc[-2] <= macd_s.iloc[-2] and macd_l.iloc[-1] > macd_s.iloc[-1]) else 0
    features["macd_cross_down"] = 1 if (len(macd_l) >= 2 and macd_l.iloc[-2] >= macd_s.iloc[-2] and macd_l.iloc[-1] < macd_s.iloc[-1]) else 0

    features["bb_pct"] = compute_bollinger(close_s).iloc[-1]
    features["bb_width"] = (close_s.rolling(20).std() * 4 / close_s.rolling(20).mean()).iloc[-1]
    atr = compute_atr(df)
    features["atr_ratio"] = atr.iloc[-1] / last["close"] if last["close"] != 0 else 0
    features["atr_change"] = atr.iloc[-1] / atr.iloc[-5] - 1 if len(atr) >= 5 else 0

    vol_sma = df["volume"].rolling(20).mean()
    features["volume_ratio"] = last["volume"] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1
    features["volume_change"] = last["volume"] / df["volume"].iloc[-2] - 1 if df["volume"].iloc[-2] > 0 else 0

    ret = df["close"].pct_change()
    for i in range(1, 11):
        features[f"ret_{i}"] = ret.iloc[-i] if len(ret) >= i else 0
        features[f"dir_{i}"] = 1 if (len(ret) >= i and ret.iloc[-i] > 0) else 0

    features["volatility_5"] = ret.iloc[-5:].std() if len(ret) >= 5 else 0
    features["volatility_10"] = ret.iloc[-10:].std() if len(ret) >= 10 else 0
    features["volatility_ratio"] = features["volatility_5"] / features["volatility_10"] if features["volatility_10"] > 0 else 1

    features["hour"] = df["timestamp"].iloc[-1].hour
    features["minute"] = df["timestamp"].iloc[-1].minute
    features["day_of_week"] = df["timestamp"].iloc[-1].dayofweek

    up_count = sum(1 for i in range(-5, 0) if close[i] > close[i - 1])
    features["up_streak"] = up_count

    doji, hammer, bullish_eng, bearish_eng, marubozu = detect_candle_pattern(last, prev)
    features["doji"] = 1 if doji else 0
    features["hammer"] = 1 if hammer else 0
    features["bullish_engulf"] = 1 if bullish_eng else 0
    features["bearish_engulf"] = 1 if bearish_eng else 0
    features["marubozu"] = 1 if marubozu else 0

    recent_high = close_s.iloc[-20:].max()
    recent_low = close_s.iloc[-20:].min()
    features["dist_high_pct"] = (recent_high - last["close"]) / last["close"] * 100
    features["dist_low_pct"] = (last["close"] - recent_low) / last["close"] * 100
    pivot = (recent_high + recent_low + last["close"]) / 3
    r1 = 2 * pivot - recent_low
    s1 = 2 * pivot - recent_high
    features["pivot_dist"] = (last["close"] - pivot) / pivot * 100
    features["r1_dist"] = (r1 - last["close"]) / last["close"] * 100
    features["s1_dist"] = (last["close"] - s1) / last["close"] * 100

    if _DAILY_DF is not None and len(_DAILY_DF) > 20:
        daily_close = _DAILY_DF["close"]
        daily_ema9 = compute_ema(daily_close, 9)
        daily_ema21 = compute_ema(daily_close, 21)
        daily_rsi = compute_rsi(daily_close, 14)
        features["daily_ema_ratio"] = daily_ema9.iloc[-1] / daily_ema21.iloc[-1] if daily_ema21.iloc[-1] != 0 else 1
        features["daily_rsi"] = daily_rsi.iloc[-1]
        features["daily_trend"] = 1 if daily_ema9.iloc[-1] > daily_ema21.iloc[-1] else -1
        daily_ret = daily_close.pct_change().iloc[-1]
        features["daily_return"] = daily_ret if not pd.isna(daily_ret) else 0
    else:
        features["daily_ema_ratio"] = 1
        features["daily_rsi"] = 50
        features["daily_trend"] = 0
        features["daily_return"] = 0

    if _VIX_DF is not None and len(_VIX_DF) > 5:
        vix = _VIX_DF["close"]
        features["vix_close"] = vix.iloc[-1]
        features["vix_change"] = vix.pct_change().iloc[-1] if len(vix) >= 2 else 0
        features["vix_ma_ratio"] = vix.iloc[-1] / vix.iloc[-10:].mean() if len(vix) >= 10 else 1
        vix_high = vix.iloc[-1] > vix.iloc[-20:].mean()
        features["vix_high"] = 1 if vix_high else 0
    else:
        features["vix_close"] = 15
        features["vix_change"] = 0
        features["vix_ma_ratio"] = 1
        features["vix_high"] = 0

    close_mean = close_s.mean()
    features["close_vs_mean"] = (last["close"] - close_mean) / close_mean * 100
    features["skew_3"] = ret.iloc[-3:].skew() if len(ret) >= 3 else 0
    features["kurt_5"] = ret.iloc[-5:].kurtosis() if len(ret) >= 5 else 0

    return features

def get_feature_names():
    return [
        "body_ratio", "upper_wick", "lower_wick", "return_pct", "range_pct",
        "prev_close_ratio", "gap_pct", "close_ma20_ratio", "close_ma50_ratio",
        "rsi_14", "rsi_7", "rsi_5", "rsi_14_change", "rsi_oversold", "rsi_overbought",
        "rsi_divergence",
        "ema_ratio_9_21", "ema_ratio_21_50", "price_vs_ema9", "price_vs_ema21",
        "price_vs_ema200", "ema_9_slope", "ema_21_slope",
        "ema_bullish", "ema_cross_up", "ema_cross_down",
        "macd", "macd_signal", "macd_hist", "macd_hist_change", "macd_bullish",
        "macd_cross_up", "macd_cross_down",
        "bb_pct", "bb_width", "atr_ratio", "atr_change",
        "volume_ratio", "volume_change",
        "ret_1", "ret_2", "ret_3", "ret_4", "ret_5", "ret_6", "ret_7", "ret_8", "ret_9", "ret_10",
        "dir_1", "dir_2", "dir_3", "dir_4", "dir_5", "dir_6", "dir_7", "dir_8", "dir_9", "dir_10",
        "volatility_5", "volatility_10", "volatility_ratio",
        "hour", "minute", "day_of_week", "up_streak",
        "doji", "hammer", "bullish_engulf", "bearish_engulf", "marubozu",
        "dist_high_pct", "dist_low_pct", "pivot_dist", "r1_dist", "s1_dist",
        "daily_ema_ratio", "daily_rsi", "daily_trend", "daily_return",
        "vix_close", "vix_change", "vix_ma_ratio", "vix_high",
        "close_vs_mean", "skew_3", "kurt_5"
    ]

def features_to_array(features_dict):
    names = get_feature_names()
    arr = np.array([features_dict.get(name, 0) for name in names])
    arr = np.nan_to_num(arr, nan=0, posinf=0, neginf=0)
    return arr

def prepare_training_data(df, target_type="next_candle"):
    df = df.copy().reset_index(drop=True)
    close = df["close"].values

    if target_type == "next_candle":
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    elif target_type == "next_3_candle":
        future_close = df["close"].shift(-3)
        df["target"] = (future_close > df["close"]).astype(int)
    elif target_type == "significant_move":
        ret = df["close"].pct_change(-1) * 100
        df["target"] = 0
        df.loc[ret > 0.05, "target"] = 1
        df.loc[ret < -0.05, "target"] = 0
        df.loc[abs(ret) <= 0.05, "target"] = -1
    elif target_type == "next_2_candle":
        future_close = df["close"].shift(-2)
        df["target"] = (future_close > df["close"]).astype(int)

    valid = df["target"].notna()
    if target_type == "significant_move":
        valid = valid & (df["target"] != -1)
    df = df[valid].reset_index(drop=True)

    X_list, y_list, w_list = [], [], []
    for i in range(len(df)):
        window = df.iloc[:i + 1]
        if len(window) < 55:
            continue
        feat = compute_features(window)
        if feat is not None:
            X_list.append(features_to_array(feat))
            y_list.append(int(df.iloc[i]["target"]))
            ret = abs(df["close"].pct_change(-1).iloc[i]) * 100 if i < len(df) - 1 else 0.05
            w_list.append(min(ret / 0.05, 3.0) if ret > 0 else 1.0)

    if not X_list:
        return None, None, None, None

    return (
        np.array(X_list),
        np.array(y_list, dtype=int),
        df.iloc[54:54 + len(X_list)]["timestamp"].values,
        np.array(w_list)
    )
