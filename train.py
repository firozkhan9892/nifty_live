import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from features import compute_features, features_to_array, prepare_training_data
from model import train_model, save_model, evaluate_model, train_ensemble
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def download_nifty_15m(days=60):
    logger.info(f"Downloading Nifty 15-min (last {days}d)...")
    df = yf.download("^NSEI", period=f"{days}d", interval="15m", progress=False)
    if df.empty:
        return None
    df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()
    df = df.rename(columns={
        "Datetime": "timestamp", "Open_^NSEI": "open", "High_^NSEI": "high",
        "Low_^NSEI": "low", "Close_^NSEI": "close", "Volume_^NSEI": "volume"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.dropna().reset_index(drop=True)

def evaluate_thresholds(proba, y_test, label=""):
    best_acc, best_th = 0, 0.5
    logger.info(f"\n{label} Threshold analysis:")
    for th in np.arange(0.50, 0.85, 0.05):
        pred = (proba >= th).astype(int)
        acc = np.mean(pred == y_test)
        cov = np.mean(proba >= th) * 100
        logger.info(f"  th={th:.2f}: acc={acc:.2%}, cov={cov:.1f}%")
        if cov > 5 and acc > best_acc:
            best_acc, best_th = acc, th
    logger.info(f"  Best: th={best_th:.2f}, acc={best_acc:.2%}")
    return best_th, best_acc

def main():
    logger.info("=== Training Nifty Predictor v2 ===")

    df = download_nifty_15m(60)
    if df is None or len(df) < 200:
        return

    target_types = [
        ("next_2_candle", "Next 2 candles direction"),
        ("next_candle", "Next 1 candle direction"),
    ]

    best_acc_overall = 0
    best_model = None
    best_target = ""

    for ttype, tname in target_types:
        logger.info(f"\n--- Target: {tname} ---")
        X, y, ts, weights = prepare_training_data(df, target_type=ttype)
        if X is None:
            logger.warning(f"  No data for target {ttype}")
            continue

        logger.info(f"  Samples: {len(X)}, Features: {X.shape[1]}, UP={y.mean():.1%}")

        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        w_tr = weights[:split] if weights is not None else None

        model = train_model(X_tr, y_tr, X_te, y_te, sample_weight=w_tr)
        acc, _ = evaluate_model(model, X_te, y_te)
        logger.info(f"  Single model accuracy: {acc:.2%}")

        proba = model.predict_proba(X_te)[:, 1]
        evaluate_thresholds(proba, y_te, tname)

        if acc > best_acc_overall:
            best_acc_overall = acc
            best_model = model
            best_target = ttype

        ensemble = train_ensemble(X_tr, y_tr, X_te, y_te, n_models=5)
        ens_pred = ensemble.predict(X_te)
        ens_acc = np.mean(ens_pred == y_te)
        logger.info(f"  Ensemble accuracy: {ens_acc:.2%}")

        if ens_acc > best_acc_overall:
            best_acc_overall = ens_acc
            best_model = model
            best_target = ttype

    if best_model:
        logger.info(f"\nBest model: target={best_target}, acc={best_acc_overall:.2%}")
        save_model(best_model, Config.MODEL_PATH)
    else:
        logger.error("No model trained!")

    feat_names = __import__("features", fromlist=["get_feature_names"]).get_feature_names()
    imp = best_model.feature_importances_
    top = np.argsort(imp)[-20:]
    logger.info("Top 20 features:")
    for idx in reversed(top):
        logger.info(f"  {feat_names[idx]}: {imp[idx]:.4f}")

    logger.info("Training complete!")

if __name__ == "__main__":
    main()
