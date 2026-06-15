import os
import pickle
import logging
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

def create_model(seed=42):
    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        min_child_weight=3,
        gamma=0.1,
        random_state=seed,
        eval_metric="logloss"
    )

def train_model(X_train, y_train, X_val=None, y_val=None, sample_weight=None):
    model = create_model()
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {acc:.4f}")
    return acc, y_pred

def predict_next(model, features_array):
    if model is None or features_array is None:
        return 0, 50.0
    f = features_array.reshape(1, -1)
    raw = model.predict(f)[0]
    proba = model.predict_proba(f)[0]
    confidence = max(proba) * 100
    return int(raw), float(confidence)

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved -> {path}")

def load_model(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded <- {path}")
    return model

class EnsembleModel:
    def __init__(self, models=None):
        self.models = models or []

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return (preds.mean(axis=0) >= 0.5).astype(int)

    def predict_proba(self, X):
        probas = np.array([m.predict_proba(X) for m in self.models])
        return probas.mean(axis=0)

    def add_model(self, model):
        self.models.append(model)

    @property
    def n_models(self):
        return len(self.models)

def train_ensemble(X_train, y_train, X_val=None, y_val=None, n_models=5, sample_weight=None):
    ensemble = EnsembleModel()
    seeds = [42, 123, 456, 789, 111]
    for i in range(min(n_models, len(seeds))):
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.025,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=1.5, min_child_weight=3,
            gamma=0.1, random_state=seeds[i], eval_metric="logloss"
        )
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        ensemble.add_model(model)
        logger.info(f"Ensemble model {i+1}/{n_models} trained (seed={seeds[i]})")
    return ensemble

def tune_hyperparams(X_train, y_train):
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.025, 0.05],
        "n_estimators": [200, 300, 400],
        "min_child_weight": [2, 3, 5],
        "subsample": [0.7, 0.8],
    }
    model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(model, param_grid, cv=skf, scoring="accuracy", n_jobs=1, verbose=0)
    grid.fit(X_train, y_train)
    logger.info(f"Best params: {grid.best_params_}, Best CV: {grid.best_score_:.4f}")
    return grid.best_estimator_

def train_with_cv(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, models = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        model = train_model(X_tr, y_tr, X_va, y_va)
        acc, _ = evaluate_model(model, X_va, y_va)
        accs.append(acc)
        models.append(model)
    mean_acc = np.mean(accs)
    logger.info(f"CV accuracies: {accs}, mean: {mean_acc:.4f}")
    best = models[int(np.argmax(accs))]
    return best, mean_acc
