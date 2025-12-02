import os, joblib, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# Survival (optional)
SURVIVAL_AVAILABLE = True
try:
    from lifelines import CoxPHFitter
except Exception:
    SURVIVAL_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_store")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_classification(df_cls: pd.DataFrame, feature_cols, target_col):
    X = df_cls[feature_cols].fillna(0.0).values
    y = df_cls[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Strong baseline
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Calibrate probabilities
    cal = CalibratedClassifierCV(clf, method="isotonic", cv=3)
    cal.fit(X_train_scaled, y_train)

    auc = roc_auc_score(y_test, cal.predict_proba(X_test_scaled)[:,1])
    meta = {"roc_auc": float(auc), "features": feature_cols}

    joblib.dump({"scaler": scaler, "model": cal, "meta": meta}, os.path.join(MODEL_DIR, "cls_reorder.pkl"))
    return meta

def train_survival(df_surv: pd.DataFrame, feature_cols):
    if not SURVIVAL_AVAILABLE:
        return {"survival_trained": False, "reason": "lifelines not available"}

    # Prepare lifelines DF: duration + event + covariates
    d = df_surv.copy()
    d = d.dropna(subset=["time_to_next"])
    d = d.rename(columns={"time_to_next": "duration"})
    d["event"] = 1  # we only included rows with known next order; in real data include censored too.

    cols = ["duration","event"] + feature_cols
    d = d[cols].copy()
    # Minimal cleaning
    d = d.fillna(0.0)

    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d, duration_col="duration", event_col="event", show_progress=False)

    joblib.dump({"cph": cph, "features": feature_cols}, os.path.join(MODEL_DIR, "surv_cox.pkl"))
    return {"survival_trained": True, "n": int(len(d))}

def predict_proba(X_df: pd.DataFrame):
    import joblib
    bundle = joblib.load(os.path.join(MODEL_DIR, "cls_reorder.pkl"))
    scaler = bundle["scaler"]
    model = bundle["model"]
    X = scaler.transform(X_df)
    return model.predict_proba(X)[:,1]
