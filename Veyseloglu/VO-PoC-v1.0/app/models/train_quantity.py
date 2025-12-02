import os, joblib, warnings, numpy as np, pandas as pd

from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_store")
os.makedirs(MODEL_DIR, exist_ok=True)

# Optional Chronos-T5
HF_AVAILABLE = True
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    HF_AVAILABLE = False

def train_regression(df_reg: pd.DataFrame, feature_cols, target_col):
    X = df_reg[feature_cols].fillna(0.0).values
    y = df_reg[target_col].values

    # Tweedie GLM (handles many zeros/heavy tails)
    glm = TweedieRegressor(power=1.5, alpha=0.1)
    glm.fit(X, y)

    # Quantile GBM for P90
    gbr_p50 = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=42)
    gbr_p90 = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=42)
    gbr_p50.fit(X, y)
    gbr_p90.fit(X, y)

    yhat = glm.predict(X)
    mae = mean_absolute_error(y, yhat)

    joblib.dump({"glm": glm, "gbr_p50": gbr_p50, "gbr_p90": gbr_p90, "features": feature_cols, "mae": float(mae)}, os.path.join(MODEL_DIR, "reg_qty.pkl"))
    return {"mae": float(mae), "features": feature_cols}

def load_chronos(model_name: str="amazon/chronos-t5-large"):
    if not HF_AVAILABLE:
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tok, mdl
    except Exception as e:
        return None, None

def chronos_forecast(tokenizer, model, series, horizon=1):
    # Minimal text-serialization of series (Chronos expects tokenized numeric sequence).
    # This is a *very* simplified demonstration; real Chronos uses specialized processing.
    if tokenizer is None or model is None:
        return None
    nums = " ".join([str(float(x)) for x in series])
    prompt = f"predict next {horizon} steps: {nums}"
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=8)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract first number found
    import re
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except:
        return None
