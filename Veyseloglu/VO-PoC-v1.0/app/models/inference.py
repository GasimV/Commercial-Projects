import os, joblib, numpy as np, pandas as pd

from .train_likelihood import predict_proba as cls_predict_proba
from .train_quantity import chronos_forecast, load_chronos

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_store")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_cls_bundle():
    import joblib
    return joblib.load(os.path.join(MODEL_DIR, "cls_reorder.pkl"))

def load_reg_bundle():
    import joblib
    return joblib.load(os.path.join(MODEL_DIR, "reg_qty.pkl"))

def load_ranker_bundle():
    import joblib
    path = os.path.join(MODEL_DIR, "ranker.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def predict_likelihood(df_current, feature_cols):
    bundle = load_cls_bundle()
    scaler = bundle["scaler"]
    model = bundle["model"]
    X = scaler.transform(df_current[feature_cols].fillna(0.0).values)
    return model.predict_proba(X)[:,1]

def predict_quantity(df_current, feature_cols):
    bundle = load_reg_bundle()
    glm = bundle["glm"]; p50 = bundle["gbr_p50"]; p90 = bundle["gbr_p90"]
    X = df_current[feature_cols].fillna(0.0).values
    return glm.predict(X), p50.predict(X), p90.predict(X)

def recommend(feat_current: pd.DataFrame, K: int = 10):
    """
    Produce Top-K recommendations for a given customer's candidate rows.
    1) compute p_reorder using the classifier (+ scaler) with its own feature list
    2) compute qty_p50/qty_p90 using the regression bundle with its feature list
    3) build margin_proxy and other fields
    4) if a trained ranker exists, score with it; else use heuristic
    """
    import joblib, os
    df = feat_current.copy()

    # --- 1) likelihood
    cls_bundle = joblib.load(os.path.join(MODEL_DIR, "cls_reorder.pkl"))
    cls_feats = cls_bundle["meta"]["features"]  # saved during training
    scaler = cls_bundle["scaler"]
    cls_model = cls_bundle["model"]
    Xc = scaler.transform(df[cls_feats].fillna(0.0).values)
    df["p_reorder"] = cls_model.predict_proba(Xc)[:, 1]

    # --- 2) quantity
    reg_bundle = joblib.load(os.path.join(MODEL_DIR, "reg_qty.pkl"))
    reg_feats = reg_bundle["features"]
    Xr = df[reg_feats].fillna(0.0).values
    df["qty_mean"] = reg_bundle["glm"].predict(Xr)
    df["qty_p50"]  = reg_bundle["gbr_p50"].predict(Xr)
    df["qty_p90"]  = reg_bundle["gbr_p90"].predict(Xr)

    # --- 3) helper/business features
    if "discount_pct" not in df.columns:
        df["discount_pct"] = 0.0
    df["margin_proxy"] = 1 - df["discount_pct"].fillna(0.0)
    for c in ["on_hand", "lead_time", "stockout_flag", "dow", "is_weekend"]:
        if c not in df.columns:
            df[c] = 0

    # --- 4) score & rank
    # default heuristic (works even if no ranker)
    df["score"] = (
        df["p_reorder"]
        * (0.7 * df["qty_p50"] + 0.3 * df["qty_p90"])
        * (0.5 + 0.5 * df["margin_proxy"])
        * (1.0 - 0.9 * df["stockout_flag"])
    )

    # if we trained a ranker, use it
    rank_path = os.path.join(MODEL_DIR, "ranker.pkl")
    if os.path.exists(rank_path):
        rb = joblib.load(rank_path)
        if rb.get("model") is not None:
            feats = rb["features"]  # ['p_reorder','qty_p50','qty_p90','margin_proxy','on_hand','lead_time','stockout_flag','dow','is_weekend']
            Xrank = df[feats].fillna(0.0).values
            df["score"] = rb["model"].predict(Xrank)

    top = df.sort_values("score", ascending=False).head(K).copy()
    top["suggested_qty"] = np.maximum(1, np.round(top["qty_p50"]).astype(int))
    return top[[
        "customer_id", "sku_id",
        "p_reorder", "qty_p50", "qty_p90",
        "score", "suggested_qty",
        "on_hand", "stockout_flag", "lead_time", "discount_pct"
    ]]
