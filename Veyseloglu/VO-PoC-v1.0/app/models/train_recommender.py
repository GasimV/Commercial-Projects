import os, joblib, numpy as np, pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_store")
os.makedirs(MODEL_DIR, exist_ok=True)

# Optional ranking model
LGBM_AVAILABLE = True
try:
    import lightgbm as lgb
except Exception:
    LGBM_AVAILABLE = False

def build_training_frame(feat_current: pd.DataFrame, p_prob: np.ndarray, p_qty_p50: np.ndarray, p_qty_p90: np.ndarray):
    df = feat_current.copy()
    df["p_reorder"] = p_prob
    df["qty_p50"] = p_qty_p50
    df["qty_p90"] = p_qty_p90
    # simple margin proxy: 1 - discount
    df["margin_proxy"] = 1 - df["discount_pct"].fillna(0.0)
    # heuristic label: if last order qty>0 mark as relevant
    df["label"] = (df["qty"]>0).astype(int)
    # group by customer for ranking
    return df

def train_ranker(train_df: pd.DataFrame):
    features = ["p_reorder","qty_p50","qty_p90","margin_proxy","on_hand","lead_time","stockout_flag","dow","is_weekend"]
    if LGBM_AVAILABLE:
        X = train_df[features].values
        y = train_df["label"].values
        # group by customer
        groups = train_df.groupby("customer_id").size().values
        dtrain = lgb.Dataset(X, label=y, group=groups)
        params = {"objective":"lambdarank","metric":"ndcg","ndcg_eval_at":[5,10],"verbosity":-1}
        gbm = lgb.train(params, dtrain, num_boost_round=100)
        joblib.dump({"model": gbm, "features": features}, os.path.join(MODEL_DIR, "ranker.pkl"))
        return {"trained": True, "algo":"lightgbm_lambdarank", "features": features}
    else:
        # Fallback: save feature list; we'll use heuristic scoring at inference
        joblib.dump({"model": None, "features": features}, os.path.join(MODEL_DIR, "ranker.pkl"))
        return {"trained": False, "algo":"heuristic", "features": features}

def heuristic_score(row):
    return row["p_reorder"] * (0.7*row["qty_p50"] + 0.3*row["qty_p90"]) * (0.5 + 0.5*row["margin_proxy"]) * (1.0 if row["stockout_flag"]==0 else 0.1)
