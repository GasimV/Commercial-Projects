from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json

from app.utils.data_loader import load_all
from app.models.features import build_snapshots_for_reorder, latest_snapshot_per_customer
from app.models.train_likelihood import train_classification, train_survival
from app.models.train_quantity import train_regression, load_chronos, chronos_forecast
from app.models.train_recommender import build_training_frame, train_ranker
from app.models.inference import predict_likelihood, predict_quantity, recommend

app = FastAPI(title="AI Order Recommendation Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    N_days: int = 14

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/train")
def train(req: TrainRequest):
    data = load_all()
    orders = data["orders"]
    prices = data["prices"]
    inventory = data["inventory"]
    calendar = data["calendar"]

    snaps = build_snapshots_for_reorder(orders, prices, inventory, calendar, N_days=req.N_days)

    # 1) Classification
    df_cls, cls_cols, cls_target = snaps["classification"]
    meta_cls = train_classification(df_cls, cls_cols, cls_target)

    # 1b) Survival
    df_surv, s_cols = snaps["survival"]
    meta_surv = train_survival(df_surv, s_cols)

    # 2) Quantity regression
    df_reg, reg_cols, reg_target = snaps["regression"]
    meta_reg = train_regression(df_reg, reg_cols, reg_target)

    # 3) Ranker (uses current latest snapshot as pseudo training; for demo purposes)
    last, feature_cols = latest_snapshot_per_customer(orders, prices, inventory, calendar)
    # Use classifier & regressors to create features for ranker
    # Use same columns for simplicity
    from app.models.train_likelihood import predict_proba as cls_proba
    import joblib, os
    import numpy as np
    from app.models.train_quantity import load_chronos
    # predict likelihood
    import joblib
    bundle = joblib.load(os.path.join("app","models_store","cls_reorder.pkl"))
    scaler = bundle["scaler"]; model = bundle["model"]
    Xc = scaler.transform(last[cls_cols].fillna(0.0).values)
    p = model.predict_proba(Xc)[:,1]
    # predict quantities
    regb = joblib.load(os.path.join("app","models_store","reg_qty.pkl"))
    glm = regb["glm"]; p50 = regb["gbr_p50"]; p90 = regb["gbr_p90"]
    Xr = last[reg_cols].fillna(0.0).values
    qmean = glm.predict(Xr); q50 = p50.predict(Xr); q90 = p90.predict(Xr)
    train_df = build_training_frame(last, p, q50, q90)
    meta_rank = train_ranker(train_df)

    return {"classification": meta_cls, "survival": meta_surv, "regression": meta_reg, "ranker": meta_rank}

@app.get("/likelihood")
def likelihood(customer_id: int = Query(...), N_days: int = Query(14)):
    data = load_all()
    last, cls_cols = latest_snapshot_per_customer(data["orders"], data["prices"], data["inventory"], data["calendar"])
    cand = last[last["customer_id"]==customer_id].copy()
    from app.models.train_likelihood import predict_proba
    import joblib, os
    bundle = joblib.load(os.path.join("app","models_store","cls_reorder.pkl"))
    scaler = bundle["scaler"]; model = bundle["model"]
    X = scaler.transform(cand[cls_cols].fillna(0.0).values)
    prob = model.predict_proba(X)[:,1]
    cand["prob_reorder_N"] = prob
    return json.loads(cand[["customer_id","sku_id","prob_reorder_N"]].to_json(orient="records"))

@app.get("/quantity")
def quantity(customer_id: int = Query(...)):
    data = load_all()
    last, cols = latest_snapshot_per_customer(data["orders"], data["prices"], data["inventory"], data["calendar"])
    cand = last[last["customer_id"]==customer_id].copy()
    import joblib, os
    regb = joblib.load(os.path.join("app","models_store","reg_qty.pkl"))
    glm = regb["glm"]; p50 = regb["gbr_p50"]; p90 = regb["gbr_p90"]
    X = cand[cols + ["qty"]].fillna(0.0).values if "qty" in cand.columns else cand[cols].fillna(0.0).values
    cand["qty_mean"] = glm.predict(X)
    cand["qty_p50"] = p50.predict(X)
    cand["qty_p90"] = p90.predict(X)
    return json.loads(cand[["customer_id","sku_id","qty_mean","qty_p50","qty_p90"]].to_json(orient="records"))

@app.get("/recommend")
def recommend_endpoint(customer_id: int = Query(...), k: int = Query(10)):
    data = load_all()
    last, _ = latest_snapshot_per_customer(data["orders"], data["prices"], data["inventory"], data["calendar"])
    cand = last[last["customer_id"]==customer_id].copy()
    from app.models.inference import recommend as rec
    top = rec(cand, K=k)
    return json.loads(top.to_json(orient="records"))
