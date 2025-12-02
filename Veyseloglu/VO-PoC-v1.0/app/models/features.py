import pandas as pd
import numpy as np

def build_snapshots_for_reorder(orders: pd.DataFrame, prices: pd.DataFrame, inventory: pd.DataFrame, calendar: pd.DataFrame, N_days: int=14):
    # Prepare target: time to next order per (customer, sku)
    df = orders.sort_values(["customer_id","sku_id","order_dt"])
    df["next_order_dt"] = df.groupby(["customer_id","sku_id"])["order_dt"].shift(-1)
    df["time_to_next"] = (df["next_order_dt"] - df["order_dt"]).dt.days
    df["label_reorder_in_N"] = (df["time_to_next"]<=N_days).astype(int).fillna(0)

    # Features near order time
    df["prev_order_dt"] = df.groupby(["customer_id","sku_id"])["order_dt"].shift(1)
    df["time_since_prev"] = (df["order_dt"] - df["prev_order_dt"]).dt.days
    df["time_since_prev"].fillna(df["time_since_prev"].median(), inplace=True)

    # Rolling features
    win = 30
    grp = df.groupby(["customer_id","sku_id"])
    df["qty_rolling_mean_30"] = grp["qty"].rolling(window=3, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    # Merge calendar (approximate by joining on order date)
    cal = calendar.rename(columns={"dt":"order_dt"})
    features = df.merge(cal, on="order_dt", how="left")

    # Latest price info per day/customer/sku
    p = prices.rename(columns={"dt":"order_dt"})
    p = p.sort_values("order_dt").drop_duplicates(["order_dt","customer_id","sku_id"], keep="last")
    features = features.merge(p[["order_dt","customer_id","sku_id","net_price","discount_pct"]], on=["order_dt","customer_id","sku_id"], how="left")
    features["discount_pct"].fillna(0.0, inplace=True)

    # Inventory snapshot join (approximate same day)
    inv = inventory.rename(columns={"dt":"order_dt"})
    features = features.merge(inv[["order_dt","customer_id","sku_id","on_hand","stockout_flag","lead_time"]], on=["order_dt","customer_id","sku_id"], how="left")
    for c in ["on_hand","stockout_flag","lead_time"]:
        features[c].fillna(features[c].median() if c!="stockout_flag" else 0, inplace=True)

    # Basic categorical encodings
    features["dow"] = features["order_dt"].dt.weekday
    features["is_weekend"] = features["dow"].isin([5,6]).astype(int)

    # Survival targets
    survival = features.dropna(subset=["time_to_next"]).copy()
    survival["event"] = 1  # since we have a next order for these rows
    # For the last observed orders with no next order, we can later append censored rows if needed

    # Classification dataset
    cls_cols = [
        "time_since_prev","qty_rolling_mean_30","discount_pct","on_hand","stockout_flag",
        "lead_time","dow","is_weekend","temp_c","holiday_flag"
    ]
    cls_target = "label_reorder_in_N"

    # Regression dataset for quantity: predict next qty using current features (only where next order exists)
    reg = features.dropna(subset=["next_order_dt"]).copy()
    reg["target_qty_next"] = reg.groupby(["customer_id","sku_id"])["qty"].shift(-1)
    reg = reg.dropna(subset=["target_qty_next"])

    reg_cols = cls_cols + ["qty"]
    reg_target = "target_qty_next"

    # Keep keys for inference
    key_cols = ["order_id","order_dt","customer_id","sku_id"]

    return {
        "features": features,
        "classification": (features[key_cols + cls_cols + [cls_target]].dropna(), cls_cols, cls_target),
        "regression": (reg[key_cols + reg_cols + [reg_target]].dropna(), reg_cols, reg_target),
        "survival": (survival[key_cols + cls_cols + ["time_to_next","event"]], cls_cols)
    }

def latest_snapshot_per_customer(orders: pd.DataFrame, prices: pd.DataFrame, inventory: pd.DataFrame, calendar: pd.DataFrame, as_of_date=None):
    # Build candidate feature rows for *current* recommendation
    if as_of_date is None:
        as_of_date = orders["order_dt"].max()
    # Take last observation per (customer, sku)
    df = orders[orders["order_dt"]<=as_of_date].sort_values(["customer_id","sku_id","order_dt"])
    last = df.groupby(["customer_id","sku_id"]).tail(1).copy()

    # Add feature columns similarly to training
    last["prev_order_dt"] = df.groupby(["customer_id","sku_id"])["order_dt"].shift(1)
    last["time_since_prev"] = (last["order_dt"] - last["prev_order_dt"]).dt.days
    last["time_since_prev"].fillna(last["time_since_prev"].median(), inplace=True)

    win = 30
    grp = df.groupby(["customer_id","sku_id"])
    rm = grp["qty"].rolling(window=3, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    last = last.merge(rm.rename("qty_rolling_mean_30"), left_index=True, right_index=True, how="left")

    cal = calendar.rename(columns={"dt":"order_dt"})
    last = last.merge(cal, on="order_dt", how="left")

    p = prices.rename(columns={"dt":"order_dt"}).sort_values("order_dt").drop_duplicates(["order_dt","customer_id","sku_id"], keep="last")
    last = last.merge(p[["order_dt","customer_id","sku_id","net_price","discount_pct"]], on=["order_dt","customer_id","sku_id"], how="left")
    last["discount_pct"].fillna(0.0, inplace=True)

    inv = inventory.rename(columns={"dt":"order_dt"})
    last = last.merge(inv[["order_dt","customer_id","sku_id","on_hand","stockout_flag","lead_time"]], on=["order_dt","customer_id","sku_id"], how="left")
    for c in ["on_hand","stockout_flag","lead_time"]:
        last[c].fillna(last[c].median() if c!="stockout_flag" else 0, inplace=True)

    last["dow"] = last["order_dt"].dt.weekday
    last["is_weekend"] = last["dow"].isin([5,6]).astype(int)

    cls_cols = [
        "time_since_prev","qty_rolling_mean_30","discount_pct","on_hand","stockout_flag",
        "lead_time","dow","is_weekend","temp_c","holiday_flag"
    ]
    return last, cls_cols
