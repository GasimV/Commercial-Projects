import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

N_CUSTOMERS = 80
N_SKUS = 150
DAYS = 120  # simulate ~4 months
START_DATE = datetime(2025, 1, 1)

# Customers
customers = []
for cid in range(1, N_CUSTOMERS+1):
    seg = np.random.choice(["small_shop","mini_market","cafe","bar","restaurant"], p=[0.35,0.35,0.1,0.1,0.1])
    ch = np.random.choice(["retail","horeca"], p=[0.6,0.4])
    sub = np.random.choice(["direct","distributor"], p=[0.7,0.3])
    lat = 40.3 + np.random.rand()*0.7  # around Azerbaijan for fun
    lon = 49.5 + np.random.rand()*1.2
    customers.append([cid, f"Store #{cid}", f"Addr {cid}", lat, lon, seg, ch, sub, np.random.choice(["R1","R2","R3","R4"])])
customers_df = pd.DataFrame(customers, columns=[
    "customer_id","name","address","gps_lat","gps_lon","segment","channel","sub_channel","region"
])

# SKUs
skus = []
cats = ["cola","juice","water","snack","energy","iced_tea","chocolate"]
brands = ["BrandA","BrandB","BrandC","BrandD"]
for sid in range(1, N_SKUS+1):
    cat = np.random.choice(cats)
    brand = np.random.choice(brands)
    pack = np.random.choice(["330ml","500ml","1L","1.5L","6x330ml","6x500ml"])
    shelf = np.random.randint(60, 365)
    subgrp = f"{cat}_group"
    base_price = np.round(np.random.uniform(0.3, 2.5), 2)
    skus.append([sid, brand, cat, pack, shelf, subgrp, base_price])
skus_df = pd.DataFrame(skus, columns=["sku_id","brand","category","pack","shelf_life","substitution_group","base_price"])

# Calendar (external)
calendar = []
for d in range(DAYS):
    dt = START_DATE + timedelta(days=d)
    dow = dt.weekday()
    holiday = 1 if (dow==6 and np.random.rand()<0.2) else 0
    season = "winter" if dt.month in [1,2,12] else ("spring" if dt.month in [3,4,5] else ("summer" if dt.month in [6,7,8] else "fall"))
    calendar.append([dt.date().isoformat(), dow, holiday, season, np.random.normal(20, 8)])
calendar_df = pd.DataFrame(calendar, columns=["dt","dow","holiday_flag","season","temp_c"])

# Demand rates per (customer, sku)
rates = {}
for cid in customers_df["customer_id"]:
    for sid in skus_df["sku_id"]:
        base = np.random.gamma(0.5, 0.8)  # many near-zero
        # lift for cola & water in summer
        seasonal = 1.5 if (skus_df.loc[skus_df["sku_id"]==sid,"category"].iloc[0] in ["cola","water"]) else 1.0
        rates[(cid, sid)] = base * seasonal

orders = []
interactions = []
prices = []
inventory = []

for d in range(DAYS):
    dt = START_DATE + timedelta(days=d)
    # dynamic promo day flag per SKU
    promo_skus = set(np.random.choice(skus_df["sku_id"], size=np.random.randint(5, 20), replace=False))
    for _, sku in skus_df.iterrows():
        sku_id = int(sku["sku_id"])
        for _, cust in customers_df.iterrows():
            cid = int(cust["customer_id"])
            lam = rates[(cid, sku_id)]
            # seasonality simple: higher in summer months
            if dt.month in [6,7,8]:
                lam *= 1.2
            if dt.weekday() in [4,5]:  # Fri/Sat
                lam *= 1.1
            if sku_id in promo_skus:
                lam *= 1.3

            qty = np.random.poisson(lam)
            if qty>0:
                unit_price = float(sku["base_price"]) * (0.9 if sku_id in promo_skus else 1.0)
                orders.append([f"O{d}-{cid}-{sku_id}", dt.date().isoformat(), cid, sku_id, qty, unit_price, int(sku_id in promo_skus), np.random.choice(["retail","horeca"]), np.random.choice(["direct","distributor"])])
                # simple interactions
                interactions.append([cid, f"sess-{cid}-{d}", f"{dt.isoformat()}T10:00:00", "reco_accept" if np.random.rand()<0.2 else "view", sku_id])

            # inventory snapshot (toy: enough stock but with some random stockouts)
            on_hand = np.random.randint(0, 120)
            stockout = 1 if on_hand==0 else 0
            inventory.append([dt.date().isoformat(), cid, sku_id, on_hand, np.random.randint(0,30), stockout, (dt - timedelta(days=np.random.randint(1,10))).date().isoformat(), np.random.randint(1,5)])

            # price record
            list_price = float(sku["base_price"]) * 1.05
            net_price = float(sku["base_price"]) * (0.9 if sku_id in promo_skus else 1.0)
            discount_pct = 1 - (net_price / list_price)
            prices.append([dt.date().isoformat(), sku_id, cid, list_price, net_price, discount_pct])

# Campaigns (toy)
campaigns = []
for i in range(10):
    start = START_DATE + timedelta(days=int(np.random.randint(0, DAYS-15)))
    end = start + timedelta(days=int(np.random.randint(5, 15)))
    campaigns.append([f"CMP{i}", start.date().isoformat(), end.date().isoformat(), np.random.choice(["retail","horeca"]), np.random.choice(["direct","distributor"]), round(np.random.uniform(1000,5000),2), round(np.random.uniform(0.05,0.2),2), "in_stock & margin>0.1"])

# Save
orders_df = pd.DataFrame(orders, columns=["order_id","order_dt","customer_id","sku_id","qty","unit_price","promo_flag","channel","sub_channel"])
orders_df.to_csv(os.path.join(DATA_DIR, "orders.csv"), index=False)

customers_df.to_csv(os.path.join(DATA_DIR, "customers.csv"), index=False)
skus_df.to_csv(os.path.join(DATA_DIR, "skus.csv"), index=False)

pd.DataFrame(inventory, columns=["dt","customer_id","sku_id","on_hand","on_order","stockout_flag","last_delivery_dt","lead_time"]).to_csv(os.path.join(DATA_DIR, "inventory.csv"), index=False)
pd.DataFrame(prices, columns=["dt","sku_id","customer_id","list_price","net_price","discount_pct"]).to_csv(os.path.join(DATA_DIR, "prices.csv"), index=False)
calendar_df.to_csv(os.path.join(DATA_DIR, "calendar.csv"), index=False)
pd.DataFrame(campaigns, columns=["campaign_id","start_dt","end_dt","channel","sub_channel","budget","discount","eligibility"]).to_csv(os.path.join(DATA_DIR, "campaigns.csv"), index=False)
pd.DataFrame(interactions, columns=["customer_id","session_id","ts","event_type","sku_id"]).to_csv(os.path.join(DATA_DIR, "interactions.csv"), index=False)

print("Synthetic data generated in data/")
