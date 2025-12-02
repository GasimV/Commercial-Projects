import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE), "data")

def load_all():
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"), parse_dates=["order_dt"])
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    skus = pd.read_csv(os.path.join(DATA_DIR, "skus.csv"))
    inventory = pd.read_csv(os.path.join(DATA_DIR, "inventory.csv"), parse_dates=["dt","last_delivery_dt"])
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"), parse_dates=["dt"])
    calendar = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"), parse_dates=["dt"])
    campaigns = pd.read_csv(os.path.join(DATA_DIR, "campaigns.csv"), parse_dates=["start_dt","end_dt"])
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"), parse_dates=["ts"])
    return {
        "orders": orders,
        "customers": customers,
        "skus": skus,
        "inventory": inventory,
        "prices": prices,
        "calendar": calendar,
        "campaigns": campaigns,
        "interactions": interactions
    }
