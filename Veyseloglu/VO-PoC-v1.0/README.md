# AI Order Recommendation Demo (3-Component System)

This demo includes:
1) **Re-order Likelihood** within N days (classification + survival fallback)
2) **Next-order Quantity** (classical regression + optional Chronos-T5 forecasting)
3) **Targeted Recommendations** (ranking with business constraints)

It ships with a synthetic dataset generator and a tiny web UI.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Generate synthetic data
python scripts/generate_synthetic_data.py

# 2) Train all components
python scripts/train_all.py

# 3) Run the API
python scripts/serve.py  # serves at http://127.0.0.1:8000

# 4) Open the UI
# Open frontend/index.html in your browser (it talks to http://127.0.0.1:8000)
```

### Notes
- If LightGBM/XGBoost or HF models cannot be loaded, the code **falls back** to scikit-learn models.
- Survival model uses **lifelines** (Cox PH). If not available, we fall back to a calibrated classifier only.
- Chronos-T5 is optional. If it's not present or cannot be downloaded, we skip it.
