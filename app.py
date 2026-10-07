"""
app.py
FastAPI service wrapping the credit card fraud detection model.

Run locally:
    uvicorn app:app --reload

Deploy (e.g. Render):
    start command -> uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import joblib
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MODEL_PATH = "model.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
CATEGORIES = bundle["categories"]
CATEGORY_LABELS = [c.replace("category_", "").replace("_", " ").title() for c in CATEGORIES]

app = FastAPI(title="Credit Card Fraud Detection API")


class Transaction(BaseModel):
    amt: float
    zip: int
    lat: float
    long: float
    city_pop: int
    merch_lat: float
    merch_long: float
    age: int
    hour: int
    day: int
    month: int
    category: str  # one of CATEGORIES' human label, e.g. "Grocery Pos"


def build_feature_row(amt, zip_, lat, long_, city_pop, merch_lat, merch_long,
                       age, hour, day, month, category_key: str) -> pd.DataFrame:
    row = {col: 0 for col in FEATURES}
    row.update({
        "amt": amt, "zip": zip_, "lat": lat, "long": long_, "city_pop": city_pop,
        "merch_lat": merch_lat, "merch_long": merch_long, "age": age,
        "hour": hour, "day": day, "month": month,
    })
    if category_key in CATEGORIES:
        row[category_key] = 1
    return pd.DataFrame([row], columns=FEATURES)


def predict(amt, zip_, lat, long_, city_pop, merch_lat, merch_long, age, hour, day, month, category_key):
    X = build_feature_row(amt, zip_, lat, long_, city_pop, merch_lat, merch_long,
                           age, hour, day, month, category_key)
    proba = float(model.predict_proba(X)[0][1])
    return proba


@app.post("/api/predict")
def api_predict(txn: Transaction):
    category_key = "category_" + txn.category.lower().replace(" ", "_")
    proba = predict(txn.amt, txn.zip, txn.lat, txn.long, txn.city_pop,
                     txn.merch_lat, txn.merch_long, txn.age, txn.hour,
                     txn.day, txn.month, category_key)
    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": proba >= 0.5,
    }


def category_options(selected=None):
    opts = []
    for label, key in zip(CATEGORY_LABELS, CATEGORIES):
        sel = " selected" if key == selected else ""
        opts.append(f'<option value="{key}"{sel}>{label}</option>')
    return "\n".join(opts)


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Credit Card Fraud Detection Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 640px;
            margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
    h1 {{ font-size: 1.4rem; }}
    p.sub {{ color: #666; margin-top: -8px; }}
    form {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 24px; }}
    label {{ font-size: 0.85rem; color: #333; display: block; margin-bottom: 4px; }}
    input, select {{ width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 6px; }}
    .full {{ grid-column: 1 / -1; }}
    button {{ grid-column: 1 / -1; padding: 12px; background: #1a1a1a; color: white;
              border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; margin-top: 8px; }}
    .result {{ margin-top: 24px; padding: 16px; border-radius: 8px; font-size: 1.1rem; }}
    .fraud {{ background: #fde8e8; color: #a11; border: 1px solid #f5b5b5; }}
    .safe {{ background: #e8f8ee; color: #176b3a; border: 1px solid #a9e0bd; }}
  </style>
</head>
<body>
  <h1>Credit Card Fraud Detection</h1>
  <p class="sub">Random Forest model (98.6% ROC-AUC) &mdash; trained on 300K+ transactions with SMOTE-balanced classes.</p>
  {result_html}
  <form method="post" action="/">
    <div><label>Amount ($)</label><input name="amt" type="number" step="0.01" value="{amt}" required></div>
    <div><label>Category</label><select name="category">{category_opts}</select></div>
    <div><label>Cardholder Lat</label><input name="lat" type="number" step="0.0001" value="{lat}" required></div>
    <div><label>Cardholder Long</label><input name="long" type="number" step="0.0001" value="{long}" required></div>
    <div><label>Merchant Lat</label><input name="merch_lat" type="number" step="0.0001" value="{merch_lat}" required></div>
    <div><label>Merchant Long</label><input name="merch_long" type="number" step="0.0001" value="{merch_long}" required></div>
    <div><label>ZIP Code</label><input name="zip" type="number" value="{zip}" required></div>
    <div><label>City Population</label><input name="city_pop" type="number" value="{city_pop}" required></div>
    <div><label>Cardholder Age</label><input name="age" type="number" value="{age}" required></div>
    <div><label>Hour (0-23)</label><input name="hour" type="number" min="0" max="23" value="{hour}" required></div>
    <div><label>Day of Week (1-7)</label><input name="day" type="number" min="1" max="7" value="{day}" required></div>
    <div><label>Month (1-12)</label><input name="month" type="number" min="1" max="12" value="{month}" required></div>
    <button type="submit">Check Transaction</button>
  </form>
  <p class="sub" style="margin-top:24px;">JSON API: <code>POST /api/predict</code></p>
</body>
</html>
"""

DEFAULTS = dict(amt=120.50, lat=37.77, long=-122.42, merch_lat=37.80, merch_long=-122.27,
                 zip=94105, city_pop=50000, age=40, hour=14, day=3, month=6)


@app.get("/", response_class=HTMLResponse)
def form_get():
    return PAGE_TEMPLATE.format(result_html="", category_opts=category_options(), **DEFAULTS)


@app.post("/", response_class=HTMLResponse)
def form_post(
    amt: float = Form(...), zip: int = Form(...), lat: float = Form(...), long: float = Form(...),
    city_pop: int = Form(...), merch_lat: float = Form(...), merch_long: float = Form(...),
    age: int = Form(...), hour: int = Form(...), day: int = Form(...), month: int = Form(...),
    category: str = Form(...),
):
    proba = predict(amt, zip, lat, long, city_pop, merch_lat, merch_long, age, hour, day, month, category)
    css_class = "fraud" if proba >= 0.5 else "safe"
    verdict = "LIKELY FRAUD" if proba >= 0.5 else "Looks legitimate"
    result_html = f'<div class="result {css_class}"><strong>{verdict}</strong><br>Fraud probability: {proba:.1%}</div>'
    values = dict(amt=amt, zip=zip, lat=lat, long=long, city_pop=city_pop, merch_lat=merch_lat,
                  merch_long=merch_long, age=age, hour=hour, day=day, month=month)
    return PAGE_TEMPLATE.format(result_html=result_html, category_opts=category_options(category), **values)
