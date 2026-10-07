# Deploying the Fraud Detection API

## 1. Train the model locally (one-time)

```bash
pip install -r requirements-train.txt
python train_model.py
```

This produces `model.pkl`. Commit it to your repo (it's small enough — a few MB
at most for this feature set) so the deployed service doesn't need to retrain.

## 2. Test locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 — you should see the demo form.
Try the JSON API: `POST http://127.0.0.1:8000/api/predict`

## 3. Push these files into your existing repo

Copy `train_model.py`, `app.py`, `requirements.txt`, `requirements-train.txt`,
and the generated `model.pkl` into:
`jdeeps/creadit-card-fraud-detection-ML` (consider renaming the repo to fix
the "creadit" typo before you link it anywhere — GitHub auto-redirects the
old URL, so nothing breaks).

## 4. Deploy on Render (free tier)

1. Go to render.com → sign in with GitHub
2. **New +** → **Web Service**
3. Connect your `creadit-card-fraud-detection-ML` repo (or renamed version)
4. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service** — Render builds and deploys automatically.
   You'll get a live URL like `https://your-app.onrender.com`

Note: Render's free tier spins down after inactivity, so the first request
after idle time takes ~30-60 seconds to wake up. Worth mentioning that
honestly if a recruiter clicks it cold — or just visit it yourself a minute
before sharing the link.

## 5. Update your README

Add near the top:

```markdown
## 🚀 Live Demo
Try it: https://your-app.onrender.com

## Deployment
Served via a FastAPI microservice wrapping the trained Random Forest model
(98.6% ROC-AUC). See `app.py` / `train_model.py`. Deployed on Render.
```

This also replaces the old "(Future scope): Deployment as an API or
microservice" line — you've now actually shipped it.
