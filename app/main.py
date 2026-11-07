from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.explain import make_explainer, top_reasons

BUNDLE = joblib.load(Path(__file__).resolve().parent.parent / 'models' / 'bundle.joblib')
FEATURES = BUNDLE['features']
EXPLAINER = make_explainer(BUNDLE['model'])

app = FastAPI(title='PlayerPulse API', version='1.0')


class Player(BaseModel):
    matches_7d: float
    matches_14d: float
    matches_30d: float
    matches_60d: float
    days_since_last: float
    win_rate_7d: float | None = None
    win_rate_30d: float | None = None
    avg_kills_30d: float
    avg_deaths_30d: float
    avg_assists_30d: float
    avg_duration_30d: float
    kda_30d: float
    activity_trend: float
    account_age_days: float


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(p: Player):
    X = pd.DataFrame([p.model_dump()])[FEATURES].astype(float)
    raw = BUNDLE['model'].predict_proba(X)[:, 1]
    prob = float(BUNDLE['iso'].predict(raw)[0])
    return {
        'churn_probability': prob,
        'at_risk': prob >= BUNDLE['threshold'],
        'threshold': BUNDLE['threshold'],
        'top_reasons': top_reasons(EXPLAINER, X, FEATURES)[0],
    }
