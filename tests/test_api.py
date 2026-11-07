import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ACTIVE = {'matches_7d': 10, 'matches_14d': 18, 'matches_30d': 35, 'matches_60d': 70,
              'days_since_last': 0.5, 'win_rate_7d': 0.5, 'win_rate_30d': 0.5,
              'avg_kills_30d': 6, 'avg_deaths_30d': 6, 'avg_assists_30d': 9,
              'avg_duration_30d': 2400, 'kda_30d': 2.5, 'activity_trend': 0.5,
              'account_age_days': 120}
FADING = dict(ACTIVE, matches_7d=0, matches_14d=0, matches_30d=1, matches_60d=8,
              days_since_last=25, win_rate_7d=None, activity_trend=0.0)


def test_health():
    assert client.get('/health').json() == {'status': 'ok'}


def test_predict_shape():
    r = client.post('/predict', json=ACTIVE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body['churn_probability'] <= 1
    assert len(body['top_reasons']) == 3


def test_fading_player_is_riskier():
    a = client.post('/predict', json=ACTIVE).json()['churn_probability']
    f = client.post('/predict', json=FADING).json()['churn_probability']
    assert f > a


def test_bad_input_rejected():
    assert client.post('/predict', json={'matches_7d': 'abc'}).status_code == 422

from src.threshold import best_threshold


def test_threshold_separates_classes():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    t, net = best_threshold(y, p, cost=1, value=10, save_rate=0.5)
    assert 0.2 < t <= 0.8
    assert net > 0