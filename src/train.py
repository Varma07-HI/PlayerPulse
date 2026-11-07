import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import DASH_DIR, FEATURES, MODEL_DIR, PROC_DIR
from src.splits import time_split
from src.threshold import best_threshold


def metrics(y, p):
    return {'pr_auc': average_precision_score(y, p),
            'roc_auc': roc_auc_score(y, p),
            'brier': brier_score_loss(y, p)}


def main():
    df = pd.read_parquet(PROC_DIR / 'features.parquet')
    train, valid, test = time_split(df)
    Xtr, ytr = train[FEATURES], train['churn']
    Xva, yva = valid[FEATURES], valid['churn']
    Xte, yte = test[FEATURES], test['churn']
    print('churn rate  train/valid/test:', round(ytr.mean(), 3),
          round(yva.mean(), 3), round(yte.mean(), 3))

    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    xgb_params = {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
                      'subsample': 0.8, 'colsample_bytree': 0.8, 'eval_metric': 'aucpr',
                      'random_state': 42}
    models = {
        'baseline_prior': DummyClassifier(strategy='prior'),
        'logreg': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                LogisticRegression(max_iter=1000)),
        'xgb_plain': XGBClassifier(**xgb_params),
        'xgb_weighted': XGBClassifier(scale_pos_weight=spw, **xgb_params),
    }

    mlflow.set_experiment('playerpulse')
    scores = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(Xtr, ytr)
            m = metrics(yva, model.predict_proba(Xva)[:, 1])
            mlflow.log_param('model', name)
            mlflow.log_metrics({f'valid_{k}': v for k, v in m.items()})
            scores[name] = m
            print(f'{name:15s}', {k: round(v, 4) for k, v in m.items()})

    best = max(['xgb_plain', 'xgb_weighted'], key=lambda n: scores[n]['pr_auc'])
    model = models[best]
    print('chosen model:', best)

    # calibration: map raw scores to honest probabilities using the validation set
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(model.predict_proba(Xva)[:, 1], yva)
    p_va = iso.predict(model.predict_proba(Xva)[:, 1])
    p_te = iso.predict(model.predict_proba(Xte)[:, 1])

    thr, _ = best_threshold(yva.values, p_va)            # chosen on validation
    _, _net_te = best_threshold(yte.values, np.array([thr] * 0 + list(p_te)))  # for reference
    flagged = p_te >= thr
    test_m = metrics(yte, p_te)
    test_m.update({'threshold': thr,
                   'test_churn_rate': float(yte.mean()),
                   'precision_at_thr': float(yte[flagged].mean()) if flagged.any() else 0.0,
                   'recall_at_thr': float(flagged[yte.values == 1].mean())})
    print('TEST:', {k: round(v, 4) for k, v in test_m.items()})

    with mlflow.start_run(run_name='final_' + best):
        mlflow.log_param('model', best)
        mlflow.log_metrics({f'test_{k}': v for k, v in test_m.items()})

    joblib.dump({'model': model, 'iso': iso, 'features': FEATURES, 'threshold': thr},
                MODEL_DIR / 'bundle.joblib')
    meta = {'chosen_model': best, 'threshold': thr, 'features': FEATURES,
            'valid_scores': scores, 'test_metrics': test_m}
    (MODEL_DIR / 'meta.json').write_text(json.dumps(meta, indent=2, default=float))

    out = test[['account_id', 'cutoff', 'churn', 'cluster', 'skill', 'matches_60d']].copy()
    out['p'] = p_te
    out.to_parquet(DASH_DIR / 'test_predictions.parquet')
    print('saved models/bundle.joblib and dashboard_data/test_predictions.parquet')


if __name__ == '__main__':
    main()