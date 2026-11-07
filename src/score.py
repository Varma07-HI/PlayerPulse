import hashlib
import sqlite3
import joblib
import pandas as pd
from src.config import DASH_DIR, DAY, DB_PATH, MODEL_DIR, PROC_DIR
from src.explain import make_explainer, top_reasons


def hash_id(x):
    return hashlib.sha256(str(x).encode()).hexdigest()[:10]


def retention(con):
    m = pd.read_sql('SELECT account_id, start_time FROM matches', con)
    first = m.groupby('account_id').start_time.transform('min')
    m['month_idx'] = ((m.start_time - first) // (30 * DAY)).astype(int)
    m['cohort'] = pd.to_datetime(first, unit='s').dt.to_period('M').astype(str)
    r = m.groupby(['cohort', 'month_idx']).account_id.nunique().reset_index(name='active')
    size = r[r.month_idx == 0].set_index('cohort').active
    r['retention'] = r.active / r.cohort.map(size)
    return r[r.month_idx <= 6]


def score_cutoff(b, explainer, df, cutoff):
    d = df[df.cutoff == cutoff].copy()
    X = d[b['features']]
    d['churn_prob'] = b['iso'].predict(b['model'].predict_proba(X)[:, 1])
    d['at_risk'] = d.churn_prob >= b['threshold']
    rs = top_reasons(explainer, X, b['features'])
    d['reasons'] = [', '.join(r['feature'] + (' (+)' if r['impact'] > 0 else ' (-)') for r in x)
                    for x in rs]
    return d


def main():
    b = joblib.load(MODEL_DIR / 'bundle.joblib')
    df = pd.read_parquet(PROC_DIR / 'features.parquet')
    ex = make_explainer(b['model'])
    cuts = sorted(df.cutoff.unique())
    latest = score_cutoff(b, ex, df, cuts[-1])
    older = score_cutoff(b, ex, df, cuts[-6])   # old enough to have real outcomes

    cols = ['account_id', 'cutoff', 'churn_prob', 'at_risk', 'reasons']
    with sqlite3.connect(DB_PATH) as con:
        pd.concat([older, latest])[cols].to_sql('scores', con, if_exists='replace', index=False)
        retention(con).to_parquet(DASH_DIR / 'retention.parquet')

    latest['player_id'] = latest.account_id.map(hash_id)
    keep = ['player_id', 'cutoff', 'churn_prob', 'at_risk', 'reasons', 'cohort_month',
            'cluster', 'skill', 'matches_30d', 'days_since_last']
    latest[keep].to_parquet(DASH_DIR / 'scores.parquet')

    tp = pd.read_parquet(DASH_DIR / 'test_predictions.parquet')
    if pd.api.types.is_numeric_dtype(tp.account_id):   # hash once only
        tp['account_id'] = tp.account_id.map(hash_id)
        tp.to_parquet(DASH_DIR / 'test_predictions.parquet')
    print('scored', len(latest), 'players;', int(latest.at_risk.sum()), 'at risk')


if __name__ == '__main__':
    main()