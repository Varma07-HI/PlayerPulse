import json
import pandas as pd
from fairlearn.metrics import (MetricFrame, false_negative_rate,
                               false_positive_rate, selection_rate)
from sklearn.metrics import precision_score
from src.config import DASH_DIR, MODEL_DIR


def main():
    df = pd.read_parquet(DASH_DIR / 'test_predictions.parquet')
    thr = json.loads((MODEL_DIR / 'meta.json').read_text())['threshold']
    df['pred'] = (df.p >= thr).astype(int)
    df['activity'] = pd.qcut(df.matches_60d.rank(method='first'), 3,
                             labels=['low', 'mid', 'high']).astype(str)
    frames = []
    for col in ['cluster', 'skill', 'activity']:
        g = df[col].fillna(-1).astype(str).rename(col)
        mf = MetricFrame(
            metrics={'selection_rate': selection_rate,
                     'false_positive_rate': false_positive_rate,
                     'false_negative_rate': false_negative_rate,
                     'precision': lambda y, p: precision_score(y, p, zero_division=0)},
            y_true=df.churn, y_pred=df.pred, sensitive_features=g)
        t = mf.by_group.reset_index().rename(columns={col: 'group'})
        t.insert(0, 'slice', col)
        t['n'] = t['group'].map(g.value_counts())
        frames.append(t)
        gap = mf.by_group['false_negative_rate'].max() - mf.by_group['false_negative_rate'].min()
        print(f'{col}: false-negative-rate gap = {gap:.3f}')
    pd.concat(frames).to_csv(DASH_DIR / 'fairness.csv', index=False)


if __name__ == '__main__':
    main()