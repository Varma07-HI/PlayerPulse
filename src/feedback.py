import sqlite3
import pandas as pd
from sklearn.metrics import average_precision_score
from src.config import DAY, DB_PATH


def live_performance():
    with sqlite3.connect(DB_PATH) as con:
        s = pd.read_sql('SELECT account_id, cutoff, churn_prob FROM scores', con)
        t_max = con.execute('SELECT MAX(start_time) FROM matches').fetchone()[0]
        played = pd.read_sql('SELECT account_id, start_time FROM matches', con)
    s = s[s.cutoff + 30 * DAY <= t_max].copy()     # only scores whose 30 days are over
    if s.empty:
        print('no scores are old enough yet')
        return None
    m = s.merge(played, on='account_id')
    m = m[(m.start_time >= m.cutoff) & (m.start_time < m.cutoff + 30 * DAY)]
    still_playing = set(zip(m.account_id, m.cutoff))
    s['churn'] = [0 if (a, c) in still_playing else 1 for a, c in zip(s.account_id, s.cutoff)]
    pr = average_precision_score(s.churn, s.churn_prob)
    print('live PR-AUC', round(pr, 4), 'on', len(s), 'scored players, churn rate', round(s.churn.mean(), 3))
    return pr


if __name__ == '__main__':
    live_performance()