import sqlite3

import pandas as pd

from src.config import DB_PATH, PROC_DIR

SQL = '''
SELECT m.account_id, c.cutoff,
  SUM(CASE WHEN m.start_time <  c.cutoff AND m.start_time >= c.cutoff -  7*86400 THEN 1 ELSE 0 END) AS matches_7d,
  SUM(CASE WHEN m.start_time <  c.cutoff AND m.start_time >= c.cutoff - 14*86400 THEN 1 ELSE 0 END) AS matches_14d,
  SUM(CASE WHEN m.start_time <  c.cutoff AND m.start_time >= c.cutoff - 30*86400 THEN 1 ELSE 0 END) AS matches_30d,
  SUM(CASE WHEN m.start_time <  c.cutoff AND m.start_time >= c.cutoff - 60*86400 THEN 1 ELSE 0 END) AS matches_60d,
  MAX(CASE WHEN m.start_time < c.cutoff THEN m.start_time END) AS last_match_time,
  AVG(CASE WHEN m.start_time < c.cutoff AND m.start_time >= c.cutoff -  7*86400 THEN m.win END) AS win_rate_7d,
  AVG(CASE WHEN m.start_time < c.cutoff AND m.start_time >= c.cutoff - 30*86400 THEN m.win END) AS win_rate_30d,
  AVG(CASE WHEN m.start_time < c.cutoff AND m.start_time >= c.cutoff - 30*86400 THEN m.kills END) AS avg_kills_30d,
  AVG(CASE WHEN m.start_time < c.cutoff AND m.start_time >= c.cutoff - 30*86400 THEN m.deaths END) AS avg_deaths_30d,
  AVG(CASE WHEN m.start_time < c.cutoff AND m.start_time >= c.cutoff - 30*86400 THEN m.assists END) AS avg_assists_30d,
  AVG(CASE WHEN m.start_time < c.cutoff AND m.start_time >= c.cutoff - 30*86400 THEN m.duration END) AS avg_duration_30d,
  SUM(CASE WHEN m.start_time >= c.cutoff AND m.start_time < c.cutoff + 30*86400 THEN 1 ELSE 0 END) AS matches_next_30d
FROM cutoffs c
JOIN matches m
  ON m.start_time >= c.cutoff - 60*86400 AND m.start_time < c.cutoff + 30*86400
GROUP BY m.account_id, c.cutoff
HAVING matches_30d >= 1
'''


def mode_by_player(con, col):
    t = pd.read_sql(
        f'SELECT account_id, {col} AS v, COUNT(*) AS n FROM matches '
        f'WHERE {col} IS NOT NULL GROUP BY account_id, {col}', con)
    t = t.sort_values('n', ascending=False).drop_duplicates('account_id')
    return t.set_index('account_id')['v'].rename(col)


def main():
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(SQL, con)
        first = pd.read_sql('SELECT account_id, MIN(start_time) AS first_time '
                            'FROM matches GROUP BY account_id', con)
        cluster = mode_by_player(con, 'cluster')
        skill = mode_by_player(con, 'skill')

    df = df.merge(first, on='account_id')
    # group columns used ONLY for the fairness audit, never as model inputs
    df = df.join(cluster, on='account_id').join(skill, on='account_id')

    df['churn'] = (df['matches_next_30d'] == 0).astype(int)
    df['days_since_last'] = (df['cutoff'] - df['last_match_time']) / 86400
    df['kda_30d'] = (df['avg_kills_30d'] + df['avg_assists_30d']) / df['avg_deaths_30d'].clip(lower=1)
    df['activity_trend'] = df['matches_14d'] / (df['matches_30d'] + 1)
    df['account_age_days'] = (df['cutoff'] - df['first_time']) / 86400
    df['cohort_month'] = pd.to_datetime(df['first_time'], unit='s').dt.to_period('M').astype(str)

    df.to_parquet(PROC_DIR / 'features.parquet')
    print('rows', len(df), 'players', df.account_id.nunique(),
          'churn rate', round(df.churn.mean(), 3))


if __name__ == '__main__':
    main()
