import sqlite3

import pandas as pd

from src.config import CUTOFF_STEP_DAYS, DAY, DB_PATH, HISTORY_DAYS, HORIZON_DAYS, RAW_DIR


def main():
    df = pd.read_parquet(RAW_DIR / 'matches.parquet')
    n0 = len(df)
    # cleaning
    df = df.dropna(subset=['account_id', 'match_id', 'start_time', 'duration'])
    df = df.drop_duplicates(['account_id', 'match_id'])
    df = df[(df.duration >= 300) & (df.duration <= 14400)]
    for c in ['kills', 'deaths', 'assists']:
        df[c] = df[c].fillna(0)
    print(f'rows before {n0}, after cleaning {len(df)}')

    with sqlite3.connect(DB_PATH) as con:
        df.to_sql('matches', con, if_exists='replace', index=False)
        con.execute('CREATE INDEX idx_time ON matches(start_time)')
        con.execute('CREATE INDEX idx_player ON matches(account_id, start_time)')
        t0 = int(df.start_time.min()) + HISTORY_DAYS * DAY
        t1 = int(df.start_time.max()) - HORIZON_DAYS * DAY
        cutoffs = pd.DataFrame({'cutoff': list(range(t0, t1, CUTOFF_STEP_DAYS * DAY))})
        cutoffs.to_sql('cutoffs', con, if_exists='replace', index=False)
        print('cutoffs:', len(cutoffs), 'players:', df.account_id.nunique())


if __name__ == '__main__':
    main()
