import time

import pandas as pd
import requests

from src.config import RAW_DIR

BASE = 'https://api.opendota.com/api'
FIELDS = ['match_id', 'player_slot', 'radiant_win', 'duration', 'game_mode',
          'lobby_type', 'hero_id', 'start_time', 'kills', 'deaths', 'assists',
          'skill', 'party_size', 'cluster', 'leaver_status']


def get(path, params=None):
    for attempt in range(5):
        try:
            r = requests.get(BASE + path, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(5)
            continue
        if r.status_code == 200:
            time.sleep(1.1)  # stay under 60 calls per minute
            return r.json()
        if r.status_code == 429:  # too many requests: wait longer each time
            time.sleep(30 * (attempt + 1))
            continue
        time.sleep(5)
    return None


def seed_players(target):
    ids = set()
    for m in (get('/publicMatches') or []):
        mid = m['match_id']
        detail = get(f'/matches/{mid}')
        if not detail:
            continue
        for p in detail.get('players', []):
            if p.get('account_id'):
                ids.add(p['account_id'])
        if len(ids) >= target:
            break
    return list(ids)


def main(n_players=400):
    players_dir = RAW_DIR / 'players'
    players_dir.mkdir(parents=True, exist_ok=True)
    ids_file = RAW_DIR / 'player_ids.txt'
    if ids_file.exists():
        ids = [int(x) for x in ids_file.read_text().split()]
    else:
        ids = seed_players(n_players)
        ids_file.write_text(' '.join(map(str, ids)))

    for i, aid in enumerate(ids):
        out = players_dir / f'{aid}.parquet'
        if out.exists():  # resumable
            continue
        data = get(f'/players/{aid}/matches', {'date': 365, 'project': FIELDS})
        df = pd.DataFrame(data or [])
        df['account_id'] = aid
        df.to_parquet(out)
        if i % 25 == 0:
            print(f'{i}/{len(ids)} players done')

    frames = [pd.read_parquet(f) for f in players_dir.glob('*.parquet')]
    allm = pd.concat([f for f in frames if len(f) > 0], ignore_index=True)
    allm['win'] = ((allm['player_slot'] < 128) == allm['radiant_win']).astype(int)
    allm.to_parquet(RAW_DIR / 'matches.parquet')
    print('saved', allm.shape, 'players:', allm['account_id'].nunique())


if __name__ == '__main__':
    main()