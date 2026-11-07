from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
PROC_DIR = ROOT / 'data' / 'processed'
MODEL_DIR = ROOT / 'models'
DASH_DIR = ROOT / 'dashboard_data'
DB_PATH = str(ROOT / 'data' / 'playerpulse.db')

DAY = 86400                 # seconds in a day
HISTORY_DAYS = 60           # how far back features look
HORIZON_DAYS = 30           # churn = no matches in the next 30 days
CUTOFF_STEP_DAYS = 7        # one snapshot per player per week
END_TIME_SYNTH = 1790000000 # fixed 'now' for synthetic data

FEATURES = [
    'matches_7d', 'matches_14d', 'matches_30d', 'matches_60d',
    'days_since_last', 'win_rate_7d', 'win_rate_30d',
    'avg_kills_30d', 'avg_deaths_30d', 'avg_assists_30d',
    'avg_duration_30d', 'kda_30d', 'activity_trend', 'account_age_days',
]

for d in (RAW_DIR, PROC_DIR, MODEL_DIR, DASH_DIR):
    d.mkdir(parents=True, exist_ok=True)
