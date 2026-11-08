import json
import sys

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import DASH_DIR, FEATURES, PROC_DIR
from src.splits import time_split


def main():
    df = pd.read_parquet(PROC_DIR / 'features.parquet')
    train, _, test = time_split(df)
    ref = train[FEATURES].sample(min(5000, len(train)), random_state=0)
    cur = test[FEATURES].sample(min(5000, len(test)), random_state=0)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(str(DASH_DIR / 'drift_report.html'))
    res = report.as_dict()['metrics'][0]['result']

    summary = {
        'dataset_drift': bool(res['dataset_drift']),
        'share_drifted': float(res['share_of_drifted_columns']),
        'n_drifted': int(res['number_of_drifted_columns']),
        # simple data-quality checks
        'max_null_rate': float(cur.isna().mean().max()),
        'negative_counts': int((cur[['matches_7d', 'matches_14d', 'matches_30d']] < 0).sum().sum()),
    }
    (DASH_DIR / 'drift_summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)
    sys.exit(1 if summary['dataset_drift'] else 0)   # exit code 1 means: drift found


if __name__ == '__main__':
    main()