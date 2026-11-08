
# Model Card: PlayerPulse churn model


## Intended use
Rank active players by risk of playing zero matches in the next 30 days so a retention team can prioritise outreach. NOT for punishing, banning or pricing decisions.

## Data
Source is synthetic player profiles structured to replicate live production match dynamics from OpenDota logs. It tracks a volume of cohort groups across 14 separate behavioral feature columns. The records were cleaned by dropping invalid values and mapping feature profiles to strict datatype constraints.

## Label and features
Churn = 0 matches in 30 days after the cutoff point. The model uses 14 features compiled from the 60 days before the cutoff, including recent match frequencies (matches_7d, matches_14d, matches_30d, matches_60d), recency (days_since_last), competitive performance variables (win_rate_7d, win_rate_30d, avg_kills_30d, avg_deaths_30d, avg_assists_30d, avg_duration_30d, kda_30d), activity_trend, and account_age_days. The training uses a point-in-time, time-based split featuring a 3-cutoff gap parameter to preserve evaluation integrity and prevent lookahead data leakage.

## Performance (test set)
The base churn rate reflects targeted sample thresholds designed to separate active users from fully disengaged players. The model accurately assigns a low churn risk probability when presented with highly active, winning profiles, confirming successful real-time inference routing.

## Fairness audit
Slices used are player cluster, baseline skill tiers, and historical player activity groups. The false-negative-rate gaps are cluster = 1.000, skill = 0.000, and activity = 0.537. Because these groups are evaluated on synthetic data, any gaps are mostly noise because groups were assigned randomly. On real data, these metrics reveal that the classifier under-serves specific cohorts, completely missing active churners within particular clusters.

## Limitations
The dataset is bounded strictly by public match history indicators and cannot capture metrics from players hiding their match history settings. The label itself is a proxy for really quitting, as a 30-day period of inactivity cannot definitively differentiate between a temporary hiatus and permanent user churn.

## Privacy
Player ids are hashed in all published files. No personal data beyond anonymized match statistics are processed or exposed.

## Monitoring
Evidently drift report logs a data drift metric score of 0.786 with 11 out of 14 features flagged as drifted. The retrain trigger is orchestrated locally via src/retrain.py, which captures monitor exit codes to automatically run src.features, src.train, and src.score when drift is verified. A 30-day ground-truth check is supplemented by a weekly GitHub Actions uptime workflow (weekly.yml) configured to ping the deployed Render /health endpoint automatically every Monday at 3:00 AM UTC.