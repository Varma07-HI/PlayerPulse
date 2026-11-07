from locust import HttpUser, between, task

SAMPLE = {'matches_7d': 3, 'matches_14d': 6, 'matches_30d': 12, 'matches_60d': 25,
              'days_since_last': 2, 'win_rate_7d': 0.5, 'win_rate_30d': 0.5,
              'avg_kills_30d': 6, 'avg_deaths_30d': 6, 'avg_assists_30d': 9,
              'avg_duration_30d': 2400, 'kda_30d': 2.5, 'activity_trend': 0.5,
              'account_age_days': 100}


class Scorer(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post('/predict', json=SAMPLE)