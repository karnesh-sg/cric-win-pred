from locust import HttpUser, between, task


class CricketAPIUser(HttpUser):
    wait_time = between(3, 5)

    @task(3)
    def predict_chase(self):
        self.client.post(
            "/predict/chase",
            json={
                "batting_team": "Chennai Super Kings",
                "bowling_team": "Mumbai Indians",
                "venue": "MA Chidambaram Stadium",
                "target": 180,
                "runs": 90,
                "wickets": 3,
                "balls_bowled": 60,
            },
            name="/predict/chase",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health")
