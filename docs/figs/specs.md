# Cricket Win Probability - Spec (v1)
## 1. Goal
A web app that shows, ball by ball, the probability that each team wins a men's T20 match and,
in the first innings, a projected final-score range.
## 2. Who it is for
Cricket fans and analysts who want a probability they can trust; recruiters who want to see a
complete ML project (data -> model -> API -> live app).
## 3. Scope
IN: men's T20 (IPL + T20 internationals); chase win probability; first-innings projected score
 (10th/50th/90th percentile) and batting-first win probability; REST API; dashboard with
 live-scenario mode and historical replay mode.
OUT: ODI/Test, player-level models, toss/dew/pitch effects, live scoreboard feeds.
## 4. Data
Cricsheet ball-by-ball JSON. Exclude: no result, tie/super over, D/L-affected matches.
## 5. Prediction tasks (one row = state after each legal ball)
A) Chase: label = chasing team won.
B) First-innings score: label = final innings total (quantile models).
C) Batting-first win: label = batting-first team won.
## 6. Features (only information available at that ball)
Match state (runs, wickets, balls left, run rates), momentum (last 12 balls, partnership),
venue priors from earlier matches, team Elo before the match.
## 7. Evaluation
Split by match date: train to 2023, validation 2024, test 2025+. Test set used once.
Metrics: log loss, Brier, AUC, ECE; score model: MAE and coverage of the 10-90 band (about 80%).
Success: beats the logistic baseline on test log loss and Brier; ECE lower after calibration;
curves behave sensibly (wicket lowers, boundary raises).
## 8. Leakage rules
No random ball-level splits. No feature may use later balls or later matches. A unit test
checks that features at ball t do not change when later balls are removed.
## 9. Product requirements
API: /predict/chase, /predict/first-innings, /teams, /venues, /matches, /matches/{id}/replay,
/health, /model-info. Every prediction returns BOTH teams' win probability (sum to 1).
Dashboard: probability bar, curve, projected-score band, key moments, top reasons, phone friendly.
## 10. Deployment
API in Docker on Render; dashboard on Streamlit Community Cloud; uptime monitor; smoke test
after every deploy.
