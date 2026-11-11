import os
import html
import requests
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Cricket Win Probability",
    layout="wide",
)

st.title("🏏 Cricket Win Probability")
st.caption("Live scenario analysis powered by your ML model")


# --------------------------------------------------
# 2. API URL
# --------------------------------------------------


def api_url():
    try:
        return st.secrets["API_URL"]
    except Exception:
        return os.environ.get(
            "API_URL",
            "http://127.0.0.1:8000",
        )


# --------------------------------------------------
# 3. API GET helper
# --------------------------------------------------


@st.cache_data(ttl=3600, show_spinner="Waking the API...")
def get(path):
    response = requests.get(
        api_url() + path,
        timeout=90,
    )

    response.raise_for_status()

    return response.json()


# --------------------------------------------------
# 4. API POST helper
# --------------------------------------------------


def post(path, payload):
    response = requests.post(
        api_url() + path,
        json=payload,
        timeout=90,
    )

    response.raise_for_status()

    return response.json()


# --------------------------------------------------
# 5. Overs to balls conversion
# --------------------------------------------------


def balls(overs):
    total = round(overs * 10)

    completed_overs, extra_balls = divmod(total, 10)

    if extra_balls > 5:
        raise ValueError("Invalid overs notation. The ball component must be between 0 and 5.")

    return completed_overs * 6 + extra_balls


# --------------------------------------------------
# 6. Probability bar
# --------------------------------------------------


def prob_bar(batting_team, batting_prob, bowling_team, bowling_prob):

    batting_prob = max(0.0, min(1.0, batting_prob))
    bowling_prob = max(0.0, min(1.0, bowling_prob))

    batting_width = batting_prob * 100
    bowling_width = bowling_prob * 100

    batting_team = html.escape(batting_team)
    bowling_team = html.escape(bowling_team)

    html_content = (
        "<div style='display:flex;height:42px;"
        "border-radius:8px;overflow:hidden;"
        "font-weight:600;color:white;'>"
        f"<div style='width:{batting_width:.1f}%;"
        "background:#0F766E;padding:10px;"
        "box-sizing:border-box;overflow:hidden;"
        "white-space:nowrap;'>"
        f"{batting_team} {batting_prob:.1%}"
        "</div>"
        f"<div style='width:{bowling_width:.1f}%;"
        "background:#B45309;padding:10px;"
        "box-sizing:border-box;text-align:right;"
        "overflow:hidden;white-space:nowrap;'>"
        f"{bowling_team} {bowling_prob:.1%}"
        "</div>"
        "</div>"
    )

    st.markdown(
        html_content,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 7. Display top factors
# --------------------------------------------------


def show_factors(factors):

    st.subheader("Why this prediction?")

    if isinstance(factors, dict):
        for name, value in factors.items():
            st.write(f"**{name}:** {value}")

    elif isinstance(factors, list):
        for factor in factors:
            st.write(f"• {factor}")

    else:
        st.write(factors)


# --------------------------------------------------
# 8. API error handler
# --------------------------------------------------


def show_api_error():

    st.error("API is waking up or unavailable. Please retry in a moment.")

    if st.button("Retry API"):
        st.cache_data.clear()
        st.rerun()


# --------------------------------------------------
# 9. Live scenario mode
# --------------------------------------------------

mode = st.sidebar.radio(
    "Mode",
    ["Live scenario", "Match replay"],
)


if mode == "Live scenario":
    # Fetch teams and venues

    try:
        teams_data = get("/teams")
        venues_data = get("/venues")

        teams = teams_data["teams"]
        venues = venues_data["venues"]

    except requests.RequestException:
        show_api_error()
        st.stop()

    innings = st.radio(
        "Select innings",
        [1, 2],
        horizontal=True,
    )

    # --------------------------------------------------
    # Form
    # --------------------------------------------------

    with st.form("scenario"):
        st.subheader("Match situation")

        c1, c2, c3 = st.columns(3)

        batting_team = c1.selectbox(
            "Batting team",
            teams,
        )

        bowling_team = c2.selectbox(
            "Bowling team",
            teams,
            index=1 if len(teams) > 1 else 0,
        )

        venue = c3.selectbox(
            "Venue",
            venues,
        )

        c4, c5, c6, c7 = st.columns(4)

        runs = c4.number_input(
            "Runs",
            min_value=0,
            max_value=300,
            value=90,
        )

        wickets = c5.number_input(
            "Wickets lost",
            min_value=0,
            max_value=10,
            value=3,
        )

        overs = c6.number_input(
            "Overs (e.g. 12.3)",
            min_value=0.0,
            max_value=20.0,
            value=10.0,
            step=0.1,
            format="%.1f",
        )

        target = c7.number_input(
            "Target (chase only)",
            min_value=1,
            max_value=300,
            value=180,
        )

        st.subheader("Recent performance")

        c8, c9, c10, c11 = st.columns(4)

        runs_last2 = c8.number_input(
            "Runs in last 2 overs",
            min_value=0,
            max_value=60,
            value=12,
        )

        wickets_last2 = c9.number_input(
            "Wickets in last 2 overs",
            min_value=0,
            max_value=10,
            value=0,
        )

        dot_pct = c10.number_input(
            "Dot-ball percentage (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=0.4167,
            step=0.01,
        )

        partnership = c11.number_input(
            "Partnership runs",
            min_value=0,
            max_value=200,
            value=20,
        )

        go = st.form_submit_button(
            "Update prediction",
            type="primary",
        )

    # --------------------------------------------------
    # 10. Prediction logic
    # --------------------------------------------------

    if go:
        if batting_team == bowling_team:
            st.warning("Batting team and bowling team must be different.")

        else:
            try:
                balls_bowled = balls(overs)

                body = {
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "venue": venue,
                    "runs": runs,
                    "wickets": wickets,
                    "balls_bowled": balls_bowled,
                    "runs_last12": runs_last2,
                    "wkts_last12": wickets_last2,
                    "dot_pct_last12": dot_pct,
                    "partnership_runs": partnership,
                }

                # Innings 2: Chase prediction
                if innings == 2:
                    response = post(
                        "/predict/chase",
                        {
                            **body,
                            "target": target,
                        },
                    )

                # Innings 1: First innings prediction
                else:
                    response = post(
                        "/predict/first-innings",
                        body,
                    )

                # ------------------------------------------
                # Display scoreboard
                # ------------------------------------------

                st.divider()

                st.subheader(f"{batting_team} {runs}/{wickets}")

                st.caption(f"Overs: {overs:.1f} | Venue: {venue}")

                batting_prob = response["batting_team_win_prob"]

                bowling_prob = response["bowling_team_win_prob"]

                prob_bar(
                    batting_team,
                    batting_prob,
                    bowling_team,
                    bowling_prob,
                )

                # ------------------------------------------
                # Display explanation
                # ------------------------------------------

                factors = response.get(
                    "top_factors",
                    [],
                )

                show_factors(factors)

                # ------------------------------------------
                # First innings projected score
                # ------------------------------------------

                if innings == 1:
                    st.subheader("Projected score range")

                    projected_score = response.get("projected_score")

                    if isinstance(projected_score, dict):
                        q10 = projected_score.get("q10")
                        q50 = projected_score.get("q50")
                        q90 = projected_score.get("q90")

                        m1, m2, m3 = st.columns(3)

                        m1.metric("Q10", q10)
                        m2.metric("Median (Q50)", q50)
                        m3.metric("Q90", q90)

                    elif projected_score is not None:
                        st.write(projected_score)

                    else:
                        st.info("The API response does not currently include projected_score.")

            except ValueError as error:
                st.warning(str(error))

            except requests.RequestException:
                show_api_error()
