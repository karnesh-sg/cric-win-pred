import os

import requests
import streamlit as st


def api_url():
    try:
        return st.secrets["API_URL"].rstrip("/")
    except (KeyError, FileNotFoundError):
        return os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


st.set_page_config(page_title="Cricket Win Probability", page_icon="🏏", layout="centered")

st.title("🏏 Cricket Win Probability")
st.write("Test your live cricket prediction API.")

if st.button("Test the API"):
    with st.spinner("Calling API... the first request may take a while"):
        try:
            response = requests.post(
                f"{api_url()}/predict/chase",
                timeout=90,
                json={"target": 180, "runs": 90, "wickets": 3, "balls_bowled": 60},
            )

            response.raise_for_status()
            result = response.json()

            st.success("API response received!")
            st.json(result)

            if "batting_team_win_prob" in result:
                probability = result["batting_team_win_prob"]
                st.metric("Batting team win probability", f"{probability:.2%}")

        except requests.RequestException as error:
            st.error(f"Could not reach the API: {error}")
        except (ValueError, TypeError):
            st.error("The API returned an invalid response.")
