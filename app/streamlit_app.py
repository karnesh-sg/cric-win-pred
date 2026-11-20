"""Tarazu: premium dark cricket analytics dashboard."""

import html
import os
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

APP_NAME = "Tarazu · तराज़ू · தராசு"
TAGLINE = "CRICKET INTELLIGENCE PLATFORM"

TEAL = "#19C6A5"
TEAL_DARK = "#0B8F79"
AMBER = "#F5A623"
RED = "#F87171"
GREEN = "#34D399"

BG = "#0B1120"
SURFACE = "#111B2E"
SURFACE_2 = "#172338"
SURFACE_3 = "#1D2B42"
BORDER = "#27364D"

INK = "#F1F5F9"
MUTED = "#94A3B8"
SUBTLE = "#64748B"

st.set_page_config(
    page_title=f"{APP_NAME} | Cricket Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. HTML + CSS HELPERS
# ============================================================


def esc(value):
    """Escape dynamic values before inserting them into HTML."""
    return html.escape(str(value))


def html_block(markup):
    """
    Render HTML without Markdown interpreting indented nested tags
    as a code block. This is the key fix for the raw HTML in the UI.
    """
    cleaned = "\n".join(line.lstrip() for line in str(markup).strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
    --bg: {BG};
    --surface: {SURFACE};
    --surface-2: {SURFACE_2};
    --surface-3: {SURFACE_3};
    --border: {BORDER};
    --ink: {INK};
    --muted: {MUTED};
    --subtle: {SUBTLE};
    --teal: {TEAL};
    --amber: {AMBER};
}}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
}}
.stApp {{
    background:
        radial-gradient(ellipse at 12% 0%, rgba(25,198,165,.075), transparent 32%),
        radial-gradient(ellipse at 90% 8%, rgba(59,130,246,.045), transparent 28%),
        var(--bg);
    color: var(--ink);
}}
.block-container {{
    max-width: 1440px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{
    color: var(--ink);
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -.035em;
}}
[data-testid="stMarkdownContainer"] p {{ color: var(--muted); }}
[data-testid="stCaptionContainer"] {{ color: var(--subtle); }}

/* Brand header */
.cz-header {{
    display:flex; justify-content:space-between; align-items:center;
    gap:16px; flex-wrap:wrap; padding:8px 0 20px; margin-bottom:22px;
    border-bottom:1px solid var(--border);
}}
.cz-logo-wrap {{ display:flex; align-items:center; gap:12px; }}
.cz-logo {{
    width:43px; height:43px; border-radius:12px;
    background:linear-gradient(135deg,#19C6A5,#087E70);
    display:flex; justify-content:center; align-items:center;
    color:#061B19; font-family:'Space Grotesk',sans-serif;
    font-size:23px; font-weight:800;
    box-shadow:0 0 24px rgba(25,198,165,.20);
}}
.cz-brand {{
    color:#F8FAFC; font-family:'Space Grotesk',sans-serif;
    font-size:23px; font-weight:800; letter-spacing:-.045em; line-height:1.1;
}}
.cz-tag {{
    color:var(--subtle); font-size:9px; font-weight:700;
    letter-spacing:.15em; margin-top:5px;
}}
.cz-online {{
    display:flex; align-items:center; gap:8px;
    border:1px solid rgba(52,211,153,.25); background:rgba(16,185,129,.08);
    border-radius:30px; padding:8px 13px; color:#6EE7B7;
    font-size:9px; font-weight:800; letter-spacing:.09em;
}}
.cz-online-dot {{
    height:7px; width:7px; background:#34D399; border-radius:50%;
    box-shadow:0 0 10px #34D399;
}}

/* Page structure */
.cz-page-title {{
    font-family:'Space Grotesk',sans-serif; color:var(--ink);
    font-size:26px; font-weight:700; letter-spacing:-.045em; margin-bottom:3px;
}}
.cz-page-sub {{ color:var(--subtle); font-size:12px; margin-bottom:20px; }}
.cz-section {{
    font-size:9px; font-weight:800; letter-spacing:.14em;
    text-transform:uppercase; color:var(--subtle); margin:5px 0 12px;
}}
.cz-section-line {{ display:flex; align-items:center; gap:10px; }}
.cz-section-line:after {{ content:""; height:1px; background:var(--border); flex:1; }}

/* Cards */
.cz-card {{
    background:linear-gradient(145deg,var(--surface),#101A2B);
    border:1px solid var(--border); border-radius:15px;
    padding:19px; margin-bottom:14px; box-shadow:0 8px 24px rgba(0,0,0,.10);
}}
.cz-card-flat {{
    background:var(--surface); border:1px solid var(--border);
    border-radius:13px; padding:16px; margin-bottom:12px;
}}
.cz-card-label {{
    color:var(--subtle); font-size:9px; font-weight:800;
    letter-spacing:.13em; text-transform:uppercase; margin-bottom:9px;
}}

/* Scoreboard */
.cz-scoreboard {{
    background:
        radial-gradient(ellipse at 50% 0%,rgba(25,198,165,.10),transparent 65%),
        linear-gradient(140deg,#17263A,#101827 75%);
    border:1px solid #30425B; border-radius:18px;
    padding:23px; margin-bottom:15px; box-shadow:0 15px 35px rgba(0,0,0,.16);
}}
.cz-score-top {{
    display:flex; justify-content:space-between; align-items:center;
    gap:12px; flex-wrap:wrap; padding-bottom:18px;
    border-bottom:1px solid rgba(148,163,184,.15); margin-bottom:19px;
}}
.cz-live-tag {{
    display:inline-flex; align-items:center; gap:7px;
    border:1px solid rgba(248,113,113,.25); background:rgba(248,113,113,.08);
    color:#FCA5A5; padding:5px 9px; border-radius:6px;
    font-size:9px; font-weight:800; letter-spacing:.10em;
}}
.cz-score-grid {{
    display:grid; grid-template-columns:1fr 1fr 1fr; align-items:center; gap:12px;
}}
.cz-team-name {{
    font-family:'Space Grotesk',sans-serif; color:#F8FAFC;
    font-size:17px; font-weight:700; overflow-wrap:anywhere;
}}
.cz-team-score {{
    font-family:'Space Grotesk',sans-serif; font-size:42px;
    font-weight:700; color:#FFF; letter-spacing:-.06em; line-height:1.2; margin-top:4px;
}}
.cz-team-overs {{ color:var(--muted); font-size:10px; margin-top:3px; }}
.cz-target {{
    text-align:center; border-left:1px solid rgba(148,163,184,.15);
    border-right:1px solid rgba(148,163,184,.15); padding:4px 7px;
}}
.cz-target-number {{
    font-family:'Space Grotesk',sans-serif; font-size:31px;
    font-weight:700; color:#FFF;
}}
.cz-target-caption {{
    font-size:9px; color:var(--subtle); font-weight:800; letter-spacing:.12em;
}}
.cz-need {{
    display:inline-block; margin-top:7px; color:#FCD34D;
    background:rgba(245,166,35,.10); border:1px solid rgba(245,166,35,.2);
    border-radius:6px; padding:4px 8px; font-size:9px; font-weight:800;
}}
.cz-score-meta {{
    display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px;
    border-top:1px solid rgba(148,163,184,.15); padding-top:15px; margin-top:19px;
}}
.cz-meta-item {{ color:var(--muted); font-size:9px; }}
.cz-meta-item b {{
    display:block; color:#F8FAFC; font-family:'Space Grotesk',sans-serif;
    font-size:17px; margin-top:3px;
}}

/* Probability */
.cz-prob {{
    background:linear-gradient(145deg,#13263A,#101A2B);
    border:1px solid #2C4559; border-radius:16px;
    padding:20px; margin-bottom:15px;
}}
.cz-prob-top {{
    display:flex; justify-content:space-between; align-items:center;
    gap:10px; margin-bottom:19px;
}}
.cz-prob-title {{ font-size:10px; font-weight:800; letter-spacing:.13em; color:var(--muted); }}
.cz-model-tag {{
    font-size:9px; font-weight:800; color:#6EE7B7;
    background:rgba(25,198,165,.09); border:1px solid rgba(25,198,165,.2);
    padding:5px 8px; border-radius:5px;
}}
.cz-prob-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-bottom:15px; }}
.cz-prob-side:last-child {{ text-align:right; }}
.cz-prob-team {{ color:var(--muted); font-size:11px; font-weight:700; overflow-wrap:anywhere; }}
.cz-prob-number {{
    font-family:'Space Grotesk',sans-serif; font-size:40px; font-weight:700;
    line-height:1.2; letter-spacing:-.055em; margin-top:5px;
}}
.cz-prob-bar {{ height:13px; display:flex; border-radius:30px; overflow:hidden; background:var(--surface-3); }}
.cz-prob-bar i {{ height:100%; display:block; transition:width .35s ease; }}
.cz-prob-foot {{
    display:flex; justify-content:space-between; gap:10px; margin-top:10px;
    color:var(--subtle); font-size:8px;
}}

/* What-if outcomes */
.cz-outcome {{
    background:linear-gradient(145deg,#17263A,#111C2D);
    border:1px solid #2A3C52; border-radius:13px;
    padding:15px; min-height:120px; margin-bottom:10px;
    transition:border-color .15s ease,transform .15s ease;
}}
.cz-outcome:hover {{ border-color:var(--teal); transform:translateY(-2px); }}
.cz-outcome-name {{ color:var(--muted); font-size:10px; font-weight:800; letter-spacing:.10em; }}
.cz-outcome-prob {{
    color:#F8FAFC; font-family:'Space Grotesk',sans-serif;
    font-size:29px; font-weight:700; margin-top:7px; letter-spacing:-.045em;
}}
.cz-outcome-change {{ font-size:10px; font-weight:700; margin-top:1px; }}
.cz-positive {{ color:#6EE7B7; }}
.cz-negative {{ color:#FCA5A5; }}
.cz-neutral {{ color:var(--muted); }}
.cz-outcome-accent {{ height:3px; border-radius:10px; margin-top:12px; }}

/* TreeSHAP */
.cz-factor {{
    display:grid; grid-template-columns:minmax(110px,1fr) 2fr;
    gap:10px; align-items:center; padding:11px 0;
    border-bottom:1px solid rgba(148,163,184,.10);
}}
.cz-factor:last-child {{ border-bottom:0; }}
.cz-factor-name {{ color:#E2E8F0; font-size:11px; font-weight:700; overflow-wrap:anywhere; }}
.cz-factor-detail {{ color:var(--muted); font-size:10px; }}
.cz-factor-track {{ height:6px; border-radius:8px; background:var(--surface-3); overflow:hidden; margin-bottom:5px; }}
.cz-factor-fill {{ height:100%; border-radius:8px; }}
.cz-note {{ color:var(--subtle); font-size:10px; line-height:1.7; }}

/* Quantile projection */
.cz-range {{ position:relative; height:9px; background:var(--surface-3); border-radius:10px; margin:24px 6px 17px; }}
.cz-range-band {{ position:absolute; height:9px; background:linear-gradient(90deg,#087E70,#19C6A5); border-radius:10px; }}
.cz-range-mid {{ position:absolute; top:-5px; height:19px; width:3px; background:#FFF; border-radius:3px; }}
.cz-range-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; text-align:center; gap:8px; }}
.cz-range-value {{ color:#F8FAFC; font-family:'Space Grotesk',sans-serif; font-size:24px; font-weight:700; }}
.cz-range-label {{ color:var(--subtle); font-size:9px; font-weight:800; letter-spacing:.08em; }}

/* Empty state and footer */
.cz-empty {{
    background:var(--surface); border:1px dashed #34445C;
    border-radius:15px; padding:43px 22px; text-align:center;
    color:var(--muted); font-size:12px;
}}
.cz-empty b {{ display:block; color:#F8FAFC; font-size:16px; margin-bottom:8px; }}
.cz-divider {{ border-top:1px solid var(--border); margin:18px 0; }}
.cz-footer {{
    border-top:1px solid var(--border); padding-top:18px; margin-top:36px;
    color:var(--subtle); font-size:10px; line-height:1.8;
}}

/* Streamlit controls: dark styling */
div[data-testid="stForm"] {{
    background:rgba(17,27,46,.85); border:1px solid var(--border);
    border-radius:14px; padding:18px;
}}
.stButton > button,.stFormSubmitButton > button {{
    border-radius:9px; font-weight:700; min-height:40px;
}}
.stButton > button[kind="primary"],.stFormSubmitButton > button[kind="primary"] {{
    background:var(--teal-dark); border:1px solid var(--teal-dark); color:white;
}}
.stButton > button[kind="primary"]:hover,.stFormSubmitButton > button[kind="primary"]:hover {{
    background:#0AA58C; border-color:#0AA58C; color:white;
}}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {{
    background:#172338 !important; border-color:#34445C !important;
}}
div[data-testid="stSelectbox"] [data-baseweb="select"] *,
div[data-testid="stMultiSelect"] [data-baseweb="select"] * {{
    color:#F1F5F9 !important;
}}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {{
    background:#172338 !important; color:#F1F5F9 !important;
    border-color:#34445C !important;
}}
div[data-testid="stMetric"] {{
    background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:13px;
}}
div[data-testid="stExpander"] {{
    background:rgba(17,27,46,.65); border:1px solid var(--border); border-radius:11px;
}}
div[data-testid="stAlert"] {{ border-radius:11px; }}
div[data-testid="stRadio"] label,div[data-testid="stCheckbox"] label {{ color:var(--muted); }}
div[data-testid="stSlider"] [data-baseweb="slider"] {{ color:var(--teal); }}
[data-testid="stDataFrame"],[data-testid="stTable"] {{ border:1px solid var(--border); border-radius:10px; }}

@media(max-width:700px) {{
    .cz-team-score {{ font-size:30px; }}
    .cz-target-number {{ font-size:24px; }}
    .cz-prob-number {{ font-size:32px; }}
    .cz-scoreboard {{ padding:16px; }}
    .cz-card {{ padding:15px; }}
    .cz-factor {{ grid-template-columns:1fr; gap:5px; }}
    .cz-prob-foot {{ font-size:7px; }}
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. HEADER
# ============================================================

html_block(f"""
<div class="cz-header">
    <div class="cz-logo-wrap">
        <div class="cz-logo">त</div>
        <div>
            <div class="cz-brand">{esc(APP_NAME)}</div>
            <div class="cz-tag">{esc(TAGLINE)}</div>
        </div>
    </div>
    <div class="cz-online">
        <span class="cz-online-dot"></span>
        PREDICTION ENGINE
    </div>
</div>
""")


# ============================================================
# 4. API HELPERS
# ============================================================


def api_url():
    try:
        return st.secrets["API_URL"].rstrip("/")
    except Exception:
        return os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


@st.cache_data(ttl=3600, show_spinner=False)
def get(path):
    response = requests.get(api_url() + path, timeout=90)
    response.raise_for_status()
    return response.json()


def post(path, payload):
    response = requests.post(api_url() + path, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()


def show_api_error(error=None):
    st.error("Tarazu could not connect to the prediction service.")
    if isinstance(error, requests.Timeout):
        st.caption("The request timed out. The API may be waking up. Wait briefly and retry.")
    elif isinstance(error, requests.ConnectionError):
        st.caption("No connection to the API. Check that it is running and API_URL is correct.")
    elif isinstance(error, requests.HTTPError):
        code = error.response.status_code if error.response is not None else "unknown"
        st.caption(f"The API returned HTTP {code}.")
    elif error is not None:
        st.caption(f"Technical detail: {error}")
    if st.button("Retry connection", key="retry_api"):
        st.cache_data.clear()
        st.rerun()


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def show_table(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


# ============================================================
# 5. GENERAL HELPERS
# ============================================================


def over_text(balls_bowled):
    balls_bowled = int(balls_bowled)
    return f"{balls_bowled // 6}.{balls_bowled % 6}"


def chase_numbers(runs, balls_bowled, target):
    balls_left = max(0, 120 - int(balls_bowled))
    need = max(0, int(target) - int(runs))
    crr = runs / (balls_bowled / 6) if balls_bowled else 0.0
    rrr = need / (balls_left / 6) if balls_left else None
    return need, balls_left, crr, rrr


def meta_item(label, value):
    return f'<div class="cz-meta-item">{esc(label)}<b>{esc(value)}</b></div>'


def empty_state(title, message):
    html_block(f"""
<div class="cz-empty">
    <b>{esc(title)}</b>
    {esc(message)}
</div>
""")


# ============================================================
# 6. SCOREBOARD
# ============================================================


def score_card(
    team,
    runs,
    wickets,
    balls_bowled,
    venue,
    meta_html="",
    target=None,
    need=None,
    balls_left=None,
):
    target_display = "—" if target is None else str(int(target))
    need_display = ""
    if need is not None and balls_left is not None:
        need_display = f'<div class="cz-need">{int(need)} RUNS OFF {int(balls_left)} BALLS</div>'

    html_block(f"""
<div class="cz-scoreboard">
    <div class="cz-score-top">
        <div class="cz-card-label" style="margin:0">MATCH SITUATION</div>
        <div class="cz-live-tag"><span>●</span> SCENARIO ANALYSIS</div>
    </div>
    <div class="cz-score-grid">
        <div>
            <div class="cz-team-name">{esc(team)}</div>
            <div class="cz-team-score">{int(runs)}/{int(wickets)}</div>
            <div class="cz-team-overs">{over_text(balls_bowled)} OVERS · {esc(venue)}</div>
        </div>
        <div class="cz-target">
            <div class="cz-target-caption">TARGET</div>
            <div class="cz-target-number">{target_display}</div>
            {need_display}
        </div>
        <div style="text-align:right">
            <div class="cz-team-name">MATCH STATE</div>
            <div class="cz-team-score" style="font-size:27px">{max(0, 10 - int(wickets))}</div>
            <div class="cz-team-overs">WICKETS IN HAND</div>
        </div>
    </div>
    <div class="cz-score-meta">{meta_html}</div>
</div>
""")


# ============================================================
# 7. WIN PROBABILITY
# ============================================================


def prob_panel(team_a, pa, team_b, pb):
    pa = max(0.0, min(1.0, float(pa)))
    pb = max(0.0, min(1.0, float(pb)))
    total = pa + pb
    if total > 0:
        pa, pb = pa / total, pb / total
    aria = f"{team_a} {pa:.1%}, {team_b} {pb:.1%}"

    html_block(f"""
<div class="cz-prob">
    <div class="cz-prob-top">
        <div class="cz-prob-title">WIN PROBABILITY</div>
        <div class="cz-model-tag">MODEL ESTIMATE</div>
    </div>
    <div class="cz-prob-grid">
        <div class="cz-prob-side">
            <div class="cz-prob-team"><span style="color:{TEAL}">●</span> {esc(team_a)}</div>
            <div class="cz-prob-number" style="color:{TEAL}">{pa:.1%}</div>
        </div>
        <div class="cz-prob-side">
            <div class="cz-prob-team">{esc(team_b)} <span style="color:{AMBER}">●</span></div>
            <div class="cz-prob-number" style="color:{AMBER}">{pb:.1%}</div>
        </div>
    </div>
    <div class="cz-prob-bar" role="img" aria-label="{esc(aria)}">
        <i style="width:{pa * 100:.2f}%;background:{TEAL}"></i>
        <i style="width:{pb * 100:.2f}%;background:{AMBER}"></i>
    </div>
    <div class="cz-prob-foot">
        <span>BATTING TEAM</span>
        <span>ESTIMATED OUTCOME · NOT A GUARANTEE</span>
        <span>BOWLING TEAM</span>
    </div>
</div>
""")


# ============================================================
# 8. WHAT-IF SCENARIOS
# ============================================================


def render_what_if(what_if_res):
    html_block('<div class="cz-section cz-section-line">NEXT-BALL LAB</div>')
    st.caption(
        "Explore how a hypothetical next legal delivery changes the batting team's "
        "estimated win probability. These are conditional model estimates, not "
        "probabilities of the delivery occurring."
    )

    if not isinstance(what_if_res, dict):
        st.info("Next-ball scenarios are unavailable. Run a new chase analysis to retry.")
        return

    outcomes = what_if_res.get("outcomes") or []
    base = what_if_res.get("current_batting_team_win_prob")

    if not outcomes:
        st.info("No next-ball scenarios are available. The chase may already be terminal.")
        return
    if base is None:
        st.warning("The what-if response is missing its baseline probability.")
        return

    try:
        base = float(base)
    except (TypeError, ValueError):
        st.warning("The what-if baseline probability is invalid.")
        return

    expected_order = ["Dot", "1", "2", "4", "6", "Wicket"]
    by_name = {str(item.get("outcome")): item for item in outcomes if isinstance(item, dict)}
    ordered = [by_name[name] for name in expected_order if name in by_name]

    if not ordered:
        st.info("The simulator returned no recognised delivery outcomes.")
        return

    columns = st.columns(3)
    for index, item in enumerate(ordered):
        name = str(item.get("outcome", "Outcome"))
        try:
            probability = max(0.0, min(1.0, float(item["batting_team_win_prob"])))
            change = float(item["change"])
        except (KeyError, TypeError, ValueError):
            continue

        if change > 0.000001:
            css, change_text, accent = "cz-positive", f"+{change * 100:.1f} pp", GREEN
        elif change < -0.000001:
            css, change_text, accent = "cz-negative", f"{change * 100:.1f} pp", RED
        else:
            css, change_text, accent = "cz-neutral", "No material change", SUBTLE

        with columns[index % 3]:
            html_block(f"""
<div class="cz-outcome">
    <div class="cz-outcome-name">{esc(name.upper())}</div>
    <div class="cz-outcome-prob">{probability:.1%}</div>
    <div class="cz-outcome-change {css}">{esc(change_text)}</div>
    <div class="cz-outcome-accent" style="background:{accent}"></div>
</div>
""")

    st.caption(
        f"Baseline batting-team win probability: {base:.1%}. "
        "pp = percentage points. Each card shows the estimated win probability "
        "after that hypothetical outcome."
    )
    with st.expander("Simulator methodology and limitations"):
        st.markdown(
            """
            - The existing trained chase model is reused for each hypothetical outcome.
            - The simulator evaluates a dot, 1, 2, 4, 6 and wicket.
            - Recent rolling statistics are approximated by shrinking the previous
              12-ball values and adding the hypothetical delivery.
            - A wicket resets the current partnership.
            - Wides and no-balls are not simulated.
            - These are conditional win-probability estimates, not delivery probabilities.
            """
        )


# ============================================================
# 9. TREE SHAP
# ============================================================


def render_factors(factors, team_a, team_b, explanation_note=None):
    html_block('<div class="cz-section cz-section-line">MODEL INTELLIGENCE · TREE SHAP</div>')

    if not factors:
        st.caption(
            "No model-specific feature contributions were returned. "
            "No heuristic explanation is substituted."
        )
        if explanation_note:
            st.caption(str(explanation_note))
        return

    rows = []
    for item in factors:
        if not isinstance(item, dict):
            continue
        feature = item.get("feature", item.get("name", "Feature"))
        try:
            value = float(item.get("shap_value", item.get("impact", 0.0)) or 0.0)
        except (TypeError, ValueError):
            continue
        direction = str(item.get("direction", "")).lower()
        if "batting" in direction:
            side, color = team_a, TEAL
        elif "bowling" in direction:
            side, color = team_b, AMBER
        elif value > 0:
            side, color = team_a, TEAL
        elif value < 0:
            side, color = team_b, AMBER
        else:
            side, color = "Neither team", MUTED
        rows.append((feature, value, side, color))

    if not rows:
        st.caption("No valid feature contributions were returned.")
        return

    maximum = max((abs(row[1]) for row in rows), default=0.0) or 1.0
    parts = []
    for feature, value, side, color in rows:
        width = max(2, int(100 * abs(value) / maximum))
        sign = "+" if value > 0 else ""
        parts.append(f"""
<div class="cz-factor">
    <div class="cz-factor-name">{esc(feature)}</div>
    <div>
        <div class="cz-factor-track">
            <div class="cz-factor-fill" style="width:{width}%;background:{color}"></div>
        </div>
        <div class="cz-factor-detail">{sign}{value:.4f} log-odds · favours {esc(side)}</div>
    </div>
</div>
""")

    note = explanation_note or (
        "TreeSHAP values explain the classifier's raw margin (log-odds), "
        "before isotonic calibration and terminal-state rules. They are not "
        "percentage-point changes."
    )
    html_block(
        '<div class="cz-card">'
        + "".join(parts)
        + '<div class="cz-note" style="margin-top:13px">'
        + esc(note)
        + "</div></div>"
    )


# ============================================================
# 10. FIRST-INNINGS QUANTILE PROJECTION
# ============================================================


def render_projected(projected):
    html_block('<div class="cz-section cz-section-line">FIRST-INNINGS SCORE PROJECTION</div>')

    if not isinstance(projected, dict):
        st.caption("Predictions from the trained score quantile models were not returned.")
        return

    low, mid, high = projected.get("q10"), projected.get("q50"), projected.get("q90")
    if None in (low, mid, high):
        st.caption("The API response is missing one or more quantiles.")
        return

    low, mid, high = int(low), int(mid), int(high)
    if not low <= mid <= high:
        st.warning("The returned quantiles are not in ascending order.")
        return

    lo_bound = max(0, low - 25)
    hi_bound = high + 25
    span = max(1, hi_bound - lo_bound)
    left = 100 * (low - lo_bound) / span
    width = 100 * (high - low) / span
    centre = 100 * (mid - lo_bound) / span

    html_block(f"""
<div class="cz-card">
    <div class="cz-card-label">ESTIMATED FINAL SCORE DISTRIBUTION</div>
    <div class="cz-range">
        <div class="cz-range-band" style="left:{left:.2f}%;width:{width:.2f}%"></div>
        <div class="cz-range-mid" style="left:calc({centre:.2f}% - 1px)"></div>
    </div>
    <div class="cz-range-grid">
        <div><div class="cz-range-value">{low}</div><div class="cz-range-label">Q10 · LOWER</div></div>
        <div><div class="cz-range-value" style="color:{TEAL}">{mid}</div><div class="cz-range-label">Q50 · MEDIAN</div></div>
        <div><div class="cz-range-value">{high}</div><div class="cz-range-label">Q90 · UPPER</div></div>
    </div>
    <div class="cz-divider"></div>
    <div class="cz-note">
        Predictions from the trained score quantile models. These estimates
        are not guaranteed final scores.
    </div>
</div>
""")


# ============================================================
# 11. RENDER SAVED SCENARIO
# ============================================================


def render_scenario(saved):
    res = saved["res"]
    ctx = saved["ctx"]
    pa = float(res["batting_team_win_prob"])
    pb = float(res.get("bowling_team_win_prob", 1 - pa))

    if ctx["innings"] == 2 and ctx["target"] is not None:
        need, left_balls, crr, rrr = chase_numbers(ctx["runs"], ctx["balls"], ctx["target"])
        rrr_text = f"{rrr:.2f}" if rrr is not None else "n/a"
        meta = (
            meta_item("CURRENT RUN RATE", f"{crr:.2f}")
            + meta_item("REQUIRED RUN RATE", rrr_text)
            + meta_item("BALLS REMAINING", str(left_balls))
        )
        score_card(
            ctx["bat"],
            ctx["runs"],
            ctx["wkts"],
            ctx["balls"],
            ctx["venue"],
            meta,
            target=ctx["target"],
            need=need,
            balls_left=left_balls,
        )
    else:
        crr = ctx["runs"] / (ctx["balls"] / 6) if ctx["balls"] else 0.0
        meta = (
            meta_item("CURRENT RUN RATE", f"{crr:.2f}")
            + meta_item("WICKETS IN HAND", str(10 - ctx["wkts"]))
            + meta_item("OVERS REMAINING", f"{max(0, 20 - ctx['balls'] / 6):.1f}")
        )
        score_card(ctx["bat"], ctx["runs"], ctx["wkts"], ctx["balls"], ctx["venue"], meta)

    prob_panel(ctx["bat"], pa, ctx["bowl"], pb)

    # What-if is deliberately placed directly below the chase probability panel.
    if ctx["innings"] == 2:
        render_what_if(saved.get("what_if_res"))

    render_factors(res.get("top_factors"), ctx["bat"], ctx["bowl"], res.get("explanation_note"))
    if ctx["innings"] == 1:
        render_projected(res.get("projected_score"))


# ============================================================
# 12. MATCH SCENARIO FORM + API SUBMISSION
# ============================================================


def scenario_view():
    html_block("""
<div class="cz-page-title">Match Centre</div>
<div class="cz-page-sub">Configure a match situation and explore model-driven outcomes.</div>
""")

    try:
        with st.spinner("Connecting to prediction services..."):
            teams = get("/teams")["teams"]
            venues = get("/venues")["venues"]
    except requests.RequestException as error:
        show_api_error(error)
        st.stop()
    except (KeyError, TypeError) as error:
        st.error("The API returned unexpected team or venue data.")
        st.caption(str(error))
        st.stop()

    left, right = st.columns([1, 1.65], gap="large")

    with left:
        html_block('<div class="cz-section cz-section-line">MATCH CONFIGURATION</div>')
        innings = st.radio(
            "Innings",
            [1, 2],
            format_func=lambda i: "1st innings" if i == 1 else "2nd innings · chase",
            horizontal=True,
            label_visibility="collapsed",
            key="scenario_innings",
        )

        with st.form("scenario"):
            batting_team = st.selectbox("Batting team", teams)
            default_bowling = 1 if len(teams) > 1 else 0
            bowling_team = st.selectbox("Bowling team", teams, index=default_bowling)
            venue = st.selectbox("Venue", venues)

            c1, c2 = st.columns(2)
            runs = c1.number_input("Runs", 0, 300, 90)
            wickets = c2.number_input("Wickets lost", 0, 10, 3)

            c3, c4 = st.columns(2)
            overs_done = c3.number_input("Overs completed", 0, 20, 10)
            balls_in_over = c4.number_input(
                "Balls in this over",
                0,
                5,
                0,
                help="Legal balls bowled in the current over.",
            )

            target = None
            if innings == 2:
                target = st.number_input("Target", 1, 300, 180)

            with st.expander("Advanced match statistics"):
                st.caption("Describe the most recent 12 legal deliveries.")
                runs_last12 = st.number_input("Runs in last 12 balls", 0, 60, 12)
                wkts_last12 = st.number_input("Wickets in last 12 balls", 0, 10, 0)
                dot_pct = st.slider("Dot balls in last 12 balls (%)", 0, 100, 42)
                partnership = st.number_input("Current partnership", 0, 200, 20)

            submitted = st.form_submit_button(
                "Analyse match", type="primary", use_container_width=True
            )

    balls_bowled = int(overs_done) * 6 + int(balls_in_over)

    if submitted:
        problem = None
        if batting_team == bowling_team:
            problem = "Batting and bowling teams must be different."
        elif balls_bowled > 120:
            problem = "An innings cannot exceed 20 overs."
        elif int(wickets) >= 10 and int(balls_bowled) < 0:
            problem = "Invalid innings state."

        if problem:
            with right:
                st.warning(problem)
        else:
            body = {
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "venue": venue,
                "runs": int(runs),
                "wickets": int(wickets),
                "balls_bowled": balls_bowled,
                "runs_last12": int(runs_last12),
                "wkts_last12": int(wkts_last12),
                "dot_pct_last12": dot_pct / 100,
                "partnership_runs": int(partnership),
            }

            try:
                with right, st.spinner("Running match probability analysis..."):
                    what_if_res = None
                    if innings == 2:
                        chase_body = {**body, "target": int(target)}
                        res = post("/predict/chase", chase_body)

                        # A what-if failure must not discard a successful base prediction.
                        try:
                            what_if_res = post("/what-if/chase", chase_body)
                        except requests.RequestException as error:
                            st.warning(
                                "The main chase prediction succeeded, but the next-ball "
                                "simulator could not be reached."
                            )
                            st.caption(str(error))
                    else:
                        res = post("/predict/first-innings", body)

                st.session_state["scenario_result"] = {
                    "res": res,
                    "what_if_res": what_if_res,
                    "ctx": {
                        "innings": innings,
                        "bat": batting_team,
                        "bowl": bowling_team,
                        "venue": venue,
                        "runs": int(runs),
                        "wkts": int(wickets),
                        "balls": balls_bowled,
                        "target": int(target) if target is not None else None,
                    },
                }
            except requests.RequestException as error:
                with right:
                    show_api_error(error)
            except (KeyError, TypeError, ValueError) as error:
                with right:
                    st.error("The API response has unexpected fields.")
                    st.caption(str(error))

    with right:
        saved = st.session_state.get("scenario_result")
        if not saved:
            empty_state(
                "Your match analysis starts here",
                "Select teams, configure the match state and run the prediction engine.",
            )
            return
        try:
            render_scenario(saved)
        except (KeyError, TypeError, ValueError) as error:
            st.error("The prediction response could not be displayed.")
            st.caption(str(error))


# ============================================================
# 13. REPLAY CHART
# ============================================================


def build_curve(rep, upto, team_one):
    fig = go.Figure()
    fig.add_hline(y=0.5, line_dash="dot", line_color="#475569")

    start2 = rep.loc[rep["innings"] == 2, "n"].min()
    if pd.notna(start2):
        fig.add_vline(x=start2, line_dash="dot", line_color="#64748B")
        fig.add_annotation(
            x=start2,
            y=1,
            text="INNINGS 2",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=6,
            font=dict(color=MUTED, size=10),
        )

    fig.add_trace(
        go.Scatter(
            x=rep["n"],
            y=rep["p_bat_first_win"],
            mode="lines",
            name=f"{team_one} win probability",
            line=dict(width=3, color=TEAL),
            fill="tozeroy",
            fillcolor="rgba(25,198,165,.09)",
            customdata=rep["over_ball"],
            hovertemplate="Over %{customdata}<br>Win probability %{y:.1%}<extra></extra>",
        )
    )

    wickets_df = rep[rep["is_wicket"] == 1]
    fig.add_trace(
        go.Scatter(
            x=wickets_df["n"],
            y=wickets_df["p_bat_first_win"],
            mode="markers",
            name="Wicket",
            marker=dict(symbol="x", color=RED, size=10, line=dict(width=2)),
            customdata=wickets_df["over_ball"],
            hovertemplate="Wicket · over %{customdata}<extra></extra>",
        )
    )

    sixes_df = rep[rep["is_six"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sixes_df["n"],
            y=sixes_df["p_bat_first_win"],
            mode="markers",
            name="Six",
            marker=dict(symbol="triangle-up", color=AMBER, size=9),
            customdata=sixes_df["over_ball"],
            hovertemplate="Six · over %{customdata}<extra></extra>",
        )
    )

    current = rep.iloc[[upto - 1]]
    fig.add_trace(
        go.Scatter(
            x=current["n"],
            y=current["p_bat_first_win"],
            mode="markers",
            name="Current delivery",
            marker=dict(
                symbol="circle",
                color="#FFFFFF",
                size=12,
                line=dict(color=TEAL, width=3),
            ),
            hoverinfo="skip",
        )
    )
    fig.add_vline(x=upto, line_color="#FFFFFF", line_width=1.5)

    fig.update_layout(
        height=390,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="DM Sans", color=MUTED, size=11),
        margin=dict(l=12, r=12, t=35, b=15),
        xaxis_title="DELIVERY",
        yaxis=dict(
            tickformat=".0%",
            range=[0, 1],
            title=None,
            gridcolor="#27364D",
            zeroline=False,
        ),
        xaxis=dict(gridcolor="#1D2B42", zeroline=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=10),
        ),
        hovermode="closest",
    )
    return fig


# ============================================================
# 14. MATCH REPLAY
# ============================================================


def replay_view():
    html_block("""
<div class="cz-page-title">Match Replay</div>
<div class="cz-page-sub">Explore how win probability evolved throughout a historical match.</div>
""")

    try:
        with st.spinner("Loading match archive..."):
            matches_data = get("/matches")

        matches = (
            matches_data.get("matches", []) if isinstance(matches_data, dict) else matches_data
        )
        if not matches:
            empty_state("No matches available", "The replay API returned no matches.")
            st.stop()

        def match_label(match):
            return match.get("label") or (
                f"{match['date']} | {match['batting_team']} | {match['venue']}"
            )

        top1, top2 = st.columns([3, 1])
        pick = top1.selectbox("SELECT MATCH", matches, format_func=match_label)
        speed = top2.select_slider(
            "PLAYBACK SPEED",
            options=[1, 2, 4, 8],
            value=2,
            format_func=lambda value: f"{value}x",
        )

        with st.spinner("Loading delivery data..."):
            replay_data = get(f"/matches/{pick['match_id']}/replay")

        rep = pd.DataFrame(replay_data["replay"] if isinstance(replay_data, dict) else replay_data)
        if rep.empty:
            empty_state("No replay data", "This match does not have ball-by-ball data.")
            st.stop()

        required = {
            "innings",
            "batting_team",
            "runs",
            "wkts",
            "balls_bowled",
            "p_bat_first_win",
            "is_wicket",
            "is_six",
            "is_four",
        }
        missing = required.difference(rep.columns)
        if missing:
            raise KeyError(f"Replay response is missing columns: {', '.join(sorted(missing))}")

        rep = rep.reset_index(drop=True)
        rep["n"] = range(1, len(rep) + 1)
        rep["over_ball"] = rep["balls_bowled"].apply(over_text)

        innings_one = rep[rep["innings"] == 1]
        innings_two = rep[rep["innings"] == 2]
        if innings_one.empty or innings_two.empty:
            raise ValueError("Replay must contain both innings.")

        team_one = innings_one["batting_team"].iloc[0]
        team_two = innings_two["batting_team"].iloc[0]
        innings_one_last = innings_one.iloc[-1]

        total = len(rep)
        session_key = f"ball_{pick['match_id']}"
        st.session_state.setdefault(session_key, max(1, total // 2))

        upto = st.slider("DELIVERY TIMELINE", 1, total, key=session_key)
        play = st.button("▶  PLAY FROM HERE", type="primary")
        st.caption("Use the timeline to explore individual deliveries, or play the match forward.")

        slot = st.empty()

        def draw(n_now):
            row = rep.iloc[n_now - 1]
            probability_one = float(row["p_bat_first_win"])

            with slot.container():
                c1, c2 = st.columns(2)
                if row["innings"] == 1:
                    score_one = (int(row["runs"]), int(row["wkts"]), int(row["balls_bowled"]))
                else:
                    score_one = (
                        int(innings_one_last["runs"]),
                        int(innings_one_last["wkts"]),
                        int(innings_one_last["balls_bowled"]),
                    )

                with c1:
                    score_card(
                        team_one,
                        *score_one,
                        venue="INNINGS 1",
                        meta_html=meta_item(
                            "INNINGS", "COMPLETED" if row["innings"] == 2 else "IN PROGRESS"
                        ),
                    )

                with c2:
                    if row["innings"] == 2:
                        target_value = int(innings_one_last["runs"]) + 1
                        need, balls_left, _, rrr = chase_numbers(
                            int(row["runs"]), int(row["balls_bowled"]), target_value
                        )
                        rrr_text = f"{rrr:.2f}" if rrr is not None else "n/a"
                        meta = meta_item("REQUIRED RATE", rrr_text) + meta_item(
                            "BALLS LEFT", str(balls_left)
                        )
                        score_card(
                            team_two,
                            int(row["runs"]),
                            int(row["wkts"]),
                            int(row["balls_bowled"]),
                            "INNINGS 2",
                            meta,
                            target=target_value,
                            need=need,
                            balls_left=balls_left,
                        )
                    else:
                        html_block(f"""
<div class="cz-scoreboard">
    <div class="cz-team-name">{esc(team_two)}</div>
    <div class="cz-team-overs" style="margin-top:12px">YET TO BAT</div>
                </div>
""")

                prob_panel(team_one, probability_one, team_two, 1 - probability_one)
                html_block('<div class="cz-section">PROBABILITY TIMELINE</div>')
                show_chart(build_curve(rep, n_now, team_one))

        if play:
            for delivery in range(upto, total + 1):
                draw(delivery)
                time.sleep(0.25 / speed)
        else:
            draw(upto)

        # Key moments
        html_block('<div class="cz-section cz-section-line">KEY MATCH MOMENTS</div>')
        key_moments = rep.copy()
        key_moments["before"] = key_moments.groupby("innings")["p_bat_first_win"].shift(1)
        key_moments["swing"] = (key_moments["p_bat_first_win"] - key_moments["before"]).abs()
        key_moments["event"] = "Runs"
        key_moments.loc[key_moments["is_four"] == 1, "event"] = "Four"
        key_moments.loc[key_moments["is_six"] == 1, "event"] = "Six"
        key_moments.loc[key_moments["is_wicket"] == 1, "event"] = "Wicket"
        key_moments = key_moments.dropna(subset=["swing"]).nlargest(5, "swing")

        def jump(state_key, delivery):
            st.session_state[state_key] = delivery

        button_columns = st.columns(len(key_moments)) if len(key_moments) else []
        for column, (_, moment) in zip(button_columns, key_moments.iterrows()):
            column.button(
                f"{moment['over_ball']} · {moment['event']}",
                key=f"km_{pick['match_id']}_{int(moment['n'])}",
                on_click=jump,
                args=(session_key, int(moment["n"])),
                help="Jump to this delivery",
            )

        table = pd.DataFrame(
            {
                "Innings": key_moments["innings"].astype(int),
                "Over": key_moments["over_ball"],
                "Event": key_moments["event"],
                f"{team_one} win % before": key_moments["before"] * 100,
                f"{team_one} win % after": key_moments["p_bat_first_win"] * 100,
                "Swing (pts)": key_moments["swing"] * 100,
            }
        )
        config = {
            col: st.column_config.NumberColumn(format="%.1f")
            for col in table.columns
            if col.endswith("before") or col.endswith("after")
        }
        config["Swing (pts)"] = st.column_config.NumberColumn(format="%.1f")
        show_table(table, column_config=config)

    except requests.RequestException as error:
        show_api_error(error)
        st.stop()
    except (KeyError, ValueError, TypeError, IndexError) as error:
        st.error(
            "Replay data could not be displayed because the API response has an unexpected format."
        )
        st.caption(str(error))
        st.stop()


# ============================================================
# 15. NAVIGATION
# ============================================================

html_block('<div class="cz-section">WORKSPACE</div>')
MODES = ["Live Scenario", "Match Replay", "Can I Trust This?"]
segmented = getattr(st, "segmented_control", None)

if segmented:
    choice = segmented(
        "Workspace",
        MODES,
        default=MODES[0],
        key="mode_seg_v2",
        label_visibility="collapsed",
    )
else:
    choice = st.radio(
        "Workspace",
        MODES,
        horizontal=True,
        key="mode_seg_v2",
        label_visibility="collapsed",
    )

mode = choice or st.session_state.get("_last_mode", MODES[0])
st.session_state["_last_mode"] = mode
st.write("")

if mode == "Live Scenario":
    scenario_view()
elif mode == "Match Replay":
    replay_view()


# ============================================================
# 16. MODEL TRUST PANEL
# ============================================================


def render_model_trust_panel():
    """
    Display the production chase-model trust report from the /trust API.
    Includes held-out test metrics, baseline comparison and reliability curve.
    """

    # --------------------------------------------------------
    # 1. LOAD TRUST REPORT FROM API
    # --------------------------------------------------------

    try:
        trust_data = get("/trust")
    except requests.RequestException as error:
        st.error("Unable to load the model trust report.")
        show_api_error(error)
        st.stop()

    metrics = trust_data["metrics"]
    baseline = trust_data["baseline"]
    bins = sorted(trust_data["bins"], key=lambda item: item["pred"])

    version = trust_data["version"]
    evaluation_date = trust_data["evaluation_date"]
    n_states = trust_data["n_states"]
    n_matches = trust_data["n_matches"]

    # --------------------------------------------------------
    # 2. PAGE HEADER
    # --------------------------------------------------------

    html_block('<div class="cz-section cz-section-line">MODEL TRUST & TRANSPARENCY</div>')

    st.markdown(
        "Evaluation evidence, calibration context, model explanations and known limitations."
    )

    # --------------------------------------------------------
    # 3. MODEL PERFORMANCE
    # --------------------------------------------------------

    with st.container(border=True):
        st.markdown("#### Chase model · held-out test set")

        st.caption(
            f"Model version {version} · "
            f"Evaluation date {evaluation_date} · "
            f"{n_states:,} ball-by-ball states across "
            f"{n_matches:,} matches"
        )

        metric_cols = st.columns(4)

        metric_cols[0].metric(
            "ROC AUC",
            f"{metrics['auc']:.4f}",
            help=("Higher is better. Measures how well the model ranks wins above losses."),
        )

        metric_cols[1].metric(
            "Log loss",
            f"{metrics['logloss']:.4f}",
            help=("Lower is better. Penalises confident incorrect probability estimates."),
        )

        metric_cols[2].metric(
            "Brier score",
            f"{metrics['brier']:.4f}",
            help=("Lower is better. Measures squared error of predicted probabilities."),
        )

        metric_cols[3].metric(
            "ECE",
            f"{metrics['ece']:.4f}",
            help=(
                "Lower is better. Measures the average gap "
                "between predicted confidence and observed frequency."
            ),
        )

        st.markdown(
            """
            **How to read these results**

            - **ROC AUC:** measures the model's ability to distinguish
              winning and losing outcomes across thresholds.
            - **Log loss:** evaluates the quality of probability estimates
              and strongly penalises confident incorrect predictions.
            - **Brier score:** measures the squared difference between
              predicted probabilities and actual outcomes.
            - **ECE:** summarises the gap between predicted confidence
              and observed outcome frequency across probability groups.

            These are historical held-out test-set metrics, not a
            guarantee that an individual live-match prediction will be correct.
            """
        )

    # --------------------------------------------------------
    # 4. BASELINE COMPARISON
    # --------------------------------------------------------

    with st.container(border=True):
        st.markdown("#### Model vs baseline")

        st.markdown(
            """
            The baseline predicts the same chase-win probability for every
            test state: the chase-win frequency observed in the training set.

            A useful model should improve on this baseline in predictive
            discrimination and probability quality.
            """
        )

        comparison = pd.DataFrame(
            {
                "Tarazu model": [
                    metrics["logloss"],
                    metrics["brier"],
                    metrics["auc"],
                    metrics["ece"],
                ],
                "Constant baseline": [
                    baseline["logloss"],
                    baseline["brier"],
                    baseline["auc"],
                    baseline["ece"],
                ],
            },
            index=["Log loss", "Brier score", "ROC AUC", "ECE"],
        )

        st.dataframe(
            comparison.style.format("{:.4f}"),
            use_container_width=True,
        )

        st.caption(
            trust_data.get(
                "baseline_definition",
                "Constant probability based on training-set chase-win frequency.",
            )
        )

        st.info(
            "The baseline's lower ECE does not mean it is a better predictor. "
            "A constant prediction can have low calibration error while "
            "having no ability to distinguish between winning and losing "
            "situations. Its ROC AUC is 0.5."
        )

    # --------------------------------------------------------
    # 5. RELIABILITY / CALIBRATION CURVE
    # --------------------------------------------------------

    with st.container(border=True):
        st.markdown("#### Reliability curve")

        st.markdown(
            """
            This chart compares the model's average predicted probability
            with the actual chase-win frequency in each probability group.

            A perfectly calibrated model would follow the diagonal:
            a group predicted at 70% should win approximately 70% of the time.
            """
        )

        reliability_df = pd.DataFrame(
            {
                "Observed win rate": [item["actual"] for item in bins],
                "Perfect calibration": [item["pred"] for item in bins],
            },
            index=[item["pred"] for item in bins],
        )

        reliability_df.index.name = "Mean predicted probability"

        st.line_chart(
            reliability_df,
            y=["Observed win rate", "Perfect calibration"],
            x_label="Mean predicted probability",
            y_label="Observed chase-win frequency",
        )

        st.caption(
            f"Quantile-based reliability groups · "
            f"{len(bins)} groups shown. Duplicate probability "
            "boundaries can reduce the number of groups."
        )

        with st.expander("View reliability bin data"):
            bin_table = pd.DataFrame(
                [
                    {
                        "Predicted probability": item["pred"],
                        "Observed win rate": item["actual"],
                        "States": item["n"],
                    }
                    for item in bins
                ]
            )

            st.dataframe(
                bin_table.style.format(
                    {
                        "Predicted probability": "{:.3f}",
                        "Observed win rate": "{:.3f}",
                        "States": "{:,.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # 6. CALIBRATION EXPLANATION
    # --------------------------------------------------------

    with st.container(border=True):
        st.markdown("#### Calibration")

        st.markdown(
            """
            The production chase probability uses isotonic calibration.
            Calibration is intended to make predicted probabilities better
            match observed frequencies over groups of similar predictions.

            For example, among many predictions near 70%, a well-calibrated
            model should see the corresponding outcome occur roughly 70%
            of the time.

            Calibration quality is population-level behaviour. It does not
            mean an individual 70% prediction is certain or that every
            subgroup is equally calibrated.

            The trust report evaluates the production prediction pipeline,
            including calibration and terminal-state rules.
            """
        )

    # --------------------------------------------------------
    # 7. FIRST-INNINGS MODEL
    # --------------------------------------------------------

    with st.container(border=True):
        st.markdown("#### First-innings model · score ranges")

        st.markdown(
            """
            First-innings score projections use trained quantile models:

            - **Q10:** lower conditional score estimate.
            - **Q50:** median conditional score estimate.
            - **Q90:** upper conditional score estimate.

            These quantiles describe a modelled score range; they are not
            guaranteed minimum, expected and maximum scores.

            The shared trust report contains chase-model test metrics only.
            No first-innings AUC, loss, MAE or coverage figure is claimed here.
            """
        )

    # --------------------------------------------------------
    # 8. EXPLAINABILITY
    # --------------------------------------------------------

    with st.container(border=True):
        st.markdown("#### Explainability · TreeSHAP")

        st.markdown(
            """
            TreeSHAP values show how input features contribute to the
            underlying tree model's raw margin (log-odds).

            Positive and negative values indicate direction on that model
            scale. They are **not** direct percentage-point changes in
            calibrated win probability.

            The dashboard displays feature contributions returned by
            the API. Explanations describe model behaviour; they do not
            establish causation.
            """
        )

    # --------------------------------------------------------
    # 9. WHAT-IF ASSUMPTIONS
    # --------------------------------------------------------

    with st.expander("What-if simulator · assumptions"):
        st.markdown(
            """
            - The existing trained chase model is reused for each hypothetical
              next-ball state.
            - The simulator evaluates Dot, 1, 2, 4, 6 and Wicket.
            - Recent rolling statistics are approximated from the previous
              12-ball values and the hypothetical delivery.
            - A wicket resets the current partnership.
            - Wides and no-balls are not simulated.
            - Each card is a conditional win-probability estimate after the
              hypothetical outcome, not the probability of that delivery
              occurring.
            """
        )

    # --------------------------------------------------------
    # 10. KNOWN LIMITATIONS
    # --------------------------------------------------------

    with st.expander("Known limitations and intended use"):
        st.markdown(
            """
            - The models learn patterns from historical men's T20 data;
              future matches can differ from the training distribution.
            - The model does not account for individual player skill, toss,
              pitch condition, dew, weather or live player availability
              unless those variables are explicitly included in its inputs.
            - Data quality, missing inputs, unseen teams or venues and
              unusual match conditions can affect reliability.
            - Ball-by-ball states from the same match are correlated;
              they are not independent match-level observations.
            - Reliability bins use quantile grouping, and duplicate
              probability boundaries can reduce the number of bins.
            - Shortened-innings handling limitation F-01 remains unresolved.
            - A probability is an estimate under the model, not a guarantee.
            - TreeSHAP explains model contributions, not real-world causation.

            Tarazu is an analytical and research tool, not betting advice.
            """
        )


if mode == "Can I Trust This?":
    render_model_trust_panel()


# ============================================================
# 17. FOOTER
# ============================================================

html_block(f"""
<div class="cz-footer">
    <b style="color:{TEAL};letter-spacing:.12em">{esc(APP_NAME)}</b>
    &nbsp; · &nbsp; {esc(TAGLINE)}
    <br>
    Powered by machine learning · Historical men's T20 data
    <br>
    Predictions are estimates subject to model limitations and data quality.
</div>
""")
