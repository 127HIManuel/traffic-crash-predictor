"""
Traffic Crash Severity Prediction System — Streamlit Web Application
Osun State, Nigeria | FRSC + Kaggle Ensemble Model
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components


# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OsunCrash — Traffic Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# GLOBAL STYLING
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:         #080B10;
    --surface:    #0E1219;
    --surface2:   #141922;
    --surface3:   #1A2030;
    --border:     #1E2736;
    --border2:    #243040;
    --accent:     #00D4FF;
    --accent2:    #FF6B35;
    --accent3:    #00FF88;
    --fatal:      #FF3355;
    --serious:    #FF9500;
    --minor:      #00CC6A;
    --text:       #E2E8F0;
    --text2:      #94A3B8;
    --text3:      #64748B;
    --font-head:  'Bebas Neue', sans-serif;
    --font-body:  'Space Grotesk', sans-serif;
    --font-mono:  'JetBrains Mono', monospace;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg);
    color: var(--text);
}

.block-container {
    padding: 0 2rem 3rem 2rem;
    max-width: 1500px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border2) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.25rem;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: var(--font-head);
    letter-spacing: 0.05em;
}

/* ── Top hero bar ── */
.hero-bar {
    background: linear-gradient(135deg, #080B10 0%, #0D1520 50%, #080B10 100%);
    border-bottom: 1px solid var(--border2);
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-bar::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -10%;
    width: 40%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(0,212,255,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero-bar::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 40%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(255,107,53,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: var(--font-head);
    font-size: 3.5rem;
    letter-spacing: 0.12em;
    line-height: 1;
    color: var(--text);
    margin: 0;
}
.hero-title .accent { color: var(--accent); }
.hero-title .accent2 { color: var(--accent2); }
.hero-sub {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text3);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 4px;
    padding: 0.25rem 0.75rem;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    margin-top: 0.75rem;
}
.hero-badge::before {
    content: '●';
    font-size: 0.5rem;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border2);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border2);
    margin-bottom: 2rem;
}
.kpi-card {
    background: var(--surface);
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: background 0.2s;
}
.kpi-card:hover { background: var(--surface2); }
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-total::after   { background: linear-gradient(90deg, var(--accent), transparent); }
.kpi-fatal::after   { background: linear-gradient(90deg, var(--fatal), transparent); }
.kpi-serious::after { background: linear-gradient(90deg, var(--serious), transparent); }
.kpi-minor::after   { background: linear-gradient(90deg, var(--minor), transparent); }

.kpi-tag {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 0.75rem;
}
.kpi-num {
    font-family: var(--font-head);
    font-size: 3rem;
    letter-spacing: 0.05em;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.kpi-desc { font-size: 0.78rem; color: var(--text3); }

/* ── Section headers ── */
.sec-head {
    font-family: var(--font-head);
    font-size: 1.4rem;
    letter-spacing: 0.1em;
    color: var(--text);
    margin: 0 0 1.25rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border2);
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.sec-head .line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border2), transparent);
}

/* ── Sidebar Logo ── */
.sb-logo {
    font-family: var(--font-head);
    font-size: 2rem;
    letter-spacing: 0.1em;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.sb-logo .c1 { color: var(--accent); }
.sb-logo .c2 { color: var(--text); }
.sb-logo .c3 { color: var(--accent2); }
.sb-tagline {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text3);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ── Nav items ── */
.stRadio label { font-family: var(--font-body) !important; }

/* ── Risk result container ── */
.risk-wrap {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 2rem;
    height: 100%;
}
.risk-icon-large {
    font-size: 4rem;
    margin-bottom: 0.5rem;
    display: block;
}
.risk-label-text {
    font-family: var(--font-head);
    font-size: 1.1rem;
    letter-spacing: 0.15em;
    color: var(--text3);
    margin-bottom: 0.3rem;
}
.risk-class {
    font-family: var(--font-head);
    font-size: 3.5rem;
    letter-spacing: 0.08em;
    line-height: 1;
    margin-bottom: 1rem;
}
.risk-badge-pill {
    display: inline-block;
    padding: 0.35rem 1.25rem;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.25rem;
}
.rbp-low      { background: rgba(0,204,106,0.12); color: #00CC6A; border: 1px solid rgba(0,204,106,0.3); }
.rbp-moderate { background: rgba(255,149,0,0.12);  color: #FF9500; border: 1px solid rgba(255,149,0,0.3); }
.rbp-high     { background: rgba(255,51,85,0.12);  color: #FF3355; border: 1px solid rgba(255,51,85,0.3); }
.rbp-critical { background: rgba(255,51,85,0.18);  color: #FF3355; border: 1px solid #FF3355; }

/* ── Probability bars ── */
.prob-section { margin-top: 1.25rem; }
.prob-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
}
.prob-lbl {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text3);
}
.prob-pct {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 600;
}
.prob-track {
    height: 6px;
    background: var(--border2);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 0.75rem;
}
.prob-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

/* ── Advice panel ── */
.advice-wrap {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 2rem;
    height: 100%;
}
.advice-title {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 1rem;
}
.advice-text {
    font-size: 0.9rem;
    line-height: 1.8;
    color: var(--text);
    margin-bottom: 1.5rem;
}
.trip-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.75rem;
}
.trip-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 0.75rem;
}
.trip-item-key {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 0.2rem;
}
.trip-item-val {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text);
}

/* ── Route pills ── */
.route-pill {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
    border-left: 3px solid transparent;
    transition: background 0.15s;
}
.route-pill:hover { background: var(--surface2); }
.rp-fatal    { border-left-color: var(--fatal); }
.rp-serious  { border-left-color: var(--serious); }
.rp-moderate { border-left-color: #FFD60A; }
.rp-low      { border-left-color: var(--minor); }
.route-name { font-weight: 600; font-size: 0.87rem; }
.route-meta { font-family: var(--font-mono); font-size: 0.68rem; color: var(--text3); margin-top: 0.15rem; }
.route-pct  { font-family: var(--font-head); font-size: 1.4rem; letter-spacing: 0.05em; }
.route-risk-tag { font-family: var(--font-mono); font-size: 0.62rem; letter-spacing: 0.1em; text-transform: uppercase; }

/* ── Info box ── */
.info-box {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-size: 0.84rem;
    line-height: 1.7;
    margin-top: 1.5rem;
    color: var(--text2);
}
.info-box b { color: var(--text); }

/* ── Model insight cards ── */
.model-card {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 1.25rem;
    height: 100%;
}
.model-card-title {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
}
.model-card-body { font-size: 0.84rem; line-height: 1.7; color: var(--text2); }

/* ── Data rows ── */
.data-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.83rem;
}
.dr-key { color: var(--text3); font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.05em; flex-shrink: 0; }
.dr-val { color: var(--text); text-align: right; }

/* ── Inputs & selects ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div {
    background: var(--surface2) !important;
    border-color: var(--border2) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
.stSlider > div > div > div {
    background: var(--accent) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-bottom: 1px solid var(--border2);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text3);
    padding: 0.75rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ── Button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent), #0099BB) !important;
    color: #000 !important;
    font-family: var(--font-head) !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.7rem 2rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border-color: var(--border2); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS & STATIC DATA
# ══════════════════════════════════════════════════════════════════════

OSUN_ROUTES = {
    "Gbongan – Ibadan":         {"fatal_pct": 18.2, "total": 22, "zone": "Gbongan"},
    "Ilesa – Ipetu":            {"fatal_pct":  9.1, "total": 22, "zone": "Ilesa"},
    "Ilesa – Ife":              {"fatal_pct": 18.8, "total": 16, "zone": "Ilesa"},
    "Ipetu – Ilesa":            {"fatal_pct": 38.5, "total": 13, "zone": "Ipetu"},
    "Ife – Ilesa":              {"fatal_pct": 25.0, "total": 12, "zone": "Ife"},
    "Ife – Ondo":               {"fatal_pct": 11.1, "total":  9, "zone": "Ife"},
    "Wasinmi – Asejire":        {"fatal_pct": 33.3, "total":  9, "zone": "Wasinmi"},
    "Osogbo – Sekona":          {"fatal_pct":  0.0, "total":  8, "zone": "Osogbo"},
    "Gbongan – Ife":            {"fatal_pct": 14.3, "total":  7, "zone": "Gbongan"},
    "Ikirun – Osogbo":          {"fatal_pct":  0.0, "total":  6, "zone": "Ikirun"},
    "Ikirun – Ilaodo":          {"fatal_pct": 16.7, "total":  6, "zone": "Ikirun"},
    "Ipetu – Ilesa (Alt)":      {"fatal_pct": 50.0, "total":  6, "zone": "Ipetu"},
    "Ikire – Gbongan":          {"fatal_pct": 20.0, "total":  5, "zone": "Ikire"},
    "Ile Ife – Ondo Expressway":{"fatal_pct":100.0, "total":  1, "zone": "Ife"},
    "Ipetu – Akure":            {"fatal_pct":  0.0, "total":  5, "zone": "Ipetu"},
    "Ife – Gbongan":            {"fatal_pct":  0.0, "total":  4, "zone": "Ife"},
    "Ipetu – Ijesa":            {"fatal_pct": 25.0, "total":  4, "zone": "Ipetu"},
    "Iwo – Ibadan":             {"fatal_pct":  0.0, "total":  3, "zone": "Iwo"},
    "Ilobu – Osogbo":           {"fatal_pct":  0.0, "total":  3, "zone": "Ilobu"},
    "Osogbo – Gbongan":         {"fatal_pct": 16.7, "total":  6, "zone": "Osogbo"},
    "Osogbo – Ikirun":          {"fatal_pct": 33.3, "total":  3, "zone": "Osogbo"},
    "Gbongan – Osogbo":         {"fatal_pct":  0.0, "total":  1, "zone": "Gbongan"},
    "Ibadan – Ife Highway":     {"fatal_pct":  0.0, "total":  1, "zone": "Ibadan"},
    "Ife – Sekona":             {"fatal_pct": 50.0, "total":  2, "zone": "Ife"},
    "Asejire – Wasinmi":        {"fatal_pct": 33.3, "total":  3, "zone": "Asejire"},
}

CRASH_STATS = {
    "total_frsc": 298, "total_all": 8509,
    "fatal": 139, "serious": 1244, "minor": 7126,
    "fatal_frsc": 56, "serious_frsc": 198, "minor_frsc": 44,
}

HOUR_COUNTS = {
    0:6,1:2,2:3,3:5,4:4,5:3,6:15,7:16,8:13,9:20,10:17,11:26,
    12:17,13:10,14:20,15:32,16:21,17:22,18:17,19:8,20:11,21:6,22:2,23:2
}

DAY_COUNTS = {
    "Monday":54,"Tuesday":33,"Wednesday":43,
    "Thursday":38,"Friday":47,"Saturday":42,"Sunday":41
}

VEHICLE_TYPES = [
    "Bus / Minibus (Commercial)",
    "Car / Saloon (Private)",
    "Motorcycle / Okada",
    "Truck / Lorry",
    "Tricycle (Keke NAPEP)",
    "Other",
]

CRASH_CAUSES = [
    "Speeding",
    "Wrongful Overtaking",
    "Loss of Control",
    "Tyre Burst",
    "Brake Failure",
    "Dangerous Driving",
    "Mechanical Defect",
    "Obstruction",
    "Overloading",
    "Unknown / Other",
]

WEATHER_OPTIONS = ["Clear / Normal","Partly Cloudy","Cloudy","Raining","Foggy","Thunderstorm"]
INV_MAP = {0:"Minor", 1:"Serious", 2:"Fatal"}

RISK_LEVELS = {
    "Minor":   ("LOW",      "🟢", "#00CC6A", "LOW",      "rbp-low",
                 "The predicted crash severity is <b>Minor</b>. Road conditions appear manageable for this journey. Maintain standard safe driving practices — keep your speed appropriate for conditions, wear your seatbelt, and stay alert."),
    "Serious": ("MODERATE", "🟡", "#FF9500", "MODERATE", "rbp-moderate",
                 "The predicted severity is <b>Serious</b>. Several risk factors are elevated for this trip. Exercise heightened caution — reduce speed, increase following distance, avoid distractions, and be extra vigilant at road junctions."),
    "Fatal":   ("CRITICAL", "🔴", "#FF3355", "CRITICAL", "rbp-critical",
                 "<b>CRITICAL risk detected.</b> The combination of conditions entered is associated with fatal crashes in the Osun State dataset. Strongly consider delaying travel, choosing an alternative route, or addressing the identified risk factors before departing."),
}

Model_File_ID = "1b_8QGQtngUr1hHjMiE1AdEvDIhs8G9fx"

# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    model_path     = "ensemble_recall_fix.pkl"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={Model_File_ID}"
            gdown.download(url, model_path, quiet=False)
    encoders_path  = "recall_fix_encoders.pkl"
    features_path  = "recall_fix_features.pkl"
    threshold_path = "serious_threshold.pkl"
    missing = [p for p in [model_path, encoders_path, features_path, threshold_path]
               if not os.path.exists(p)]
    if missing:
        return None, None, None, None, missing
    ensemble  = joblib.load(model_path)
    encoders  = joblib.load(encoders_path)
    features  = joblib.load(features_path)
    threshold = joblib.load(threshold_path)
    return ensemble, encoders, features, threshold, []


# ══════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════
def build_feature_row(hour, day_name, weather, light_cond, vehicle_type,
                       crash_cause, people_involved, encoders, features):
    is_night    = 1 if (hour >= 20 or hour <= 5) else 0
    is_rush     = 1 if (6 <= hour <= 9 or 16 <= hour <= 20) else 0
    is_weekend  = 1 if day_name in ["Saturday","Sunday"] else 0
    is_dark        = 1 if light_cond == "Darkness" else 0
    dark_and_night = 1 if (is_dark == 1 and is_night == 1) else 0
    bad_weather    = 1 if weather in ["Raining","Foggy","Thunderstorm"] else 0
    speed_related  = 1 if crash_cause in ["Speeding","Loss of Control","Wrongful Overtaking"] else 0
    mech_failure   = 1 if crash_cause in ["Brake Failure","Mechanical Defect","Tyre Burst"] else 0
    multi_vehicle  = 1 if people_involved > 2 else 0

    veh_map = {
        "Bus / Minibus (Commercial)": "Bus",
        "Car / Saloon (Private)"    : "Car",
        "Motorcycle / Okada"        : "Motorcycle",
        "Truck / Lorry"             : "Truck/Lorry",
        "Tricycle (Keke NAPEP)"     : "Motorcycle",
        "Other"                     : "Other",
    }
    weather_map = {
        "Clear / Normal": "Normal","Partly Cloudy": "PARTLY CLOUDY",
        "Cloudy": "CLOUDY","Raining": "Raining",
        "Foggy": "Fog or mist","Thunderstorm": "Raining and Windy",
    }
    light_map   = {"Daylight": "Daylight","Darkness": "Darkness - lights lit"}
    cause_map   = {
        "Speeding": "Driving at high speed","Wrongful Overtaking": "Overtaking",
        "Loss of Control": "Overturning","Tyre Burst": "Other",
        "Brake Failure": "Other","Dangerous Driving": "Driving carelessly",
        "Mechanical Defect": "Other","Obstruction": "Other",
        "Overloading": "Overloading","Unknown / Other": "Unknown",
    }

    def safe_encode(le, val):
        return le.transform([val])[0] if val in le.classes_ else 0

    weather_enc = safe_encode(encoders["weather_cat"], weather_map.get(weather, "Normal"))
    light_enc   = safe_encode(encoders["light_cat"],   light_map.get(light_cond, "Daylight"))
    vehicle_enc = safe_encode(encoders["vehicle_cat"], veh_map.get(vehicle_type, "Other"))
    cause_enc   = safe_encode(encoders["cause_cat"],   cause_map.get(crash_cause, "Unknown"))

    row = {
        "hour": hour, "is_night": is_night, "is_rush_hour": is_rush,
        "is_weekend": is_weekend, "is_dark": is_dark,
        "dark_and_night": dark_and_night, "bad_weather": bad_weather,
        "high_casualty": 0, "multi_vehicle": multi_vehicle,
        "speed_related": speed_related, "casualty_count": people_involved,
        "num_killed": 0, "people_involved": people_involved,
        "wet_road": 1 if bad_weather else 0, "at_junction": 0,
        "impairment": 0, "overtaking": 1 if crash_cause == "Wrongful Overtaking" else 0,
        "heavy_vehicle": 1 if "Truck" in vehicle_type else 0,
        "two_wheeler": 1 if "Motorcycle" in vehicle_type or "Tricycle" in vehicle_type else 0,
        "is_festive": 0, "mech_failure": mech_failure,
        "weather_cat": weather_enc, "light_cat": light_enc,
        "vehicle_cat": vehicle_enc, "cause_cat": cause_enc,
    }
    X = np.array([[row[f] for f in features]], dtype=np.float32)
    return X


def apply_threshold(y_proba, threshold, class_idx=1):
    return np.where(
        y_proba[:, class_idx] >= threshold,
        class_idx,
        np.argmax(
            np.where(np.arange(y_proba.shape[1]) == class_idx, -np.inf, y_proba),
            axis=1,
        ),
    )


def run_prediction(ensemble, encoders, features, threshold,
                    hour, day_name, weather, light_cond,
                    vehicle_type, crash_cause, people_involved):
    X = build_feature_row(hour, day_name, weather, light_cond,
                           vehicle_type, crash_cause, people_involved, encoders, features)
    proba  = ensemble.predict_proba(X)[0]
    pred_i = apply_threshold(proba.reshape(1,-1), threshold, class_idx=1)[0]
    label  = INV_MAP[pred_i]
    level, icon, color, badge, pill_cls, advice = RISK_LEVELS[label]
    return {
        "label": label, "level": level, "icon": icon, "color": color,
        "badge": badge, "pill_cls": pill_cls, "advice": advice,
        "p_minor": float(proba[0]), "p_serious": float(proba[1]), "p_fatal": float(proba[2]),
    }


# ══════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#64748B", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)


def severity_donut():
    labels = ["Minor","Serious","Fatal"]
    values = [CRASH_STATS["minor"], CRASH_STATS["serious"], CRASH_STATS["fatal"]]
    colors = ["#00CC6A","#FF9500","#FF3355"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.7,
        marker=dict(colors=colors, line=dict(color="#080B10", width=4)),
        textinfo="label+percent",
        textfont=dict(family="Space Grotesk", size=11, color="#E2E8F0"),
        showlegend=False,
    ))
    fig.add_annotation(
        text=f"<b>{CRASH_STATS['total_all']:,}</b>",
        font=dict(size=24, family="Bebas Neue", color="#E2E8F0"),
        showarrow=False, y=0.05)
    fig.add_annotation(
        text="TOTAL", font=dict(size=10, family="JetBrains Mono", color="#64748B"),
        showarrow=False, y=-0.12)
    fig.update_layout(title=dict(text="All Data — Severity Split",
                                  font=dict(size=12, family="JetBrains Mono", color="#94A3B8")),
                       **PLOT_THEME)
    return fig


def hourly_crashes_chart():
    hours  = list(HOUR_COUNTS.keys())
    counts = list(HOUR_COUNTS.values())
    colors = ["#FF3355" if (h >= 20 or h <= 5) else
              "#FF9500" if (6 <= h <= 9 or 16 <= h <= 20) else
              "#00D4FF" for h in hours]
    fig = go.Figure(go.Bar(
        x=hours, y=counts,
        marker_color=colors, marker_line_width=0,
        hovertemplate="<b>%{x}:00</b><br>Crashes: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Crashes by Hour of Day — FRSC Osun 2025",
                    font=dict(size=12, family="JetBrains Mono", color="#94A3B8")),
        xaxis=dict(tickmode="array", tickvals=list(range(0,24,3)),
                    ticktext=[f"{h:02d}h" for h in range(0,24,3)],
                    gridcolor="#1A2030", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1A2030"),
        bargap=0.2,
        **PLOT_THEME,
    )
    for x0, x1 in [(20,24),(0,5)]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(255,51,85,0.06)", line_width=0)
    for x0, x1 in [(6,9),(16,20)]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(255,149,0,0.05)", line_width=0)
    return fig


def day_of_week_chart():
    days   = list(DAY_COUNTS.keys())
    counts = list(DAY_COUNTS.values())
    colors = ["#FF6B35" if d in ["Saturday","Sunday"] else "#00D4FF" for d in days]
    fig = go.Figure(go.Bar(
        x=days, y=counts, marker_color=colors, marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Crashes: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Crashes by Day of Week — FRSC Osun 2025",
                    font=dict(size=12, family="JetBrains Mono", color="#94A3B8")),
        xaxis=dict(gridcolor="#1A2030"),
        yaxis=dict(gridcolor="#1A2030"),
        bargap=0.25,
        **PLOT_THEME,
    )
    return fig


def route_danger_chart():
    route_data = [(r, d["fatal_pct"], d["total"])
                   for r, d in OSUN_ROUTES.items() if d["total"] >= 3]
    route_data.sort(key=lambda x: x[1], reverse=True)
    routes = [r[0] for r in route_data[:14]]
    fatals = [r[1] for r in route_data[:14]]
    totals = [r[2] for r in route_data[:14]]
    colors = ["#FF3355" if f >= 30 else "#FF9500" if f >= 15 else "#FFD60A" if f >= 5 else "#00CC6A"
               for f in fatals]
    fig = go.Figure(go.Bar(
        x=fatals[::-1], y=routes[::-1], orientation="h",
        marker_color=colors[::-1], marker_line_width=0,
        text=[f"  {f:.0f}%" for f in fatals[::-1]],
        textposition="outside",
        textfont=dict(size=10, color="#E2E8F0", family="JetBrains Mono"),
        customdata=totals[::-1],
        hovertemplate="<b>%{y}</b><br>Fatal rate: %{x:.1f}%<br>Total crashes: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Route Fatal Crash Rate — FRSC Osun 2025  (routes with ≥3 crashes)",
                    font=dict(size=12, family="JetBrains Mono", color="#94A3B8")),
        xaxis=dict(title="Fatal Crash %", gridcolor="#1A2030", range=[0,75]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
        height=500,
        **PLOT_THEME,
    )
    return fig


def prob_bars_html(p_minor, p_serious, p_fatal):
    bars = [("MINOR", p_minor, "#00CC6A"), ("SERIOUS", p_serious, "#FF9500"), ("FATAL", p_fatal, "#FF3355")]
    html = '<div class="prob-section">'
    for label, prob, color in bars:
        pct = prob * 100
        html += f"""
        <div class="prob-label-row">
          <span class="prob-lbl">{label}</span>
          <span class="prob-pct" style="color:{color};">{pct:.1f}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width:{pct:.1f}%;background:{color};"></div>
        </div>"""
    html += "</div>"
    return html


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <span class="c1">OSUN</span><span class="c2">·</span><span class="c3">CRASH</span>
    </div>
    <div class="sb-tagline">Traffic Safety Intelligence</div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Dashboard", "🔮  Risk Predictor", "🗺️  Route Analysis", "📊  Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;line-height:1.9;color:#64748B;">
    <span style="color:#94A3B8;font-family:'JetBrains Mono',mono;font-size:0.65rem;letter-spacing:0.1em;">DATA SOURCES</span><br>
    FRSC Osun Command — 2025<br>
    Kaggle Global Dataset<br>
    Gharsalli, 2024<br><br>
    <span style="color:#94A3B8;font-family:'JetBrains Mono',mono;font-size:0.65rem;letter-spacing:0.1em;">MODEL</span><br>
    HistGBM + RF Ensemble<br>
    Threshold-tuned · Serious recall<br><br>
    <span style="color:#94A3B8;font-family:'JetBrains Mono',mono;font-size:0.65rem;letter-spacing:0.1em;">COVERAGE</span><br>
    Osun State, Nigeria<br>
    298 FRSC · 8,509 total records
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════
ensemble, encoders, features, threshold, missing_files = load_model()
model_loaded = ensemble is not None

# ══════════════════════════════════════════════════════════════════════
# MOBILE AUTO-COLLAPSE
# ══════════════════════════════════════════════════════════════════════
components.html("""
<script>
const doc = window.parent.document;
if (window.parent.innerWidth < 992) {
    const sidebar = doc.querySelector('[data-testid="stSidebar"]');
    if (sidebar && sidebar.getBoundingClientRect().width > 0) {
        const btn = doc.querySelector('[data-testid="stSidebarCollapseButton"]') ||
                    doc.querySelector('[aria-label="Collapse sidebar"]') ||
                    doc.querySelector('[data-testid="stSidebar"] button');
        if (btn) btn.click();
    }
}
</script>""", height=0, width=0)


# ══════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════
if "Dashboard" in page:

    st.markdown("""
    <div class="hero-bar">
      <p class="hero-title">TRAFFIC SAFETY<br><span class="accent">INTELLIGENCE</span> <span class="accent2">SYS.</span></p>
      <p class="hero-sub">Osun State, Nigeria — Historical Crash Intelligence 2025</p>
      <div class="hero-badge">FRSC OSUN COMMAND · LIVE DATASET · 2025</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    cards = [
        (k1, "kpi-total", "TOTAL RECORDS", f"{CRASH_STATS['total_all']:,}", "#00D4FF", "Combined FRSC + Kaggle"),
        (k2, "kpi-fatal", "FATAL CRASHES", f"{CRASH_STATS['fatal']}", "#FF3355", f"{CRASH_STATS['fatal']/CRASH_STATS['total_all']*100:.1f}% of all records"),
        (k3, "kpi-serious", "SERIOUS CRASHES", f"{CRASH_STATS['serious']:,}", "#FF9500", f"{CRASH_STATS['serious']/CRASH_STATS['total_all']*100:.1f}% of all records"),
        (k4, "kpi-minor", "MINOR CRASHES", f"{CRASH_STATS['minor']:,}", "#00CC6A", f"{CRASH_STATS['minor']/CRASH_STATS['total_all']*100:.1f}% of all records"),
    ]
    for col, cls, tag, val, color, desc in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}" style="border-radius:10px;border:1px solid #1E2736;margin-bottom:1rem;">
              <div class="kpi-tag">{tag}</div>
              <div class="kpi-num" style="color:{color};">{val}</div>
              <div class="kpi-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(severity_donut(), use_container_width=True)
    with c2:
        st.plotly_chart(hourly_crashes_chart(), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(day_of_week_chart(), use_container_width=True)
    with c4:
        labels = ["Fatal","Serious","Minor"]
        values = [CRASH_STATS["fatal_frsc"], CRASH_STATS["serious_frsc"], CRASH_STATS["minor_frsc"]]
        colors = ["#FF3355","#FF9500","#00CC6A"]
        fig_frsc = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.7,
            marker=dict(colors=colors, line=dict(color="#080B10", width=4)),
            textinfo="label+percent",
            textfont=dict(family="Space Grotesk", size=11, color="#E2E8F0"),
            showlegend=False,
        ))
        fig_frsc.add_annotation(
            text=f"<b>{CRASH_STATS['total_frsc']}</b>",
            font=dict(size=24, family="Bebas Neue", color="#E2E8F0"),
            showarrow=False, y=0.05)
        fig_frsc.add_annotation(
            text="FRSC", font=dict(size=10, family="JetBrains Mono", color="#64748B"),
            showarrow=False, y=-0.12)
        fig_frsc.update_layout(
            title=dict(text="FRSC Osun Severity — 2025",
                        font=dict(size=12, family="JetBrains Mono", color="#94A3B8")),
            **PLOT_THEME)
        st.plotly_chart(fig_frsc, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <b>KEY FINDINGS FROM THE DATA</b><br>
    🔴 <b>15:00 – 17:00</b> is the single most dangerous window — 32 crashes recorded in the 15:00 hour alone across Osun State.<br>
    🟡 <b>Monday</b> records the highest crash count (54), followed by Friday (47) — consistent with work-week travel patterns.<br>
    🟠 <b>Ipetu – Ilesa</b> corridor has the highest fatal crash rate (38.5%) among routes with meaningful sample size.<br>
    🟢 FRSC Osun data shows <b>66.4% of crashes are Serious</b> — far higher than the global Kaggle average of 12.7%.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════
elif "Predictor" in page:

    st.markdown("""
    <div class="hero-bar">
      <p class="hero-title">CRASH RISK <span class="accent">PREDICTOR</span></p>
      <p class="hero-sub">Enter your trip details to receive a predicted crash severity score</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"⚠️ Model files not found: {', '.join(missing_files)}")
        st.info("Run the training pipeline first to generate model artefacts.")
        st.stop()

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="sec-head">TRIP DETAILS <div class="line"></div></div>', unsafe_allow_html=True)

        travel_time = st.slider(
            "Planned departure time",
            min_value=0, max_value=23, value=8, format="%d:00",
        )
        time_label = "🌙 Night window" if (travel_time >= 20 or travel_time <= 5) else \
                     "⚡ Rush hour" if (6 <= travel_time <= 9 or 16 <= travel_time <= 20) else "☀️ Daytime"
        st.caption(f"**{travel_time:02d}:00** — {time_label}")

        travel_day = st.selectbox("Day of travel",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        light_cond  = st.selectbox("Light conditions", ["Daylight","Darkness"])
        weather     = st.selectbox("Weather conditions", WEATHER_OPTIONS)

    with right:
        st.markdown('<div class="sec-head">VEHICLE & ROUTE <div class="line"></div></div>', unsafe_allow_html=True)

        vehicle_type = st.selectbox("Vehicle type", VEHICLE_TYPES)
        crash_cause  = st.selectbox("Primary risk / concern on this route", CRASH_CAUSES)
        people_count = st.number_input("Number of people in vehicle",
                                        min_value=1, max_value=50, value=2)
        route = st.selectbox("Route / Corridor (optional)",
                              ["Select a route..."] + list(OSUN_ROUTES.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1,3])
    with col_btn:
        predict_clicked = st.button("PREDICT CRASH RISK", use_container_width=True, type="primary")

    if predict_clicked:
        with st.spinner("Running prediction model..."):
            result = run_prediction(
                ensemble, encoders, features, threshold,
                travel_time, travel_day, weather, light_cond,
                vehicle_type, crash_cause, people_count,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        res_l, res_r = st.columns(2, gap="large")

        with res_l:
            st.markdown(f"""
            <div class="risk-wrap">
              <div class="risk-label-text">PREDICTED SEVERITY</div>
              <div style="font-size:3.5rem;margin:0.4rem 0;">{result['icon']}</div>
              <div class="risk-class" style="color:{result['color']};">{result['label'].upper()}</div>
              <div class="risk-badge-pill {result['pill_cls']}">{result['level']} RISK</div>
              <hr style="border-color:#1E2736;margin:1rem 0;">
              <div style="font-family:'JetBrains Mono',mono;font-size:0.65rem;letter-spacing:0.15em;
                          text-transform:uppercase;color:#64748B;margin-bottom:0.4rem;">
                CLASS PROBABILITIES
              </div>
              {prob_bars_html(result['p_minor'], result['p_serious'], result['p_fatal'])}
            </div>
            """, unsafe_allow_html=True)

        with res_r:
            st.markdown(f"""
            <div class="advice-wrap">
              <div class="advice-title">SAFETY ASSESSMENT</div>
              <div class="advice-text">{result['advice']}</div>
              <hr style="border-color:#1E2736;margin:1rem 0;">
              <div style="font-family:'JetBrains Mono',mono;font-size:0.65rem;letter-spacing:0.15em;
                          text-transform:uppercase;color:#64748B;margin-bottom:0.75rem;">
                TRIP PROFILE
              </div>
              <div class="trip-grid">
                <div class="trip-item"><div class="trip-item-key">TIME</div><div class="trip-item-val">{travel_time:02d}:00</div></div>
                <div class="trip-item"><div class="trip-item-key">DAY</div><div class="trip-item-val">{travel_day}</div></div>
                <div class="trip-item"><div class="trip-item-key">WEATHER</div><div class="trip-item-val">{weather}</div></div>
                <div class="trip-item"><div class="trip-item-key">LIGHT</div><div class="trip-item-val">{light_cond}</div></div>
                <div class="trip-item"><div class="trip-item-key">VEHICLE</div><div class="trip-item-val">{vehicle_type}</div></div>
                <div class="trip-item"><div class="trip-item-key">OCCUPANTS</div><div class="trip-item-val">{people_count}</div></div>
                <div class="trip-item" style="grid-column:1/-1;"><div class="trip-item-key">PRIMARY RISK</div><div class="trip-item-val">{crash_cause}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if route != "Select a route...":
            rdata = OSUN_ROUTES[route]
            fp    = rdata["fatal_pct"]
            clr   = "#FF3355" if fp >= 30 else "#FF9500" if fp >= 15 else "#FFD60A" if fp >= 5 else "#00CC6A"
            tag   = "⚠️ HIGH-DANGER CORRIDOR" if fp >= 30 else "🟡 Moderate history" if fp >= 10 else "🟢 Relatively low fatal rate"
            st.markdown(f"""
            <div class="info-box" style="border-left-color:{clr};margin-top:1rem;">
            <b>HISTORICAL PROFILE — {route}</b><br>
            This corridor has recorded <b>{rdata['total']} crashes</b> in the FRSC Osun 2025 dataset.
            Fatal crash rate: <b style="color:{clr};">{fp:.1f}%</b> &nbsp;·&nbsp; {tag}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: ROUTE ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif "Route" in page:

    st.markdown("""
    <div class="hero-bar">
      <p class="hero-title">ROUTE <span class="accent">ANALYSIS</span></p>
      <p class="hero-sub">Historical crash profile for Osun State transit corridors — FRSC 2025</p>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(route_danger_chart(), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">ALL ROUTES — RISK CLASSIFICATION <div class="line"></div></div>', unsafe_allow_html=True)

    sorted_routes = sorted(OSUN_ROUTES.items(), key=lambda x: x[1]["fatal_pct"], reverse=True)
    col1, col2 = st.columns(2)

    for i, (route_name, data) in enumerate(sorted_routes):
        fp = data["fatal_pct"]
        if fp >= 30:
            pill_cls, risk_text, color = "rp-fatal",    "CRITICAL", "#FF3355"
        elif fp >= 15:
            pill_cls, risk_text, color = "rp-serious",  "ELEVATED",  "#FF9500"
        elif fp >= 5:
            pill_cls, risk_text, color = "rp-moderate", "MODERATE",  "#FFD60A"
        else:
            pill_cls, risk_text, color = "rp-low",      "LOW",       "#00CC6A"

        pill_html = f"""
        <div class="route-pill {pill_cls}">
          <div>
            <div class="route-name">{route_name}</div>
            <div class="route-meta">{data['total']} crashes · {data['zone']} zone</div>
          </div>
          <div style="text-align:right;">
            <div class="route-pct" style="color:{color};">{fp:.0f}%</div>
            <div class="route-risk-tag" style="color:{color};">{risk_text}</div>
          </div>
        </div>"""
        if i % 2 == 0:
            with col1: st.markdown(pill_html, unsafe_allow_html=True)
        else:
            with col2: st.markdown(pill_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>NOTE ON ROUTE DATA</b><br>
    Fatal crash percentages are computed from FRSC Osun 2025 records only (298 crash records).
    Routes with fewer than 3 crashes should be interpreted with caution due to small sample size.
    The most statistically reliable corridors are <b>Gbongan–Ibadan</b> (22 crashes),
    <b>Ilesa–Ipetu</b> (22), and <b>Ilesa–Ife</b> (16).
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════
elif "Insights" in page:

    st.markdown("""
    <div class="hero-bar">
      <p class="hero-title">MODEL <span class="accent">INSIGHTS</span></p>
      <p class="hero-sub">Technical transparency — how the prediction model works</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">MODEL ARCHITECTURE <div class="line"></div></div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    for col, title, color, body in [
        (a1, "HistGradientBoosting", "#00D4FF",
         "Sequential gradient boosting that corrects errors from previous trees. Better at learning complex non-linear patterns. Weighted 2× in the final ensemble for higher influence."),
        (a2, "Random Forest", "#FF6B35",
         "Parallel bagging of 600 decision trees. Reduces variance through diversity. Uses balanced class weights to counteract the class imbalance in training data."),
        (a3, "Soft-Vote Ensemble", "#00FF88",
         "Averages probability outputs from both models. Final class = highest average probability after Serious-class threshold tuning to boost recall on non-minor crashes."),
    ]:
        with col:
            st.markdown(f"""
            <div class="model-card">
              <div class="model-card-title" style="color:{color};">{title}</div>
              <div class="model-card-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">TRAINING DATA & TECHNIQUES <div class="line"></div></div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)

    with d1:
        data_rows = {
            "FRSC Osun 2025": "298 crash records (56 Fatal, 198 Serious, 44 Minor)",
            "Kaggle Global":  "8,210 crash records (82 Fatal, 1,046 Serious, 7,082 Minor)",
            "Combined total": "8,508 records after merging",
            "After SMOTE":    "Serious ×4 and Fatal ×6 oversampling on training split",
        }
        for k, v in data_rows.items():
            st.markdown(f'<div class="data-row"><span class="dr-key">{k}</span><span class="dr-val">{v}</span></div>',
                        unsafe_allow_html=True)

    with d2:
        technique_rows = {
            "Imbalance handling": "Targeted SMOTE + amplified class weights (Serious:6, Fatal:20)",
            "Threshold tuning":   "Serious class threshold optimised on validation set (recall ≥ 0.60)",
            "Validation split":   "Stratified 15% held-out validation set",
            "Leakage prevention": "num_killed and num_injured excluded — derived from target label",
        }
        for k, v in technique_rows.items():
            st.markdown(f'<div class="data-row"><span class="dr-key">{k}</span><span class="dr-val">{v}</span></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">FEATURES USED FOR PREDICTION <div class="line"></div></div>', unsafe_allow_html=True)

    feat_groups = {
        "TEMPORAL":     ["hour","is_night","is_rush_hour","is_weekend","is_festive"],
        "ENVIRONMENTAL":["is_dark","dark_and_night","bad_weather","wet_road"],
        "CRASH TYPE":   ["speed_related","mech_failure","overtaking","impairment","at_junction"],
        "VEHICLE":      ["vehicle_cat","heavy_vehicle","two_wheeler"],
        "MAGNITUDE":    ["casualty_count","people_involved","multi_vehicle","high_casualty"],
        "CATEGORICAL":  ["weather_cat","light_cat","cause_cat"],
    }
    fg1, fg2, fg3 = st.columns(3)
    cols_cycle = [fg1, fg2, fg3]
    for i, (group, feats) in enumerate(feat_groups.items()):
        with cols_cycle[i % 3]:
            html = f'<div style="margin-bottom:1.25rem;"><div class="kpi-tag" style="margin-bottom:0.5rem;">{group}</div>'
            for f in feats:
                html += f'<div style="font-family:JetBrains Mono,mono;font-size:0.74rem;padding:0.25rem 0;color:#94A3B8;border-bottom:1px solid #1A2030;">· {f}</div>'
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="border-left-color:#FF9500;">
    <b>LIMITATIONS & DISCLAIMER</b><br><br>
    · <b>Small local dataset:</b> FRSC Osun 2025 contains 298 records — statistical patterns for individual routes may be unreliable.<br>
    · <b>Feature signal:</b> Without outcome data the model relies on contextual factors with weak-to-moderate predictive power. Accuracy on unseen data is 60–70%.<br>
    · <b>Not a substitute for official guidance:</b> This tool is an academic research prototype. Do not use it as the sole basis for travel decisions.<br>
    · <b>Model scope:</b> Trained on Osun State data + global patterns. Predictions outside Osun corridors should be treated with extra caution.<br><br>
    <i>Developed as a final-year undergraduate project. For official road safety advice, consult the Federal Road Safety Corps (FRSC).</i>
    </div>
    """, unsafe_allow_html=True)
