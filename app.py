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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Root variables ── */
:root {
    --bg:         #0D0F14;
    --surface:    #151820;
    --surface2:   #1C2030;
    --border:     #252A3A;
    --accent:     #FF5C35;
    --accent2:    #FFB547;
    --green:      #2ECC71;
    --red:        #E74C3C;
    --orange:     #FF8C00;
    --text:       #E8EAF0;
    --muted:      #7B829A;
    --font-head:  'Syne', sans-serif;
    --font-body:  'DM Sans', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg);
    color: var(--text);
}

/* ── Remove Streamlit chrome but KEEP mobile sidebar toggle ── */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stHeader"] { padding-top: 0; }
/* Hide header elements EXCEPT the sidebar toggle button */
[data-testid="stHeader"] > div { display: none; }
[data-testid="collapsedControl"] { 
    visibility: visible !important; 
    display: flex !important;
    color: var(--accent);
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-top: 10px;
    margin-left: 10px;
    z-index: 999999;
}
.block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: var(--font-head); }

/* ── Metric cards ── */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-fatal::before   { background: var(--red); }
.kpi-serious::before { background: var(--orange); }
.kpi-minor::before   { background: var(--green); }
.kpi-total::before   { background: var(--accent); }

.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: var(--font-head);
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-sub { font-size: 0.78rem; color: var(--muted); }

/* ── Section headers ── */
.section-header {
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: var(--text);
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* ── Risk gauge container ── */
.risk-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    font-family: var(--font-head);
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}
.risk-LOW      { background: rgba(46,204,113,0.15); color: #2ECC71; border: 2px solid #2ECC71; }
.risk-MODERATE { background: rgba(255,180,71,0.15); color: #FFB547; border: 2px solid #FFB547; }
.risk-HIGH     { background: rgba(255,92,53,0.15);  color: #FF5C35; border: 2px solid #FF5C35; }
.risk-CRITICAL { background: rgba(231,76,60,0.2);   color: #E74C3C; border: 2px solid #E74C3C; }

/* ── Probability bars ── */
.prob-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.5rem 0; }
.prob-label { font-size: 0.82rem; font-weight: 500; width: 70px; color: var(--muted); text-align: right; }
.prob-track { flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
.prob-fill   { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
.prob-pct    { font-size: 0.82rem; font-weight: 600; width: 42px; }

/* ── Info panel ── */
.info-panel {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-size: 0.85rem;
    line-height: 1.6;
    margin-top: 1rem;
}

/* ── Route pill ── */
.route-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.route-pill-fatal    { border-left: 4px solid var(--red); }
.route-pill-serious  { border-left: 4px solid var(--orange); }
.route-pill-moderate { border-left: 4px solid var(--accent2); }
.route-pill-low      { border-left: 4px solid var(--green); }

/* ── Inputs ── */
.stSelectbox > div, .stSlider > div { background: var(--surface2); }
div[data-testid="stSelectbox"] > div { border-color: var(--border) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"]      { font-family: var(--font-head); font-weight: 600; color: var(--muted); }
.stTabs [aria-selected="true"]    { color: var(--text) !important; border-bottom: 2px solid var(--accent) !important; }

/* ── Divider ── */
hr { border-color: var(--border); }

/* ── Logo / header area ── */
.app-logo {
    font-family: var(--font-head);
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
}
.app-logo span { color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS & STATIC DATA
# ══════════════════════════════════════════════════════════════════════

# Key Osun State routes with their historical crash danger profile
OSUN_ROUTES = {
    "Gbongan – Ibadan":        {"fatal_pct": 18.2, "total": 22, "zone": "Gbongan"},
    "Ilesa – Ipetu":           {"fatal_pct":  9.1, "total": 22, "zone": "Ilesa"},
    "Ilesa – Ife":             {"fatal_pct": 18.8, "total": 16, "zone": "Ilesa"},
    "Ipetu – Ilesa":           {"fatal_pct": 38.5, "total": 13, "zone": "Ipetu"},
    "Ife – Ilesa":             {"fatal_pct": 25.0, "total": 12, "zone": "Ife"},
    "Ife – Ondo":              {"fatal_pct": 11.1, "total":  9, "zone": "Ife"},
    "Wasinmi – Asejire":       {"fatal_pct": 33.3, "total":  9, "zone": "Wasinmi"},
    "Osogbo – Sekona":         {"fatal_pct":  0.0, "total":  8, "zone": "Osogbo"},
    "Gbongan – Ife":           {"fatal_pct": 14.3, "total":  7, "zone": "Gbongan"},
    "Ikirun – Osogbo":         {"fatal_pct":  0.0, "total":  6, "zone": "Ikirun"},
    "Ikirun – Ilaodo":         {"fatal_pct": 16.7, "total":  6, "zone": "Ikirun"},
    "Ipetu – Ilesa (Alt)":     {"fatal_pct": 50.0, "total":  6, "zone": "Ipetu"},
    "Ikire – Gbongan":         {"fatal_pct": 20.0, "total":  5, "zone": "Ikire"},
    "Ile Ife – Ondo Expressway":{"fatal_pct":100.0,"total":  1, "zone": "Ife"},
    "Ipetu – Akure":           {"fatal_pct":  0.0, "total":  5, "zone": "Ipetu"},
    "Ife – Gbongan":           {"fatal_pct":  0.0, "total":  4, "zone": "Ife"},
    "Ipetu – Ijesa":           {"fatal_pct": 25.0, "total":  4, "zone": "Ipetu"},
    "Iwo – Ibadan":            {"fatal_pct":  0.0, "total":  3, "zone": "Iwo"},
    "Ilobu – Osogbo":          {"fatal_pct":  0.0, "total":  3, "zone": "Ilobu"},
    "Osogbo – Gbongan":        {"fatal_pct": 16.7, "total":  6, "zone": "Osogbo"},
    "Osogbo – Ikirun":         {"fatal_pct": 33.3, "total":  3, "zone": "Osogbo"},
    "Gbongan – Osogbo":        {"fatal_pct":  0.0, "total":  1, "zone": "Gbongan"},
    "Ibadan – Ife Highway":    {"fatal_pct":  0.0, "total":  1, "zone": "Ibadan"},
    "Ife – Sekona":            {"fatal_pct": 50.0, "total":  2, "zone": "Ife"},
    "Asejire – Wasinmi":       {"fatal_pct": 33.3, "total":  3, "zone": "Asejire"},
}

# Crash statistics for dashboard
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

# Vehicle types for prediction form
VEHICLE_TYPES = [
    "Bus / Minibus (Commercial)",
    "Car / Saloon (Private)",
    "Motorcycle / Okada",
    "Truck / Lorry",
    "Tricycle (Keke NAPEP)",
    "Other",
]

# Crash causes for prediction form
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

WEATHER_OPTIONS   = ["Clear / Normal", "Partly Cloudy", "Cloudy", "Raining", "Foggy", "Thunderstorm"]
LABEL_MAP         = {"Minor":0, "Serious":1, "Fatal":2}
INV_MAP           = {0:"Minor", 1:"Serious", 2:"Fatal"}

RISK_LEVELS = {
    "Minor":   ("LOW",      "🟢", "#2ECC71", "LOW",      "The predicted crash severity is Minor. Road conditions appear manageable. Exercise standard caution."),
    "Serious": ("MODERATE", "🟡", "#FFB547", "MODERATE", "The predicted severity is Serious. Exercise heightened caution — slow down, maintain distance, and stay alert to road conditions."),
    "Fatal":   ("CRITICAL", "🔴", "#E74C3C", "CRITICAL", "CRITICAL risk detected. Conditions associated with fatal crashes are present. Consider delaying travel or choosing an alternative route."),
}


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════
Model_File_ID = "1b_8QGQtngUr1hHjMiE1AdEvDIhs8G9fx"
@st.cache_resource
def load_model():
    """Load the trained ensemble model and artefacts."""
    model_path     = "ensemble_recall_fix.pkl"
    
    # Check if the model file is already downloaded
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive... (This may take a minute)"):
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
    """
    Build the feature vector that the ensemble model expects.
    Maps user-friendly form inputs to the exact numeric encoding
    the model was trained on.
    """
    # ── Derived temporal features ────────────────────────────────────
    is_night    = 1 if (hour >= 20 or hour <= 5)               else 0
    is_rush     = 1 if (6 <= hour <= 9 or 16 <= hour <= 20)    else 0
    is_weekend  = 1 if day_name in ["Saturday","Sunday"]        else 0
    is_festive  = 0   # cannot know at prediction time — set conservative default

    # ── Environmental features ───────────────────────────────────────
    is_dark        = 1 if light_cond == "Darkness"            else 0
    dark_and_night = 1 if (is_dark == 1 and is_night == 1)    else 0
    bad_weather    = 1 if weather in ["Raining","Foggy","Thunderstorm"] else 0

    # ── Crash characteristics ────────────────────────────────────────
    speed_related  = 1 if crash_cause in ["Speeding","Loss of Control","Wrongful Overtaking"] else 0
    mech_failure   = 1 if crash_cause in ["Brake Failure","Mechanical Defect","Tyre Burst"]   else 0
    multi_vehicle  = 1 if people_involved > 2 else 0
    high_casualty  = 0   # not known pre-crash — set to 0

    # ── Vehicle type mapping ─────────────────────────────────────────
    veh_map = {
        "Bus / Minibus (Commercial)"   : "Bus",
        "Car / Saloon (Private)"       : "Car",
        "Motorcycle / Okada"           : "Motorcycle",
        "Truck / Lorry"                : "Truck/Lorry",
        "Tricycle (Keke NAPEP)"        : "Motorcycle",
        "Other"                        : "Other",
    }
    vehicle_cat_str = veh_map.get(vehicle_type, "Other")

    # ── Weather category mapping ─────────────────────────────────────
    weather_map = {
        "Clear / Normal" : "Normal",
        "Partly Cloudy"  : "PARTLY CLOUDY",
        "Cloudy"         : "CLOUDY",
        "Raining"        : "Raining",
        "Foggy"          : "Fog or mist",
        "Thunderstorm"   : "Raining and Windy",
    }
    weather_cat_str = weather_map.get(weather, "Normal")

    # ── Light category mapping ───────────────────────────────────────
    light_map = {
        "Daylight" : "Daylight",
        "Darkness" : "Darkness - lights lit",
    }
    light_cat_str = light_map.get(light_cond, "Daylight")

    # ── Cause category mapping ───────────────────────────────────────
    cause_map = {
        "Speeding"             : "Driving at high speed",
        "Wrongful Overtaking"  : "Overtaking",
        "Loss of Control"      : "Overturning",
        "Tyre Burst"           : "Other",
        "Brake Failure"        : "Other",
        "Dangerous Driving"    : "Driving carelessly",
        "Mechanical Defect"    : "Other",
        "Obstruction"          : "Other",
        "Overloading"          : "Overloading",
        "Unknown / Other"      : "Unknown",
    }
    cause_cat_str = cause_map.get(crash_cause, "Unknown")

    # ── Encode categoricals using saved LabelEncoders ────────────────
    def safe_encode(le, val):
        return le.transform([val])[0] if val in le.classes_ else 0

    weather_enc = safe_encode(encoders["weather_cat"], weather_cat_str)
    light_enc   = safe_encode(encoders["light_cat"],   light_cat_str)
    vehicle_enc = safe_encode(encoders["vehicle_cat"], vehicle_cat_str)
    cause_enc   = safe_encode(encoders["cause_cat"],   cause_cat_str)

    # ── Assemble the feature dict ────────────────────────────────────
    row = {
        "hour"          : hour,
        "is_night"      : is_night,
        "is_rush_hour"  : is_rush,
        "is_weekend"    : is_weekend,
        "is_dark"       : is_dark,
        "dark_and_night": dark_and_night,
        "bad_weather"   : bad_weather,
        "high_casualty" : high_casualty,
        "multi_vehicle" : multi_vehicle,
        "speed_related" : speed_related,
        "casualty_count": people_involved,
        "num_killed"    : 0,
        "people_involved": people_involved,
        "wet_road"      : 1 if bad_weather else 0,
        "at_junction"   : 0,
        "impairment"    : 0,
        "overtaking"    : 1 if crash_cause == "Wrongful Overtaking" else 0,
        "heavy_vehicle" : 1 if "Truck" in vehicle_type else 0,
        "two_wheeler"   : 1 if "Motorcycle" in vehicle_type or "Tricycle" in vehicle_type else 0,
        "is_festive"    : is_festive,
        "mech_failure"  : mech_failure,
        "weather_cat"   : weather_enc,
        "light_cat"     : light_enc,
        "vehicle_cat"   : vehicle_enc,
        "cause_cat"     : cause_enc,
    }

    X = np.array([[row[f] for f in features]], dtype=np.float32)
    return X


def apply_threshold(y_proba, threshold, class_idx=1):
    """Apply the Serious-class threshold tuned during training."""
    return np.where(
        y_proba[:, class_idx] >= threshold,
        class_idx,
        np.argmax(
            np.where(
                np.arange(y_proba.shape[1]) == class_idx,
                -np.inf,
                y_proba,
            ),
            axis=1,
        ),
    )


def run_prediction(ensemble, encoders, features, threshold,
                    hour, day_name, weather, light_cond,
                    vehicle_type, crash_cause, people_involved):
    """Run the full prediction pipeline and return result dict."""
    X = build_feature_row(hour, day_name, weather, light_cond,
                           vehicle_type, crash_cause, people_involved,
                           encoders, features)
    proba  = ensemble.predict_proba(X)[0]
    pred_i = apply_threshold(proba.reshape(1,-1), threshold, class_idx=1)[0]
    label  = INV_MAP[pred_i]
    level, icon, color, badge, advice = RISK_LEVELS[label]
    return {
        "label"    : label,
        "level"    : level,
        "icon"     : icon,
        "color"    : color,
        "badge"    : badge,
        "advice"   : advice,
        "p_minor"  : float(proba[0]),
        "p_serious": float(proba[1]),
        "p_fatal"  : float(proba[2]),
    }


# ══════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#7B829A", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)

def severity_donut():
    labels = ["Minor","Serious","Fatal"]
    values = [CRASH_STATS["minor"], CRASH_STATS["serious"], CRASH_STATS["fatal"]]
    colors = ["#2ECC71","#FF8C00","#E74C3C"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0D0F14", width=3)),
        textinfo="label+percent",
        textfont=dict(family="Syne, sans-serif", size=12, color="#E8EAF0"),
        showlegend=False,
    ))
    fig.add_annotation(text=f"<b>{CRASH_STATS['total_all']:,}</b>",
                        font=dict(size=22, family="Syne", color="#E8EAF0"),
                        showarrow=False)
    fig.update_layout(title=dict(text="Severity Distribution — All Data",
                                  font=dict(size=13, family="Syne")),
                       **PLOT_THEME)
    return fig


def hourly_crashes_chart():
    hours  = list(HOUR_COUNTS.keys())
    counts = list(HOUR_COUNTS.values())
    colors = ["#E74C3C" if (h >= 20 or h <= 5) else
              "#FFB547" if (6 <= h <= 9 or 16 <= h <= 20) else
              "#4A90D9" for h in hours]
    fig = go.Figure(go.Bar(
        x=hours, y=counts,
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(
        title=dict(text="Crashes by Hour of Day (FRSC Osun)",
                    font=dict(size=13, family="Syne")),
        xaxis=dict(tickmode="array",
                    tickvals=list(range(0,24,2)),
                    ticktext=[f"{h:02d}:00" for h in range(0,24,2)],
                    gridcolor="#1C2030"),
        yaxis=dict(gridcolor="#1C2030"),
        **PLOT_THEME,
    )
    fig.add_vrect(x0=20, x1=24, fillcolor="rgba(231,76,60,0.08)", line_width=0)
    fig.add_vrect(x0=0,  x1=5,  fillcolor="rgba(231,76,60,0.08)", line_width=0)
    fig.add_vrect(x0=6,  x1=9,  fillcolor="rgba(255,181,71,0.06)", line_width=0)
    fig.add_vrect(x0=16, x1=20, fillcolor="rgba(255,181,71,0.06)", line_width=0)
    return fig


def day_of_week_chart():
    days   = list(DAY_COUNTS.keys())
    counts = list(DAY_COUNTS.values())
    colors = ["#FF5C35" if d in ["Saturday","Sunday"] else "#4A90D9" for d in days]
    fig = go.Figure(go.Bar(
        x=days, y=counts,
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(
        title=dict(text="Crashes by Day of Week (FRSC Osun)",
                    font=dict(size=13, family="Syne")),
        xaxis=dict(gridcolor="#1C2030"),
        yaxis=dict(gridcolor="#1C2030"),
        **PLOT_THEME,
    )
    return fig


def route_danger_chart():
    route_data = [(r, d["fatal_pct"], d["total"])
                   for r, d in OSUN_ROUTES.items() if d["total"] >= 3]
    route_data.sort(key=lambda x: x[1], reverse=True)
    routes  = [r[0] for r in route_data[:15]]
    fatals  = [r[1] for r in route_data[:15]]
    totals  = [r[2] for r in route_data[:15]]
    colors  = ["#E74C3C" if f >= 30 else "#FF8C00" if f >= 15 else "#FFB547" if f >= 5 else "#2ECC71"
                for f in fatals]
    fig = go.Figure(go.Bar(
        x=fatals[::-1], y=routes[::-1],
        orientation="h",
        marker_color=colors[::-1],
        marker_line_width=0,
        text=[f"{f:.0f}%" for f in fatals[::-1]],
        textposition="outside",
        textfont=dict(size=11, color="#E8EAF0"),
        customdata=totals[::-1],
        hovertemplate="<b>%{y}</b><br>Fatal rate: %{x:.1f}%<br>Total crashes: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Route Fatal Crash Rate — FRSC Osun (routes with ≥3 crashes)",
                    font=dict(size=13, family="Syne")),
        xaxis=dict(title="Fatal Crash % ", gridcolor="#1C2030", range=[0, 70]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=480,
        **PLOT_THEME,
    )
    return fig


def probability_bar(p_minor, p_serious, p_fatal):
    """Render HTML probability bars."""
    bars = [
        ("Minor",   p_minor,   "#2ECC71"),
        ("Serious", p_serious, "#FFB547"),
        ("Fatal",   p_fatal,   "#E74C3C"),
    ]
    html = '<div style="margin-top:1rem;">'
    for label, prob, color in bars:
        pct = prob * 100
        html += f"""
        <div class="prob-row">
          <div class="prob-label">{label}</div>
          <div class="prob-track">
            <div class="prob-fill" style="width:{pct:.1f}%;background:{color};"></div>
          </div>
          <div class="prob-pct" style="color:{color};">{pct:.1f}%</div>
        </div>"""
    html += "</div>"
    return html


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="app-logo">Osun<span>Crash</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#7B829A;margin-bottom:1.5rem;">Traffic Safety Intelligence System</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Dashboard", "🔮  Risk Predictor", "🗺️  Route Analysis", "📊  Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#7B829A;line-height:1.7;">
    <b style="color:#E8EAF0;">Data Sources</b><br>
    FRSC Osun Command — 2025<br>
    Kaggle Global Dataset<br>
    (Gharsalli, 2024)<br><br>
    <b style="color:#E8EAF0;">Model</b><br>
    HistGBM + RF Ensemble<br>
    Threshold-tuned for Serious recall<br><br>
    <b style="color:#E8EAF0;">Coverage</b><br>
    Osun State, Nigeria<br>
    298 local crash records<br>
    8,509 combined records
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════

ensemble, encoders, features, threshold, missing_files = load_model()
model_loaded = ensemble is not None


# ══════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════

if "Dashboard" in page:
    st.markdown('<h1 style="font-family:Syne;font-size:2rem;font-weight:800;margin-bottom:0.25rem;">Traffic Safety Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7B829A;font-size:0.9rem;margin-bottom:2rem;">Osun State, Nigeria — Historical Crash Intelligence (2025)</p>', unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card kpi-total">
          <div class="kpi-label">Total Records</div>
          <div class="kpi-value" style="color:#FF5C35;">{CRASH_STATS['total_all']:,}</div>
          <div class="kpi-sub">Combined FRSC + Kaggle dataset</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card kpi-fatal">
          <div class="kpi-label">Fatal Crashes</div>
          <div class="kpi-value" style="color:#E74C3C;">{CRASH_STATS['fatal']}</div>
          <div class="kpi-sub">{CRASH_STATS['fatal']/CRASH_STATS['total_all']*100:.1f}% of all records</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card kpi-serious">
          <div class="kpi-label">Serious Crashes</div>
          <div class="kpi-value" style="color:#FF8C00;">{CRASH_STATS['serious']:,}</div>
          <div class="kpi-sub">{CRASH_STATS['serious']/CRASH_STATS['total_all']*100:.1f}% of all records</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card kpi-minor">
          <div class="kpi-label">Minor Crashes</div>
          <div class="kpi-value" style="color:#2ECC71;">{CRASH_STATS['minor']:,}</div>
          <div class="kpi-sub">{CRASH_STATS['minor']/CRASH_STATS['total_all']*100:.1f}% of all records</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────────────────────────────
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(severity_donut(), use_container_width=True)
    with c2:
        st.plotly_chart(hourly_crashes_chart(), use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────────────
    c3, c4 = st.columns([1, 1])
    with c3:
        st.plotly_chart(day_of_week_chart(), use_container_width=True)
    with c4:
        # Osun-specific FRSC severity donut
        labels = ["Fatal","Serious","Minor"]
        values = [CRASH_STATS["fatal_frsc"], CRASH_STATS["serious_frsc"], CRASH_STATS["minor_frsc"]]
        colors = ["#E74C3C","#FF8C00","#2ECC71"]
        fig_frsc = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.65,
            marker=dict(colors=colors, line=dict(color="#0D0F14", width=3)),
            textinfo="label+percent",
            textfont=dict(family="Syne", size=12, color="#E8EAF0"),
            showlegend=False,
        ))
        fig_frsc.add_annotation(
            text=f"<b>{CRASH_STATS['total_frsc']}</b>",
            font=dict(size=22, family="Syne", color="#E8EAF0"),
            showarrow=False)
        fig_frsc.update_layout(
            title=dict(text="FRSC Osun Severity (2025)",
                        font=dict(size=13, family="Syne")),
            **PLOT_THEME)
        st.plotly_chart(fig_frsc, use_container_width=True)

    # ── Key insights box ─────────────────────────────────────────────
    st.markdown("""
    <div class="info-panel">
    <b>Key Findings from the Data</b><br>
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
    st.markdown('<h1 style="font-family:Syne;font-size:2rem;font-weight:800;margin-bottom:0.25rem;">Crash Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7B829A;font-size:0.9rem;margin-bottom:2rem;">Enter your trip details to receive a predicted crash severity score.</p>', unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"⚠️ Model files not found. Please ensure these files are in the working directory: {', '.join(missing_files)}")
        st.info("Run the training pipeline first (`ensemble_staged.py`) to generate the model artefacts.")
        st.stop()

    # ── Input form ───────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-header">🕐 Trip Details</div>', unsafe_allow_html=True)

        travel_time = st.slider(
            "Planned departure time",
            min_value=0, max_value=23, value=8,
            format="%d:00",
            help="Hour of day you plan to travel (0 = midnight, 12 = noon)"
        )
        st.caption(f"Selected: **{travel_time:02d}:00** {'🌙 Night' if travel_time >= 20 or travel_time <= 5 else '⚡ Rush hour' if (6 <= travel_time <= 9 or 16 <= travel_time <= 20) else '☀️ Daytime'}")

        travel_day = st.selectbox(
            "Day of travel",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
            index=0,
        )

        light_cond = st.selectbox(
            "Light conditions",
            ["Daylight", "Darkness"],
            help="Expected lighting on the road at time of travel",
        )

        weather = st.selectbox(
            "Weather conditions",
            WEATHER_OPTIONS,
            help="Expected weather during your journey",
        )

    with right:
        st.markdown('<div class="section-header">🚗 Vehicle & Route Details</div>', unsafe_allow_html=True)

        vehicle_type = st.selectbox(
            "Vehicle type",
            VEHICLE_TYPES,
        )

        crash_cause = st.selectbox(
            "Primary risk / concern on this route",
            CRASH_CAUSES,
            help="Select the main risk factor you are concerned about on this route",
        )

        people_count = st.number_input(
            "Number of people in the vehicle",
            min_value=1, max_value=50, value=2,
            help="Total number of occupants including driver",
        )

        route = st.selectbox(
            "Route / Corridor (optional — for reference)",
            ["Select a route..."] + list(OSUN_ROUTES.keys()),
            help="Select your Osun State route to see its historical crash profile",
        )

    # ── Predict button ────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1,3])
    with col_btn:
        predict_clicked = st.button("🔮  Predict Crash Risk", use_container_width=True, type="primary")

    if predict_clicked:
        with st.spinner("Running prediction model..."):
            result = run_prediction(
                ensemble, encoders, features, threshold,
                travel_time, travel_day, weather, light_cond,
                vehicle_type, crash_cause, people_count,
            )

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        res_left, res_right = st.columns([1,1], gap="large")

        with res_left:
            st.markdown(f"""
            <div class="risk-container">
              <div style="font-size:0.75rem;font-weight:600;letter-spacing:0.12em;
                          text-transform:uppercase;color:#7B829A;margin-bottom:0.75rem;">
                Predicted Severity
              </div>
              <div style="font-size:3.5rem;margin-bottom:0.5rem;">{result['icon']}</div>
              <div class="risk-badge risk-{result['badge']}">{result['level']} RISK</div>
              <div style="font-family:Syne;font-size:1.5rem;font-weight:700;
                          color:{result['color']};margin-bottom:1rem;">
                {result['label']}
              </div>
              <hr style="border-color:#252A3A;margin:1rem 0;">
              <div style="font-size:0.82rem;font-weight:600;letter-spacing:0.08em;
                          text-transform:uppercase;color:#7B829A;margin-bottom:0.5rem;">
                Class Probabilities
              </div>
              {probability_bar(result['p_minor'], result['p_serious'], result['p_fatal'])}
            </div>
            """, unsafe_allow_html=True)

        with res_right:
            st.markdown(f"""
            <div class="risk-container" style="text-align:left;height:100%;">
              <div style="font-size:0.75rem;font-weight:600;letter-spacing:0.12em;
                          text-transform:uppercase;color:#7B829A;margin-bottom:1rem;">
                Safety Advice
              </div>
              <div style="font-size:0.92rem;line-height:1.8;color:#E8EAF0;margin-bottom:1.5rem;">
                {result['advice']}
              </div>
              <hr style="border-color:#252A3A;margin:1rem 0;">
              <div style="font-size:0.78rem;font-weight:600;text-transform:uppercase;
                          letter-spacing:0.1em;color:#7B829A;margin-bottom:0.75rem;">
                Your Trip Profile
              </div>
              <div style="font-size:0.82rem;line-height:2;color:#E8EAF0;">
                🕐 <b>Time:</b> {travel_time:02d}:00 &nbsp;|&nbsp;
                📅 <b>Day:</b> {travel_day}<br>
                🌤 <b>Weather:</b> {weather} &nbsp;|&nbsp;
                💡 <b>Light:</b> {light_cond}<br>
                🚗 <b>Vehicle:</b> {vehicle_type}<br>
                ⚠️ <b>Risk factor:</b> {crash_cause}<br>
                👥 <b>Occupants:</b> {people_count}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Show route historical data if selected
        if route != "Select a route...":
            rdata = OSUN_ROUTES[route]
            fp = rdata["fatal_pct"]
            color_r = "#E74C3C" if fp >= 30 else "#FF8C00" if fp >= 15 else "#FFB547" if fp >= 5 else "#2ECC71"
            st.markdown(f"""
            <div class="info-panel" style="margin-top:1rem;border-left-color:{color_r};">
            <b>Historical Profile — {route}</b><br>
            This corridor has recorded <b>{rdata['total']} crashes</b> in the FRSC Osun 2025 dataset.
            Fatal crash rate: <b style="color:{color_r};">{fp:.1f}%</b>.
            {'⚠️ This is a <b>HIGH-DANGER</b> corridor. Extra caution is strongly advised.' if fp >= 30
             else '🟡 This route has a moderate crash history.' if fp >= 10
             else '🟢 This route has a relatively low fatal crash rate.'}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: ROUTE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

elif "Route" in page:
    st.markdown('<h1 style="font-family:Syne;font-size:2rem;font-weight:800;margin-bottom:0.25rem;">Route Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7B829A;font-size:0.9rem;margin-bottom:2rem;">Historical crash profile for Osun State transit corridors — FRSC 2025 data.</p>', unsafe_allow_html=True)

    st.plotly_chart(route_danger_chart(), use_container_width=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📍 All Routes — Risk Classification</div>', unsafe_allow_html=True)

    # Sort routes by danger
    sorted_routes = sorted(OSUN_ROUTES.items(), key=lambda x: x[1]["fatal_pct"], reverse=True)

    col1, col2 = st.columns(2)
    for i, (route_name, data) in enumerate(sorted_routes):
        fp = data["fatal_pct"]
        if fp >= 30:
            pill_class, risk_text = "route-pill-fatal",    "🔴 HIGH DANGER"
        elif fp >= 15:
            pill_class, risk_text = "route-pill-serious",  "🟠 ELEVATED"
        elif fp >= 5:
            pill_class, risk_text = "route-pill-moderate", "🟡 MODERATE"
        else:
            pill_class, risk_text = "route-pill-low",      "🟢 LOW"

        pill_html = f"""
        <div class="route-pill {pill_class}">
          <div>
            <div style="font-weight:600;font-size:0.87rem;">{route_name}</div>
            <div style="font-size:0.75rem;color:#7B829A;">{data['total']} crashes recorded</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.75rem;">{risk_text}</div>
            <div style="font-weight:700;font-size:1rem;">{fp:.0f}% fatal</div>
          </div>
        </div>"""
        if i % 2 == 0:
            with col1:
                st.markdown(pill_html, unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(pill_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel" style="margin-top:2rem;">
    <b>Note on Route Data</b><br>
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
    st.markdown('<h1 style="font-family:Syne;font-size:2rem;font-weight:800;margin-bottom:0.25rem;">Model Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7B829A;font-size:0.9rem;margin-bottom:2rem;">Technical transparency — how the prediction model works.</p>', unsafe_allow_html=True)

    # ── Model architecture ────────────────────────────────────────────
    st.markdown('<div class="section-header">🏗️ Model Architecture</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    for col, title, body in [
        (a1, "HistGradientBoosting", "Sequential gradient boosting. Corrects errors from previous trees. Better at learning complex non-linear patterns. Weighted 2× in the ensemble."),
        (a2, "Random Forest", "Parallel bagging of 600 decision trees. Reduces variance through diversity. Uses balanced class weights."),
        (a3, "Soft-Vote Ensemble", "Averages probability outputs from both models. Final class = highest average probability after Serious threshold tuning."),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="height:100%;">
              <div class="kpi-label">{title}</div>
              <div style="font-size:0.85rem;line-height:1.7;margin-top:0.5rem;color:#C8CAD4;">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # ── Training data ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📚 Training Data</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        data_rows = {
            "FRSC Osun 2025": "298 crash records (56 Fatal, 198 Serious, 44 Minor)",
            "Kaggle Global":  "8,210 crash records (82 Fatal, 1,046 Serious, 7,082 Minor)",
            "Combined total": "8,508 records after merging",
            "After SMOTE":    "Serious ×4 and Fatal ×6 oversampling applied to training split",
        }
        for k, v in data_rows.items():
            st.markdown(f"""
            <div style="border-bottom:1px solid #252A3A;padding:0.6rem 0;
                        font-size:0.85rem;display:flex;justify-content:space-between;">
              <span style="color:#7B829A;">{k}</span>
              <span style="color:#E8EAF0;text-align:right;max-width:55%;">{v}</span>
            </div>""", unsafe_allow_html=True)

    with d2:
        technique_rows = {
            "Imbalance handling": "Targeted SMOTE (Serious ×4, Fatal ×6) + amplified class weights (Serious:6, Fatal:20)",
            "Threshold tuning":   "Serious class threshold optimised on validation set (recall floor ≥ 0.60)",
            "Validation split":   "Stratified 15% held-out validation set",
            "Leakage prevention": "num_killed and num_injured excluded — derived from target label",
        }
        for k, v in technique_rows.items():
            st.markdown(f"""
            <div style="border-bottom:1px solid #252A3A;padding:0.6rem 0;
                        font-size:0.85rem;display:flex;justify-content:space-between;">
              <span style="color:#7B829A;">{k}</span>
              <span style="color:#E8EAF0;text-align:right;max-width:55%;">{v}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # ── Feature list ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔧 Features Used for Prediction</div>', unsafe_allow_html=True)
    feat_groups = {
        "Temporal"     : ["hour","is_night","is_rush_hour","is_weekend","is_festive"],
        "Environmental": ["is_dark","dark_and_night","bad_weather","wet_road"],
        "Crash Type"   : ["speed_related","mech_failure","overtaking","impairment","at_junction"],
        "Vehicle"      : ["vehicle_cat","heavy_vehicle","two_wheeler"],
        "Magnitude"    : ["casualty_count","people_involved","multi_vehicle","high_casualty"],
        "Categorical"  : ["weather_cat","light_cat","cause_cat"],
    }
    fg1, fg2, fg3 = st.columns(3)
    cols_cycle = [fg1, fg2, fg3]
    for i, (group, feats) in enumerate(feat_groups.items()):
        with cols_cycle[i % 3]:
            feat_html = f'<div style="margin-bottom:1rem;"><div class="kpi-label">{group}</div>'
            for f in feats:
                feat_html += f'<div style="font-size:0.8rem;padding:0.2rem 0;color:#C8CAD4;border-bottom:1px solid #1C2030;">• {f}</div>'
            feat_html += "</div>"
            st.markdown(feat_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # ── Limitations ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠️ Limitations & Disclaimer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-panel">
    <b>Important limitations of this system:</b><br><br>
    • <b>Small local dataset:</b> FRSC Osun 2025 contains 298 records — statistical patterns for individual routes may be unreliable.<br>
    • <b>Feature signal:</b> Without outcome data (injuries/fatalities) the model relies on contextual factors with weak-to-moderate predictive power. Accuracy on unseen data is 60–70%.<br>
    • <b>Not a substitute for official guidance:</b> This tool is an academic research prototype. Do not use it as the sole basis for travel decisions.<br>
    • <b>Model scope:</b> Trained on Osun State data + global patterns. Predictions outside Osun State corridors should be treated with extra caution.<br><br>
    <i>This system was developed as a final-year undergraduate project. For official road safety advice, consult the Federal Road Safety Corps (FRSC).</i>
    </div>
    """, unsafe_allow_html=True)
