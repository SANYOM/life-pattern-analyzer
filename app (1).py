import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Life Pattern Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #090920 0%, #12123a 50%, #0b1120 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stHeader"] { background: transparent; }

/* ── Typography ── */
.hero-title {
    font-size: 2.8rem; font-weight: 900; text-align: center;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.15; margin-bottom: 4px;
}
.hero-sub {
    text-align: center; color: rgba(255,255,255,0.45);
    font-size: 1.05rem; margin-top: 0; margin-bottom: 20px;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px; padding: 22px; margin-bottom: 12px;
}
.metric-label {
    color: rgba(255,255,255,0.5); font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px;
}
.metric-value {
    font-size: 2.4rem; font-weight: 800; color: white; line-height: 1;
}
.metric-sub { color: rgba(255,255,255,0.4); font-size: 0.82rem; margin-top: 4px; }

/* ── Status badges ── */
.badge {
    display: inline-block; padding: 4px 14px; border-radius: 99px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.5px;
}
.badge-red   { background: rgba(239,68,68,0.2);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.4); }
.badge-amber { background: rgba(251,191,36,0.2); color: #fde68a; border: 1px solid rgba(251,191,36,0.4); }
.badge-green { background: rgba(52,211,153,0.2); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.4); }

/* ── Insight cards ── */
.insight-danger {
    background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.25);
    border-radius: 14px; padding: 16px 20px; margin: 8px 0; color: #fca5a5;
}
.insight-warn {
    background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.25);
    border-radius: 14px; padding: 16px 20px; margin: 8px 0; color: #fde68a;
}
.insight-good {
    background: rgba(52,211,153,0.07); border: 1px solid rgba(52,211,153,0.25);
    border-radius: 14px; padding: 16px 20px; margin: 8px 0; color: #6ee7b7;
}
.insight-blue {
    background: rgba(96,165,250,0.07); border: 1px solid rgba(96,165,250,0.25);
    border-radius: 14px; padding: 16px 20px; margin: 8px 0; color: #93c5fd;
}

/* ── Sidebar mental-load bar ── */
.ml-bar-bg {
    background: rgba(255,255,255,0.08); border-radius: 99px;
    height: 10px; width: 100%; margin-top: 6px;
}
.ml-bar-fill {
    height: 10px; border-radius: 99px; transition: width 0.4s ease;
}

/* ── Divider ── */
.divider {
    border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 18px 0;
}

/* ── Tab styling ── */
[data-testid="stTabs"] button {
    color: rgba(255,255,255,0.5) !important; font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #a78bfa !important;
}

/* ── Slider & Select ── */
.stSlider > div > div > div > div { background: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FILES = [
    "model_pressure.pkl", "model_burnout.pkl",
    "model_productivity.pkl", "model_wellbeing.pkl",
    "feature_importance.pkl", "feature_cols.pkl", "peer_data.pkl",
]

@st.cache_resource(show_spinner=False)
def load_all_models():
    if not all(os.path.exists(f) for f in MODEL_FILES):
        with st.spinner("🧠 Training your Digital Twin AI… (first run only, ~15 s)"):
            try:
                import model as m
                m.train_all_models()
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                return None
    try:
        return {
            "pressure":    joblib.load("model_pressure.pkl"),
            "burnout":     joblib.load("model_burnout.pkl"),
            "productivity":joblib.load("model_productivity.pkl"),
            "wellbeing":   joblib.load("model_wellbeing.pkl"),
            "importance":  joblib.load("feature_importance.pkl"),
            "feature_cols":joblib.load("feature_cols.pkl"),
            "peer":        joblib.load("peer_data.pkl"),
        }
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def predict_all(mdls, inputs):
    X = np.array([inputs])
    pressure_raw  = int(mdls["pressure"].predict(X)[0])
    burnout       = float(np.clip(mdls["burnout"].predict(X)[0],      0, 100))
    productivity  = float(np.clip(mdls["productivity"].predict(X)[0], 0, 100))
    wellbeing     = float(np.clip(mdls["wellbeing"].predict(X)[0],    0, 100))
    mental_load   = float(burnout * 0.40 + (100 - productivity) * 0.30 + (100 - wellbeing) * 0.30)
    pressure_map  = {1: ("Low", "green"), 2: ("Medium", "amber"), 3: ("High", "red")}
    label, color  = pressure_map.get(pressure_raw, ("Medium", "amber"))
    return dict(
        pressure=pressure_raw, pressure_label=label, pressure_color=color,
        burnout=burnout, productivity=productivity,
        wellbeing=wellbeing, mental_load=mental_load,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
TRANSPARENT = "rgba(0,0,0,0)"

def _base_layout(**kwargs):
    return dict(
        paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
        font=dict(color="white"), margin=dict(l=20, r=20, t=30, b=20),
        **kwargs
    )

def gauge_chart(value, title, low_good=True):
    """Gauge where low_good=True means low value is safe (e.g. burnout)."""
    if low_good:
        steps = [
            dict(range=[0,  33], color="rgba(52,211,153,0.18)"),
            dict(range=[33, 66], color="rgba(251,191,36,0.18)"),
            dict(range=[66,100], color="rgba(239,68,68,0.18)"),
        ]
        bar_color = "#34d399" if value < 33 else "#fbbf24" if value < 66 else "#f87171"
    else:
        steps = [
            dict(range=[0,  33], color="rgba(239,68,68,0.18)"),
            dict(range=[33, 66], color="rgba(251,191,36,0.18)"),
            dict(range=[66,100], color="rgba(52,211,153,0.18)"),
        ]
        bar_color = "#f87171" if value < 33 else "#fbbf24" if value < 66 else "#34d399"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        title=dict(text=title, font=dict(color="rgba(255,255,255,0.7)", size=13)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="rgba(255,255,255,0.3)",
                      tickfont=dict(color="rgba(255,255,255,0.4)", size=10)),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor=TRANSPARENT,
            bordercolor="rgba(255,255,255,0.08)",
            steps=steps,
        ),
        number=dict(font=dict(color="white", size=34), suffix="%"),
    ))
    fig.update_layout(**_base_layout(height=210))
    return fig


def radar_chart(sleep, study, screen, stress, activity, caffeine):
    cats = ["Sleep", "Study Focus", "Screen Ctrl", "Low Stress", "Active", "Low Caffeine"]
    vals = [
        (sleep / 12) * 100,
        (study / 12) * 100,
        (1 - screen / 12) * 100,
        (1 - (stress - 1) / 2) * 100,
        activity * 100,
        (1 - caffeine / 5) * 100,
    ]
    # Close polygon
    cats_c = cats + [cats[0]]
    vals_c = vals + [vals[0]]
    ideal_c = [75, 72, 78, 82, 80, 75, 75]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_c, theta=cats_c, name="You",
        fill="toself", fillcolor="rgba(167,139,250,0.15)",
        line=dict(color="#a78bfa", width=2.5),
        marker=dict(color="#a78bfa", size=7),
    ))
    fig.add_trace(go.Scatterpolar(
        r=ideal_c, theta=cats_c, name="Ideal Zone",
        fill="toself", fillcolor="rgba(52,211,153,0.05)",
        line=dict(color="rgba(52,211,153,0.5)", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=TRANSPARENT,
            radialaxis=dict(
                visible=True, range=[0, 100],
                color="rgba(255,255,255,0.2)",
                tickfont=dict(color="rgba(255,255,255,0.3)", size=9),
            ),
            angularaxis=dict(
                color="rgba(255,255,255,0.6)",
                tickfont=dict(color="white", size=11),
            ),
        ),
        showlegend=True,
        legend=dict(font=dict(color="white"), bgcolor=TRANSPARENT,
                    orientation="h", x=0.2, y=-0.08),
        **_base_layout(height=380),
    )
    return fig


def projection_chart(df_proj):
    fig = go.Figure()
    traces = [
        ("Burnout Risk",  "burnout",      "#f87171", True),
        ("Productivity",  "productivity", "#60a5fa", False),
        ("Wellbeing",     "wellbeing",    "#34d399", False),
    ]
    for name, col, color, filled in traces:
        kw = dict(fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba")) if filled else {}
        fig.add_trace(go.Scatter(
            x=df_proj["day"], y=df_proj[col],
            name=name, mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            **kw,
        ))
    fig.add_hline(y=75, line_dash="dot", line_color="rgba(239,68,68,0.45)",
                  annotation_text="⚠️ Burnout threshold",
                  annotation_font=dict(color="rgba(239,68,68,0.7)", size=11))
    fig.update_layout(
        xaxis=dict(color="white", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(color="white", gridcolor="rgba(255,255,255,0.05)", range=[0, 100]),
        legend=dict(font=dict(color="white"), bgcolor="rgba(255,255,255,0.04)",
                    orientation="h", x=0, y=1.12),
        **_base_layout(height=360, margin=dict(l=40, r=30, t=50, b=40)),
    )
    return fig


def importance_chart(imp_df):
    colors = [
        f"rgba(167,139,250,{0.4 + 0.6 * v / imp_df['importance'].max()})"
        for v in imp_df["importance"]
    ]
    fig = go.Figure(go.Bar(
        x=imp_df["importance"] * 100,
        y=imp_df["feature"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v*100:.1f}%" for v in imp_df["importance"]],
        textfont=dict(color="white", size=11),
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(color="white", gridcolor="rgba(255,255,255,0.05)",
                   title="Influence on Academic Pressure (%)", range=[0, 40]),
        yaxis=dict(color="white"),
        **_base_layout(height=310, margin=dict(l=10, r=60, t=10, b=40)),
    )
    return fig


def comparison_chart(current, simulated, label):
    cats = ["Burnout Risk", "Productivity", "Wellbeing"]
    c_vals = [current["burnout"], current["productivity"], current["wellbeing"]]
    s_vals = [simulated["burnout"], simulated["productivity"], simulated["wellbeing"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current",   x=cats, y=c_vals,
                         marker_color="rgba(148,163,184,0.5)"))
    fig.add_trace(go.Bar(name=f"If: {label}", x=cats, y=s_vals,
                         marker_color=["rgba(248,113,113,0.7)",
                                       "rgba(96,165,250,0.7)",
                                       "rgba(52,211,153,0.7)"]))
    fig.update_layout(
        barmode="group",
        xaxis=dict(color="white"),
        yaxis=dict(color="white", range=[0, 100],
                   gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(color="white"), bgcolor=TRANSPARENT),
        **_base_layout(height=320, margin=dict(l=30, r=20, t=10, b=40)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7-DAY SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def simulate_7_days(mdls, base):
    age, gender, sleep, screen, stress, study, activity, caffeine = base
    fatigue = 0
    rows = []
    for i, day in enumerate(DAYS):
        adj_sleep    = max(3.0, sleep    - fatigue * 0.18)
        adj_stress   = min(3.0, stress   + fatigue * 0.09)
        adj_activity = float(np.clip(activity - fatigue * 0.07, 0, 1))
        inputs = [age, gender, adj_sleep, screen, adj_stress, study, adj_activity, caffeine]
        p = predict_all(mdls, inputs)
        if p["burnout"] > 65:
            fatigue += 0.45
        elif p["burnout"] > 40:
            fatigue += 0.18
        else:
            fatigue = max(0.0, fatigue - 0.25)
        rows.append(dict(day=day, burnout=round(p["burnout"], 1),
                         productivity=round(p["productivity"], 1),
                         wellbeing=round(p["wellbeing"], 1)))
    return pd.DataFrame(rows)


def burnout_countdown(proj_df):
    for i, row in proj_df.iterrows():
        if row["burnout"] >= 75:
            return i + 1
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SMART RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def top_recommendation(mdls, inputs):
    age, gender, sleep, screen, stress, study, activity, caffeine = inputs
    base = predict_all(mdls, inputs)
    base_score = base["wellbeing"] + base["productivity"] - base["burnout"]

    options = [
        ("Sleep 1 more hour 🛏️",         [age, gender, min(sleep+1, 12),    screen,          stress,          study,          activity, caffeine]),
        ("Reduce screen time by 1h 📵",   [age, gender, sleep,               max(screen-1, 0), stress,          study,          activity, caffeine]),
        ("Exercise every day 🏃",          [age, gender, sleep,               screen,          stress,          study,          1,        caffeine]),
        ("Cut 1 cup of caffeine ☕",       [age, gender, sleep,               screen,          stress,          study,          activity, max(caffeine-1, 0)]),
        ("Study 1 more hour 📚",           [age, gender, sleep,               screen,          stress,          min(study+1,12), activity, caffeine]),
        ("Manage stress (meditate) 🧘",   [age, gender, sleep,               screen,          max(stress-1,1), study,          activity, caffeine]),
        ("Cut screen time by 2h 🎯",      [age, gender, sleep,               max(screen-2, 0), stress,         study,          activity, caffeine]),
        ("Sleep + exercise combo 💪",     [age, gender, min(sleep+1, 12),    screen,          stress,          study,          1,        caffeine]),
    ]

    best = None
    best_gain = -999
    for label, new_inp in options:
        p = predict_all(mdls, new_inp)
        score = p["wellbeing"] + p["productivity"] - p["burnout"]
        if score > best_gain:
            best_gain = score
            best = (label, p, score - base_score)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# PEER PERCENTILE
# ─────────────────────────────────────────────────────────────────────────────
def percentile(value, arr):
    return float(np.mean(arr < value)) * 100


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
mdls = load_all_models()
if mdls is None:
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — USER INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Your Profile")
    age    = st.slider("Age", 18, 35, 22)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    gender_enc = {"male": 0, "female": 1, "other": 2}[gender.lower()]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 🌙 Daily Habits")
    sleep  = st.slider("Sleep Hours",          0, 12,  7)
    study  = st.slider("Study Hours",          0, 12,  4)
    screen = st.slider("Screen Time (hrs)",    0, 12,  5)
    caffeine = st.slider("Caffeine (cups/day)", 0,  5,  1)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 🧠 Lifestyle")
    stress_raw = st.selectbox("Stress Level",       ["Low", "Medium", "High"])
    phys_raw   = st.selectbox("Physical Activity",  ["Yes", "No"])

    stress_enc   = {"low": 1, "medium": 2, "high": 3}[stress_raw.lower()]
    activity_enc = 1 if phys_raw == "Yes" else 0

    INPUTS = [age, gender_enc, sleep, screen, stress_enc, study, activity_enc, caffeine]
    preds  = predict_all(mdls, INPUTS)

    # Real-time mental load bar in sidebar
    ml   = preds["mental_load"]
    mlc  = "#34d399" if ml < 33 else "#fbbf24" if ml < 55 else "#f87171"
    mll  = "Sustainable 🟢" if ml < 33 else "Moderate 🟡" if ml < 55 else ("High 🔴" if ml < 75 else "Critical 🚨")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### ⚡ Mental Load Index")
    st.markdown(f"""
    <div style='font-size:1.6rem;font-weight:800;color:{mlc}'>{ml:.0f}<span style='font-size:1rem;color:rgba(255,255,255,0.4)'>/100</span></div>
    <div class='ml-bar-bg'><div class='ml-bar-fill' style='width:{ml}%;background:{mlc}'></div></div>
    <div style='color:rgba(255,255,255,0.5);font-size:0.82rem;margin-top:6px'>{mll}</div>
    """, unsafe_allow_html=True)

    if ml >= 75:
        st.error("🚨 Critical overload detected! Check the Insights tab.")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 AI Life Pattern Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Your personal Digital Twin — live predictions · 7-day simulation · smart insights</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Digital Twin Dashboard",
    "⚡  What-If Simulator",
    "📅  7-Day Projection",
    "💡  AI Insights",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DIGITAL TWIN DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Row 1: 4 KPI cards ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    badge_cls = {"green": "badge-green", "amber": "badge-amber", "red": "badge-red"}

    with c1:
        bc = "badge-green" if preds["burnout"] < 33 else "badge-amber" if preds["burnout"] < 66 else "badge-red"
        st.markdown(f"""
        <div class='glass-card'>
          <div class='metric-label'>🔥 Burnout Risk</div>
          <div class='metric-value'>{preds['burnout']:.0f}%</div>
          <div class='metric-sub'><span class='badge {bc}'>{'Low' if preds['burnout']<33 else 'Moderate' if preds['burnout']<66 else 'High'}</span></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        pc = "badge-red" if preds["productivity"] < 33 else "badge-amber" if preds["productivity"] < 66 else "badge-green"
        st.markdown(f"""
        <div class='glass-card'>
          <div class='metric-label'>⚡ Productivity</div>
          <div class='metric-value'>{preds['productivity']:.0f}%</div>
          <div class='metric-sub'><span class='badge {pc}'>{'Low' if preds['productivity']<33 else 'Moderate' if preds['productivity']<66 else 'High'}</span></div>
        </div>""", unsafe_allow_html=True)

    with c3:
        wc = "badge-red" if preds["wellbeing"] < 33 else "badge-amber" if preds["wellbeing"] < 66 else "badge-green"
        st.markdown(f"""
        <div class='glass-card'>
          <div class='metric-label'>💚 Wellbeing</div>
          <div class='metric-value'>{preds['wellbeing']:.0f}%</div>
          <div class='metric-sub'><span class='badge {wc}'>{'Low' if preds['wellbeing']<33 else 'Moderate' if preds['wellbeing']<66 else 'High'}</span></div>
        </div>""", unsafe_allow_html=True)

    with c4:
        ac = badge_cls[preds["pressure_color"]]
        st.markdown(f"""
        <div class='glass-card'>
          <div class='metric-label'>🎓 Academic Pressure</div>
          <div class='metric-value'>{preds['pressure_label']}</div>
          <div class='metric-sub'><span class='badge {ac}'>AI Classified</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Gauges ────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge_chart(preds["burnout"],      "🔥 Burnout Risk",  low_good=True),  use_container_width=True)
    with g2:
        st.plotly_chart(gauge_chart(preds["productivity"], "⚡ Productivity",  low_good=False), use_container_width=True)
    with g3:
        st.plotly_chart(gauge_chart(preds["wellbeing"],    "💚 Wellbeing",     low_good=False), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Habit DNA radar + Feature importance ──────────────────────────
    r1, r2 = st.columns([1.1, 0.9])
    with r1:
        st.markdown("<div class='glass-card'><div class='metric-label'>🧬 Your Habit DNA Fingerprint</div>", unsafe_allow_html=True)
        st.plotly_chart(
            radar_chart(sleep, study, screen, stress_enc, activity_enc, caffeine),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='glass-card'><div class='metric-label'>🔍 What Drives Your Academic Pressure</div>", unsafe_allow_html=True)
        st.plotly_chart(importance_chart(mdls["importance"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — WHAT-IF SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### ⚡ What-If Simulator — Change One Habit, See Your Future")
    st.markdown("<p style='color:rgba(255,255,255,0.45)'>Pick any habit below and drag the slider to instantly see how that single change reshapes all your metrics.</p>", unsafe_allow_html=True)

    habit = st.selectbox("Which habit do you want to change?", [
        "Sleep Hours 🛏️",
        "Screen Time 📱",
        "Study Hours 📚",
        "Caffeine Intake ☕",
        "Stress Level 🧠",
        "Physical Activity 🏃",
    ])

    sim_inputs = INPUTS.copy()

    if habit == "Sleep Hours 🛏️":
        new_val = st.slider("New Sleep Hours", 0, 12, sleep)
        sim_inputs[2] = new_val
        change_desc = f"Sleep {sleep}h → {new_val}h"
    elif habit == "Screen Time 📱":
        new_val = st.slider("New Screen Time (hrs)", 0, 12, screen)
        sim_inputs[3] = new_val
        change_desc = f"Screen {screen}h → {new_val}h"
    elif habit == "Study Hours 📚":
        new_val = st.slider("New Study Hours", 0, 12, study)
        sim_inputs[5] = new_val
        change_desc = f"Study {study}h → {new_val}h"
    elif habit == "Caffeine Intake ☕":
        new_val = st.slider("New Caffeine (cups)", 0, 5, caffeine)
        sim_inputs[7] = new_val
        change_desc = f"Caffeine {caffeine} → {new_val} cups"
    elif habit == "Stress Level 🧠":
        new_str = st.selectbox("New Stress Level", ["Low", "Medium", "High"])
        new_val = {"low": 1, "medium": 2, "high": 3}[new_str.lower()]
        sim_inputs[4] = new_val
        change_desc = f"Stress {stress_raw} → {new_str}"
    else:
        new_pa = st.selectbox("New Physical Activity", ["Yes", "No"])
        new_val = 1 if new_pa == "Yes" else 0
        sim_inputs[6] = new_val
        change_desc = f"Activity {phys_raw} → {new_pa}"

    sim_preds = predict_all(mdls, sim_inputs)

    # ── Comparison bars ───────────────────────────────────────────────────────
    st.plotly_chart(comparison_chart(preds, sim_preds, change_desc), use_container_width=True)

    # ── Delta cards ───────────────────────────────────────────────────────────
    d1, d2, d3 = st.columns(3)
    def delta_html(label, cur, sim, low_good=True):
        diff = sim - cur
        better = (diff < 0) if low_good else (diff > 0)
        arrow = "↑" if diff > 0 else "↓"
        color = "#34d399" if better else "#f87171"
        return f"""
        <div class='glass-card' style='text-align:center'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value' style='font-size:1.8rem'>{sim:.0f}%</div>
          <div style='color:{color};font-weight:700;font-size:1.1rem'>{arrow} {abs(diff):.1f}%</div>
        </div>"""

    with d1: st.markdown(delta_html("🔥 Burnout Risk",  preds["burnout"],      sim_preds["burnout"],      True),  unsafe_allow_html=True)
    with d2: st.markdown(delta_html("⚡ Productivity",  preds["productivity"],  sim_preds["productivity"],  False), unsafe_allow_html=True)
    with d3: st.markdown(delta_html("💚 Wellbeing",     preds["wellbeing"],     sim_preds["wellbeing"],     False), unsafe_allow_html=True)

    # ── Net impact verdict ────────────────────────────────────────────────────
    net = (sim_preds["wellbeing"] + sim_preds["productivity"] - sim_preds["burnout"]) - \
          (preds["wellbeing"]     + preds["productivity"]     - preds["burnout"])

    if net > 5:
        st.markdown(f"<div class='insight-good'>✅ <b>Positive impact detected.</b> This change improves your overall life score by <b>{net:.1f} points</b>. Highly recommended!</div>", unsafe_allow_html=True)
    elif net < -5:
        st.markdown(f"<div class='insight-danger'>⚠️ <b>Negative impact.</b> This change worsens your overall score by <b>{abs(net):.1f} points</b>. Think twice.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='insight-warn'>🔄 <b>Neutral impact.</b> This change has minimal effect on your overall score.</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — 7-DAY PROJECTION
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📅 7-Day Digital Twin Projection")
    st.markdown("<p style='color:rgba(255,255,255,0.45)'>Simulates how your metrics evolve over the next 7 days if you maintain your current habits. Fatigue accumulates, stress rises — watch where you're headed.</p>", unsafe_allow_html=True)

    proj = simulate_7_days(mdls, INPUTS)
    st.plotly_chart(projection_chart(proj), use_container_width=True)

    # ── Burnout countdown ─────────────────────────────────────────────────────
    countdown = burnout_countdown(proj)
    if countdown:
        st.markdown(f"""
        <div class='insight-danger'>
          🚨 <b>Burnout Warning:</b> At your current pace, burnout risk crosses the critical threshold on
          <b>Day {countdown} ({DAYS[countdown-1]})</b>. You need to intervene NOW.
        </div>""", unsafe_allow_html=True)
    else:
        peak_burnout = proj["burnout"].max()
        st.markdown(f"""
        <div class='insight-good'>
          ✅ <b>Stable week ahead.</b> Your burnout risk peaks at <b>{peak_burnout:.0f}%</b> — 
          well below the danger threshold. Keep it up!
        </div>""", unsafe_allow_html=True)

    # ── Day-by-day table ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Day-by-day breakdown")
    display_proj = proj.copy()
    display_proj.columns = ["Day", "Burnout Risk %", "Productivity %", "Wellbeing %"]
    for col in ["Burnout Risk %", "Productivity %", "Wellbeing %"]:
        display_proj[col] = display_proj[col].apply(lambda x: f"{x:.1f}")
    st.dataframe(
        display_proj,
        use_container_width=True,
        hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 💡 Personalized AI Insights")

    peer = mdls["peer"]
    n = peer["n_students"]

    # ── Peer comparison ───────────────────────────────────────────────────────
    b_pct  = percentile(preds["burnout"],      peer["burnout_values"])
    p_pct  = percentile(preds["productivity"], peer["productivity_values"])
    w_pct  = percentile(preds["wellbeing"],    peer["wellbeing_values"])

    st.markdown("#### 👥 How You Compare to Peers")
    p1, p2, p3 = st.columns(3)

    def peer_card(title, pct, low_good=False):
        better = pct < 50 if low_good else pct > 50
        color  = "#34d399" if better else "#f87171"
        word   = "better" if better else "worse"
        vs_pct = 100 - pct if low_good else pct
        return f"""
        <div class='glass-card' style='text-align:center'>
          <div class='metric-label'>{title}</div>
          <div style='font-size:2rem;font-weight:800;color:{color}'>{vs_pct:.0f}th</div>
          <div style='color:rgba(255,255,255,0.45);font-size:0.82rem'>percentile among {n} students</div>
          <div style='color:{color};font-size:0.85rem;margin-top:4px'>
            {'⬆️' if better else '⬇️'} {word} than {pct if low_good else 100-pct:.0f}% of peers
          </div>
        </div>"""

    with p1: st.markdown(peer_card("🔥 Burnout Risk",  b_pct, low_good=True),  unsafe_allow_html=True)
    with p2: st.markdown(peer_card("⚡ Productivity",  p_pct, low_good=False), unsafe_allow_html=True)
    with p3: st.markdown(peer_card("💚 Wellbeing",     w_pct, low_good=False), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Personalized recommendations ──────────────────────────────────────────
    st.markdown("#### 🎯 Your Top Recommendations")

    insights = []

    if sleep < 6:
        insights.append(("danger", "🛏️ <b>Critical Sleep Deficit.</b> You're sleeping under 6 hours. Sleep deprivation is the #1 predictor of burnout in our dataset. Aim for at least 7–8 hours."))
    elif sleep < 7:
        insights.append(("warn", "🛏️ <b>Sleep slightly low.</b> Adding just 1 hour of sleep per night can reduce burnout risk by up to 15 points."))
    else:
        insights.append(("good", "🛏️ <b>Solid sleep routine.</b> Your sleep hours are in the healthy range. This is your biggest protective factor."))

    if screen > 8:
        insights.append(("danger", "📱 <b>Screen time overload.</b> 8+ hours of screen time correlates strongly with elevated stress. Try a 2-hour reduction."))
    elif screen > 6:
        insights.append(("warn", "📱 <b>Screen time is elevated.</b> Consider reducing by 1–2 hours, especially before bed."))

    if stress_enc == 3:
        insights.append(("danger", "🧠 <b>High stress detected.</b> This is your biggest risk factor. Even 10 minutes of daily meditation can lower perceived stress significantly."))
    elif stress_enc == 2:
        insights.append(("warn", "🧠 <b>Moderate stress.</b> Stress management techniques (journaling, breathing exercises) can shift this to Low within 2 weeks."))

    if activity_enc == 0:
        insights.append(("warn", "🏃 <b>No physical activity.</b> Even a 20-minute daily walk increases wellbeing by an estimated 10–15 points and reduces caffeine dependency."))
    else:
        insights.append(("good", "🏃 <b>Active lifestyle.</b> Physical activity is one of your strongest wellbeing boosters. Keep it consistent."))

    if caffeine > 3:
        insights.append(("warn", f"☕ <b>High caffeine intake ({caffeine} cups).</b> Excess caffeine raises cortisol, compounding stress. Try to stay under 2–3 cups."))

    if study > 9:
        insights.append(("warn", "📚 <b>Study overload risk.</b> Studying 9+ hours daily without recovery accelerates cognitive fatigue. Consider Pomodoro breaks."))

    style_map = {"danger": "insight-danger", "warn": "insight-warn",
                 "good": "insight-good", "blue": "insight-blue"}
    for sty, text in insights:
        st.markdown(f"<div class='{style_map[sty]}'>{text}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── #1 Power Move ─────────────────────────────────────────────────────────
    st.markdown("#### 🚀 Your Single Highest-Impact Change")
    rec = top_recommendation(mdls, INPUTS)
    if rec:
        label, new_p, gain = rec
        st.markdown(f"""
        <div class='insight-blue'>
          <div style='font-size:1.3rem;font-weight:800;color:#93c5fd;margin-bottom:8px'>
            {label}
          </div>
          <div style='color:rgba(255,255,255,0.7);line-height:1.7'>
            This single change would shift your overall life score by <b style='color:#34d399'>+{gain:.1f} points</b>.<br>
            Burnout: <b>{new_p['burnout']:.0f}%</b> &nbsp;·&nbsp;
            Productivity: <b>{new_p['productivity']:.0f}%</b> &nbsp;·&nbsp;
            Wellbeing: <b>{new_p['wellbeing']:.0f}%</b>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("<br><hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center;color:rgba(255,255,255,0.25);font-size:0.8rem'>
      Digital Twin trained on {n} real student records &nbsp;·&nbsp;
      4 AI models &nbsp;·&nbsp; Real-time simulation engine
    </div>""", unsafe_allow_html=True)
