import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import shap

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CricMetric AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-base:       #080C16;
    --bg-surface:    #0E1525;
    --bg-raised:     #141D2E;
    --bg-highlight:  #1A2540;
    --accent-teal:   #00D4AA;
    --accent-orange: #FF6B35;
    --accent-blue:   #4A9EFF;
    --text-primary:  #F0F4FF;
    --text-secondary:#8892A4;
    --text-muted:    #4E5A6E;
    --border:        #1E2A3E;
    --border-bright: #2A3A55;
    --positive:      #00D4AA;
    --negative:      #FF4757;
    --font-body:     'Inter', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
    --radius-sm:     6px;
    --radius-md:     10px;
    --radius-lg:     16px;
}

html, body, [class*="css"], .stApp {
    font-family: var(--font-body) !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

[data-testid="stSidebar"] .stRadio label {
    font-size: 0.8rem !important;
    letter-spacing: 0.04em;
    color: var(--text-secondary) !important;
    padding: 0.4rem 0.6rem !important;
    border-radius: var(--radius-sm);
    transition: all 0.15s ease;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--text-primary) !important;
    background: var(--bg-highlight) !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.8rem 0 !important;
}

.page-header { margin-bottom: 2rem; }
.page-header h1 {
    font-size: 1.7rem; font-weight: 700;
    letter-spacing: -0.02em; color: var(--text-primary);
    margin: 0 0 0.25rem 0;
}
.page-header p {
    font-size: 0.82rem; color: var(--text-secondary);
    margin: 0; letter-spacing: 0.02em;
}

.stat-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent-teal);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.3rem;
    position: relative; overflow: hidden;
}
.stat-card::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(0,212,170,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.stat-card-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-secondary); margin-bottom: 0.5rem;
}
.stat-card-value {
    font-size: 1.8rem; font-weight: 800; color: var(--text-primary);
    font-variant-numeric: tabular-nums; line-height: 1; letter-spacing: -0.03em;
}
.stat-card-sub {
    font-size: 0.72rem; color: var(--text-muted);
    margin-top: 0.3rem; font-family: var(--font-mono);
}

.pillar-card {
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 1.6rem; height: 100%;
}
.pillar-tag {
    display: inline-block; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 0.25rem 0.6rem; border-radius: 4px; margin-bottom: 0.9rem;
}
.tag-ppi { background: rgba(0,212,170,0.15);  color: var(--accent-teal); }
.tag-cfs { background: rgba(74,158,255,0.15); color: var(--accent-blue); }
.tag-ens { background: rgba(255,107,53,0.15); color: var(--accent-orange); }

.pillar-card h3 {
    font-size: 1rem; font-weight: 700;
    color: var(--text-primary); margin: 0 0 0.25rem 0;
}
.pillar-weight {
    font-size: 0.75rem; color: var(--text-secondary);
    font-family: var(--font-mono); margin-bottom: 0.9rem;
}
.pillar-card p, .pillar-card li {
    font-size: 0.82rem; color: var(--text-secondary); line-height: 1.6;
}
.pillar-card ul { padding-left: 1rem; }
.pillar-highlight {
    font-size: 0.78rem; background: var(--bg-highlight);
    border-left: 2px solid var(--accent-teal);
    padding: 0.5rem 0.8rem;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    color: var(--text-secondary); margin-top: 1rem;
    font-family: var(--font-mono);
}

.leader-banner {
    background: linear-gradient(90deg, rgba(0,212,170,0.08) 0%, transparent 80%);
    border: 1px solid rgba(0,212,170,0.25);
    border-left: 3px solid var(--accent-teal);
    border-radius: var(--radius-md);
    padding: 0.9rem 1.2rem;
    display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;
}
.leader-banner-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--accent-teal);
}
.leader-banner-name { font-size: 1.1rem; font-weight: 700; color: var(--text-primary); }
.leader-banner-score { font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-secondary); }

.section-heading {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-muted);
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}

[data-testid="metric-container"] {
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: var(--radius-md); padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    font-size: 0.68rem !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important; font-weight: 800 !important;
    color: var(--text-primary) !important; letter-spacing: -0.02em !important;
}

[data-testid="stSlider"] label {
    font-size: 0.72rem !important; font-weight: 600 !important;
    letter-spacing: 0.06em !important; color: var(--text-secondary) !important;
}
[data-testid="stSelectbox"] label {
    font-size: 0.72rem !important; font-weight: 600 !important;
    letter-spacing: 0.06em !important; color: var(--text-secondary) !important;
}

[data-baseweb="tab-list"] {
    background: transparent !important; gap: 0.5rem !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-baseweb="tab"] {
    font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 0.05em !important; color: var(--text-secondary) !important;
    background: transparent !important; border: none !important;
    padding: 0.6rem 1rem !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent-teal) !important;
    border-bottom: 2px solid var(--accent-teal) !important;
}

.stAlert { background: var(--bg-surface) !important; border-radius: var(--radius-md) !important; font-size: 0.82rem !important; }
.stCaption, [data-testid="caption"] { color: var(--text-muted) !important; font-size: 0.72rem !important; }

.home-hero { padding: 2rem 0 1.5rem; border-bottom: 1px solid var(--border); margin-bottom: 1.2rem; }
.home-hero-eyebrow {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--accent-teal); margin-bottom: 1rem;
}
.home-hero h1 {
    font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 900;
    letter-spacing: -0.04em; line-height: 1.05;
    color: var(--text-primary); margin: 0 0 1rem 0;
}
.home-hero h1 span { color: var(--accent-teal); -webkit-text-stroke: 0; }
.home-hero-sub {
    font-size: 0.95rem; color: var(--text-secondary);
    max-width: 580px; line-height: 1.65; margin: 0;
}

.validation-strip {
    display: flex; gap: 2rem; padding: 1.2rem 0;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    margin: 2rem 0; flex-wrap: wrap;
}
.val-item-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.2rem; }
.val-item-value { font-size: 1rem; font-weight: 700; color: var(--text-primary); font-family: var(--font-mono); }
.val-item-desc  { font-size: 0.72rem; color: var(--text-secondary); }

.score-ring-wrap {
    display: flex; flex-direction: column; align-items: center; gap: 0.3rem;
    padding: 1.5rem; background: var(--bg-surface);
    border: 1px solid var(--border); border-radius: var(--radius-lg);
}
.score-ring-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted); }
.score-ring-value { font-size: 3rem; font-weight: 900; color: var(--accent-teal); letter-spacing: -0.05em; line-height: 1; }
.score-ring-rank  { font-size: 0.82rem; color: var(--text-secondary); font-family: var(--font-mono); }

.vs-player-block {
    text-align: center; padding: 1.5rem;
    background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
}
.vs-player-name  { font-size: 1.3rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; }
.vs-player-rank  { font-size: 0.78rem; color: var(--text-secondary); font-family: var(--font-mono); margin-top: 0.3rem; }
.vs-player-score { font-size: 2rem; font-weight: 900; margin-top: 0.5rem; letter-spacing: -0.04em; }
.vs-divider { display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 800; color: var(--text-muted); letter-spacing: 0.1em; }

.rising-info-box {
    background: var(--bg-surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent-orange); border-radius: var(--radius-md);
    padding: 1.1rem 1.4rem; font-size: 0.82rem; color: var(--text-secondary);
    line-height: 1.65; margin-bottom: 1.5rem;
}
.rising-info-box strong { color: var(--accent-orange); }

.app-footer {
    margin-top: 4rem; padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-size: 0.7rem; color: var(--text-muted); letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)


# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
# Base layout WITHOUT xaxis/yaxis/legend — pass those per-chart to avoid duplicate-kwarg errors
_AXIS = dict(gridcolor="#1E2A3E", linecolor="#1E2A3E", tickcolor="#1E2A3E", zerolinecolor="#1E2A3E")

PLOTLY_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8892A4", size=11),
    margin=dict(l=20, r=20, t=40, b=40),
)

def plotly_layout(**overrides):
    """Merge base layout with per-chart axis/legend overrides safely."""
    layout = dict(
        **PLOTLY_BASE,
        xaxis=_AXIS.copy(),
        yaxis=_AXIS.copy(),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1E2A3E", borderwidth=1),
    )
    # Deep-merge xaxis/yaxis so callers can extend them
    for key in ("xaxis", "yaxis"):
        if key in overrides:
            merged = {**layout[key], **overrides.pop(key)}
            layout[key] = merged
    layout.update(overrides)
    return layout

COLORS = dict(teal="#00D4AA", orange="#FF6B35", blue="#4A9EFF", red="#FF4757")


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    master = pd.read_csv(r"D:\CricMetric-AI\data\processed\master_player_features.csv")
    rising = pd.read_csv(r"D:\CricMetric-AI\data\processed\rising_star_matches.csv")
    ens    = pd.read_csv(r"D:\CricMetric-AI\data\processed\ens_scores.csv")
    return master, rising, ens

@st.cache_resource
def load_models():
    with open(r"D:\CricMetric-AI\data\processed\xgb_ranking_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(r"D:\CricMetric-AI\data\processed\shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return model, explainer

master_df, rising_df, ens_df = load_data()
model, explainer = load_models()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem;">
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.3rem;">
            <span style="font-size:1.4rem;">🏏</span>
            <span style="font-size:1rem; font-weight:800; letter-spacing:-0.02em; color:#F0F4FF;">CricMetric</span>
            <span style="font-size:0.6rem; font-weight:700; letter-spacing:0.1em; color:#00D4AA;
                         background:rgba(0,212,170,0.12); padding:2px 6px; border-radius:4px;">AI</span>
        </div>
        <div style="font-size:0.68rem; color:#4E5A6E; letter-spacing:0.04em;">
            Context-Aware ODI Batter Rankings
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Navigation</div>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠  Overview", "🏆  Rankings", "🔍  Player Profile",
         "⚔️  Head to Head", "⭐  Rising Stars", "📊  Era Explorer"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;">
        <div><div style="font-size:1rem;font-weight:800;color:#F0F4FF;">475</div><div style="font-size:0.65rem;color:#4E5A6E;">Batters</div></div>
        <div><div style="font-size:1rem;font-weight:800;color:#F0F4FF;">2,544</div><div style="font-size:0.65rem;color:#4E5A6E;">Matches</div></div>
        <div><div style="font-size:1rem;font-weight:800;color:#F0F4FF;">1.3M</div><div style="font-size:0.65rem;color:#4E5A6E;">Balls</div></div>
        <div><div style="font-size:1rem;font-weight:800;color:#F0F4FF;">2002–26</div><div style="font-size:0.65rem;color:#4E5A6E;">Coverage</div></div>
    </div>
    """, unsafe_allow_html=True)


# ─── PAGE: OVERVIEW ───────────────────────────────────────────────────────────
if page == "🏠  Overview":

    st.markdown("""
    <div class="home-hero">
        <div class="home-hero-eyebrow">ODI Batting Intelligence Platform</div>
        <h1>Rankings built on<br><span>context,</span> not just runs.</h1>
        <p class="home-hero-sub">
            CricMetric AI evaluates every ODI batter across 1.3 million balls
            through situational pressure, clutch influence, and era-adjusted quality —
            three signals traditional averages cannot capture.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="validation-strip">
        <div><div class="val-item-label">ICC Correlation</div><div class="val-item-value">r = 0.74</div><div class="val-item-desc">Independent validation</div></div>
        <div><div class="val-item-label">Win Prob Model</div><div class="val-item-value">AUC 0.85</div><div class="val-item-desc">Random Forest · 1.3M balls</div></div>
        <div><div class="val-item-label">XGBoost R²</div><div class="val-item-value">0.59</div><div class="val-item-desc">Ranking model fit</div></div>
        <div><div class="val-item-label">Explainability</div><div class="val-item-value">SHAP</div><div class="val-item-desc">Every ranking explained</div></div>
        <div><div class="val-item-label">Bayesian Shrinkage</div><div class="val-item-value">All metrics</div><div class="val-item-desc">No arbitrary cutoffs</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Three Pillars</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-tag tag-ppi">PPI · 35%</div>
            <h3>Pressure Performance Index</h3>
            <p class="pillar-weight">Bayesian-shrunk scores across 4 pressure scenarios</p>
            <p>Measures how a batter performs when the match is on the line — not when it's already won.</p>
            <ul>
                <li><strong style="color:#F0F4FF">S1</strong> Early collapse — 3+ wickets down in first 20 overs</li>
                <li><strong style="color:#F0F4FF">S2</strong> Death chase — overs 30–50, target 250+</li>
                <li><strong style="color:#F0F4FF">S3</strong> World Cup — ICC tournament matches</li>
                <li><strong style="color:#F0F4FF">S4</strong> Chase recovery — early wickets in winning chases</li>
            </ul>
            <div class="pillar-highlight">Kohli S2 death-chase score: 90.62 / 100</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-tag tag-cfs">CFS · 35%</div>
            <h3>Clutch Factor Score</h3>
            <p class="pillar-weight">Three independent win-impact signals</p>
            <p>Measures whether a batter's presence actually changes match outcomes, not just scoreboards.</p>
            <ul>
                <li><strong style="color:#F0F4FF">C1</strong> Win Probability Added — causal match-state shifts</li>
                <li><strong style="color:#F0F4FF">C2</strong> Match-Winning Ratio — team-strength adjusted</li>
                <li><strong style="color:#F0F4FF">C3</strong> Dominance Score — individual share vs team total</li>
            </ul>
            <div class="pillar-highlight">Tendulkar C3 Dominance: 88.19 — greatness without team results</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-tag tag-ens">ENS · 30%</div>
            <h3>Era Normalisation Score</h3>
            <p class="pillar-weight">Because 300 in 2003 ≠ 300 in 2023</p>
            <p>Adjusts for two decades of rule changes, T20 influence, and shifting run-scoring environments.</p>
            <ul>
                <li><strong style="color:#F0F4FF">E1</strong> Era-adjusted average — normalised to yearly rates</li>
                <li><strong style="color:#F0F4FF">E2</strong> Career volume — log-scaled for sustained excellence</li>
                <li><strong style="color:#F0F4FF">E3</strong> Opposition quality — era-adjusted bowling strength</li>
            </ul>
            <div class="pillar-highlight">Tendulkar era boost: +2.54 (harder scoring environment)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Why Traditional Rankings Fall Short</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div style="background:var(--bg-surface); border:1px solid var(--border);
                    border-radius:var(--radius-lg); padding:1.6rem; font-size:0.85rem;
                    color:var(--text-secondary); line-height:1.7;">
            A Sachin Tendulkar century in the 2003 World Cup and a century in a dead bilateral
            series count identically in career averages. They should not.
            <br><br>
            ICC rankings use a points-decay algorithm that rewards recent form but ignores
            situational context entirely. A boundary in over 45 of a death chase carries the
            same weight as one in over 5 of a comfortable win.
            <br><br>
            <span style="color:#F0F4FF; font-weight:600;">CricMetric AI corrects this at the
            ball level, across 24 years of ODI cricket.</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=["Career Avg", "ICC Points", "CricMetric AI"],
            y=[1, 2, 5],
            text=["No context", "Recent bias", "Full context"],
            textposition="outside",
            marker_color=[COLORS["red"], COLORS["orange"], COLORS["teal"]],
        ))
        fig.update_layout(
            **plotly_layout(
                height=280,
                title=dict(text="Contextual depth by ranking system", font=dict(size=11), x=0),
                yaxis=dict(visible=False),
                showlegend=False,
            )
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("""
    <div class="app-footer">
        Built by Kadiyapu Durgesh Kumar · VIT-AP University 2028 ·
        <a href="https://github.com/Durgeshkadiyapu/CricMetric-AI"
           style="color:#00D4AA; text-decoration:none;">github.com/Durgeshkadiyapu/CricMetric-AI</a>
    </div>
    """, unsafe_allow_html=True)


# ─── PAGE: RANKINGS ───────────────────────────────────────────────────────────
elif page == "🏆  Rankings":

    st.markdown("""
    <div class="page-header">
        <h1>All-Time ODI Batter Rankings</h1>
        <p>Composite score: 35% PPI · 35% CFS · 30% ENS · Bayesian-shrunk across all 475 qualified batters</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Filters</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1: top_n   = st.slider("Show top N", 10, len(master_df), 100)
    with f2: min_ppi = st.slider("Min PPI", 0, 100, 0)
    with f3: min_cfs = st.slider("Min CFS", 0, 100, 0)
    with f4: sort_by = st.selectbox("Sort by", ["FINAL_SCORE", "PPI", "CFS", "ENS"])

    filtered = (
        master_df.copy()
        .pipe(lambda d: d[(d["PPI"] >= min_ppi) & (d["CFS"] >= min_cfs)])
        .sort_values(sort_by, ascending=False)
        .head(top_n)
        .assign(Rank=lambda d: range(1, len(d) + 1))
    )

    leader = filtered.iloc[0]
    st.markdown(f"""
    <div class="leader-banner">
        <div>
            <div class="leader-banner-label">👑 Current Leader</div>
            <div class="leader-banner-name">{leader['batter']}</div>
        </div>
        <div style="border-left:1px solid #1E2A3E; height:36px;"></div>
        <div><div class="leader-banner-label">Final Score</div>
             <div class="leader-banner-score" style="color:#00D4AA;font-size:1.3rem;font-weight:800;">{leader['FINAL_SCORE']:.2f}</div></div>
        <div><div class="leader-banner-label">PPI</div><div class="leader-banner-score">{leader['PPI']:.2f}</div></div>
        <div><div class="leader-banner-label">CFS</div><div class="leader-banner-score">{leader['CFS']:.2f}</div></div>
        <div><div class="leader-banner-label">ENS</div><div class="leader-banner-score">{leader['ENS']:.2f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Rankings Table</div>', unsafe_allow_html=True)
    display = filtered[["Rank","batter","FINAL_SCORE","PPI","CFS","ENS","career_avg","career_runs"]].copy()
    display.columns = ["Rank","Batter","Final Score","PPI","CFS","ENS","Career Avg","Career Runs"]
    display = display.round(2)
    st.dataframe(
        display, hide_index=True, width="stretch", height=600,
        column_config={
            "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=100, format="%.2f"),
            "PPI":         st.column_config.ProgressColumn("PPI",         min_value=0, max_value=100, format="%.2f"),
            "CFS":         st.column_config.ProgressColumn("CFS",         min_value=0, max_value=100, format="%.2f"),
            "ENS":         st.column_config.ProgressColumn("ENS",         min_value=0, max_value=100, format="%.2f"),
        }
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Top 20 — Score Breakdown</div>', unsafe_allow_html=True)
    top20 = filtered.head(20)
    fig = go.Figure()
    for label, col, color in [("PPI", "PPI", COLORS["teal"]), ("CFS", "CFS", COLORS["blue"]), ("ENS", "ENS", COLORS["orange"])]:
        fig.add_trace(go.Bar(name=label, x=top20["batter"], y=top20[col], marker_color=color))
    fig.update_layout(**plotly_layout(barmode="group", height=480, xaxis_tickangle=-40, legend=dict(orientation="h", y=1.08)))
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-heading">Score Distribution — All 475 Batters</div>', unsafe_allow_html=True)
    fig2 = px.histogram(master_df, x="FINAL_SCORE", nbins=30, color_discrete_sequence=[COLORS["teal"]])
    fig2.update_traces(marker_line_color=COLORS["teal"], marker_line_width=0.5, opacity=0.85)
    fig2.update_layout(**plotly_layout(height=340, xaxis=dict(title="Final Score"), yaxis=dict(title="Batters")))
    st.plotly_chart(fig2, width="stretch")


# ─── PAGE: PLAYER PROFILE ─────────────────────────────────────────────────────
elif page == "🔍  Player Profile":

    st.markdown("""
    <div class="page-header">
        <h1>Player Profile</h1>
        <p>Full situational breakdown with SHAP-explained ranking drivers</p>
    </div>
    """, unsafe_allow_html=True)

    player_list = master_df.sort_values("FINAL_SCORE", ascending=False)["batter"].tolist()
    selected_player = st.selectbox("Select batter", player_list)

    if selected_player:
        player = master_df[master_df["batter"] == selected_player].iloc[0]
        st.markdown("<br>", unsafe_allow_html=True)

        ring_col, m1, m2, m3, m4 = st.columns([1.2, 1, 1, 1, 1])
        with ring_col:
            st.markdown(f"""
            <div class="score-ring-wrap">
                <div class="score-ring-label">Overall Score</div>
                <div class="score-ring-value">{player['FINAL_SCORE']:.1f}</div>
                <div class="score-ring-rank">Rank #{int(player['FINAL_RANK'])}</div>
            </div>
            """, unsafe_allow_html=True)
        with m1: st.metric("PPI",        f"{player['PPI']:.2f}")
        with m2: st.metric("CFS",        f"{player['CFS']:.2f}")
        with m3: st.metric("ENS",        f"{player['ENS']:.2f}")
        with m4: st.metric("Career Avg", f"{player['career_avg']:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-heading">Radar Profile</div>', unsafe_allow_html=True)
            cats = ["PPI", "CFS", "ENS"]
            vals = [player["PPI"], player["CFS"], player["ENS"]]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(0,212,170,0.12)",
                line=dict(color=COLORS["teal"], width=2), name=selected_player,
            ))
            fig_r.update_layout(
                **PLOTLY_BASE,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1E2A3E", tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor="#1E2A3E"),
                ),
                height=380, showlegend=False,
            )
            st.plotly_chart(fig_r, width="stretch")

        with col2:
            st.markdown('<div class="section-heading">Situation Scores</div>', unsafe_allow_html=True)
            situations = {
                "S1 Early Collapse": player["s1_score"], "S2 Death Chase":    player["s2_score"],
                "S3 World Cup":      player["s3_score"], "S4 Chase Recovery": player["s4_score"],
                "C1 Win Prob Added": player["c1_score"], "C2 Match Winning":  player["c2_score"],
                "C3 Dominance":      player["c3_score"],
            }
            colors = [COLORS["teal"]] * 4 + [COLORS["blue"]] * 3
            fig_b = go.Figure(go.Bar(
                x=list(situations.values()), y=list(situations.keys()),
                orientation="h", marker_color=colors, marker_line_width=0,
            ))
            fig_b.update_layout(**plotly_layout(height=380, xaxis=dict(range=[0, 100]), showlegend=False))
            st.plotly_chart(fig_b, width="stretch")

        st.markdown('<div class="section-heading">Career Statistics</div>', unsafe_allow_html=True)
        cs1, cs2, cs3 = st.columns(3)
        cs1.metric("Career Runs",          f"{int(player['career_runs']):,}")
        cs2.metric("Career Average",       f"{player['career_avg']:.2f}")
        cs3.metric("Era-Adjusted Average", f"{player['era_adj_avg']:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">SHAP — Ranking Drivers</div>', unsafe_allow_html=True)

        feature_cols_xgb = [
            "s1_score","s2_score","s3_score","s4_score",
            "c1_score","c2_score","c3_score",
            "era_adj_avg_norm","volume_norm","opp_quality_norm"
        ]
        try:
            player_features = master_df[master_df["batter"] == selected_player][feature_cols_xgb]
            shap_vals = explainer.shap_values(player_features)
            shap_df = pd.DataFrame({
                "Feature":       feature_cols_xgb,
                "SHAP Impact":   shap_vals[0],
                "Feature Value": player_features.values[0],
            }).sort_values("SHAP Impact", key=abs, ascending=False)
            shap_colors = [COLORS["teal"] if x > 0 else COLORS["red"] for x in shap_df["SHAP Impact"]]
            fig_s = go.Figure(go.Bar(
                x=shap_df["SHAP Impact"], y=shap_df["Feature"],
                orientation="h", marker_color=shap_colors, marker_line_width=0,
            ))
            fig_s.update_layout(**plotly_layout(
                height=380,
                xaxis=dict(title="SHAP value — positive pushes rank up"),
                showlegend=False,
            ))
            st.plotly_chart(fig_s, width="stretch")
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")


# ─── PAGE: HEAD TO HEAD ───────────────────────────────────────────────────────
elif page == "⚔️  Head to Head":

    st.markdown("""
    <div class="page-header">
        <h1>Head to Head</h1>
        <p>Compare any two batters across all context-aware metrics</p>
    </div>
    """, unsafe_allow_html=True)

    player_list = master_df.sort_values("FINAL_SCORE", ascending=False)["batter"].tolist()
    sel1, sel2 = st.columns(2)
    with sel1: player1 = st.selectbox("Batter A", player_list, index=0)
    with sel2: player2 = st.selectbox("Batter B", player_list, index=1)

    if player1 and player2 and player1 != player2:
        p1 = master_df[master_df["batter"] == player1].iloc[0]
        p2 = master_df[master_df["batter"] == player2].iloc[0]

        st.markdown("<br>", unsafe_allow_html=True)
        hc1, hc_mid, hc2 = st.columns([2, 0.8, 2])
        with hc1:
            st.markdown(f"""
            <div class="vs-player-block" style="border-top:2px solid #00D4AA;">
                <div class="vs-player-name">{player1}</div>
                <div class="vs-player-rank">Rank #{int(p1['FINAL_RANK'])}</div>
                <div class="vs-player-score" style="color:#00D4AA;">{p1['FINAL_SCORE']:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with hc_mid:
            st.markdown('<div class="vs-divider" style="height:100%;padding-top:2rem;">VS</div>', unsafe_allow_html=True)
        with hc2:
            st.markdown(f"""
            <div class="vs-player-block" style="border-top:2px solid #4A9EFF;">
                <div class="vs-player-name">{player2}</div>
                <div class="vs-player-rank">Rank #{int(p2['FINAL_RANK'])}</div>
                <div class="vs-player-score" style="color:#4A9EFF;">{p2['FINAL_SCORE']:.2f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Full Profile Comparison</div>', unsafe_allow_html=True)

        categories = ["PPI","CFS","ENS","S1","S2","S3","S4","C1","C2","C3"]
        p1_vals = [p1["PPI"],p1["CFS"],p1["ENS"],p1["s1_score"],p1["s2_score"],p1["s3_score"],p1["s4_score"],p1["c1_score"],p1["c2_score"],p1["c3_score"]]
        p2_vals = [p2["PPI"],p2["CFS"],p2["ENS"],p2["s1_score"],p2["s2_score"],p2["s3_score"],p2["s4_score"],p2["c1_score"],p2["c2_score"],p2["c3_score"]]

        fig_r = go.Figure()
        for name, vals, color, fill in [
            (player1, p1_vals, COLORS["teal"], "rgba(0,212,170,0.12)"),
            (player2, p2_vals, COLORS["blue"], "rgba(74,158,255,0.10)"),
        ]:
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                fill="toself", fillcolor=fill,
                line=dict(color=color, width=2), name=name,
            ))
        fig_r.update_layout(
            **PLOTLY_BASE,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,100], gridcolor="#1E2A3E", tickfont=dict(size=9)),
                angularaxis=dict(gridcolor="#1E2A3E"),
            ),
            height=480,
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_r, width="stretch")

        st.markdown('<div class="section-heading">Metric by Metric</div>', unsafe_allow_html=True)
        metrics = {
            "Final Score":      ("FINAL_SCORE","🏆"), "PPI":              ("PPI","🎯"),
            "CFS":              ("CFS","⚡"),          "ENS":              ("ENS","📈"),
            "S1 Early Collapse":("s1_score","🔴"),     "S2 Death Chase":   ("s2_score","💀"),
            "S3 World Cup":     ("s3_score","🏆"),     "S4 Chase Recovery":("s4_score","🔄"),
            "C1 Win Prob Added":("c1_score","📊"),     "C2 Match Winning": ("c2_score","🎯"),
            "C3 Dominance":     ("c3_score","👑"),     "Career Average":   ("career_avg","📋"),
            "Era Adj Average":  ("era_adj_avg","⏳"),  "Career Runs":      ("career_runs","🏏"),
        }
        rows = []
        for label, (col, icon) in metrics.items():
            v1, v2 = p1[col], p2[col]
            winner = player1 if v1 > v2 else player2 if v2 > v1 else "—"
            rows.append({"Metric": f"{icon} {label}", player1: round(v1,2), player2: round(v2,2), "Winner": winner})
        comp_df = pd.DataFrame(rows)

        def highlight_winner(row):
            styles = ["","","",""]
            if row["Winner"] == player1:
                styles[1] = "background-color:rgba(0,212,170,0.12);color:#00D4AA;font-weight:600"
            elif row["Winner"] == player2:
                styles[2] = "background-color:rgba(74,158,255,0.12);color:#4A9EFF;font-weight:600"
            return styles

        st.dataframe(comp_df.style.apply(highlight_winner, axis=1), width="stretch", hide_index=True)

        p1_wins = sum(1 for _, (col,_) in metrics.items() if p1[col] > p2[col])
        p2_wins = sum(1 for _, (col,_) in metrics.items() if p2[col] > p1[col])
        st.markdown("<br>", unsafe_allow_html=True)
        w1, w2, w3 = st.columns(3)
        w1.metric(f"{player1} wins", f"{p1_wins} metrics")
        w2.metric("Tied",            f"{len(metrics)-p1_wins-p2_wins} metrics")
        w3.metric(f"{player2} wins", f"{p2_wins} metrics")


# ─── PAGE: RISING STARS ───────────────────────────────────────────────────────
elif page == "⭐  Rising Stars":

    st.markdown("""
    <div class="page-header">
        <h1>Rising Stars</h1>
        <p>Which active young batters are tracking like all-time ODI legends?</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rising-info-box">
        <strong>Methodology:</strong> 20-dimensional career trajectory vectors
        (avg, SR, WPA, balls faced, batting position × 4 years) were built for 274 rising stars
        and compared against top-100 legends using cosine similarity.
        Qualification: debut 2019+, 117+ career balls.
    </div>
    """, unsafe_allow_html=True)

    rising_top = rising_df[rising_df["match_rank"] == 1].copy()
    rising_top = rising_top.merge(
        master_df[["batter","FINAL_RANK","FINAL_SCORE"]],
        left_on="legend_match", right_on="batter", how="left"
    ).drop("batter", axis=1)

    col1, col2 = st.columns(2)
    with col1: min_sim = st.slider("Minimum similarity", 0.0, 1.0, 0.70, 0.01)
    with col2:
        legend_filter = st.multiselect("Filter by legend",
                                       options=sorted(rising_top["legend_match"].unique()), default=[])

    filtered_r = rising_top[rising_top["similarity"] >= min_sim]
    if legend_filter:
        filtered_r = filtered_r[filtered_r["legend_match"].isin(legend_filter)]
    filtered_r = filtered_r.sort_values("similarity", ascending=False)

    st.markdown(f'<div class="section-heading">{len(filtered_r)} Players Found</div>', unsafe_allow_html=True)

    disp = filtered_r[["rising_star","legend_match","similarity","FINAL_RANK"]].copy()
    disp.columns = ["Rising Star","Closest Legend","Similarity","Legend Rank"]
    disp["Similarity"]  = disp["Similarity"].round(4)
    disp["Legend Rank"] = disp["Legend Rank"].apply(lambda x: f"#{int(x)}" if pd.notna(x) else "N/A")
    st.dataframe(
        disp, width="stretch", hide_index=True,
        column_config={"Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.4f")}
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Top 15 Similarity Matches</div>', unsafe_allow_html=True)
    top15 = filtered_r.head(15)
    fig = go.Figure(go.Bar(
        x=top15["similarity"],
        y=top15["rising_star"] + " → " + top15["legend_match"],
        orientation="h", marker_color=COLORS["teal"], marker_line_width=0,
        text=top15["similarity"].round(3), textposition="outside",
        textfont=dict(color="#8892A4", size=10),
    ))
    fig.update_layout(**plotly_layout(height=480, xaxis=dict(range=[0, 1.1], title="Cosine Similarity")))
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-heading">Which Legends Are Most Matched?</div>', unsafe_allow_html=True)
    top20_legends = master_df.sort_values("FINAL_SCORE", ascending=False).head(20)["batter"].tolist()
    legend_counts = rising_top["legend_match"].value_counts()
    legend_full   = pd.Series({l: legend_counts.get(l, 0) for l in top20_legends}).sort_values(ascending=True)
    fig2 = go.Figure(go.Bar(
        x=legend_full.values, y=legend_full.index, orientation="h",
        marker_color=[COLORS["red"] if v == 0 else COLORS["blue"] for v in legend_full.values],
        marker_line_width=0,
        text=legend_full.values, textposition="outside",
        textfont=dict(color="#8892A4", size=10),
    ))
    fig2.update_layout(**plotly_layout(height=520, xaxis=dict(title="Rising stars matched")))
    st.plotly_chart(fig2, width="stretch")
    st.caption("Red — unmatched legends (Kohli, ABdV). No current rising star has replicated their early-career trajectory.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Notable Findings</div>', unsafe_allow_html=True)
    n1, n2 = st.columns(2)
    with n1:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-tag tag-cfs">Unmatched</div>
            <h3>Kohli &amp; AB de Villiers</h3>
            <p>No rising star matched either as their primary trajectory match.
            Their early-career combination of quality, volume, WPA, and positional
            dominance remains historically unreplicated.</p>
        </div>""", unsafe_allow_html=True)
    with n2:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-tag tag-ppi">Closest Match</div>
            <h3>Shubman Gill → S. Dhawan</h3>
            <p>Gill's trajectory most closely matches Shikhar Dhawan's early career
            (similarity 0.9376). Both are stylish Indian openers with similar
            average, strike rate, and WPA patterns across their first four years.</p>
        </div>""", unsafe_allow_html=True)


# ─── PAGE: ERA EXPLORER ───────────────────────────────────────────────────────
elif page == "📊  Era Explorer":

    st.markdown("""
    <div class="page-header">
        <h1>Era Explorer</h1>
        <p>How ODI scoring environments changed from 2002 to 2026 — and how we account for it</p>
    </div>
    """, unsafe_allow_html=True)

    era_data = pd.DataFrame({
        "Year": list(range(2002, 2027)),
        "Runs_per_ball": [
            0.8056,0.7059,0.7422,0.7751,0.7411,0.7819,0.7791,
            0.8148,0.8079,0.7898,0.7984,0.8051,0.8421,0.8898,
            0.8763,0.8702,0.8390,0.8708,0.8666,0.7843,0.7920,
            0.8772,0.7983,0.8547,0.8269
        ],
        "Era_difficulty": [
            1.0097,1.1523,1.0959,1.0494,1.0976,1.0403,1.0440,
            0.9983,1.0068,1.0299,1.0188,1.0103,0.9659,0.9141,
            0.9282,0.9347,0.9695,0.9341,0.9386,1.0371,1.0270,
            0.9273,1.0189,0.9517,0.9837
        ]
    })

    tab1, tab2, tab3 = st.tabs(["Scoring Trends", "Era Difficulty", "Top Performers by Era"])

    with tab1:
        st.markdown('<div class="section-heading">Average Runs Per Over — ODI, 2002–2026</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=era_data["Year"], y=era_data["Runs_per_ball"] * 6,
            mode="lines+markers", name="Runs / Over",
            line=dict(color=COLORS["teal"], width=2.5),
            marker=dict(size=7, color=COLORS["teal"]),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
        ))
        for x0, x1, label, col in [
            (2002, 2008, "Pre-T20",       "rgba(0,212,170,0.07)"),
            (2008, 2016, "T20 Influence", "rgba(74,158,255,0.06)"),
            (2016, 2026, "Modern",        "rgba(255,107,53,0.05)"),
        ]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=col,
                          annotation_text=label,
                          annotation_font=dict(size=9, color="#4E5A6E"),
                          annotation_position="top left")
        fig.update_layout(**plotly_layout(
            height=440,
            xaxis=dict(title="Year"),
            yaxis=dict(title="Runs per Over"),
            hovermode="x unified",
        ))
        st.plotly_chart(fig, width="stretch")

        st.markdown("""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-top:0.5rem;">
            <div class="stat-card"><div class="stat-card-label">Pre-T20 avg</div><div class="stat-card-value">4.7</div><div class="stat-card-sub">RPO · 2002–2008</div></div>
            <div class="stat-card"><div class="stat-card-label">Peak scoring</div><div class="stat-card-value">5.34</div><div class="stat-card-sub">RPO · 2015</div></div>
            <div class="stat-card"><div class="stat-card-label">World Cup year</div><div class="stat-card-value">5.26</div><div class="stat-card-sub">RPO · 2023</div></div>
            <div class="stat-card"><div class="stat-card-label">COVID dip</div><div class="stat-card-value">2021–22</div><div class="stat-card-sub">Fewer matches, altered conditions</div></div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-heading">Era Difficulty Factor — Values above 1.0 mean harder scoring</div>', unsafe_allow_html=True)
        bar_colors = [COLORS["teal"] if x > 1 else COLORS["red"] for x in era_data["Era_difficulty"]]
        fig2 = go.Figure(go.Bar(
            x=era_data["Year"], y=era_data["Era_difficulty"],
            marker_color=bar_colors, marker_line_width=0,
            text=era_data["Era_difficulty"].round(3), textposition="outside",
            textfont=dict(color="#8892A4", size=9),
        ))
        fig2.add_hline(y=1.0, line_dash="dot", line_color="#4E5A6E",
                       annotation_text="League average", annotation_font=dict(size=9))
        fig2.update_layout(**plotly_layout(
            height=440,
            xaxis=dict(title="Year"),
            yaxis=dict(title="Difficulty Factor"),
        ))
        st.plotly_chart(fig2, width="stretch")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("""
            <div class="pillar-card">
                <div class="pillar-tag tag-ppi">Hardest Year</div>
                <h3>2003 — Factor 1.1523</h3>
                <p>Pre-powerplay, restrictive fielding rules. Batters from this era get a significant scoring boost in ENS.</p>
            </div>""", unsafe_allow_html=True)
        with d2:
            st.markdown("""
            <div class="pillar-card">
                <div class="pillar-tag tag-ens">Easiest Year</div>
                <h3>2015 — Factor 0.9141</h3>
                <p>Peak T20 influence on ODI batting. Modern techniques fully adopted, fielding restrictions heavily exploited.</p>
            </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-heading">ENS-Ranked Batters by Era</div>', unsafe_allow_html=True)
        era_choice = st.selectbox("Era", ["Pre-T20 (2002–2008)", "T20 Influence (2009–2015)", "Modern (2016–2026)"])
        era_top = (
            master_df.sort_values("ENS", ascending=False)
            .head(50)[["batter","FINAL_RANK","FINAL_SCORE","ENS","era_adj_avg","career_runs"]]
            .head(15).round(2)
        )
        st.dataframe(era_top, width="stretch", hide_index=True)
        st.caption("Era filtering by debut/retirement year requires birth date data not in Cricsheet. ENS-ranked players shown as proxy for era-adjusted performance.")