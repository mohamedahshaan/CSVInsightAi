import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSVInsight AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #F4F7FB !important;
    color: #1A2340 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; }

/* ── Remove default top padding ── */
.main .block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1280px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1730 0%, #0F2050 55%, #0B1730 100%) !important;
    border-right: 1px solid rgba(99,130,255,0.10) !important;
    min-width: 235px !important;
    max-width: 255px !important;
}
[data-testid="stSidebar"] * { color: #B8CCF0 !important; }

[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #7A96C8 !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    text-align: left !important;
    padding: 0.55rem 0.85rem !important;
    border-radius: 9px !important;
    width: 100% !important;
    transition: all 0.16s ease !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,130,255,0.13) !important;
    color: #FFFFFF !important;
    transform: translateX(3px) !important;
}

/* Remove extra gaps between sidebar elements */
[data-testid="stSidebar"] .stButton { margin: 0 !important; padding: 0 !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div { gap: 0 !important; }

/* ── Mobile sidebar toggle ── */
[data-testid="collapsedControl"] {
    background: #112244 !important;
    border-radius: 0 8px 8px 0 !important;
    border: 1px solid rgba(99,130,255,0.35) !important;
    top: 10px !important;
    z-index: 999 !important;
}
[data-testid="collapsedControl"] svg { color: white !important; fill: white !important; }

/* ── Top header ── */
.top-header {
    background: linear-gradient(135deg, #1230A0 0%, #2450D8 55%, #4070F0 100%);
    border-radius: 13px;
    padding: 0.85rem 1.4rem;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 5px 22px rgba(36,80,216,0.26);
}
.hd-sub   { font-size:0.62rem; color:rgba(255,255,255,0.6); font-weight:700; letter-spacing:0.09em; text-transform:uppercase; margin-bottom:0.12rem; }
.hd-title { font-size:1.1rem; font-weight:800; color:white; letter-spacing:-0.01em; }
.hd-badge { background:rgba(255,255,255,0.15); color:white; padding:0.22rem 0.65rem; border-radius:20px; font-size:0.68rem; font-weight:700; letter-spacing:0.05em; border:1px solid rgba(255,255,255,0.2); }
.hd-av    { width:32px; height:32px; background:linear-gradient(135deg,#A78BFA,#38BDF8); border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.75rem; color:white; border:2px solid rgba(255,255,255,0.3); margin-left:0.6rem; }

/* ── Section titles ── */
.sec-title { font-size:1.2rem; font-weight:800; color:#0F1C3F; letter-spacing:-0.02em; margin-bottom:0.14rem; }
.sec-sub   { font-size:0.78rem; color:#64748B; margin-bottom:0.9rem; line-height:1.5; }

/* ── Card ── */
.card {
    background: white;
    border-radius: 13px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 9px rgba(0,0,0,0.05);
    border: 1px solid rgba(226,232,240,0.9);
    margin-bottom: 0.8rem;
    transition: box-shadow 0.2s;
}
.card:hover { box-shadow: 0 5px 20px rgba(36,80,216,0.09); }
.card-title {
    font-size:0.82rem; font-weight:700; color:#1E293B;
    margin-bottom:0.75rem; padding-bottom:0.45rem;
    border-bottom:1px solid #F1F5F9;
    display:flex; align-items:center; gap:0.35rem;
}

/* ── Metric cards ── */
.metric-card {
    background:white; border-radius:11px; padding:0.85rem 1rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.04);
    border:1px solid rgba(226,232,240,0.9);
    transition:transform 0.18s, box-shadow 0.18s;
}
.metric-card:hover { transform:translateY(-2px); box-shadow:0 5px 18px rgba(36,80,216,0.1); }
.m-lbl { font-size:0.62rem; color:#94A3B8; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; }
.m-ico { font-size:1.3rem; margin:0.22rem 0 0.1rem; }
.m-val { font-size:1.65rem; font-weight:800; color:#0F1C3F; line-height:1; }
.m-sub { font-size:0.68rem; color:#64748B; margin-top:0.18rem; }

/* ── Badges ── */
.b-ok   { background:#DCFCE7; color:#15803D; padding:0.18rem 0.6rem; border-radius:20px; font-size:0.7rem; font-weight:700; border:1px solid #86EFAC; display:inline-block; }
.b-warn { background:#FEF3C7; color:#B45309; padding:0.18rem 0.6rem; border-radius:20px; font-size:0.7rem; font-weight:700; border:1px solid #FCD34D; display:inline-block; }
.b-err  { background:#FEE2E2; color:#B91C1C; padding:0.18rem 0.6rem; border-radius:20px; font-size:0.7rem; font-weight:700; border:1px solid #FCA5A5; display:inline-block; }

/* ── Info box ── */
.info-box { background:linear-gradient(135deg,#EFF6FF,#DBEAFE); border:1px solid #BFDBFE; border-radius:11px; padding:0.7rem 1rem; font-size:0.8rem; color:#1E40AF; margin-bottom:0.75rem; line-height:1.5; }

/* ── Welcome hero ── */
.welcome-hero {
    background: linear-gradient(135deg, #0a0f2e 0%, #0f1c5a 35%, #1a3a9a 70%, #2450d8 100%);
    border-radius: 18px;
    padding: 2.4rem 2.5rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 36px rgba(10,15,46,0.38);
    position: relative;
    overflow: hidden;
}
.welcome-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,130,255,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.wh-eyebrow { font-size:0.7rem; color:rgba(255,255,255,0.45); font-weight:700; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.5rem; }
.wh-title   { font-size:2.3rem; font-weight:900; color:white; letter-spacing:-0.03em; line-height:1.1; margin-bottom:0.65rem; }
.wh-title span { color:#93C5FD; }
.wh-desc    { font-size:0.92rem; color:rgba(255,255,255,0.62); line-height:1.7; max-width:560px; margin-bottom:1.2rem; }
.wh-pills   { display:flex; gap:0.55rem; flex-wrap:wrap; }
.wh-pill    { background:rgba(255,255,255,0.09); border:1px solid rgba(255,255,255,0.15); border-radius:20px; padding:0.32rem 0.85rem; font-size:0.74rem; color:rgba(255,255,255,0.78); font-weight:600; }

/* ── Upload result box ── */
.upload-result {
    background: linear-gradient(135deg,#f0fdf4,#dcfce7);
    border:1px solid #86efac; border-radius:13px;
    padding:1rem 1.2rem; margin-top:0.6rem;
    display:flex; align-items:flex-start; gap:0.75rem;
}
.ur-icon { font-size:1.5rem; }
.ur-title { font-weight:700; color:#15803D; font-size:0.88rem; }
.ur-sub   { color:#166534; font-size:0.78rem; margin-top:0.12rem; }
.ur-stats { display:flex; gap:0.8rem; margin-top:0.6rem; flex-wrap:wrap; }
.ur-stat  { background:white; border:1px solid #bbf7d0; border-radius:8px; padding:0.3rem 0.7rem; font-size:0.75rem; font-weight:700; color:#166534; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#2450D8,#4070F0) !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important; font-weight: 700 !important;
    font-size: 0.84rem !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 13px rgba(36,80,216,0.3) !important;
    transition: all 0.18s !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 20px rgba(36,80,216,0.4) !important;
}

/* ── Insight card ── */
.insight-card { background:linear-gradient(135deg,#1E3A8A,#1D4ED8); border-radius:13px; padding:1.1rem 1.3rem; color:white; box-shadow:0 5px 20px rgba(30,58,138,0.28); }
.insight-card h4 { font-size:0.88rem; font-weight:700; margin-bottom:0.5rem; opacity:0.92; }
.insight-card p  { font-size:0.79rem; opacity:0.76; line-height:1.6; }

/* ── Sidebar labels ── */
.sb-logo  { padding:1rem 0.9rem 0.8rem; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:0.4rem; }
.sb-lname { font-size:0.95rem; font-weight:800; color:white !important; letter-spacing:-0.01em; }
.sb-lsub  { font-size:0.61rem; color:#3D5A8A !important; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; margin-top:2px; }
.sb-nlbl  { padding:0.5rem 0.9rem 0.2rem; font-size:0.6rem; font-weight:700; color:#2D4470 !important; letter-spacing:0.1em; text-transform:uppercase; }
.sb-status{ margin:0.5rem 0.6rem; border-radius:9px; padding:0.55rem 0.75rem; border:1px solid; }
.sb-status.on  { background:rgba(34,197,94,0.07); border-color:rgba(34,197,94,0.2); }
.sb-status.off { background:rgba(148,163,184,0.06); border-color:rgba(148,163,184,0.14); }

/* ── Divider ── */
.divider { height:1px; background:linear-gradient(90deg,transparent,#CBD5E1,transparent); margin:0.8rem 0; }

/* ── Footer ── */
.footer { background:#0B1730; border-radius:12px; padding:0.85rem 1.4rem; margin-top:1.5rem; text-align:center; border:1px solid rgba(99,130,255,0.08); }
.footer p { font-size:0.74rem; color:#3D5080; margin:0; letter-spacing:0.01em; }

/* ── Founder showcase ── */
.founder-hero {
    background:linear-gradient(135deg,#0F1C3F,#1A3A8A,#1E3A7A);
    border-radius:16px; padding:2.2rem 2rem;
    text-align:center; color:white;
    box-shadow:0 10px 36px rgba(15,28,63,0.35);
    margin-bottom:1rem;
}
.f-av   { width:80px;height:80px;background:linear-gradient(135deg,#A78BFA,#38BDF8);border-radius:50%;margin:0 auto 1rem;display:flex;align-items:center;justify-content:center;font-size:2rem;font-weight:800;border:3px solid rgba(255,255,255,0.22);box-shadow:0 4px 16px rgba(0,0,0,0.25); }
.f-name { font-size:1.5rem; font-weight:800; letter-spacing:-0.01em; }
.f-role { font-size:0.82rem; color:rgba(255,255,255,0.6); margin-top:0.3rem; }
.f-uni  { font-size:0.78rem; color:#93C5FD; margin-top:0.18rem; font-weight:600; }
.f-bio  { font-size:0.84rem; color:rgba(255,255,255,0.62); margin-top:1rem; line-height:1.7; max-width:520px; margin-left:auto; margin-right:auto; }
.f-tags { display:flex; justify-content:center; gap:0.5rem; margin-top:1rem; flex-wrap:wrap; }
.f-tag  { background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.14); border-radius:20px; padding:0.28rem 0.8rem; font-size:0.74rem; color:rgba(255,255,255,0.75); font-weight:600; }
.f-links{ display:flex; justify-content:center; gap:0.6rem; margin-top:1.1rem; flex-wrap:wrap; }
.f-link { display:inline-flex;align-items:center;gap:0.38rem;padding:0.45rem 0.95rem;border-radius:20px;font-size:0.77rem;font-weight:600;border:1px solid rgba(255,255,255,0.15);text-decoration:none;transition:all 0.16s; }
.f-link:hover { transform:translateY(-2px); border-color:rgba(255,255,255,0.3); }
.f-link.wa { background:rgba(37,211,102,0.18); color:#4ADE80 !important; }
.f-link.em { background:rgba(239,68,68,0.18);  color:#FCA5A5 !important; }
.f-link.li { background:rgba(10,102,194,0.22);  color:#60A5FA !important; }
.f-link.gh { background:rgba(255,255,255,0.07); color:white !important; }

/* ── Workflow steps ── */
.wf-step { background:white; border-radius:11px; padding:0.9rem 1.05rem; border:1px solid #E2E8F0; border-left:4px solid #3B82F6; box-shadow:0 2px 7px rgba(0,0,0,0.04); transition:all 0.18s; margin-bottom:0.6rem; }
.wf-step:hover { box-shadow:0 5px 16px rgba(36,80,216,0.09); transform:translateY(-2px); }
.wf-num  { font-size:0.62rem; font-weight:800; color:#3B82F6; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.18rem; }
.wf-name { font-size:0.9rem; font-weight:700; color:#0F1C3F; }
.wf-desc { font-size:0.78rem; color:#64748B; margin-top:0.25rem; line-height:1.5; }

/* ── Selects/inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    border-radius:9px !important; border-color:#E2E8F0 !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
}
[data-testid="stDataFrame"] { border-radius:9px !important; overflow:hidden !important; }

/* ── Remove Streamlit element spacing ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] { gap: 0 !important; }
.element-container { margin-bottom: 0 !important; }
div[data-testid="stFileUploader"] { margin-top: -0.5rem; }
div[data-testid="stFileUploader"] label { display:none !important; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .main .block-container { padding:0.5rem 0.7rem 1.2rem !important; }
    .top-header { padding:0.7rem 0.9rem !important; }
    .hd-title   { font-size:0.9rem !important; }
    .welcome-hero { padding:1.6rem 1.3rem 1.4rem !important; }
    .wh-title   { font-size:1.6rem !important; }
    .founder-hero { padding:1.5rem 1rem !important; }
    .f-links, .f-tags { gap:0.4rem; }
}
@media (max-width: 480px) {
    .hd-badge { display:none; }
    .wh-pill  { font-size:0.68rem; padding:0.26rem 0.7rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
for k, v in {
    "page": "Upload Data", "df": None, "df_clean": None,
    "trained_models": {}, "model_metrics": {},
    "target_col": None, "le_map": {}, "feature_cols": [],
    "best_model_name": None, "X_test": None, "y_test": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div style="font-size:1.5rem;margin-bottom:0.28rem;">🧠</div>
        <div class="sb-lname">CSVInsight AI</div>
        <div class="sb-lsub">Data Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-nlbl">Main Menu</div>', unsafe_allow_html=True)
    for icon, label in [
        ("📁", "Upload Data"), ("📊", "Data Overview"), ("🧹", "Data Cleaning"),
        ("📈", "EDA Visualization"), ("🧠", "Model Training"), ("🏆", "Results Dashboard")
    ]:
        if st.session_state.page == label:
            st.markdown(f"""
            <div style="background:rgba(99,130,255,0.17);border-radius:9px;padding:0.52rem 0.85rem;
                margin:0.08rem 0;border-left:3px solid #6382FF;">
                <span style="font-size:0.84rem;font-weight:600;color:#FFFFFF;">{icon}&nbsp; {label}</span>
            </div>""", unsafe_allow_html=True)
        else:
            if st.button(f"{icon}  {label}", key=f"nav_{label}"):
                st.session_state.page = label
                st.rerun()

    st.markdown('<div class="sb-nlbl" style="margin-top:0.2rem;">Info</div>', unsafe_allow_html=True)
    for icon, label in [("ℹ️", "About"), ("👤", "Founder")]:
        if st.session_state.page == label:
            st.markdown(f"""
            <div style="background:rgba(99,130,255,0.17);border-radius:9px;padding:0.52rem 0.85rem;
                margin:0.08rem 0;border-left:3px solid #6382FF;">
                <span style="font-size:0.84rem;font-weight:600;color:#FFFFFF;">{icon}&nbsp; {label}</span>
            </div>""", unsafe_allow_html=True)
        else:
            if st.button(f"{icon}  {label}", key=f"nav_{label}"):
                st.session_state.page = label
                st.rerun()

    if st.session_state.df is not None:
        d = st.session_state.df
        st.markdown(f"""
        <div class="sb-status on">
            <div style="font-size:0.63rem;font-weight:700;color:#4ADE80;letter-spacing:0.05em;">✓ DATA LOADED</div>
            <div style="font-size:0.73rem;color:#94A3B8;margin-top:2px;">{d.shape[0]:,} rows × {d.shape[1]} cols</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sb-status off">
            <div style="font-size:0.63rem;font-weight:700;color:#64748B;letter-spacing:0.03em;">○ NO DATA</div>
            <div style="font-size:0.73rem;color:#4A5A72;margin-top:2px;">Upload a CSV to begin</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="padding:0.5rem;font-size:0.62rem;color:#1E2E4A;text-align:center;">CSVInsight AI v2.1</div>', unsafe_allow_html=True)


# ── Top header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-header">
    <div>
        <div class="hd-sub">CSVInsight AI · Intelligent Data Analysis</div>
        <div class="hd-title">{st.session_state.page}</div>
    </div>
    <div style="display:flex;align-items:center;gap:0.55rem;">
        <div class="hd-badge">FREE TIER</div>
        <div class="hd-av">AI</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Footer helper ──────────────────────────────────────────────────────────────
def footer():
    st.markdown("""
    <div class="footer">
        <p>© 2026 CSVInsight AI · Built by H. Mohamed Ahshaan · All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Upload Data":

    # ── Welcome hero ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="welcome-hero">
        <div class="wh-eyebrow">Welcome to</div>
        <div class="wh-title">CSVInsight <span>AI</span></div>
        <div class="wh-desc">
            Your intelligent data analysis companion. Upload any CSV file and get instant
            insights — explore patterns, clean your data, train machine learning models,
            and compare results. No coding required.
        </div>
        <div class="wh-pills">
            <div class="wh-pill">📁 Upload CSV</div>
            <div class="wh-pill">📊 Explore Data</div>
            <div class="wh-pill">🧹 Clean</div>
            <div class="wh-pill">📈 Visualise</div>
            <div class="wh-pill">🧠 Train Models</div>
            <div class="wh-pill">🏆 Get Results</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Upload Your CSV File</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Drag and drop or click to select your CSV file — analysis starts instantly.</div>', unsafe_allow_html=True)

    col_up, col_info = st.columns([3, 2], gap="medium")

    with col_up:
        uploaded = st.file_uploader(
            "Drop your CSV file here or click to browse",
            type=["csv"],
            help="Supports .csv files up to 200 MB",
        )

        if uploaded:
            with st.spinner("Reading your file…"):
                time.sleep(0.2)
                try:
                    df = pd.read_csv(uploaded)
                    st.session_state.df = df
                    st.session_state.df_clean = df.copy()
                    st.session_state.trained_models = {}
                    st.session_state.model_metrics = {}
                    st.session_state.best_model_name = None

                    # ── Rows / Columns preview ─────────────────────────────
                    num_cols_count  = df.select_dtypes(include=np.number).shape[1]
                    cat_cols_count  = df.select_dtypes(include="object").shape[1]
                    miss_total      = int(df.isnull().sum().sum())

                    st.markdown(f"""
                    <div class="upload-result">
                        <div class="ur-icon">✅</div>
                        <div style="flex:1;">
                            <div class="ur-title">File uploaded successfully!</div>
                            <div class="ur-sub">📄 {uploaded.name}</div>
                            <div class="ur-stats">
                                <div class="ur-stat">📋 {df.shape[0]:,} Rows</div>
                                <div class="ur-stat">📐 {df.shape[1]} Columns</div>
                                <div class="ur-stat">🔢 {num_cols_count} Numeric</div>
                                <div class="ur-stat">🏷️ {cat_cols_count} Categorical</div>
                                <div class="ur-stat">{"⚠️" if miss_total else "✓"} {miss_total:,} Missing</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Column names preview
                    st.markdown("""
                    <div class="info-box" style="margin-top:0.7rem;">
                        👈 Use the sidebar to navigate — start with <strong>Data Overview</strong> to explore your dataset.
                    </div>
                    """, unsafe_allow_html=True)

                    # Quick column list
                    st.markdown('<div class="card" style="margin-top:0.5rem;"><div class="card-title">🗂️ Column Preview</div>', unsafe_allow_html=True)
                    col_preview_data = []
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        nulls = int(df[col].isnull().sum())
                        uniq  = int(df[col].nunique())
                        col_preview_data.append({"Column": col, "Type": dtype, "Nulls": nulls, "Unique": uniq})
                    st.dataframe(pd.DataFrame(col_preview_data), use_container_width=True, height=220, hide_index=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Could not read file: {e}")

        elif st.session_state.df is not None:
            df = st.session_state.df
            st.markdown(f"""
            <div class="upload-result">
                <div class="ur-icon">✅</div>
                <div style="flex:1;">
                    <div class="ur-title">Dataset already loaded</div>
                    <div class="ur-stats">
                        <div class="ur-stat">📋 {df.shape[0]:,} Rows</div>
                        <div class="ur-stat">📐 {df.shape[1]} Columns</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="card"><div class="card-title">📋 How to Use</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("1", "Upload your CSV", "Drag & drop or click the upload area."),
            ("2", "Check Data Overview", "Review shape, types, and statistics."),
            ("3", "Clean the Data", "Handle missing values and duplicates."),
            ("4", "Visualise Patterns", "Explore charts and correlations."),
            ("5", "Train ML Models", "Pick a target column and train models."),
            ("6", "View Results", "Compare models and get actionable insights."),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:0.65rem;align-items:flex-start;margin-bottom:0.6rem;">
                <div style="min-width:22px;height:22px;background:#DBEAFE;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.67rem;font-weight:800;color:#1E40AF;margin-top:1px;flex-shrink:0;">{num}</div>
                <div>
                    <div style="font-size:0.8rem;font-weight:700;color:#1E293B;">{title}</div>
                    <div style="font-size:0.73rem;color:#64748B;margin-top:0.08rem;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">💡 CSV Requirements</div>', unsafe_allow_html=True)
        for r in [
            "Header row in the first line",
            "Columns separated by commas",
            "At least one numeric column",
            "A target column for ML training",
            "UTF-8 encoding preferred",
        ]:
            st.markdown(f'<div style="font-size:0.78rem;color:#475569;padding:0.26rem 0;border-bottom:1px solid #F8FAFC;">✓ {r}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Data Overview":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a CSV file first from the Upload Data page.")
        st.stop()

    df = st.session_state.df
    st.markdown('<div class="sec-title">Data Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">A full structural snapshot of your dataset — rows, columns, types, and statistics.</div>', unsafe_allow_html=True)

    total_miss = df.isnull().sum().sum()
    pct_miss   = total_miss / (df.shape[0] * df.shape[1]) * 100
    num_c = df.select_dtypes(include=np.number).shape[1]
    cat_c = df.select_dtypes(include="object").shape[1]

    cols5 = st.columns(5, gap="small")
    for col, (lbl, val, sub, ico) in zip(cols5, [
        ("ROWS",      f"{df.shape[0]:,}", "Total records",   "📋"),
        ("COLUMNS",   f"{df.shape[1]}",   "Total features",  "📐"),
        ("NUMERIC",   f"{num_c}",          "Numeric cols",    "🔢"),
        ("CATEGORIC", f"{cat_c}",          "Text cols",       "🏷️"),
        ("MISSING",   f"{total_miss:,}",   f"{pct_miss:.1f}% of cells", "⚠️"),
    ]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="m-lbl">{lbl}</div>
            <div style="display:flex;align-items:center;gap:0.35rem;margin-top:0.22rem;">
                <span style="font-size:1.2rem;">{ico}</span>
                <span class="m-val">{val}</span>
            </div>
            <div class="m-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📋 Dataset Preview — First 10 Rows</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="card"><div class="card-title">🗂️ Column Info</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Column":   df.columns,
            "Type":     df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Nulls":    df.isnull().sum().values,
            "Unique":   [df[c].nunique() for c in df.columns],
        }), use_container_width=True, height=255, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">📊 Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().round(3), use_container_width=True, height=255)
        st.markdown("</div>", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Data Cleaning":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a CSV file first.")
        st.stop()

    df = st.session_state.df.copy()
    st.markdown('<div class="sec-title">Data Cleaning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Detect and fix missing values and duplicates before modelling.</div>', unsafe_allow_html=True)

    missing  = df.isnull().sum()
    miss_pct = (missing / len(df) * 100).round(2)
    miss_df  = pd.DataFrame({
        "Column": missing.index, "Count": missing.values, "Pct": miss_pct.values
    }).sort_values("Pct", ascending=False)

    st.markdown('<div class="card"><div class="card-title">🔍 Missing Value Analysis</div>', unsafe_allow_html=True)
    if missing.sum() == 0:
        st.markdown('<div style="text-align:center;padding:1.2rem;color:#15803D;font-weight:700;font-size:0.9rem;">✅ No missing values detected — your dataset is clean!</div>', unsafe_allow_html=True)
    else:
        for _, row in miss_df.iterrows():
            pct = row["Pct"]
            if pct == 0:   badge = '<span class="b-ok">✓ Clean</span>';    bar_c = "#22C55E"
            elif pct < 10: badge = f'<span class="b-warn">⚠ {pct}%</span>'; bar_c = "#F59E0B"
            else:          badge = f'<span class="b-err">✗ {pct}%</span>';  bar_c = "#EF4444"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.8rem;padding:0.42rem 0;border-bottom:1px solid #F1F5F9;">
                <div style="width:130px;font-size:0.77rem;font-weight:600;color:#334155;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{row["Column"]}</div>
                <div style="flex:1;background:#F1F5F9;border-radius:4px;height:6px;overflow:hidden;">
                    <div style="width:{min(pct,100)}%;background:{bar_c};height:100%;border-radius:4px;"></div>
                </div>
                <div style="width:52px;font-size:0.76rem;color:#64748B;text-align:right;">{int(row["Count"])}</div>
                <div style="width:90px;text-align:right;">{badge}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🛠️ Cleaning Strategy</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1: num_s = st.selectbox("Numeric missing values", ["Fill with Median", "Fill with Mean", "Fill with 0", "Drop rows"])
    with c2: cat_s = st.selectbox("Categorical missing values", ["Fill with Mode", "Fill with 'Unknown'", "Drop rows"])
    drop_d = st.checkbox("🗑️ Remove duplicate rows", value=True)

    if st.button("⚡ Apply Cleaning", type="primary"):
        df_c = st.session_state.df.copy()
        nc = df_c.select_dtypes(include=np.number).columns.tolist()
        cc = df_c.select_dtypes(include="object").columns.tolist()
        if   num_s == "Fill with Median": df_c[nc] = df_c[nc].fillna(df_c[nc].median())
        elif num_s == "Fill with Mean":   df_c[nc] = df_c[nc].fillna(df_c[nc].mean())
        elif num_s == "Fill with 0":      df_c[nc] = df_c[nc].fillna(0)
        elif num_s == "Drop rows":        df_c = df_c.dropna(subset=nc)
        if   cat_s == "Fill with Mode":
            for c in cc:
                df_c[c] = df_c[c].fillna(df_c[c].mode()[0] if not df_c[c].mode().empty else "Unknown")
        elif cat_s == "Fill with 'Unknown'": df_c[cc] = df_c[cc].fillna("Unknown")
        elif cat_s == "Drop rows":           df_c = df_c.dropna(subset=cc)
        removed = 0
        if drop_d:
            before = len(df_c); df_c = df_c.drop_duplicates(); removed = before - len(df_c)
        st.session_state.df_clean = df_c
        st.markdown(f"""
        <div style="background:#DCFCE7;border:1px solid #86EFAC;border-radius:10px;padding:0.8rem 1rem;margin-top:0.6rem;">
            <div style="font-weight:700;color:#15803D;font-size:0.86rem;">✅ Cleaning Applied</div>
            <div style="color:#166534;font-size:0.77rem;margin-top:0.15rem;">
                {df_c.shape[0]:,} rows remaining · {df_c.isnull().sum().sum()} missing values · {removed} duplicates removed
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    dups = df.duplicated().sum()
    st.markdown(f'<div class="info-box">🔁 <strong>{dups}</strong> duplicate rows found ({dups/len(df)*100:.2f}% of dataset)</div>', unsafe_allow_html=True)
    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "EDA Visualization":
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
    if df is None:
        st.warning("⚠️ Please upload a CSV file first.")
        st.stop()

    st.markdown('<div class="sec-title">EDA Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Explore distributions, relationships, and patterns in your data visually.</div>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found for visualization.")
        st.stop()

    s1, s2 = st.columns(2, gap="medium")
    with s1: hc = st.selectbox("📊 Histogram column", num_cols)
    with s2: bc = st.selectbox("📦 Boxplot column", num_cols, index=min(1, len(num_cols) - 1))

    r1, r2 = st.columns(2, gap="medium")
    PLOT_H = 265
    LAYOUT = dict(margin=dict(l=0, r=0, t=12, b=0), height=PLOT_H,
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                  font=dict(family="Plus Jakarta Sans"))

    with r1:
        st.markdown('<div class="card"><div class="card-title">📊 Distribution Histogram</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x=hc, nbins=30, color_discrete_sequence=["#3B82F6"], template="plotly_white")
        fig.update_layout(**LAYOUT, bargap=0.06)
        fig.update_traces(marker_line_color="#2D5BE3", marker_line_width=0.5)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="card"><div class="card-title">📦 Box Plot</div>', unsafe_allow_html=True)
        grp = st.selectbox("Group by", ["None"] + cat_cols, key="bgrp")
        fig = px.box(df, y=bc, x=grp if grp != "None" else None,
                     color=grp if grp != "None" else None,
                     template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(**LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    r3, r4 = st.columns(2, gap="medium")
    with r3:
        st.markdown('<div class="card"><div class="card-title">🌡️ Correlation Heatmap</div>', unsafe_allow_html=True)
        if len(num_cols) >= 2:
            fig = px.imshow(df[num_cols].corr(), text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_white")
            fig.update_layout(**{**LAYOUT, "height": 290, "font": dict(family="Plus Jakarta Sans", size=9)})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns.")
        st.markdown("</div>", unsafe_allow_html=True)

    with r4:
        st.markdown('<div class="card"><div class="card-title">🔵 Scatter Plot</div>', unsafe_allow_html=True)
        if len(num_cols) >= 2:
            sx = st.selectbox("X axis", num_cols, key="sx")
            sy = st.selectbox("Y axis", num_cols, index=1, key="sy")
            sc = st.selectbox("Color", ["None"] + cat_cols, key="sc")
            fig = px.scatter(df, x=sx, y=sy, color=sc if sc != "None" else None,
                             template="plotly_white", opacity=0.72,
                             color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(**{**LAYOUT, "height": 230})
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if cat_cols:
        st.markdown('<div class="card"><div class="card-title">🏷️ Categorical Distribution</div>', unsafe_allow_html=True)
        cs = st.selectbox("Select categorical column", cat_cols)
        vc = df[cs].value_counts().reset_index()
        vc.columns = [cs, "Count"]
        fig = px.bar(vc, x=cs, y="Count", template="plotly_white", color="Count", color_continuous_scale="Blues")
        fig.update_layout(**{**LAYOUT, "height": 270, "coloraxis_showscale": False})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING  (fixed)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Model Training":
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
    if df is None:
        st.warning("⚠️ Please upload a CSV file first.")
        st.stop()

    st.markdown('<div class="sec-title">Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Select a target column, choose algorithms, and train with one click.</div>', unsafe_allow_html=True)

    MODEL_MAP = {
        "Logistic Regression":    LogisticRegression(max_iter=1000, solver="lbfgs"),
        "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":      GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors":    KNeighborsClassifier(),
        "Decision Tree":          DecisionTreeClassifier(random_state=42),
    }

    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        st.markdown('<div class="card"><div class="card-title">⚙️ Configuration</div>', unsafe_allow_html=True)
        target = st.selectbox("🎯 Target column (what to predict)", df.columns.tolist())
        st.session_state.target_col = target
        sel_m  = st.multiselect("🤖 Algorithms", list(MODEL_MAP.keys()),
                                default=["Logistic Regression", "Random Forest", "Decision Tree"])
        ts = st.slider("📐 Test set size", 0.10, 0.40, 0.20, 0.05)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">📊 Class Distribution</div>', unsafe_allow_html=True)
        vc2 = df[target].value_counts()
        fig = px.pie(values=vc2.values, names=vc2.index.astype(str), hole=0.55,
                     template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(l=0, r=0, t=5, b=0), height=170,
                          paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(font=dict(size=10, family="Plus Jakarta Sans")))
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div style="font-size:0.77rem;color:#64748B;">
            <span class="b-ok">Train</span> {int(len(df)*(1-ts)):,} &nbsp;
            <span class="b-warn">Test</span> {int(len(df)*ts):,}
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀  Start Training", type="primary", use_container_width=True):
        if not sel_m:
            st.error("Please select at least one algorithm.")
        else:
            try:
                prog = st.progress(0)
                stat = st.empty()

                # ── Prepare data ───────────────────────────────────────────
                df_m = df.copy().dropna(subset=[target])
                if len(df_m) < 10:
                    st.error("❌ Not enough rows after removing missing target values (need at least 10).")
                    st.stop()

                feat = [c for c in df_m.columns if c != target]
                X = df_m[feat].copy()
                y = df_m[target].copy()

                # Encode categorical features
                le_map = {}
                for c in X.select_dtypes(include="object").columns:
                    le = LabelEncoder()
                    X[c] = le.fit_transform(X[c].astype(str))
                    le_map[c] = le

                # Encode target
                le_y = None
                if y.dtype == "object" or str(y.dtype) == "category":
                    le_y = LabelEncoder()
                    y = pd.Series(le_y.fit_transform(y.astype(str)), index=y.index)
                else:
                    y = y.reset_index(drop=True)
                    X = X.reset_index(drop=True)

                # Fill remaining NaNs
                X = X.fillna(X.median(numeric_only=True))
                X = X.fillna(0)  # catch any remaining non-numeric NaN

                # Scale
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)

                # ── Stratify only when safe ────────────────────────────────
                unique_classes, class_counts = np.unique(y, return_counts=True)
                n_classes = len(unique_classes)
                min_class_count = class_counts.min()
                n_test = max(1, int(len(df_m) * ts))

                # Stratify requires at least 2 samples per class in both splits
                can_stratify = (n_classes > 1) and (min_class_count >= 2) and (n_test >= n_classes)

                Xtr, Xte, ytr, yte = train_test_split(
                    Xs, y,
                    test_size=ts,
                    random_state=42,
                    stratify=y if can_stratify else None,
                )

                st.session_state.X_test = Xte
                st.session_state.y_test = yte
                st.session_state.feature_cols = feat
                st.session_state.le_map = le_map

                if not can_stratify:
                    st.info("ℹ️ Stratified split skipped — some classes have too few samples. Using random split instead.")

                # ── Train each model ───────────────────────────────────────
                metrics = {}
                trained = {}

                for i, name in enumerate(sel_m):
                    stat.markdown(
                        f'<div style="text-align:center;font-size:0.86rem;font-weight:600;color:#1E293B;padding:0.4rem;">'
                        f'⚙️ Training <span style="color:#3B82F6;">{name}</span>  {i+1}/{len(sel_m)}</div>',
                        unsafe_allow_html=True,
                    )
                    prog.progress((i) / len(sel_m))

                    try:
                        clf = MODEL_MAP[name]
                        clf.fit(Xtr, ytr)
                        yp = clf.predict(Xte)

                        # ROC-AUC only for binary classification
                        ypr = None
                        is_binary = len(np.unique(y)) == 2
                        if is_binary and hasattr(clf, "predict_proba"):
                            try:
                                ypr = clf.predict_proba(Xte)[:, 1]
                            except Exception:
                                ypr = None

                        m = {
                            "Accuracy":  round(accuracy_score(yte, yp), 4),
                            "Precision": round(precision_score(yte, yp, average="weighted", zero_division=0), 4),
                            "Recall":    round(recall_score(yte, yp, average="weighted", zero_division=0), 4),
                            "F1-Score":  round(f1_score(yte, yp, average="weighted", zero_division=0), 4),
                            "ROC-AUC":   "N/A",
                            "_cm":       confusion_matrix(yte, yp),
                            "_y_pred":   yp,
                        }
                        if ypr is not None:
                            try:
                                fpr, tpr, _ = roc_curve(yte, ypr)
                                m["ROC-AUC"] = round(auc(fpr, tpr), 4)
                                m["_fpr"] = fpr
                                m["_tpr"] = tpr
                            except Exception:
                                pass

                        metrics[name] = m
                        trained[name] = clf

                    except Exception as model_err:
                        st.warning(f"⚠️ {name} failed to train: {model_err}")

                    time.sleep(0.15)

                prog.progress(1.0)
                stat.empty()

                if not metrics:
                    st.error("❌ All models failed to train. Check your data and target column.")
                else:
                    st.session_state.trained_models = trained
                    st.session_state.model_metrics  = metrics
                    best = max(metrics, key=lambda x: metrics[x]["Accuracy"])
                    st.session_state.best_model_name = best
                    st.success(
                        f"✅ Training complete! {len(metrics)} model(s) trained. "
                        f"Best: **{best}** — {metrics[best]['Accuracy']*100:.2f}% accuracy"
                    )

            except Exception as e:
                st.error(f"❌ Training error: {e}")

    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Results Dashboard":
    if not st.session_state.model_metrics:
        st.warning("⚠️ No trained models yet. Please go to Model Training first.")
        st.stop()

    metrics = st.session_state.model_metrics
    best    = st.session_state.best_model_name
    bm      = metrics[best]

    st.markdown('<div class="sec-title">Results Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Compare model performance, inspect charts, and extract actionable insights.</div>', unsafe_allow_html=True)

    cm4 = st.columns(4, gap="small")
    for col, (lbl, val, sub, ico) in zip(cm4, [
        ("BEST ACCURACY", f"{bm['Accuracy']*100:.1f}%",  best,           "🏆"),
        ("F1-SCORE",      f"{bm['F1-Score']*100:.1f}%",  "Weighted avg", "🎯"),
        ("PRECISION",     f"{bm['Precision']*100:.1f}%", "Weighted avg", "🔬"),
        ("TEST SAMPLES",  str(st.session_state.y_test.shape[0]), "Held-out", "📐"),
    ]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="m-lbl">{lbl}</div>
            <div style="display:flex;align-items:center;gap:0.35rem;margin-top:0.2rem;">
                <span style="font-size:1.1rem;">{ico}</span>
                <span class="m-val">{val}</span>
            </div>
            <div class="m-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🏆 Model Comparison Table</div>', unsafe_allow_html=True)
    rows = []
    for n, m in metrics.items():
        rows.append({
            "Model":     ("⭐ " if n == best else "") + n,
            "Accuracy":  f"{m['Accuracy']*100:.2f}%",
            "Precision": f"{m['Precision']*100:.2f}%",
            "Recall":    f"{m['Recall']*100:.2f}%",
            "F1-Score":  f"{m['F1-Score']*100:.2f}%",
            "ROC-AUC":   f"{m['ROC-AUC']*100:.2f}%" if m["ROC-AUC"] != "N/A" else "N/A",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📊 Accuracy Comparison</div>', unsafe_allow_html=True)
    mn   = list(metrics)
    accs = [metrics[n]["Accuracy"] * 100 for n in mn]
    fig  = go.Figure(go.Bar(
        x=mn, y=accs,
        marker_color=["#3B82F6" if n == best else "#CBD5E1" for n in mn],
        text=[f"{a:.1f}%" for a in accs], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white", height=250, margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans"),
        yaxis=dict(range=[0, max(accs) * 1.18], showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="card"><div class="card-title">🔢 Confusion Matrix</div>', unsafe_allow_html=True)
        sm = st.selectbox("Model", list(metrics.keys()))
        fig = px.imshow(metrics[sm]["_cm"], text_auto=True, aspect="auto",
                        color_continuous_scale="Blues", template="plotly_white",
                        labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=250,
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">📈 ROC Curves</div>', unsafe_allow_html=True)
        fig = go.Figure()
        cr = ["#3B82F6", "#22C55E", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
        has_roc = False
        for i, (n, m) in enumerate(metrics.items()):
            if "_fpr" in m:
                fig.add_trace(go.Scatter(
                    x=m["_fpr"], y=m["_tpr"],
                    name=f"{n} ({m['ROC-AUC']:.3f})",
                    line=dict(color=cr[i % len(cr)], width=2.4), mode="lines",
                ))
                has_roc = True
        if has_roc:
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="#CBD5E1", width=1), showlegend=False))
            fig.update_layout(
                template="plotly_white", height=250, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="FPR", showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(title="TPR", showgrid=True, gridcolor="#F1F5F9"),
                legend=dict(font=dict(size=9)), font=dict(family="Plus Jakarta Sans"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC curves are available for binary classification only.")
        st.markdown("</div>", unsafe_allow_html=True)

    if best in st.session_state.trained_models:
        mo = st.session_state.trained_models[best]
        if hasattr(mo, "feature_importances_"):
            st.markdown('<div class="card"><div class="card-title">🌟 Feature Importance</div>', unsafe_allow_html=True)
            fi = pd.DataFrame({
                "Feature": st.session_state.feature_cols,
                "Importance": mo.feature_importances_,
            }).sort_values("Importance", ascending=True).tail(15)
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues", template="plotly_white")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=290,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Plus Jakarta Sans"), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    all_accs = [metrics[n]["Accuracy"] * 100 for n in metrics]
    st.markdown(f"""
    <div class="insight-card">
        <h4>🧠 Summary Insights</h4>
        <p>
            <strong>{best}</strong> achieved the best accuracy of <strong>{bm['Accuracy']*100:.2f}%</strong>
            across {len(metrics)} model(s) (average: {np.mean(all_accs):.2f}%). Target column:
            <strong>{st.session_state.target_col}</strong> · Features used: <strong>{len(st.session_state.feature_cols)}</strong> ·
            {("Strong performance — model is ready for deployment." if bm['Accuracy'] > 0.85 else "Moderate performance — consider more feature engineering or tuning.")}
        </p>
    </div>
    """, unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "About":
    st.markdown('<div class="sec-title">About CSVInsight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">End-to-end machine learning platform — from raw CSV to trained model, all in your browser. No coding required.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-hero">
        <div class="wh-eyebrow">What is</div>
        <div class="wh-title">CSVInsight <span>AI</span></div>
        <div class="wh-desc">
            CSVInsight AI is an intelligent, browser-based data analysis and machine learning platform
            built for students, researchers, and data professionals who want to extract insights from
            their data without writing a single line of code.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🔄 Application Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("01", "📁", "Upload Data",          "Upload any CSV file using the drag-and-drop interface. The platform instantly reads your file and stores it in memory ready for analysis."),
        ("02", "📊", "Data Overview",         "Get a full structural snapshot — row count, column types, null values, unique counts, and a complete statistical summary table."),
        ("03", "🧹", "Data Cleaning",         "Detect missing values with a colour-coded chart. Choose fill strategies (median, mean, mode) or drop rows. Remove duplicates with one click."),
        ("04", "📈", "EDA Visualization",     "Explore your data with interactive Plotly charts — histograms, box plots, scatter plots, correlation heatmaps, and categorical distributions."),
        ("05", "🧠", "Model Training",        "Pick your target column, select ML algorithms, set the test split ratio, and train. The platform handles all encoding, scaling, and splitting automatically."),
        ("06", "🏆", "Results Dashboard",     "Compare all models side by side. View confusion matrices, ROC curves, and feature importance. A summary card highlights your best model."),
    ]
    for i in range(0, len(steps), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j < len(steps):
                num, ico, name, desc = steps[i + j]
                col.markdown(f"""
                <div class="wf-step">
                    <div class="wf-num">Step {num}</div>
                    <div class="wf-name">{ico} {name}</div>
                    <div class="wf-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="card"><div class="card-title">⚙️ Technology Stack</div>', unsafe_allow_html=True)
        for ico, name, desc in [
            ("🎈", "Streamlit",      "Web application framework"),
            ("🐍", "Python 3.11",   "Core programming language"),
            ("🤖", "Scikit-learn",  "Machine learning algorithms"),
            ("📊", "Plotly",        "Interactive data visualisations"),
            ("🐼", "Pandas",        "Data manipulation & analysis"),
            ("🔢", "NumPy",         "Numerical computing"),
        ]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.7rem;padding:0.4rem 0;border-bottom:1px solid #F1F5F9;">
                <span style="font-size:1.1rem;">{ico}</span>
                <div>
                    <div style="font-size:0.8rem;font-weight:700;color:#1E293B;">{name}</div>
                    <div style="font-size:0.72rem;color:#64748B;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">🤖 Supported Algorithms</div>', unsafe_allow_html=True)
        for name, desc in [
            ("Logistic Regression",    "Linear classifier — great baseline for binary problems."),
            ("Random Forest",          "Ensemble of trees — robust and high accuracy."),
            ("Gradient Boosting",      "Sequential boosting — excellent for structured data."),
            ("Support Vector Machine", "Finds optimal decision boundary."),
            ("K-Nearest Neighbors",    "Instance-based, no training phase."),
            ("Decision Tree",          "Single tree — highly interpretable."),
        ]:
            st.markdown(f"""
            <div style="padding:0.4rem 0;border-bottom:1px solid #F1F5F9;">
                <div style="display:inline-block;background:#DBEAFE;color:#1E40AF;border-radius:20px;padding:0.15rem 0.6rem;font-size:0.71rem;font-weight:700;margin-bottom:0.14rem;">{name}</div>
                <div style="font-size:0.77rem;color:#475569;">{desc}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FOUNDER
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Founder":
    st.markdown('<div class="sec-title">Meet the Founder</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">The person who built CSVInsight AI.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="founder-hero">
        <div class="f-av">HA</div>
        <div class="f-name">H. Mohamed Ahshaan</div>
        <div class="f-role">Founder &amp; Developer · CSVInsight AI</div>
        <div class="f-uni">🎓 BSc Artificial Intelligence &amp; Data Science · Year 2</div>
        <div class="f-uni" style="color:rgba(255,255,255,0.5);font-weight:400;font-size:0.74rem;">Robert Gordon University, Aberdeen, Scotland</div>
        <div class="f-bio">
            Passionate about making machine learning accessible to everyone.
            CSVInsight AI was built to bridge the gap between raw data and real insight —
            giving anyone the power to analyse, visualise, and model their data without code.
        </div>
        <div class="f-tags">
            <div class="f-tag">Machine Learning</div>
            <div class="f-tag">Data Science</div>
            <div class="f-tag">Python</div>
            <div class="f-tag">AI Research</div>
            <div class="f-tag">Data Visualisation</div>
        </div>
        <div class="f-links">
            <a class="f-link wa" href="https://wa.me/440742663484" target="_blank">📱 WhatsApp · +44 0742 663 484</a>
            <a class="f-link em" href="mailto:mohamedahshaan@gmail.com">✉️ mohamedahshaan@gmail.com</a>
            <a class="f-link li" href="https://linkedin.com/in/ahshaan" target="_blank">💼 LinkedIn</a>
            <a class="f-link gh" href="https://github.com/ahshaan" target="_blank">🐙 GitHub</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">🎓</div>
            <div style="font-size:0.88rem;font-weight:700;color:#0F1C3F;">Education</div>
            <div style="font-size:0.78rem;color:#64748B;margin-top:0.4rem;line-height:1.55;">
                BSc AI &amp; Data Science<br>Robert Gordon University<br>
                <span style="color:#3B82F6;font-weight:600;">2nd Year · Aberdeen, UK</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">🛠️</div>
            <div style="font-size:0.88rem;font-weight:700;color:#0F1C3F;">Built With</div>
            <div style="font-size:0.78rem;color:#64748B;margin-top:0.4rem;line-height:1.55;">
                Python · Streamlit<br>Scikit-learn · Plotly<br>
                <span style="color:#3B82F6;font-weight:600;">Pandas · NumPy</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">🚀</div>
            <div style="font-size:0.88rem;font-weight:700;color:#0F1C3F;">This Project</div>
            <div style="font-size:0.78rem;color:#64748B;margin-top:0.4rem;line-height:1.55;">
                CSVInsight AI v2.1<br>Personal portfolio project<br>
                <span style="color:#3B82F6;font-weight:600;">Open to collaboration</span>
            </div>
        </div>""", unsafe_allow_html=True)

    footer()
