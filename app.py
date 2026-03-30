"""
╔══════════════════════════════════════════════════════╗
║   AI-Powered Data Analysis & Machine Learning System ║
║   Built by Mohamed Ahshaan  ·  v4.0 Cloud           ║
╚══════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
try:
    matplotlib.use('Agg')
except Exception:
    pass
import warnings, io, base64, json
warnings.filterwarnings('ignore')

# ── Sklearn ──────────────────────────────────────────
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

# ── XGBoost ──────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_OK = True
except Exception:
    XGBOOST_OK = False
    class XGBClassifier: pass
    class XGBRegressor: pass

# ── SHAP ─────────────────────────────────────────────
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ── Imbalanced-learn ─────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataSci Studio v4.0 · Ahshaan",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ─────────────────────────────────────
for k, v in [("dark_mode", True), ("page", "app"),
              ("train_split", 80), ("use_smote", False),
              ("use_class_weight", True)]:
    if k not in st.session_state:
        st.session_state[k] = v

dm = st.session_state.dark_mode

# ═══════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════
if dm:
    BG="#07070e"; SBG="#0c0c18"; CARD="#0f0f1c"; BDR="#1c1c35"
    T1="#e8e4ff"; T2="#7878a0"; T3="#333358"
    A1="#7c6af7"; A2="#00d4d0"; A3="#ff6b9d"
    SUC="#00c471"; WRN="#ffbe3d"; ERR="#ff5f5f"
    IB="#0a1220"; IBC="#00d4d0"
    WB="#120e00"; WBC="#ffbe3d"
    SB="#03120a"; SBC="#00c471"
    PBG="#07070e"; PAX="#0f0f1c"; PGR="#181830"; PT="#7878a0"; PTI="#c8baff"
else:
    BG="#f3f2ff"; SBG="#eceaff"; CARD="#ffffff"; BDR="#dbd8f5"
    T1="#160f40"; T2="#5a5870"; T3="#b0aecb"
    A1="#6c5ce7"; A2="#00a8a5"; A3="#c9296b"
    SUC="#00914f"; WRN="#a06600"; ERR="#c0392b"
    IB="#ebf6ff"; IBC="#00a8a5"
    WB="#fff8e6"; WBC="#a06600"
    SB="#e8fff3"; SBC="#00914f"
    PBG="#f3f2ff"; PAX="#ffffff"; PGR="#e8e5ff"; PT="#5a5870"; PTI="#160f40"

# ═══════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');

*,html,body,[class*="css"]{{font-family:'Plus Jakarta Sans',sans-serif!important;transition:background .25s,color .25s,border-color .25s;}}
.stApp{{background:{BG}!important;color:{T1}!important;}}
[data-testid="stSidebar"]{{background:{SBG}!important;border-right:1px solid {BDR}!important;}}
[data-testid="stSidebar"] *{{color:{T1}!important;}}
[data-testid="stSidebar"] .stMarkdown p{{color:{T2}!important;}}

/* ── Hero ── */
.hero{{background:linear-gradient(145deg,{CARD},{BG});border:1px solid {BDR};border-radius:22px;padding:2.8rem 2.5rem;margin-bottom:1.8rem;position:relative;overflow:hidden;}}
.hero::before{{content:'';position:absolute;top:-80px;right:-80px;width:280px;height:280px;border-radius:50%;background:radial-gradient({A1}22,transparent 70%);pointer-events:none;}}
.hero::after{{content:'';position:absolute;bottom:-60px;left:-40px;width:200px;height:200px;border-radius:50%;background:radial-gradient({A2}14,transparent 70%);pointer-events:none;}}
.htitle{{font-size:2.6rem;font-weight:800;letter-spacing:-1.5px;background:linear-gradient(110deg,{A1},{A2} 60%,{A3});-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 .45rem;line-height:1.1;}}
.hsub{{color:{T2};font-size:.82rem;font-family:'Fira Code',monospace;letter-spacing:2.5px;text-transform:uppercase;margin:0;}}
.hbadge{{display:inline-block;background:{A1}22;border:1px solid {A1}55;color:{A1};font-size:.72rem;font-weight:700;padding:3px 14px;border-radius:20px;letter-spacing:1px;margin-top:.8rem;}}

/* ── Animated pipeline steps ── */
.pipe-row{{display:flex;gap:.5rem;margin-top:1.2rem;flex-wrap:wrap;}}
.pipe-step{{display:inline-flex;align-items:center;gap:5px;background:{A1}12;border:1px solid {A1}33;color:{T2};font-size:.72rem;font-weight:600;padding:4px 12px;border-radius:20px;font-family:'Fira Code',monospace;}}
.pipe-step span{{color:{A1};}}

/* ── Stat row ── */
.stat-row{{display:flex;gap:1.2rem;margin-top:1rem;flex-wrap:wrap;}}
.stat-item{{text-align:center;}}
.stat-val{{font-size:1.4rem;font-weight:800;color:{A1};font-family:'Fira Code',monospace;line-height:1;}}
.stat-lbl{{font-size:.62rem;color:{T3};text-transform:uppercase;letter-spacing:1.5px;font-weight:700;}}

/* ── Section header ── */
.shdr{{display:flex;align-items:center;gap:.75rem;margin:2.4rem 0 1.2rem;padding-bottom:.7rem;border-bottom:1px solid {BDR};}}
.sico{{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,{A1},{A2});display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;}}
.stit{{font-size:1.12rem;font-weight:700;color:{T1};margin:0;letter-spacing:-.3px;}}

/* ── Metric cards ── */
.mgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:.85rem;margin:1rem 0;}}
.mc{{background:{CARD};border:1px solid {BDR};border-radius:14px;padding:1.1rem .9rem;text-align:center;transition:border-color .2s,transform .2s;cursor:default;}}
.mc:hover{{border-color:{A1}99;transform:translateY(-2px);}}
.mv{{font-size:1.65rem;font-weight:800;color:{A1};font-family:'Fira Code',monospace;line-height:1;margin-bottom:.3rem;}}
.ml{{font-size:.68rem;color:{T3};text-transform:uppercase;letter-spacing:1.5px;font-weight:700;}}

/* ── Alert boxes ── */
.ib{{background:{IB};border-left:3px solid {IBC};border-radius:0 9px 9px 0;padding:.8rem 1.1rem;margin:.4rem 0;font-size:.83rem;color:{T2};font-family:'Fira Code',monospace;line-height:1.55;}}
.wb{{background:{WB};border-left:3px solid {WBC};border-radius:0 9px 9px 0;padding:.8rem 1.1rem;margin:.4rem 0;font-size:.83rem;color:{WRN};font-family:'Fira Code',monospace;line-height:1.55;}}
.sb{{background:{SB};border-left:3px solid {SBC};border-radius:0 9px 9px 0;padding:.8rem 1.1rem;margin:.4rem 0;font-size:.83rem;color:{SUC};font-family:'Fira Code',monospace;line-height:1.55;}}

/* ── Skill chips ── */
.chip{{display:inline-block;background:{A1}18;border:1px solid {A1}40;color:{A1};padding:3px 11px;border-radius:20px;font-size:.76rem;font-weight:600;margin:2px;}}
.chip2{{display:inline-block;background:{A2}18;border:1px solid {A2}40;color:{A2};padding:3px 11px;border-radius:20px;font-size:.76rem;font-weight:600;margin:2px;}}

/* ── Founder / About cards ── */
.fcard{{background:{CARD};border:1px solid {BDR};border-radius:20px;padding:2.2rem;position:relative;overflow:hidden;}}
.fcard::before{{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,{A1},{A2},{A3});}}

.pcard{{background:{BG};border:1px solid {BDR};border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:.8rem;transition:border-color .2s;}}
.pcard:hover{{border-color:{A1}70;}}
.ptitle{{font-size:.9rem;font-weight:700;color:{T1};margin-bottom:.25rem;}}
.pdesc{{font-size:.8rem;color:{T2};line-height:1.6;}}
.alink{{font-size:.76rem;color:{A2};font-family:'Fira Code',monospace;text-decoration:none;}}

/* ── Social buttons ── */
.soc-row{{display:flex;flex-wrap:wrap;gap:.6rem;margin-top:1rem;}}
.soc{{display:inline-flex;align-items:center;gap:6px;background:{CARD};border:1px solid {BDR};border-radius:30px;padding:7px 16px;font-size:.8rem;color:{T1};font-family:'Fira Code',monospace;text-decoration:none;transition:border-color .2s,transform .2s;cursor:pointer;}}
.soc:hover{{border-color:{A1};transform:translateY(-1px);}}

/* ── Steps ── */
.step{{background:{CARD};border:1px solid {BDR};border-radius:14px;padding:1.2rem;display:flex;gap:.9rem;margin-bottom:.7rem;align-items:flex-start;}}
.snum{{width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,{A1},{A2});color:#fff;font-weight:800;font-size:.88rem;display:flex;align-items:center;justify-content:center;flex-shrink:0;}}
.sbody h4{{margin:0 0 3px;font-size:.92rem;font-weight:700;color:{T1};}}
.sbody p{{margin:0;font-size:.8rem;color:{T2};line-height:1.6;}}

/* ── Best badge ── */
.bbadge{{display:inline-block;background:linear-gradient(90deg,{A1},{A2});color:#fff;font-size:.65rem;font-weight:700;padding:2px 10px;border-radius:20px;letter-spacing:1px;text-transform:uppercase;margin-left:8px;vertical-align:middle;}}

/* ── Nav buttons ── */
.stButton>button{{background:linear-gradient(135deg,{A1},{A1}cc);color:#fff;border:none;border-radius:10px;padding:.55rem 1.5rem;font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:.86rem;transition:opacity .2s,transform .2s;}}
.stButton>button:hover{{opacity:.88;transform:translateY(-1px);}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{{background:{CARD};border-radius:11px;border:1px solid {BDR};}}
.stTabs [data-baseweb="tab"]{{color:{T2}!important;font-weight:600;}}
.stTabs [aria-selected="true"]{{color:{A1}!important;background:{A1}20;border-radius:9px;}}

/* ── Dataframe ── */
[data-testid="stDataFrame"]{{border:1px solid {BDR};border-radius:11px;overflow:hidden;}}

/* ── File uploader ── */
[data-testid="stFileUploader"]{{border:2px dashed {BDR};border-radius:14px;padding:.5rem;background:{CARD};}}
[data-testid="stFileUploader"]:hover{{border-color:{A1}88;}}

/* ── Expander ── */
details summary{{color:{T2}!important;font-size:.85rem!important;}}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"]{{background:{A1};color:#fff;}}

/* ── Toggle ── */
.tog-wrap{{display:flex;align-items:center;justify-content:flex-end;}}
.tog-btn{{background:{CARD};border:1.5px solid {BDR};border-radius:30px;padding:5px 14px;color:{T1};font-family:'Plus Jakarta Sans',sans-serif;font-size:.8rem;font-weight:700;cursor:pointer;display:inline-flex;align-items:center;gap:5px;transition:border-color .2s;}}
.tog-btn:hover{{border-color:{A1};}}

/* ── Footer ── */
.footer{{text-align:center;padding:2rem 0 1rem;border-top:1px solid {BDR};margin-top:3rem;}}
.footer p{{font-size:.75rem;color:{T3};font-family:'Fira Code',monospace;letter-spacing:.8px;margin:0;}}

/* ── Download button ── */
.dl-btn{{display:inline-flex;align-items:center;gap:6px;background:{A1}18;border:1px solid {A1}44;color:{A1};padding:7px 16px;border-radius:10px;font-size:.82rem;font-weight:700;cursor:pointer;text-decoration:none;transition:background .2s;}}
.dl-btn:hover{{background:{A1}30;}}

hr{{border-color:{BDR}!important;}}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════
def shdr(icon, title):
    st.markdown(f'<div class="shdr"><div class="sico">{icon}</div><p class="stit">{title}</p></div>', unsafe_allow_html=True)

def ib(m): st.markdown(f'<div class="ib">{m}</div>', unsafe_allow_html=True)
def wb(m): st.markdown(f'<div class="wb">{m}</div>', unsafe_allow_html=True)
def sb(m): st.markdown(f'<div class="sb">{m}</div>', unsafe_allow_html=True)

def mgrid(*cards):
    inner = "".join([f'<div class="mc"><div class="mv" style="{s}">{v}</div><div class="ml">{l}</div></div>'
                     for v, l, s in cards])
    st.markdown(f'<div class="mgrid">{inner}</div>', unsafe_allow_html=True)

def df_to_csv_b64(df):
    return base64.b64encode(df.to_csv(index=False).encode()).decode()

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')  # lowered dpi: 150→120 to save RAM
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ═══════════════════════════════════════════════════════
# PLOT STYLE
# ═══════════════════════════════════════════════════════
def pstyle():
    plt.rcParams.update({
        'figure.facecolor': PBG, 'axes.facecolor': PAX, 'axes.edgecolor': PGR,
        'axes.labelcolor': PT,   'text.color': PT,       'xtick.color': PT,
        'ytick.color': PT,       'grid.color': PGR,      'grid.linestyle': '--',
        'grid.alpha': .45,       'axes.spines.top': False,'axes.spines.right': False,
        'font.family': 'DejaVu Sans',
    })

PAL = [A1, A2, A3, SUC, WRN, "#a29bfe", "#55efc4", "#ffeaa7", "#fab1a0", "#74b9ff", "#fd79a8", "#00b894"]

# ═══════════════════════════════════════════════════════
# DATA FUNCTIONS
# ═══════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        if df.empty: return None, "File is empty."
        return df, None
    except Exception as e:
        return None, str(e)

def data_quality(df):
    miss = df.isnull().sum()
    pct  = (miss / len(df) * 100).round(2)
    dup  = df.duplicated().sum()
    const = [c for c in df.columns if df[c].nunique() <= 1]
    return {"miss": miss, "pct": pct, "dup": dup, "const": const}

def clean_data(df):
    df = df.copy(); exps = []; drop = []
    for col in df.columns:
        mc = df[col].isnull().sum(); mp = mc / len(df) * 100
        if mc == 0: continue
        if mp > 30:
            drop.append(col); exps.append(f"⚠️ **{col}** — {mp:.1f}% missing → flagged for removal."); continue
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            fv = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(fv)
            exps.append(f"✅ **{col}** — {mp:.1f}% missing (categorical) → mode: `{fv}`.")
        else:
            sk = abs(df[col].skew())
            fv = df[col].median() if sk > 1 or mp >= 5 else df[col].mean()
            st2 = "median" if sk > 1 or mp >= 5 else "mean"
            df[col] = df[col].fillna(fv)
            exps.append(f"✅ **{col}** — {mp:.1f}% missing → {st2}: `{fv:.4g}`.")
    q = data_quality(df)
    for col in q["const"]:
        if col in df.columns: drop.append(col); exps.append(f"🗑️ **{col}** — constant, removed.")
    df.drop(columns=list(set(drop)), errors='ignore', inplace=True)
    nb = len(df); df.drop_duplicates(inplace=True)
    nd = nb - len(df)
    if nd > 0: exps.append(f"🗑️ Removed **{nd}** duplicate row(s).")
    return df, exps

def engineer_features(df, target):
    df = df.copy(); enc = []
    cats = [c for c in df.select_dtypes(include=['object', 'category']).columns if c != target]
    for col in cats:
        if df[col].nunique() <= 10:
            d = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), d], axis=1)
            enc.append(f"{col} → one-hot ({d.shape[1]} cols)")
        else:
            le = LabelEncoder(); df[col] = le.fit_transform(df[col].astype(str))
            enc.append(f"{col} → label encoded")
    if df[target].dtype == 'object' or str(df[target].dtype) == 'category':
        le = LabelEncoder(); df[target] = le.fit_transform(df[target].astype(str))
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target]
    return X, y, enc

# ═══════════════════════════════════════════════════════
# MODEL DEFINITIONS  — n_estimators reduced for cloud RAM
# ═══════════════════════════════════════════════════════
def build_clf_models(use_cw):
    cw = "balanced" if use_cw else None
    m = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42, class_weight=cw),
        "Random Forest":        RandomForestClassifier(n_estimators=50, random_state=42, class_weight=cw),   # was 120
        "SVM":                  SVC(probability=True, random_state=42, class_weight=cw),
        "KNN":                  KNeighborsClassifier(),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=50, random_state=42),                # was 100
        "MLP":                  MLPClassifier(max_iter=300, random_state=42),                                # was 500
    }
    if XGBOOST_OK:
        m["XGBoost"] = XGBClassifier(
            n_estimators=50,                # was default (100)
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
            tree_method='hist',             # faster & lower RAM than default
        )
    return m

def build_reg_models():
    m = {
        "Linear Regression":  LinearRegression(),
        "Random Forest":      RandomForestRegressor(n_estimators=50, random_state=42),   # was 120
        "SVR":                SVR(),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=50, random_state=42),  # was 100
        "KNN":                KNeighborsRegressor(),
        "MLP":                MLPRegressor(max_iter=300, random_state=42),               # was 500
    }
    if XGBOOST_OK:
        m["XGBoost"] = XGBRegressor(
            n_estimators=50,                # was default (100)
            random_state=42,
            verbosity=0,
            tree_method='hist',
        )
    return m

NEEDS_SCALE = {"SVM", "SVR", "KNN", "MLP"}

def train_models(X, y, selected, ptype, split_pct, use_smote, use_cw):
    test_size = 1 - split_pct / 100
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

    # SMOTE for imbalanced classification
    if ptype == "Classification" and use_smote and SMOTE_OK:
        try:
            sm = SMOTE(random_state=42)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        except Exception:
            pass

    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr); Xs_te = sc.transform(X_te)

    mm = build_clf_models(use_cw) if ptype == "Classification" else build_reg_models()
    results = {}
    for name in selected:
        if name not in mm: continue
        model = mm[name]
        _Xtr = Xs_tr if name in NEEDS_SCALE else X_tr
        _Xte = Xs_te if name in NEEDS_SCALE else X_te
        try:
            model.fit(_Xtr, y_tr)
            yp = model.predict(_Xte)
            y_prob = None
            try:
                if hasattr(model, "predict_proba") and len(np.unique(y_tr)) == 2:
                    y_prob = model.predict_proba(_Xte)[:, 1]
            except Exception:
                pass
            results[name] = {
                "model": model, "y_test": y_te, "y_pred": yp, "y_prob": y_prob,
                "X_test": _Xte, "X_train": _Xtr, "y_train": y_tr,
                "X_columns": list(X.columns), "scaler": sc
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results, X_te, y_te

def evaluate_clf(results):
    rows = []
    for n, r in results.items():
        if "error" in r: rows.append({"Model": n, "Error": r["error"]}); continue
        yt, yp = r["y_test"], r["y_pred"]
        avg = 'binary' if len(np.unique(yt)) == 2 else 'macro'
        row = {"Model": n,
            "Accuracy":  round(accuracy_score(yt, yp), 4),
            "Precision": round(precision_score(yt, yp, average=avg, zero_division=0), 4),
            "Recall":    round(recall_score(yt, yp, average=avg, zero_division=0), 4),
            "F1 Score":  round(f1_score(yt, yp, average=avg, zero_division=0), 4),
        }
        if len(np.unique(yt)) == 2 and "y_prob" in r and r["y_prob"] is not None:
            try:
                from sklearn.metrics import roc_auc_score
                row["ROC-AUC"] = round(roc_auc_score(yt, r["y_prob"]), 4)
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)

def evaluate_reg(results):
    rows = []
    for n, r in results.items():
        if "error" in r: rows.append({"Model": n, "Error": r["error"]}); continue
        yt, yp = r["y_test"], r["y_pred"]
        rows.append({"Model": n,
            "MAE":  round(mean_absolute_error(yt, yp), 4),
            "MSE":  round(mean_squared_error(yt, yp), 4),
            "RMSE": round(np.sqrt(mean_squared_error(yt, yp)), 4),
            "R²":   round(r2_score(yt, yp), 4),
        })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════
def p_hist(df, cols):
    pstyle(); nc = min(3, len(cols)); nr = -(-len(cols) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 3.5*nr))
    axes = np.array(axes).flatten() if nr*nc > 1 else [axes]
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=30, color=PAL[i % len(PAL)], alpha=.85, edgecolor='none')
        axes[i].set_title(col, fontsize=10, fontweight='bold', color=PTI)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.tight_layout(pad=1.5); return fig

def p_box(df, cols):
    pstyle(); nc = min(3, len(cols)); nr = -(-len(cols) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 3.5*nr))
    axes = np.array(axes).flatten() if nr*nc > 1 else [axes]
    for i, col in enumerate(cols):
        bp = axes[i].boxplot(df[col].dropna(), patch_artist=True, medianprops=dict(color=A2, linewidth=2))
        for p in bp['boxes']: p.set_facecolor(A1); p.set_alpha(.7)
        for el in ['whiskers','caps','fliers']: plt.setp(bp[el], color=PGR)
        axes[i].set_title(col, fontsize=10, fontweight='bold', color=PTI)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.tight_layout(pad=1.5); return fig

def p_corr(df):
    pstyle()
    nd = df.select_dtypes(include=[np.number])
    if nd.shape[1] < 2: return None
    corr = nd.iloc[:, :16].corr()
    fig, ax = plt.subplots(figsize=(max(7, len(corr.columns)*.7), max(5.5, len(corr.columns)*.65)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap=sns.diverging_palette(260, 20, as_cmap=True),
                center=0, square=True, linewidths=.5, ax=ax,
                annot_kws={"size": 8, "color": PT}, cbar_kws={"shrink": .8})
    ax.set_title("Feature Correlation Matrix", color=PTI, fontweight='bold', pad=14)
    fig.tight_layout(); return fig

def p_cm(yt, yp, name):
    pstyle(); cm = confusion_matrix(yt, yp)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PuBu', ax=ax, linewidths=.5,
                linecolor=PBG, annot_kws={"size": 14, "weight": "bold", "color": "#0a0a20"})
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{name} — Confusion Matrix', color=PTI, fontweight='bold')
    fig.tight_layout(); return fig

def p_roc(model, Xte, yte, name):
    pstyle()
    if len(np.unique(yte)) != 2: return None
    try:
        ys = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xte)
        fpr, tpr, _ = roc_curve(yte, ys); ra = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color=A1, lw=2.5, label=f'AUC = {ra:.4f}')
        ax.fill_between(fpr, tpr, alpha=.12, color=A1)
        ax.plot([0, 1], [0, 1], '--', lw=1.2, color=PGR)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name} — ROC Curve', color=PTI, fontweight='bold')
        ax.legend(framealpha=.1, labelcolor=PT)
        fig.tight_layout(); return fig
    except Exception: return None

def p_prc(model, Xte, yte, name):
    pstyle()
    if len(np.unique(yte)) != 2 or not hasattr(model, "predict_proba"): return None
    try:
        ys = model.predict_proba(Xte)[:, 1]
        prec, rec, _ = precision_recall_curve(yte, ys)
        ap = average_precision_score(yte, ys)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec, prec, color=A2, lw=2.5, label=f'AP = {ap:.4f}')
        ax.fill_between(rec, prec, alpha=.12, color=A2)
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(f'{name} — Precision-Recall', color=PTI, fontweight='bold')
        ax.legend(framealpha=.1, labelcolor=PT)
        fig.tight_layout(); return fig
    except Exception: return None

def p_avp(yt, yp, name):
    pstyle(); fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(yt, yp, color=A1, alpha=.55, s=22, edgecolors='none')
    mn = min(np.min(yt), np.min(yp)); mx = max(np.max(yt), np.max(yp))
    ax.plot([mn, mx], [mn, mx], '--', color=A2, lw=1.8)
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f'{name} — Actual vs Predicted', color=PTI, fontweight='bold')
    fig.tight_layout(); return fig

def p_res(yt, yp, name):
    pstyle(); r = np.array(yt) - np.array(yp)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(yp, r, color=A3, alpha=.55, s=22, edgecolors='none')
    ax.axhline(0, color=A2, lw=1.8, linestyle='--')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Residuals')
    ax.set_title(f'{name} — Residual Plot', color=PTI, fontweight='bold')
    fig.tight_layout(); return fig

def p_fi(model, feats, name):
    pstyle()
    if not hasattr(model, "feature_importances_"): return None
    imp = model.feature_importances_; idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(6, max(4, len(idx)*.4)))
    colors = plt.cm.PuBu(np.linspace(.4, .9, len(idx)))
    ax.barh([feats[i] for i in idx[::-1]], imp[idx[::-1]], color=colors[::-1], edgecolor='none')
    ax.set_xlabel('Importance')
    ax.set_title(f'{name} — Feature Importance', color=PTI, fontweight='bold')
    fig.tight_layout(); return fig

def p_cmp(df, metric):
    pstyle()
    dp = df[["Model", metric]].dropna().copy()
    dp[metric] = pd.to_numeric(dp[metric], errors='coerce'); dp = dp.dropna()
    asc = metric not in ["Accuracy", "F1 Score", "R²", "Precision", "Recall"]
    dp = dp.sort_values(metric, ascending=asc)
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(dp)*.6)))
    cols = [A1 if (asc and i == 0) or (not asc and i == len(dp)-1) else f"{A1}38"
            for i in range(len(dp))]
    ax.barh(dp["Model"], dp[metric], color=cols, edgecolor='none', height=.52)
    for bar in ax.patches:
        w = bar.get_width()
        ax.text(w + abs(w)*.01, bar.get_y()+bar.get_height()/2,
                f'{w:.4f}', va='center', fontsize=9, color=PT)
    ax.set_xlabel(metric)
    ax.set_title(f'Model Comparison — {metric}', color=PTI, fontweight='bold')
    fig.tight_layout(); return fig

def p_shap(model, X_tr, feats, name):
    if not SHAP_OK: return None
    try:
        # Limit sample to 50 rows max — reduced from 100 to save RAM on cloud
        sample = X_tr[:50] if len(X_tr) > 50 else X_tr
        if hasattr(model, "predict_proba"):
            exp = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else shap.KernelExplainer(model.predict_proba, shap.sample(X_tr, 30))
        else:
            exp = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else shap.KernelExplainer(model.predict, shap.sample(X_tr, 30))
        sv = exp.shap_values(sample)
        if isinstance(sv, list): sv = sv[1]
        pstyle(); fig, ax = plt.subplots(figsize=(7, max(4, len(feats)*.35)))
        shap.summary_plot(sv, sample, feature_names=feats, show=False, plot_type="bar")
        plt.title(f'{name} — SHAP Feature Impact', color=PTI, fontweight='bold')
        fig.tight_layout(); return fig
    except Exception: return None

MEXPL = {
    "Random Forest":       "Random Forest ensembles many decision trees, reducing variance and overfitting while capturing complex patterns.",
    "Logistic Regression": "Logistic Regression suggests a largely linear decision boundary — excellent for interpretability and generalization.",
    "SVM":                 "SVM found an optimal separating hyperplane with maximum margin for class separation.",
    "KNN":                 "KNN detected strong local neighbourhood patterns — the data has clear local structure.",
    "MLP":                 "The Neural Network captured non-linear feature interactions across its hidden layers.",
    "Gradient Boosting":   "Gradient Boosting built trees sequentially, each correcting the last — ideal for capturing residuals.",
    "XGBoost":             "XGBoost combines boosting with regularization, delivering high accuracy with great generalization.",
    "Linear Regression":   "Linear Regression confirms a strong linear relationship between features and the target.",
    "SVR":                 "SVR fitted a tolerance tube around the regression line, minimizing sensitivity to outliers.",
}

# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    # Logo + toggle
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"""
        <p style='font-size:1.25rem;font-weight:800;color:{A1};margin:1rem 0 2px;letter-spacing:-.5px;'>⚡ DataSci Studio</p>
        <p style='font-size:.66rem;color:{T3};font-family:"Fira Code",monospace;margin:0;letter-spacing:2px;'>BY AHSHAAN · v3.0</p>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        lbl = "☀️ Light" if dm else "🌙 Dark"
        if st.button(lbl, key="theme"):
            st.session_state.dark_mode = not dm; st.rerun()

    st.markdown(f"<hr style='border-color:{BDR};margin:.8rem 0;'>", unsafe_allow_html=True)

    # Navigation
    st.markdown(f"<p style='font-size:.66rem;color:{T3};letter-spacing:2px;font-weight:700;text-transform:uppercase;margin-bottom:5px;'>Navigation</p>", unsafe_allow_html=True)
    for ico, key, label in [("🤖", "app", "ML Studio"), ("📖", "about", "About"), ("👤", "founder", "Founder")]:
        if st.button(f"{ico}  {label}", key=f"nav_{key}"):
            st.session_state.page = key; st.rerun()

    st.markdown(f"<hr style='border-color:{BDR};margin:.8rem 0;'>", unsafe_allow_html=True)

    if st.session_state.page == "app":
        uploaded = st.file_uploader("📂 Upload CSV Dataset", type=["csv"],
                                     help="Drag & drop or click to browse")
        st.markdown(f"<hr style='border-color:{BDR};margin:.8rem 0;'>", unsafe_allow_html=True)
        ph_info   = st.empty()
        ph_target = st.empty()
        ph_split  = st.empty()
        ph_opts   = st.empty()
        ph_models = st.empty()
        ph_run    = st.empty()
    else:
        uploaded = None

    st.markdown(f"""
    <div style='position:fixed;bottom:1.2rem;font-size:.66rem;color:{T3};
                font-family:"Fira Code",monospace;'>
        © 2026 Mohamed Ahshaan
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════
def page_about():
    st.markdown(f'<div class="hero"><p class="htitle">About DataSci Studio</p><p class="hsub">Your AI-Powered End-to-End ML Platform</p><span class="hbadge">v3.0 · Final Edition</span></div>', unsafe_allow_html=True)

    st.markdown(f"<p style='font-size:.98rem;color:{T2};line-height:1.85;max-width:800px;'>DataSci Studio is a complete, browser-based machine learning platform built with Python & Streamlit. Upload any CSV, get intelligent auto-cleaning, rich EDA visualizations, and train up to 7 ML models simultaneously — all without writing a single line of code. Built with SHAP explainability, SMOTE imbalance handling, XGBoost, and advanced metrics.</p>", unsafe_allow_html=True)

    shdr("⚡", "Key Features")
    feats = [
        ("🧹", "Intelligent Auto-Cleaning", "Detects missing values with skew-aware fill strategies, removes duplicates, drops constant columns, and explains every action taken."),
        ("📊", "Rich EDA Visualizations", "Histograms, boxplots, and a fully-masked lower-triangle correlation heatmap to explore your data before modelling."),
        ("🤖", "7+ ML Models", "Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting, MLP, and XGBoost — for both classification and regression."),
        ("⚖️", "Imbalance Handling", "SMOTE oversampling and class-weight balancing to improve recall on minority classes in imbalanced datasets."),
        ("📈", "Advanced Evaluation", "Confusion matrix, ROC + AUC, Precision-Recall curves, actual vs predicted, residuals, and feature importance."),
        ("🔍", "SHAP Explainability", "SHAP values reveal why each model makes predictions — essential for real-world trust and model transparency."),
        ("📥", "Export Everything", "Download predictions as CSV and plots as PNG directly from the results dashboard."),
        ("🌗", "Dark / Light Mode", "Fully themed — toggle between dark and light mode anytime from the sidebar button."),
    ]
    cols = st.columns(4)
    for i, (ico, t, d) in enumerate(feats):
        with cols[i % 4]:
            st.markdown(f'<div class="fcard" style="margin-bottom:.8rem;padding:1.2rem;"><div style="font-size:1.6rem;margin-bottom:.5rem;">{ico}</div><div style="font-size:.88rem;font-weight:700;color:{T1};margin-bottom:.3rem;">{t}</div><div style="font-size:.78rem;color:{T2};line-height:1.6;">{d}</div></div>', unsafe_allow_html=True)

    shdr("🚀", "How to Use — Step by Step")
    steps = [
        ("Upload CSV", "Click 'Upload CSV Dataset' in the sidebar or drag and drop your file. Datasets up to 100k rows are supported."),
        ("Review Data Overview", "Instantly see row/column counts, data types, missing values, and a dataset preview."),
        ("Auto-Cleaning Report", "The app cleans your data automatically — every action is logged with a plain-English explanation."),
        ("Explore EDA", "Switch between Distributions, Boxplots, and Correlation tabs to understand patterns in your data visually."),
        ("Configure Pipeline", "Select your target column, set the train/test split ratio, and enable SMOTE or class weights if needed."),
        ("Choose Models", "Pick from up to 7 ML algorithms — run multiple at once for easy side-by-side comparison."),
        ("Run & Evaluate", "Click 🚀 Run Full Pipeline — all models train, evaluate, and generate charts automatically."),
        ("Export Results", "Download predictions as CSV or save any chart as PNG from the results dashboard."),
    ]
    for i, (t, d) in enumerate(steps, 1):
        st.markdown(f'<div class="step"><div class="snum">{i}</div><div class="sbody"><h4>{t}</h4><p>{d}</p></div></div>', unsafe_allow_html=True)

    shdr("🛠️", "Tech Stack")
    t1 = ["Python 3.9+", "Streamlit", "Pandas", "NumPy", "Scikit-learn", "XGBoost", "Matplotlib", "Seaborn", "SHAP", "imbalanced-learn", "Jupyter Notebook", "Git", "VS Code"]
    st.markdown(" ".join([f'<span class="chip">{t}</span>' for t in t1]), unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-top:1.8rem;padding:1.4rem;background:{CARD};border:1px solid {BDR};border-radius:14px;'>
    <p style='margin:0;font-size:.86rem;color:{T2};line-height:1.75;'>
    <strong style='color:{A1};'>💡 Tips for best results:</strong> Use structured tabular CSV data.
    For classification with class imbalance, enable SMOTE or Class Weights in the sidebar.
    Datasets with 500–50,000 rows work best. Keep feature count under 50 for SHAP explanations to run fast.
    For regression, ensure the target is a continuous numeric column with enough variance.
    </p></div>""", unsafe_allow_html=True)

    # Footer
    st.markdown(f'<div class="footer"><p>© 2026 Mohamed Ahshaan · DataSci Studio v3.0 · All Rights Reserved.</p></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# PAGE: FOUNDER
# ═══════════════════════════════════════════════════════
def page_founder():
    st.markdown(f'<div class="hero"><p class="htitle">Meet the Founder</p><p class="hsub">The Developer Behind DataSci Studio</p></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2], gap="large")
    with col_a:
        st.markdown(f"""
        <div style='width:155px;height:155px;border-radius:50%;
            background:linear-gradient(135deg,{A1},{A2});
            display:flex;align-items:center;justify-content:center;
            font-size:3.2rem;font-weight:900;color:#fff;
            margin:0 auto 1.4rem;border:4px solid {BDR};
            box-shadow:0 0 50px {A1}40;'>MA
        </div>
        <p style='text-align:center;font-size:.72rem;color:{T3};font-family:"Fira Code",monospace;letter-spacing:1px;'>PHOTO PLACEHOLDER</p>
        """, unsafe_allow_html=True)

        socials = [
            ("🎬", "YouTube",  "youtube.com/@ICTwithAhshaan", "https://youtube.com/@ICTwithAhshaan"),
            ("💼", "LinkedIn", "Mohamed Ahshaan",              "https://linkedin.com/in/mohamedahshaan"),
            ("🐙", "GitHub",   "mohamedahshaan",               "https://github.com/mohamedahshaan"),
            ("💬", "WhatsApp", "+94 742 663 484",              "https://wa.me/94742663484"),
            ("📧", "Email",    "mohamedahshaan@gmail.com",     "mailto:mohamedahshaan@gmail.com"),
        ]
        for ico, platform, handle, url in socials:
            st.markdown(f'<a href="{url}" target="_blank" class="soc">{ico} <strong>{platform}</strong> · {handle}</a>', unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="fcard">
            <h2 style='margin:0 0 .2rem;font-size:2rem;font-weight:800;color:{T1};letter-spacing:-.5px;'>Mohamed Ahshaan</h2>
            <p style='margin:0 0 1rem;color:{A1};font-weight:700;font-size:.95rem;'>Aspiring Data Science Intern · Machine Learning Enthusiast</p>
            <p style='color:{T2};font-size:.88rem;line-height:1.8;margin-bottom:1.2rem;'>
                I am an AI & Data Science undergraduate with a strong foundation in machine learning,
                data analysis, and Python. I have hands-on experience developing ML models and building
                data-driven solutions using real-world datasets. I am highly motivated to apply my technical
                skills in a practical environment and am seeking a Data Science internship where I can
                contribute, learn, and grow within a professional team.
            </p>
            <p style='font-size:.7rem;color:{T3};font-family:"Fira Code",monospace;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.5rem;'>Education</p>
            <div style='display:flex;flex-direction:column;gap:.45rem;'>
                <div style='background:{BG};border:1px solid {BDR};border-radius:10px;padding:.75rem 1rem;'>
                    <p style='margin:0;font-size:.86rem;font-weight:700;color:{T1};'>BSc (Hons) in Artificial Intelligence & Data Science</p>
                    <p style='margin:0;font-size:.76rem;color:{T2};'>Robert Gordon University (RGU) · IIT Colombo · 2024 – Present</p>
                </div>
                <div style='background:{BG};border:1px solid {BDR};border-radius:10px;padding:.75rem 1rem;'>
                    <p style='margin:0;font-size:.86rem;font-weight:700;color:{T1};'>Bachelor of Information Technology (External)</p>
                    <p style='margin:0;font-size:.76rem;color:{T2};'>University of Moratuwa, Sri Lanka · 2026 – Present</p>
                </div>
                <div style='background:{BG};border:1px solid {BDR};border-radius:10px;padding:.75rem 1rem;'>
                    <p style='margin:0;font-size:.86rem;font-weight:700;color:{T1};'>Foundation Programme — <span style="color:{SUC};">Distinction ✓</span></p>
                    <p style='margin:0;font-size:.76rem;color:{T2};'>Informatics Institute of Technology (IIT), Colombo · Completed 2024</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    shdr("🛠️", "Technical Skills")
    for group, skills, cls in [
        ("Programming Languages",  ["Python", "SQL", "JavaScript"], "chip"),
        ("ML & Data Science",      ["Machine Learning", "Data Preprocessing", "Model Evaluation", "SHAP", "SMOTE"], "chip"),
        ("Libraries & Frameworks", ["NumPy", "Pandas", "Matplotlib", "Seaborn", "Scikit-learn", "TensorFlow", "Keras", "XGBoost"], "chip"),
        ("Web Technologies",       ["HTML", "CSS", "JavaScript", "React"], "chip2"),
        ("Databases",              ["MySQL", "MongoDB"], "chip2"),
        ("Tools & Cloud",          ["Jupyter", "Google Colab", "Git", "GitHub", "VS Code", "PyCharm", "AWS"], "chip2"),
    ]:
        chips = " ".join([f'<span class="{cls}">{s}</span>' for s in skills])
        st.markdown(f'<div style="margin-bottom:.7rem;"><p style="font-size:.68rem;color:{T3};font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:5px;font-family:\'Fira Code\',monospace;">{group}</p>{chips}</div>', unsafe_allow_html=True)

    shdr("🚀", "Project Experience")
    projects = [
        ("MRI Brain Tumor Chatbot", "NLP chatbot using Hugging Face transformer, fine-tuned on domain-specific medical data for natural language Q&A.", "NLP · Transformers · Hugging Face", None),
        ("MRI vs Non-MRI Classifier", "CNN-based image classifier with TensorFlow/Keras for medical image classification, with full preprocessing pipeline.", "Deep Learning · CNN · TensorFlow", None),
        ("Customer Churn Prediction", "End-to-end ML pipeline to predict churn — feature engineering, classification, and performance analysis.", "Classification · Scikit-learn", "https://github.com/mohamedahshaan/MachineLearning_CW"),
        ("ABC Bank Management System", "Menu-driven CLI banking app with account management, validation, error handling, and structured storage.", "Python · CLI", "https://github.com/mohamedahshaan/Bank-Management-System"),
        ("ICT Classes Website",  "Educational website with responsive HTML/CSS/JS layouts, clean UX, and clear navigation.", "HTML · CSS · JavaScript", "https://ictwithahshaan.edu.lk"),
        ("Organic Spices Website", "Mobile-responsive brand website with front-end components and clean UI design.", "HTML · CSS · Responsive", "https://organicspices.github.io"),
    ]
    for title, desc, tag, link in projects:
        lhtml = f'&nbsp;&nbsp;<a href="{link}" target="_blank" class="alink">🔗 {link}</a>' if link else ""
        st.markdown(f'<div class="pcard"><div class="ptitle">{title}</div><div class="pdesc">{desc}</div><div style="margin-top:.45rem;"><span class="chip" style="font-size:.7rem;">{tag}</span>{lhtml}</div></div>', unsafe_allow_html=True)

    shdr("🏆", "Achievements")
    st.markdown(f"""
    <div class="pcard" style="border-left:3px solid {A1};">
        <div class="ptitle">🥇 ModelX — ML Model Development Competition · IIT Colombo</div>
        <div class="pdesc">Team-based ML hackathon — data preprocessing, feature engineering, model development & evaluation using Python and ML libraries.</div>
        <div style="margin-top:.45rem;"><span class="chip">Hackathon</span><span class="chip">Teamwork</span><span class="chip">Python</span>
        <a href="https://github.com/mohamedahshaan/ModelX-Hackathon" target="_blank" class="alink">&nbsp;&nbsp;🔗 GitHub</a></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        shdr("🌐", "Languages")
        for lang, level, col in [("English","Intermediate",A2),("Tamil","Fluent",A1),("Sinhala","Fluent",A1)]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;background:{CARD};border:1px solid {BDR};border-radius:10px;padding:.65rem 1rem;margin-bottom:.45rem;"><span style="font-weight:700;color:{T1};font-size:.88rem;">{lang}</span><span style="color:{col};font-size:.76rem;font-weight:700;background:{col}18;padding:2px 12px;border-radius:20px;">{level}</span></div>', unsafe_allow_html=True)
    with c2:
        shdr("💡", "Soft Skills")
        soft = ["Analytical Thinking","Problem Solving","Critical Thinking","Communication","Attention to Detail","Teamwork","Time Management","Adaptability","Continuous Learning"]
        st.markdown(" ".join([f'<span class="chip2">{s}</span>' for s in soft]), unsafe_allow_html=True)

    shdr("📋", "References")
    rc1, rc2 = st.columns(2)
    for col, (name, role, org, email, phone) in zip([rc1, rc2], [
        ("Ayeshka Jayasundara","Lecturer","School of Computing, IIT Colombo","ayeshka.j@iit.ac.lk","+94 76 788 6365"),
        ("Aakib Firthows","Data Analyst","Dubai South Business Hub","aakib@dubaisouthbh.com","+94 77 328 2957")]):
        with col:
            st.markdown(f'<div style="background:{CARD};border:1px solid {BDR};border-radius:14px;padding:1.3rem;"><p style="margin:0 0 2px;font-size:.92rem;font-weight:800;color:{T1};">{name}</p><p style="margin:0 0 4px;font-size:.8rem;color:{A1};font-weight:600;">{role}</p><p style="margin:0 0 7px;font-size:.78rem;color:{T2};">{org}</p><p style="margin:0;font-size:.75rem;color:{T3};font-family:\'Fira Code\',monospace;">📧 {email}<br>📞 {phone}</p></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="footer"><p>© 2026 Mohamed Ahshaan · DataSci Studio v3.0 · All Rights Reserved.</p></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# PAGE: ML STUDIO
# ═══════════════════════════════════════════════════════
def page_app():
    st.markdown(f'<div class="hero"><p class="htitle">AI-Powered Data Analysis<br>&amp; Machine Learning System</p><p class="hsub">Upload · Clean · Explore · Model · Evaluate · Export</p><span class="hbadge">v3.0 · Final Edition</span></div>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown(f'<div style="text-align:center;padding:4rem 2rem;border:2px dashed {BDR};border-radius:20px;margin-top:1rem;background:{CARD};"><div style="font-size:3rem;margin-bottom:1rem;">📂</div><p style="font-size:1rem;color:{T2};font-family:\'Fira Code\',monospace;">Drag &amp; drop a CSV file in the sidebar, or click to browse.</p><p style="font-size:.8rem;color:{T3};">Classification &amp; Regression · 7+ ML Models · SHAP · SMOTE · XGBoost</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="footer"><p>© 2026 Mohamed Ahshaan · DataSci Studio v3.0 · All Rights Reserved.</p></div>', unsafe_allow_html=True)
        return

    # ── Load ──────────────────────────────────
    file_bytes = uploaded.read()
    df_raw, err = load_data(file_bytes, uploaded.name)
    if err: st.error(f"❌ {err}"); return
    if len(df_raw) > 100_000:
        wb(f"⚠️ Large dataset ({len(df_raw):,} rows). Using first 100,000 rows for performance.")
        df_raw = df_raw.head(100_000).reset_index(drop=True)
    elif len(df_raw) < 50:
        wb(f"⚠️ Very small dataset ({len(df_raw)} rows). Results may not be reliable.")

    # ── DATA OVERVIEW ─────────────────────────
    shdr("📋", "Data Overview")
    nn = df_raw.select_dtypes(include=[np.number]).shape[1]
    nc = df_raw.select_dtypes(include=['object','category']).shape[1]
    mgrid(
        (f"{df_raw.shape[0]:,}", "Rows",       ""),
        (f"{df_raw.shape[1]}",   "Columns",    ""),
        (f"{nn}",                "Numeric",     ""),
        (f"{nc}",                "Categorical", ""),
        (f"{df_raw.isnull().sum().sum():,}", "Missing", ""),
        (f"{df_raw.duplicated().sum():,}",   "Duplicates", ""),
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📄 Preview (first 5 rows)**")
        st.dataframe(df_raw.head())
    with c2:
        st.markdown("**🏷️ Column Summary**")
        st.dataframe(pd.DataFrame({
            "Column":   df_raw.columns,
            "Type":     df_raw.dtypes.astype(str).values,
            "Non-Null": df_raw.notnull().sum().values,
            "Unique":   df_raw.nunique().values,
            "Missing%": (df_raw.isnull().mean()*100).round(1).values,
        }), hide_index=True)

    # ── DATA QUALITY & CLEANING ───────────────
    shdr("🧹", "Data Quality & Intelligent Cleaning")
    q = data_quality(df_raw)
    miss_c = q["pct"][q["pct"] > 0]
    c1, c2 = st.columns(2)
    with c1:
        if miss_c.empty: sb("✅ No missing values detected.")
        else:
            st.markdown("**Missing Values by Column**")
            st.dataframe(pd.DataFrame({
                "Column": miss_c.index, "Missing %": miss_c.values,
                "Count": q["miss"][miss_c.index].values
            }).sort_values("Missing %", ascending=False), hide_index=True)
    with c2:
        sb("✅ No duplicate rows.") if q["dup"] == 0 else wb(f"⚠️ {q['dup']:,} duplicate row(s) found.")
        sb("✅ No constant columns.") if not q["const"] else wb(f"⚠️ Constant cols: {', '.join(q['const'])}")

    df_clean, exps = clean_data(df_raw)
    if exps:
        st.markdown("**Cleaning Actions Applied**")
        for e in exps: ib(e)
    else:
        sb("✅ Dataset was already clean. No actions required.")

    mgrid(
        (f"{df_raw.shape[0]:,}", "Before Cleaning", ""),
        (f"{df_clean.shape[0]:,}", "After Cleaning", f"color:{SUC};"),
    )

    # ── SIDEBAR CONTROLS ──────────────────────
    all_cols = df_clean.columns.tolist()

    with ph_info.container():
        _nc = df_clean.select_dtypes(include=[np.number]).shape[1]
        _cc = df_clean.select_dtypes(include=['object','category']).shape[1]
        _miss = df_clean.isnull().sum().sum()
        st.markdown(f"<p style='font-size:.72rem;color:{T3};font-family:\"Fira Code\",monospace;'><b>{len(df_clean):,} rows</b> · {len(all_cols)} cols · {_nc} numeric · {_cc} cat</p>", unsafe_allow_html=True)
        if _miss > 0:
            st.markdown(f"<p style='font-size:.7rem;color:{WRN};'>⚠️ {_miss} missing values remain</p>", unsafe_allow_html=True)
        st.caption("  \n".join([f"· {c}" for c in all_cols[:20]]))
        if len(all_cols) > 20: st.caption(f"... +{len(all_cols)-20} more")

    with ph_target.container():
        st.markdown(f"<p style='font-size:.8rem;font-weight:700;color:{T1};margin:0;'>🎯 Target Column</p>", unsafe_allow_html=True)
        tc = st.selectbox("tc", all_cols, index=len(all_cols)-1, label_visibility="collapsed")

    if tc:
        is_clf = df_clean[tc].dtype == 'object' or df_clean[tc].nunique() <= 20
        ptype = "Classification" if is_clf else "Regression"
        avail_raw = list(build_clf_models(True).keys()) if is_clf else list(build_reg_models().keys())

        with ph_split.container():
            st.markdown(f"<p style='font-size:.8rem;font-weight:700;color:{T1};margin:0;'>✂️ Train / Test Split</p>", unsafe_allow_html=True)
            split_pct = st.slider("split", 60, 90, st.session_state.train_split, 5, label_visibility="collapsed")
            st.session_state.train_split = split_pct
            st.caption(f"Train: {split_pct}% · Test: {100-split_pct}%")

        with ph_opts.container():
            st.markdown(f"<p style='font-size:.8rem;font-weight:700;color:{T1};margin:0 0 4px;'>⚙️ Options</p>", unsafe_allow_html=True)
            if is_clf:
                use_smote = st.checkbox("SMOTE (imbalance)", value=st.session_state.use_smote,
                                         help="Apply SMOTE oversampling to handle class imbalance")
                use_cw    = st.checkbox("Class Weights", value=st.session_state.use_class_weight,
                                         help="Use balanced class weights for applicable models")
                st.session_state.use_smote = use_smote
                st.session_state.use_class_weight = use_cw
            else:
                use_smote, use_cw = False, False

        with ph_models.container():
            st.markdown(f"<p style='font-size:.8rem;font-weight:700;color:{T1};margin:0;'>🤖 Models ({ptype})</p>", unsafe_allow_html=True)
            sel = st.multiselect("sel", avail_raw, default=avail_raw[:3], label_visibility="collapsed")

        with ph_run.container():
            run = st.button("🚀 Run Full Pipeline")
    else:
        run = False; sel = []; ptype = "Classification"; use_smote = False; use_cw = True; split_pct = 80

    # ── EDA ───────────────────────────────────
    shdr("📊", "Exploratory Data Analysis")
    if tc and tc in df_clean.columns:
        ib(f"🎯 Target: <strong>{tc}</strong> &nbsp;|&nbsp; Problem type: <strong>{ptype}</strong>")
        num_c = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != tc]
        if num_c:
            t1, t2, t3 = st.tabs(["📊 Distributions", "📦 Boxplots", "🔥 Correlation"])
            with t1:
                fig = p_hist(df_clean, num_c[:12]); st.pyplot(fig); plt.close(fig)
            with t2:
                fig = p_box(df_clean, num_c[:9]); st.pyplot(fig); plt.close(fig)
            with t3:
                fig = p_corr(df_clean)
                if fig: st.pyplot(fig); plt.close(fig)
                else: ib("Need ≥2 numeric columns for correlation heatmap.")
        else: ib("No numeric columns available for EDA.")

        # Outlier summary using IQR
        if num_c:
            outlier_rows = []
            for col in num_c[:10]:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                n_out = int(((df_clean[col] < Q1 - 1.5*IQR) | (df_clean[col] > Q3 + 1.5*IQR)).sum())
                pct_out = round(n_out / len(df_clean) * 100, 1)
                if n_out > 0:
                    outlier_rows.append({"Column": col, "Outliers": n_out, "Outlier %": pct_out})
            if outlier_rows:
                with st.expander("🔍 Outlier Summary (IQR Method)"):
                    st.dataframe(pd.DataFrame(outlier_rows), hide_index=True)
                    ib("💡 Outliers detected via IQR. Consider capping if they affect model performance.")

    # ── TRAINING ──────────────────────────────
    shdr("🤖", "Model Training & Evaluation")
    if not run:
        ib("👈 Configure the sidebar then click <strong>🚀 Run Full Pipeline</strong>.")
        st.markdown(f'<div class="footer"><p>© 2026 Mohamed Ahshaan · DataSci Studio v3.0 · All Rights Reserved.</p></div>', unsafe_allow_html=True)
        return
    if not sel: st.warning("Select at least one model."); return
    if tc not in df_clean.columns: st.error(f"Target '{tc}' not found."); return

    with st.spinner("⚙️ Engineering features..."):
        try: X, y, enc = engineer_features(df_clean, tc)
        except Exception as e: st.error(f"Feature engineering failed: {e}"); return

    if enc:
        with st.expander("🔧 Feature Engineering Details"):
            [ib(f"🔤 {e}") for e in enc]

    options_txt = []
    if ptype == "Classification":
        if use_smote and SMOTE_OK: options_txt.append("SMOTE oversampling ✓")
        elif use_smote: options_txt.append("SMOTE not installed — skipped")
        if use_cw: options_txt.append("Class weights ✓")
    if options_txt: ib(" &nbsp;·&nbsp; ".join(options_txt))

    with st.spinner(f"🏋️ Training {len(sel)} model(s) on {split_pct}/{100-split_pct}% split..."):
        results, X_te_raw, y_te_raw = train_models(X, y, sel, ptype, split_pct, use_smote, use_cw)

    # Force matplotlib cleanup after training
    plt.close('all')

    sb("✅ All models trained successfully!")
    summary = evaluate_clf(results) if ptype == "Classification" else evaluate_reg(results)

    # ── RESULTS DASHBOARD ─────────────────────
    shdr("🏆", "Results Dashboard")
    bm = "F1 Score" if ptype == "Classification" else "R²"
    st.markdown("**📊 All Model Scores**")
    st.dataframe(summary, hide_index=True)

    csv_b64 = df_to_csv_b64(summary)
    st.markdown(f'<a href="data:file/csv;base64,{csv_b64}" download="model_scores.csv" class="dl-btn">📥 Download Scores CSV</a>', unsafe_allow_html=True)

    if bm in summary.columns:
        v = summary.copy(); v[bm] = pd.to_numeric(v[bm], errors='coerce')
        bi = v[bm].idxmax(); bn = v.loc[bi, "Model"]; bs = v.loc[bi, bm]
        st.markdown(f'<div class="sb">🏆 <strong>Best Model: {bn}</strong><span class="bbadge">TOP PERFORMER</span><br>{bm}: <strong>{bs:.4f}</strong></div>', unsafe_allow_html=True)
        ib(f"💡 {MEXPL.get(bn, bn + ' achieved the best performance on this dataset.')}")

        fig = p_cmp(summary, bm)
        if fig:
            st.markdown("**Model Comparison Chart**")
            st.pyplot(fig); plt.close(fig)

    # ── PER-MODEL ANALYSIS ────────────────────
    shdr("🔬", "Per-Model Deep Dive")
    vn = [n for n in sel if n in results and "error" not in results[n]]
    if not vn: wb("No models completed successfully."); return

    tabs = st.tabs([f"  {n}  " for n in vn])
    for tab, name in zip(tabs, vn):
        with tab:
            r = results[name]
            yt = r["y_test"]; yp = r["y_pred"]
            model = r["model"]; Xte = r["X_test"]; feats = r["X_columns"]
            Xtr = r["X_train"]

            if ptype == "Classification":
                avg = 'binary' if len(np.unique(yt)) == 2 else 'macro'
                acc  = accuracy_score(yt, yp)
                prec = precision_score(yt, yp, average=avg, zero_division=0)
                rec  = recall_score(yt, yp, average=avg, zero_division=0)
                f1   = f1_score(yt, yp, average=avg, zero_division=0)

                roc_auc_val = None
                if len(np.unique(yt)) == 2 and hasattr(model, "predict_proba"):
                    try:
                        from sklearn.metrics import roc_auc_score
                        yprob = model.predict_proba(Xte)[:, 1]
                        roc_auc_val = roc_auc_score(yt, yprob)
                    except Exception:
                        pass

                if roc_auc_val is not None:
                    mgrid(
                        (f"{acc:.3f}",         "Accuracy",  ""),
                        (f"{prec:.3f}",         "Precision", ""),
                        (f"{rec:.3f}",          "Recall",    ""),
                        (f"{f1:.3f}",           "F1 Score",  f"color:{SUC};"),
                        (f"{roc_auc_val:.3f}",  "ROC-AUC",   f"color:{A2};"),
                    )
                else:
                    mgrid(
                        (f"{acc:.3f}",  "Accuracy",  ""),
                        (f"{prec:.3f}", "Precision", ""),
                        (f"{rec:.3f}",  "Recall",    ""),
                        (f"{f1:.3f}",   "F1 Score",  f"color:{SUC};"),
                    )

                with st.expander("📋 Full Classification Report"):
                    st.code(classification_report(yt, yp, zero_division=0))

                c1, c2 = st.columns(2)
                with c1:
                    fig = p_cm(yt, yp, name)
                    img_b64 = fig_to_b64(fig)
                    st.pyplot(fig); plt.close(fig)
                    st.markdown(f'<a href="data:image/png;base64,{img_b64}" download="{name}_confusion_matrix.png" class="dl-btn">📥 Save Plot</a>', unsafe_allow_html=True)
                with c2:
                    fig = p_roc(model, Xte, yt, name)
                    if fig:
                        img_b64 = fig_to_b64(fig)
                        st.pyplot(fig); plt.close(fig)
                        st.markdown(f'<a href="data:image/png;base64,{img_b64}" download="{name}_roc.png" class="dl-btn">📥 Save Plot</a>', unsafe_allow_html=True)
                    else: ib("ROC not available for multi-class.")

                fig = p_prc(model, Xte, yt, name)
                if fig:
                    c1, c2 = st.columns(2)
                    with c1:
                        img_b64 = fig_to_b64(fig)
                        st.markdown("**Precision-Recall Curve**")
                        st.pyplot(fig); plt.close(fig)
                        st.markdown(f'<a href="data:image/png;base64,{img_b64}" download="{name}_prc.png" class="dl-btn">📥 Save Plot</a>', unsafe_allow_html=True)

            else:  # Regression
                mae  = mean_absolute_error(yt, yp)
                mse  = mean_squared_error(yt, yp)
                rmse = np.sqrt(mse)
                r2   = r2_score(yt, yp)

                mgrid(
                    (f"{mae:.4f}",  "MAE",  ""),
                    (f"{mse:.4f}",  "MSE",  ""),
                    (f"{rmse:.4f}", "RMSE", ""),
                    (f"{r2:.4f}",   "R²",   f"color:{SUC};"),
                )

                c1, c2 = st.columns(2)
                with c1:
                    fig = p_avp(yt, yp, name)
                    img_b64 = fig_to_b64(fig)
                    st.pyplot(fig); plt.close(fig)
                    st.markdown(f'<a href="data:image/png;base64,{img_b64}" download="{name}_avp.png" class="dl-btn">📥 Save Plot</a>', unsafe_allow_html=True)
                with c2:
                    fig = p_res(yt, yp, name)
                    img_b64 = fig_to_b64(fig)
                    st.pyplot(fig); plt.close(fig)
                    st.markdown(f'<a href="data:image/png;base64,{img_b64}" download="{name}_residuals.png" class="dl-btn">📥 Save Plot</a>', unsafe_allow_html=True)

            # Feature importance
            fig = p_fi(model, feats, name)
            if fig:
                st.markdown("**🌳 Feature Importance**")
                img_b64 = fig_to_b64(fig)
                st.pyplot(fig); plt.close(fig)
                st.markdown(f'<a href="data:image/png;base64,{img_b64}" download="{name}_importance.png" class="dl-btn">📥 Save Plot</a>', unsafe_allow_html=True)

            # Cross-validation score
            with st.expander(f"🔁 Cross-Validation Score — {name}"):
                with st.spinner("Running 5-fold CV..."):
                    try:
                        from sklearn.model_selection import StratifiedKFold, cross_val_score
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if ptype=="Classification" else 5
                        scoring = "f1" if ptype=="Classification" else "r2"
                        # n_jobs=1 to avoid spawning extra processes on cloud (memory safe)
                        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
                        ib(f"📊 5-Fold CV {scoring.upper()}: <strong>{cv_scores.mean():.4f}</strong> ± {cv_scores.std():.4f} &nbsp;|&nbsp; Min: {cv_scores.min():.4f} &nbsp;|&nbsp; Max: {cv_scores.max():.4f}")
                        ib(f"💡 Low std (±{cv_scores.std():.4f}) = stable model. High std = unstable across splits.")
                    except Exception as e:
                        wb(f"CV failed: {e}")

            # SHAP
            if SHAP_OK:
                with st.expander(f"🔍 SHAP Explainability — {name}"):
                    with st.spinner("Computing SHAP values..."):
                        fig = p_shap(model, Xtr, feats, name)
                    if fig:
                        st.pyplot(fig); plt.close(fig)
                    else:
                        ib("SHAP not available for this model type.")

            # Download predictions
            pred_df = pd.DataFrame({"y_true": list(yt), "y_pred": list(yp)})
            csv_b64 = df_to_csv_b64(pred_df)
            st.markdown(f'<br><a href="data:file/csv;base64,{csv_b64}" download="{name}_predictions.csv" class="dl-btn">📥 Download {name} Predictions CSV</a>', unsafe_allow_html=True)

            # Cleanup after each model tab to free memory
            plt.close('all')

    # Footer
    st.markdown(f'<div class="footer"><p>© 2026 Mohamed Ahshaan · DataSci Studio v3.0 · All Rights Reserved.</p></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════
pg = st.session_state.page
if   pg == "app":     page_app()
elif pg == "about":   page_about()
elif pg == "founder": page_founder()
