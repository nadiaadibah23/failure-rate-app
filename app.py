import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, os
import sys
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Compressor Predictive Maintenance",
    page_icon="⚙️", layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:#0d0f14;color:#e8ecf4;}
.stApp{background-color:#0d0f14;}
h1,h2,h3{font-family:'Space Mono',monospace;}
.main-title{font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:#00d4aa;letter-spacing:-0.5px;margin-bottom:0;}
.sub-title{font-size:0.95rem;color:#7a8aa0;margin-top:4px;}
.metric-card{background:#161923;border:1px solid #2a3347;border-radius:12px;padding:20px 24px;text-align:center;}
.metric-card:hover{border-color:#00d4aa;}
.metric-label{font-size:0.75rem;color:#7a8aa0;text-transform:uppercase;letter-spacing:1px;font-family:'Space Mono',monospace;}
.metric-value{font-size:2rem;font-weight:700;font-family:'Space Mono',monospace;color:#00d4aa;line-height:1.2;}
.metric-sub{font-size:0.78rem;color:#7a8aa0;margin-top:2px;}
.risk-badge{display:inline-block;padding:6px 16px;border-radius:999px;font-family:'Space Mono',monospace;font-size:0.85rem;font-weight:700;letter-spacing:0.5px;}
.risk-low{background:#0d3d2e;color:#00d4aa;border:1px solid #00d4aa55;}
.risk-moderate{background:#3d2a0d;color:#f5a623;border:1px solid #f5a62355;}
.risk-high{background:#3d1a0d;color:#ff6b35;border:1px solid #ff6b3555;}
.risk-critical{background:#3d0d15;color:#e8445a;border:1px solid #e8445a55;}
.section-header{font-family:'Space Mono',monospace;font-size:0.7rem;text-transform:uppercase;letter-spacing:2px;color:#7a8aa0;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #2a3347;}
.pred-card{background:#161923;border:1px solid #2a3347;border-radius:12px;padding:16px 20px;margin-bottom:10px;}
.comparison-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.9rem;}
.comparison-table th{background:#1e2433;color:#7a8aa0;font-family:'Space Mono',monospace;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;padding:12px 16px;border-bottom:1px solid #2a3347;text-align:left;}
.comparison-table td{padding:12px 16px;border-bottom:1px solid #2a3347;color:#e8ecf4;}
.comparison-table tr:hover td{background:#1e2433;}
.best{color:#00d4aa;font-weight:700;}
/* Hide sidebar toggle */
[data-testid="collapsedControl"]{display:none;}
</style>
""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────────────────────
FEATURE_COLS = ['VIBRATION', 'TEMPERATURE', 'ACTUAL POWER FACTOR']
RISK_COLORS  = {"Low Risk":"#00d4aa","Critical Risk":"#e8445a",
                "Moderate Risk":"#f5a623","High Risk":"#ff6b35"}

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()

MODEL_CFG = {
    "Random Forest": {
        "r2_train":0.9998,"rmse_train":0.0041,
        "r2_val":  0.9989,"rmse_val":  0.0102,
        "r2_test": 0.9974,"rmse_test": 0.0160,
        "color":"#00d4aa","icon":"🌲",
        "pkl": BASE_DIR / "randomforest__1_.pkl",
        "scaler_pkl":None,
        "model_type":"sklearn_clf",
        "risk_dist":{"Low Risk":89.11,"Critical Risk":10.88,"Moderate Risk":0.01,"High Risk":0.01},
        "shap":{"VIBRATION":0.13,"ACTUAL POWER FACTOR":0.08,"TEMPERATURE":0.00},
    },
    "ANN": {
        "r2_train":0.9996,"rmse_train":0.0066,
        "r2_val":  0.9979,"rmse_val":  0.0144,
        "r2_test": 0.9977,"rmse_test": 0.0151,
        "color":"#4f8ef7","icon":"🧠",
        "pkl": BASE_DIR / "ann2.pkl",
        "scaler_pkl": BASE_DIR / "scaler_ann.pkl",
        "model_type":"keras_clf_from_pkl",
        "risk_dist":{"Low Risk":88.8,"Critical Risk":10.9,"Moderate Risk":0.15,"High Risk":0.15},
        "shap":{"VIBRATION":0.11,"ACTUAL POWER FACTOR":0.09,"TEMPERATURE":0.01},
    },
    "SVM": {
        "r2_train":0.9987,"rmse_train":0.0115,
        "r2_val":  0.9947,"rmse_val":  0.0227,
        "r2_test": 0.9968,"rmse_test": 0.0177,
        "color":"#f5a623","icon":"📐",
        "pkl": BASE_DIR / "svm2.pkl",
        "scaler_pkl": BASE_DIR / "scaler_svm.pkl",
        "model_type":"sklearn_clf",
        "risk_dist":{"Low Risk":87.8,"Critical Risk":11.8,"Moderate Risk":0.20,"High Risk":0.20},
        "shap":{"VIBRATION":0.12,"ACTUAL POWER FACTOR":0.07,"TEMPERATURE":0.01},
    },
    "XGBoost": {
        "r2_train":0.9978,"rmse_train":0.0145,
        "r2_val":  0.9980,"rmse_val":  0.0140,
        "r2_test": 0.9950,"rmse_test": 0.0220,
        "color":"#e8445a","icon":"⚡",
        "pkl": BASE_DIR / "xgboost2.pkl",
        "scaler_pkl":None,
        "model_type":"sklearn_clf",
        "risk_dist":{"Low Risk":87.2,"Critical Risk":12.3,"Moderate Risk":0.25,"High Risk":0.25},
        "shap":{"VIBRATION":0.10,"ACTUAL POWER FACTOR":0.08,"TEMPERATURE":0.02},
    },
}

# ── Load Models (Silent mode - logs to terminal only) ─────────────────────────
@st.cache_resource(show_spinner=False)  # Disable Streamlit spinner
def load_all():
    """Load all models silently, logging only to terminal/console"""
    out = {}
    
    print("\n" + "="*60)
    print("🚀 Loading Models for Compressor Predictive Maintenance")
    print("="*60)
    
    for name, cfg in MODEL_CFG.items():
        entry = {"model": None, "scaler": None, "err": None}
        
        # Handle ANN model loaded from pickle (Keras model inside)
        if cfg["model_type"] == "keras_clf_from_pkl":
            pkl_path = cfg["pkl"]
            
            if not pkl_path.exists():
                entry["err"] = f"Model file not found: {pkl_path.name}"
                print(f"❌ [ERROR] {name}: {entry['err']}")
                out[name] = entry
                continue
            
            try:
                with open(pkl_path, "rb") as f:
                    entry["model"] = pickle.load(f)
                print(f"✅ [SUCCESS] {name} loaded from {pkl_path.name}")
            except Exception as e:
                entry["err"] = f"Failed to load {name}: {str(e)}"
                print(f"❌ [ERROR] {name}: {entry['err']}")
        
        # Handle regular sklearn models
        elif cfg["model_type"] == "sklearn_clf":
            pkl_path = cfg["pkl"]
            if not pkl_path.exists():
                entry["err"] = f"Model file not found: {pkl_path.name}"
                print(f"❌ [ERROR] {name}: {entry['err']}")
                out[name] = entry
                continue
            
            try:
                with open(pkl_path, "rb") as f:
                    entry["model"] = pickle.load(f)
                print(f"✅ [SUCCESS] {name} loaded from {pkl_path.name}")
            except Exception as e:
                entry["err"] = f"Failed to load {name}: {str(e)}"
                print(f"❌ [ERROR] {name}: {entry['err']}")
        
        # Load scaler if specified
        if cfg.get("scaler_pkl") and cfg["scaler_pkl"].exists():
            try:
                with open(cfg["scaler_pkl"], "rb") as f:
                    entry["scaler"] = pickle.load(f)
                print(f"📊 [SCALER] {name} scaler loaded from {cfg['scaler_pkl'].name}")
            except Exception as e:
                print(f"⚠️ [WARNING] Could not load scaler for {name}: {e}")
        
        out[name] = entry
    
    # Summary
    print("\n" + "="*60)
    loaded_count = sum(1 for v in out.values() if v["model"] is not None)
    print(f"📈 Loading Complete: {loaded_count}/{len(MODEL_CFG)} models loaded successfully")
    print("="*60 + "\n")
    
    return out

# ── Helpers ────────────────────────────────────────────────────────────────────
def classify_risk(p):
    if p < 0.20: return "Low Risk",      "risk-low"
    if p < 0.50: return "Moderate Risk", "risk-moderate"
    if p < 0.80: return "High Risk",     "risk-high"
    return "Critical Risk", "risk-critical"

def predict(name, loaded, vib, temp, pf):
    cfg    = MODEL_CFG[name]
    model_data = loaded.get(name, {})
    model  = model_data.get("model")
    scaler = model_data.get("scaler")
    
    if model is None:
        error_msg = model_data.get("err", "Model not loaded")
        return {"err": error_msg}
    
    X = pd.DataFrame([[vib, temp, pf]], columns=FEATURE_COLS)
    try:
        # Apply scaler if available
        if scaler is not None:
            Xi = scaler.transform(X)
        else:
            Xi = X.values
        
        # Handle different model types
        if cfg["model_type"] == "keras_clf_from_pkl":
            # For Keras model loaded from pickle
            prediction = model.predict(Xi, verbose=0)
            # Handle different output shapes
            if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
                if prediction.shape[-1] == 1:
                    raw = float(prediction[0][0])
                else:
                    raw = float(prediction[0][1] if prediction.shape[-1] > 1 else prediction[0][0])
            else:
                raw = float(prediction[0]) if hasattr(prediction, '__getitem__') else float(prediction)
            
            op = float(np.clip(raw, 0, 1))
            fp = 1.0 - op
            state = "Operational" if op >= 0.5 else "Failure"
            
        elif cfg["model_type"] == "sklearn_clf":
            # For sklearn models
            pred = model.predict(Xi)[0]
            proba = model.predict_proba(Xi)[0]
            fp = float(proba[0])
            op = float(proba[1])
            state = "Operational" if int(pred) == 1 else "Failure"
        else:
            return {"err": f"Unknown model type: {cfg['model_type']}"}
        
        rl, rc = classify_risk(fp)
        return {"state":state, "fail":round(fp*100,2),
                "oper":round(op*100,2), "rl":rl, "rc":rc}
    except Exception as e:
        return {"err": f"Prediction error: {str(e)}"}

def dark():
    plt.rcParams.update({
        'figure.facecolor':'#161923','axes.facecolor':'#161923',
        'axes.edgecolor':'#2a3347','axes.labelcolor':'#7a8aa0',
        'xtick.color':'#7a8aa0','ytick.color':'#7a8aa0',
        'text.color':'#e8ecf4','grid.color':'#2a3347',
        'grid.alpha':0.4,'font.family':'monospace',
    })

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

# Load models (silent - logs to terminal only)
loaded = load_all()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚙️ Compressor Failure Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">RF · ANN · SVM · XGBoost — Performance & Live Prediction</p>',
            unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Live Input ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Live Prediction Input</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    vib  = st.slider("Vibration (Hz)",   0.0,  6.0,  3.5, 0.1)
with c2:
    temp = st.slider("Temperature (°C)", 25.0, 46.0, 40.0, 0.1)
with c3:
    pf   = st.slider("Power Factor",     0.0,  100.0, 98.0, 0.5)

st.markdown("<br>", unsafe_allow_html=True)

# ── Live Prediction Results ───────────────────────────────────────────────────
st.markdown('<p class="section-header">Prediction Results — All Models</p>', unsafe_allow_html=True)
pc = st.columns(4)
for i, (name, cfg) in enumerate(MODEL_CFG.items()):
    with pc[i]:
        r = predict(name, loaded, vib, temp, pf)
        if r and "state" in r:
            st.markdown(f"""
            <div class="pred-card" style="border-color:{cfg['color']}55;">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                            color:#7a8aa0;">{cfg['icon']} {name}</div>
                <div style="font-family:'Space Mono',monospace;font-size:1.2rem;
                            font-weight:700;color:{cfg['color']};margin-top:4px;">
                    {r['state']}</div>
                <span class="risk-badge {r['rc']}"
                      style="margin-top:8px;display:inline-block;">{r['rl']}</span>
                <div style="margin-top:10px;font-size:0.8rem;color:#7a8aa0;line-height:2;">
                    Failure: <b style='color:#e8445a'>{r['fail']:.1f}%</b><br>
                    Operational: <b style='color:#00d4aa'>{r['oper']:.1f}%</b>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            err = (r.get("err", "Model not available") if r else "Unknown error")
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                            color:#7a8aa0;">{cfg['icon']} {name}</div>
                <div style="color:#e8445a;font-size:0.8rem;margin-top:8px;">
                    ⚠️ Model Unavailable</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Model Performance KPIs ────────────────────────────────────────────────────
st.markdown('<p class="section-header">Test Set R² — All Models</p>', unsafe_allow_html=True)
kpi = st.columns(4)
for i, (name, cfg) in enumerate(MODEL_CFG.items()):
    with kpi[i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{cfg['icon']} {name}</div>
            <div class="metric-value" style="color:{cfg['color']}">{cfg['r2_test']:.4f}</div>
            <div class="metric-sub">RMSE: {cfg['rmse_test']:.4f}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Performance Comparison Table ──────────────────────────────────────────────
st.markdown('<p class="section-header">Model Performance Comparison</p>', unsafe_allow_html=True)
best_r2   = max(c["r2_test"]   for c in MODEL_CFG.values())
best_rmse = min(c["rmse_test"] for c in MODEL_CFG.values())
rows = ""
for name, cfg in MODEL_CFG.items():
    r2c = "best" if cfg["r2_test"]   == best_r2   else ""
    rc  = "best" if cfg["rmse_test"] == best_rmse else ""
    rows += f"""<tr>
        <td>{cfg['icon']} {name}</td>
        <td>{cfg['r2_train']:.4f}</td>
        <td>{cfg['r2_val']:.4f}</td>
        <td class="{r2c}">{cfg['r2_test']:.4f}</td>
        <td>{cfg['rmse_train']:.4f}</td>
        <td>{cfg['rmse_val']:.4f}</td>
        <td class="{rc}">{cfg['rmse_test']:.4f}</td>
    </tr>"""
st.markdown(f"""
<table class="comparison-table"><thead><tr>
    <th>Model</th>
    <th>R² Train</th><th>R² Val</th><th>R² Test</th>
    <th>RMSE Train</th><th>RMSE Val</th><th>RMSE Test</th>
</tr></thead><tbody>{rows}</tbody></table>
<p style='font-size:0.75rem;color:#7a8aa0;margin-top:8px;'>
🏆 <span style='color:#00d4aa'>Green</span> = best per metric</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── R² and RMSE Charts ────────────────────────────────────────────────────────
names  = list(MODEL_CFG.keys())
colors = [c["color"] for c in MODEL_CFG.values()]
ch1, ch2 = st.columns(2)

with ch1:
    dark()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    r2s = [c["r2_test"] for c in MODEL_CFG.values()]
    bars = ax.barh(names, r2s, color=colors, height=0.45, edgecolor='none')
    ax.set_xlim(0.99, 1.001)
    ax.set_xlabel("R² Score")
    ax.set_title("R² Score — Test Set", fontsize=10, pad=10)
    ax.axvline(x=1.0, color='#2a3347', linewidth=1, linestyle='--')
    for b, v in zip(bars, r2s):
        ax.text(v+0.0001, b.get_y()+b.get_height()/2,
                f'{v:.4f}', va='center', fontsize=8, color='#e8ecf4')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()

with ch2:
    dark()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    rmses = [c["rmse_test"] for c in MODEL_CFG.values()]
    bars  = ax.barh(names, rmses, color=colors, height=0.45, edgecolor='none')
    ax.set_xlabel("RMSE")
    ax.set_title("RMSE — Test Set (lower = better)", fontsize=10, pad=10)
    for b, v in zip(bars, rmses):
        ax.text(v+0.0002, b.get_y()+b.get_height()/2,
                f'{v:.4f}', va='center', fontsize=8, color='#e8ecf4')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("<br>", unsafe_allow_html=True)

# ── Risk Distribution ─────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Risk Level Distribution — All Models</p>', unsafe_allow_html=True)
dark()
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (name, cfg) in zip(axes, MODEL_CFG.items()):
    dist = cfg["risk_dist"]
    lbs  = list(dist.keys()); szs = list(dist.values())
    clrs = [RISK_COLORS[l] for l in lbs]
    _, _, ats = ax.pie(szs, labels=None, colors=clrs, autopct='%1.1f%%',
                       startangle=90, pctdistance=0.75,
                       wedgeprops=dict(width=0.5, edgecolor='#0d0f14', linewidth=2))
    for at in ats: at.set_fontsize(7); at.set_color('#e8ecf4')
    ax.set_title(f"{cfg['icon']} {name}", fontsize=9, pad=8, color=cfg['color'])
patches = [mpatches.Patch(color=c, label=l) for l, c in RISK_COLORS.items()]
fig.legend(handles=patches, loc='lower center', ncol=4,
           frameon=False, fontsize=8, labelcolor='#7a8aa0')
fig.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("<br>", unsafe_allow_html=True)

# ── SHAP Feature Importance ───────────────────────────────────────────────────
st.markdown('<p class="section-header">Feature Importance (SHAP) — All Models</p>', unsafe_allow_html=True)
sh1, sh2, sh3, sh4 = st.columns(4)
for col, (name, cfg) in zip([sh1, sh2, sh3, sh4], MODEL_CFG.items()):
    with col:
        dark()
        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        feats = list(cfg["shap"].keys()); vals = list(cfg["shap"].values())
        bcs = [cfg['color'], cfg['color']+'aa', cfg['color']+'55']
        bars = ax.barh(feats, vals, color=bcs, height=0.4, edgecolor='none')
        ax.set_xlabel("Mean |SHAP|", fontsize=7)
        ax.set_title(f"{cfg['icon']} {name}", fontsize=8, pad=6, color=cfg['color'])
        for b, v in zip(bars, vals):
            ax.text(v+0.002, b.get_y()+b.get_height()/2,
                    f'{v:.3f}', va='center', fontsize=7, color='#e8ecf4')
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()