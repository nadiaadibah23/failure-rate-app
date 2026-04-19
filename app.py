import streamlit as st
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")



st.set_page_config(page_title="Secondary Compressor Predictor", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0e1117; }
.metric-card {
    background: linear-gradient(145deg, #161b2e, #1c2238);
    border: 1px solid #2a3050;
    border-radius: 16px;
    padding: 22px 20px 18px 20px;
    text-align: center;
    margin-bottom: 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
}
.model-name { font-size:0.72rem; font-weight:800; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:12px; }
.status-badge { display:inline-block; padding:7px 22px; border-radius:30px; font-size:1.05rem; font-weight:800; margin-bottom:10px; }
.running  { background:#00e58815; color:#00e588; border:1.5px solid #00e58845; }
.stopped  { background:#ff6b4a15; color:#ff6b4a; border:1.5px solid #ff6b4a45; }
.raw-val  { font-size:0.73rem; color:#7a849c; margin-bottom:14px; }
.metric-row { display:flex; justify-content:space-around; border-top:1px solid #2a3050; padding-top:12px; }
.metric-label { font-size:0.65rem; color:#545e7a; text-transform:uppercase; letter-spacing:0.08em; }
.metric-value { font-size:1.0rem; font-weight:700; color:#c0c8e8; }
.input-label { font-size:0.78rem; font-weight:700; color:#8892b0; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:2px; }
.divider { border-top:1px solid #1e2438; margin:18px 0; }
.info-box { background:#0d1020; border-left:3px solid #4a5490; border-radius:8px; padding:11px 15px; font-size:0.8rem; color:#7a849c; margin-top:18px; }
.footer { margin-top:26px; font-size:0.7rem; color:#2e3858; text-align:center; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    base = os.path.dirname(__file__)
    result = {}
    for label, fname in [("Random Forest","randomforest.pkl"),("SVM","svm.pkl"),("XGBoost","xgb.pkl")]:
        with open(os.path.join(base, fname), "rb") as f:
            result[label] = pickle.load(f)
    return result

MODEL_CFG = {
    "Random Forest": {"color": "#56cfb2", "threshold": 0.5},
    "SVM":           {"color": "#f0a05a", "threshold": 0.5},
    "XGBoost":       {"color": "#e06c8c", "threshold": 0.5},
}

# Replace None with your actual computed values once you have the training dataset
MODEL_METRICS = {
    "Random Forest": {"r2": None, "rmse": None},
    "SVM":           {"r2": None, "rmse": None},
    "XGBoost":       {"r2": None, "rmse": None},
}

def run_predictions(models, temperature, power_factor, vibration):
    X = np.array([[temperature, power_factor, vibration]])
    results = {}
    for name, model in models.items():
        results[name] = float(model.predict(X)[0])
    return results

def is_running(name, val):
    return val >= MODEL_CFG[name]["threshold"]

# Header
st.markdown("""
<div style='padding:8px 0 4px 0'>
  <div style='font-size:2.1rem;font-weight:800;color:#dce4ff;letter-spacing:-0.02em'>⚙️ Secondary Compressor Monitor</div>
  <div style='font-size:0.9rem;color:#545e7a;margin-top:-4px'>Real-time multi-model prediction — adjust sliders to see instant results</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

col_in, col_out = st.columns([1, 2.5], gap="large")

with col_in:
    st.markdown("<div class='input-label'>🌡️ Temperature (°C)</div>", unsafe_allow_html=True)
    temperature = st.slider("temperature", 0.0, 150.0, 75.0, 0.5, label_visibility="collapsed", key="temp")
    st.markdown(f"<div style='text-align:right;font-size:0.82rem;color:#7b8cde;margin-top:-8px;margin-bottom:16px'><b>{temperature:.1f} °C</b></div>", unsafe_allow_html=True)

    st.markdown("<div class='input-label'>⚡ Power Factor</div>", unsafe_allow_html=True)
    power_factor = st.slider("power_factor", 0.0, 1.0, 0.85, 0.01, label_visibility="collapsed", key="pf")
    st.markdown(f"<div style='text-align:right;font-size:0.82rem;color:#56cfb2;margin-top:-8px;margin-bottom:16px'><b>{power_factor:.2f}</b></div>", unsafe_allow_html=True)

    st.markdown("<div class='input-label'>📳 Vibration (mm/s)</div>", unsafe_allow_html=True)
    vibration = st.slider("vibration", 0.0, 50.0, 5.0, 0.1, label_visibility="collapsed", key="vib")
    st.markdown(f"<div style='text-align:right;font-size:0.82rem;color:#f0a05a;margin-top:-8px;margin-bottom:16px'><b>{vibration:.1f} mm/s</b></div>", unsafe_allow_html=True)

    st.markdown("<div class='info-box'>⚡ Predictions update <b>in real-time</b> as you drag the sliders — no button needed.</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;color:#2e3858;line-height:1.8'><b style='color:#3d4a6a'>Classification rule</b><br>All models: output ≥ 0.5 → RUNNING</div>", unsafe_allow_html=True)

models = load_models()
preds  = run_predictions(models, temperature, power_factor, vibration)

with col_out:
    card_cols = st.columns(3, gap="medium")

    for idx, (name, raw_val) in enumerate(preds.items()):
        running     = is_running(name, raw_val)
        status      = "RUNNING" if running else "STOPPED"
        badge_cls   = "running" if running else "stopped"
        icon        = "🟢" if running else "🔴"
        color       = MODEL_CFG[name]["color"]
        raw_display = f"Raw output: <b>{raw_val:.4f}</b> / 1.00"

        m = MODEL_METRICS[name]
        r2_disp   = f"{m['r2']:.4f}"   if m["r2"]   is not None else "—"
        rmse_disp = f"{m['rmse']:.4f}" if m["rmse"] is not None else "—"

        html = f"""<div class='metric-card' style='border-color:{color}35'>
            <div class='model-name' style='color:{color}'>{name}</div>
            <div class='status-badge {badge_cls}'>{icon}&nbsp; {status}</div>
            <div class='raw-val'>{raw_display}</div>
            <div class='metric-row'>
                <div><div class='metric-label'>R² Score</div><div class='metric-value' style='color:{color}'>{r2_disp}</div></div>
                <div><div class='metric-label'>RMSE</div><div class='metric-value' style='color:{color}'>{rmse_disp}</div></div>
            </div>
        </div>"""
        with card_cols[idx]:
            st.markdown(html, unsafe_allow_html=True)

# Consensus
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
running_count = sum(1 for n, v in preds.items() if is_running(n, v))
total         = len(preds)
consensus     = "RUNNING" if running_count > total / 2 else "STOPPED"
c_color       = "#00e588" if consensus == "RUNNING" else "#ff6b4a"

st.markdown(f"""
<div style='background:linear-gradient(135deg,#161b2e,#1c2238);border:1.5px solid {c_color}35;border-radius:14px;padding:18px 28px;display:flex;align-items:center;justify-content:space-between;'>
    <div>
        <div style='font-size:0.7rem;color:#545e7a;letter-spacing:0.1em;text-transform:uppercase'>Ensemble Consensus</div>
        <div style='font-size:1.65rem;font-weight:800;color:{c_color};margin-top:3px'>{"🟢" if consensus=="RUNNING" else "🔴"}&nbsp; {consensus}</div>
    </div>
    <div style='text-align:right'>
        <div style='font-size:0.7rem;color:#545e7a'>Models in agreement</div>
        <div style='font-size:1.35rem;font-weight:700;color:#c0c8e8'>{running_count} / {total} predict RUNNING</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='footer'>Secondary Compressor Prediction System &nbsp;·&nbsp; Models: Random Forest · SVM · XGBoost</div>", unsafe_allow_html=True)