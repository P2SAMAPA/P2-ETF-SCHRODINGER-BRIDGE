import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Schrödinger Bridge – Entropic Transport", layout="wide")

st.markdown("""
<style>
.hero-card {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 0.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.hero-card h3 {
    font-size: 2rem;
    margin: 0;
    font-weight: bold;
}
.hero-card p {
    font-size: 1.2rem;
    margin: 0.5rem 0 0;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center;">🌉 Schrödinger Bridge</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Entropic optimal transport | Stochastic bridge between market regimes | Generative trajectory simulation</p>', unsafe_allow_html=True)

st.sidebar.markdown("## 🧩 Entropic OT")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows evaluated:** {', '.join(map(str, config.WINDOWS))} days")
st.sidebar.markdown(f"**Regularization ε:** {config.REG_EPS}")
st.sidebar.markdown(f"**Sinkhorn iterations:** {config.MAX_ITER}")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'schrodinger_bridge_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']

def display_universe(universe_name, uni_data, window_mode="best", selected_window=None):
    if not uni_data:
        st.warning(f"No data for {universe_name}")
        return
    if window_mode == "best":
        win = uni_data["best_window"]
        win_data = uni_data["best_window_data"]
        if win_data is None:
            st.warning(f"No best window data for {universe_name}")
            return
        top3 = win_data["top_etfs"]
        norm_scores = win_data["all_scores_norm"]
        raw_scores = win_data["all_scores_raw"]
        st.markdown(f'<h2 style="font-size: 1.8rem; margin-top: 1rem;">{universe_name.replace("_", " ").title()} <span style="font-size: 0.9rem; background: #e0e0e0; padding: 0.2rem 0.8rem; border-radius: 20px;">best window {win}d</span></h2>', unsafe_allow_html=True)
    else:
        win_data = next((wd for wd in uni_data["all_windows"] if wd["window"] == selected_window), None)
        if win_data is None:
            st.warning(f"No data for window {selected_window} in {universe_name}")
            return
        top3 = win_data["top_etfs"]
        norm_scores = win_data["all_scores_norm"]
        raw_scores = win_data["all_scores_raw"]
        st.markdown(f'<h2 style="font-size: 1.8rem; margin-top: 1rem;">{universe_name.replace("_", " ").title()} <span style="font-size: 0.9rem; background: #e0e0e0; padding: 0.2rem 0.8rem; border-radius: 20px;">window {selected_window}d</span></h2>', unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, etf in enumerate(top3):
        with cols[idx]:
            st.markdown(f"""
            <div class="hero-card">
                <h3>{etf['ticker']}</h3>
                <p>Bridge score: {etf['bridge_score_norm']:.3f}</p>
                <p style="font-size:0.9rem;">raw: {etf['raw_score']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    with st.expander(f"Full ranking for {universe_name}"):
        df_full = pd.DataFrame(list(norm_scores.items()), columns=["Ticker", "Normalized Score"])
        df_full["Raw Score"] = df_full["Ticker"].apply(lambda t: raw_scores[t])
        df_full = df_full.sort_values("Normalized Score", ascending=False)
        st.dataframe(df_full, use_container_width=True)

tab1, tab2 = st.tabs(["📊 Best Window (Auto)", "🔍 Choose Window (Manual)"])

with tab1:
    st.header("🌉 Top ETFs by Schrödinger Bridge Mass Transport (Auto Best Window)")
    with st.expander("📖 Interpretation", expanded=False):
        st.markdown("""
        - **Schrödinger bridge** solves the entropic optimal transport problem between two distributions (start and end returns).
        - It provides a stochastic process (a bridge) that interpolates between regimes.
        - The score for each ETF is the total probability mass **sent from** that ETF to all others in the optimal coupling.
        - High score indicates the ETF is a major source of mass in the bridge – i.e., its return distribution shifts strongly toward other assets, potentially a leading indicator.
        - The best window per universe is automatically selected (largest absolute raw signal).
        """)
    for universe_name, uni_data in data["universes"].items():
        display_universe(universe_name, uni_data, window_mode="best")

with tab2:
    st.header("🔍 Manual Window Selection")
    st.markdown("Choose a rolling window to inspect the entropic transport scores for each ETF.")
    for universe_name, uni_data in data["universes"].items():
        if not uni_data or not uni_data.get("all_windows"):
            st.warning(f"No window data for {universe_name}")
            continue
        available_windows = [wd["window"] for wd in uni_data["all_windows"]]
        sel_win = st.selectbox(f"Window for {universe_name.replace('_', ' ').title()}", available_windows, key=f"manual_{universe_name}")
        display_universe(universe_name, uni_data, window_mode="manual", selected_window=sel_win)

st.sidebar.markdown("---")
st.sidebar.caption("Schrödinger Bridge | Entropic OT & stochastic bridge matching for ETFs")
