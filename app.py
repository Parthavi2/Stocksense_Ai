"""StockSense AI - Streamlit entry point.  Run:  streamlit run app.py"""
import warnings

import streamlit as st

from pages_ui import intelligence, overview, pattern_lab, prediction_lab, trend_studio
from pages_ui.common import Context
from stocksense import config as C
from stocksense.preprocessing import DataError
from stocksense.storage import load_with_cache

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="StockSense AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
[data-testid="stMetric"] {background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:10px 14px;}
h1,h2,h3 {letter-spacing:-0.01em;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and cleaning data (first run on a large CSV takes a few seconds; later runs use a cache)...")
def load(path: str, series: tuple):
    return load_with_cache(path, series)


PAGES = {"Market Overview": overview, "Pattern Mining Lab": pattern_lab, "Trend & Variation Studio": trend_studio,
         "Prediction Lab": prediction_lab, "Market Intelligence": intelligence}

with st.sidebar:
    st.title("📈 StockSense AI")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    path = st.text_input("Data file or folder", str(C.DEFAULT_DATA_PATH), help="A NSE CSV, or a folder of daily CSVs with the same columns.")
    series = st.text_input("Series to keep", "EQ", help="Comma-separated NSE series codes; leave empty to keep all.")
    if st.button("Reload data"):
        st.cache_data.clear()
        st.rerun()

try:
    keep = tuple(s.strip() for s in series.split(",") if s.strip())
    df, rep, from_cache = load(path, keep)
except DataError as e:
    st.error(f"**Data problem:** {e}")
    st.info(f"Place your CSV at `{C.DEFAULT_DATA_PATH}` or enter its path in the sidebar.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error while loading data: {e}")
    st.stop()

ctx = Context(df=df, rep=rep, key=f"{path}|{keep}|{rep.rows_clean}|{rep.date_max}")
st.sidebar.caption(f"{rep.n_symbols:,} stocks | {rep.n_dates} day(s) | {'cache' if from_cache else 'parsed'}")

try:
    PAGES[page].render(ctx)
except Exception as e:  # last-resort guard so one bad widget state never blanks the app
    st.error(f"Something went wrong on this page: {e}")
    with st.expander("Technical details"):
        st.exception(e)
