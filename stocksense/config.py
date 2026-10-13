"""Central configuration: column mappings, thresholds and data-capability rules."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_DATA_PATH = DATA_DIR / "nse_2000000.csv"

# Raw NSE column -> internal name. Only columns that really exist in the file are used.
RAW_TO_INTERNAL = {
    "TradDt": "date", "TckrSymb": "symbol", "FinInstrmNm": "name", "ISIN": "isin",
    "SctySrs": "series", "OpnPric": "open", "HghPric": "high", "LwPric": "low",
    "ClsPric": "close", "LastPric": "last", "PrvsClsgPric": "prev_close",
    "SttlmPric": "settle", "TtlTradgVol": "volume", "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades",
}
REQUIRED_RAW = ["TradDt", "TckrSymb", "SctySrs", "OpnPric", "HghPric", "LwPric",
                "ClsPric", "PrvsClsgPric", "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd"]
PRICE_COLS = ["open", "high", "low", "close", "prev_close"]

# Data quality thresholds
OHLC_TOLERANCE = 1e-6          # tolerance when checking low <= close <= high
EXTREME_RETURN_PCT = 30.0      # |return| above this is flagged (not dropped)

# Minimum number of distinct trading days each temporal analysis needs to be meaningful.
MIN_DAYS = {
    "trend": 30,
    "weekday_seasonality": 60,
    "monthly_seasonality": 500,   # ~2 years so each month appears at least twice
    "cycles": 500,
    "residuals": 60,
    "regimes": 120,
    "forecasting": 250,
    "backtesting": 300,
    "corporate_actions": 2,
    "rule_validation_temporal": 20,
}

RANDOM_STATE = 42
