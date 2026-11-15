"""Currency universe, FRED series IDs, and constants for the carry trade screener."""

# --- G10 currencies: FRED series IDs for policy/short-term interest rates ---
G10_FRED_SERIES = {
    "USD": "FEDFUNDS",
    "EUR": "ECBDFR",
    "JPY": "IRSTCI01JPM156N",
    "GBP": "IUDSOIA",
    "CHF": "IR3TIB01CHM156N",
    "AUD": "IRSTCI01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CAD": "IRSTCI01CAM156N",
    "NOK": "IRSTCI01NOM156N",
    "SEK": "IR3TIB01SEM156N",
}

# --- EM currencies with reliable FRED coverage ---
EM_FRED_SERIES = {
    "MXN": "IRSTCI01MXM156N",
    "BRL": "IRSTCI01BRM156N",
    "ZAR": "IRSTCI01ZAM156N",
    "INR": "IRSTCI01INM156N",
    "TRY": "IRSTCI01TRM156N",
    "IDR": "IRSTCI01IDM156N",
}

# --- Emergency-only static rates ---
# Used solely as a last resort when a FRED fetch fails AND no prior cached
# value exists for that currency (e.g. first run after a series is
# discontinued). Not the primary source for any currency. Each entry's
# last_updated drives the staleness warning if it's ever actually used.
EM_FALLBACK_RATES = {
    "BRL": {"rate": 14.39, "last_updated": "2026-06"},
    "ZAR": {"rate": 7.00, "last_updated": "2026-06"},
    "INR": {"rate": 5.50, "last_updated": "2026-05"},
    "TRY": {"rate": 35.50, "last_updated": "2026-05"},
    "IDR": {"rate": 5.88, "last_updated": "2026-06"},
}

# --- Full currency universe ---
ALL_CURRENCIES = sorted(
    set(G10_FRED_SERIES) | set(EM_FRED_SERIES) | set(EM_FALLBACK_RATES)
)
G10_CURRENCIES = sorted(G10_FRED_SERIES)
EM_CURRENCIES = sorted(set(ALL_CURRENCIES) - set(G10_FRED_SERIES))

# Currencies whose rate is fetched from FRED (as opposed to the static fallback)
FRED_SERIES = {**G10_FRED_SERIES, **EM_FRED_SERIES}

# --- Volatility / carry computation constants ---
# Fetch window needs to cover the EM vol window (~5 trading years) plus a
# rolling-calc buffer, and the drift window (365 calendar days).
FX_HISTORY_DAYS = 1900
VOL_ROLLING_WINDOW = 30  # "30d vol"
VOL_ROLLING_WINDOW_LONG = 252  # ~1 trading year. G10 ranks on max(30d, this).
VOL_ROLLING_WINDOW_EM = 1260  # ~5 trading years. EM ranks on this alone, so a
# calm recent half-decade can't hide an older step-devaluation.
TRADING_DAYS_PER_YEAR = 252

# Trailing window for the drift adjustment: managed-float EM currencies (e.g.
# TRY) can depreciate on a steady, fairly predictable path that low 30d vol
# doesn't capture, which overstates their carry-to-vol. adj_carry subtracts
# this trailing depreciation from raw carry -- but only depreciation:
# appreciation must never increase carry. Not annualised -- this is the raw
# cumulative trend over the trailing window itself.
DRIFT_WINDOW_DAYS = 365  # 12 months

# A currency whose 1-year vol (vs. USD) is below this is flagged "managed" --
# its low day-to-day volatility likely understates devaluation risk.
MANAGED_CURRENCY_VOL_THRESHOLD_PCT = 3.0

# --- Signal thresholds (on carry/vol, raw for G10 and adjusted for EM) ---
SIGNAL_STRONG_THRESHOLD = 1.5
SIGNAL_MODERATE_THRESHOLD = 0.5

# --- Staleness threshold for EM static rates (days) ---
EM_STALE_DAYS = 60

# --- Caching TTLs (seconds) ---
RATE_CACHE_TTL = 3600
FX_CACHE_TTL = 900
CRASH_FEATURE_CACHE_TTL = 86400

# =====================================================================
# Crash early-warning research (Part B). G10 only -- see research/.
# =====================================================================

# Daily spot FX series for the research carry basket (USD + 9 currencies).
# "invert" = True means the raw FRED series is foreign-per-USD and must be
# inverted to get USD per 1 unit of foreign currency.
RESEARCH_FX_SERIES = {
    "EUR": {"series_id": "DEXUSEU", "invert": False},
    "GBP": {"series_id": "DEXUSUK", "invert": False},
    "AUD": {"series_id": "DEXUSAL", "invert": False},
    "NZD": {"series_id": "DEXUSNZ", "invert": False},
    "JPY": {"series_id": "DEXJPUS", "invert": True},
    "CHF": {"series_id": "DEXSZUS", "invert": True},
    "CAD": {"series_id": "DEXCAUS", "invert": True},
    "NOK": {"series_id": "DEXNOUS", "invert": True},
    "SEK": {"series_id": "DEXSDUS", "invert": True},
}

# 3-month (OECD) policy-proxy rate series for the research basket. "primary"
# is preferred; if its last observation is more than RESEARCH_STALE_SPLICE_
# MONTHS old, the tail is spliced from "splice" instead (logged when used).
# EUR has no OECD pair that's still live as of Nov 2026 (both IR3TIB01EZM156N
# and IRSTCI01EZM156N stopped updating 2026-01) -- it uses ECBDFR (daily,
# resampled to monthly) with no splice fallback.
RESEARCH_RATE_SERIES = {
    "USD": {"primary": "IR3TIB01USM156N", "splice": "IRSTCI01USM156N"},
    "EUR": {"primary": "ECBDFR", "splice": None},
    "JPY": {"primary": "IR3TIB01JPM156N", "splice": "IRSTCI01JPM156N"},
    "GBP": {"primary": "IR3TIB01GBM156N", "splice": "IRSTCI01GBM156N"},
    "CHF": {"primary": "IR3TIB01CHM156N", "splice": "IRSTCI01CHM156N"},
    "AUD": {"primary": "IR3TIB01AUM156N", "splice": "IRSTCI01AUM156N"},
    "NZD": {"primary": "IR3TIB01NZM156N", "splice": "IRSTCI01NZM156N"},
    "CAD": {"primary": "IR3TIB01CAM156N", "splice": "IRSTCI01CAM156N"},
    "NOK": {"primary": "IR3TIB01NOM156N", "splice": "IRSTCI01NOM156N"},
    "SEK": {"primary": "IR3TIB01SEM156N", "splice": "IRSTCI01SEM156N"},
}

RESEARCH_STALE_SPLICE_MONTHS = 6
RESEARCH_PUBLICATION_LAG_MONTHS = 1

# Macro / risk series (daily).
RESEARCH_RISK_SERIES = {
    "vix": "VIXCLS",
    "baa10y": "BAA10Y",
    "brent": "DCOILBRENTEU",
}

RESEARCH_SAMPLE_START = "2000-01-01"
RESEARCH_WEEKLY_RULE = "W-FRI"
RESEARCH_CACHE_DIR = "research/cache"

# Carry basket construction.
RESEARCH_N_LONG = 3
RESEARCH_N_SHORT = 3

# Crash label.
CRASH_FORWARD_WEEKS = 4
CRASH_LABEL_PERCENTILE = 10
CRASH_ROBUSTNESS_THRESHOLD = -0.03

# Feature windows.
FEATURE_VOL_WINDOW_WEEKS = 12
FEATURE_SHORT_CHANGE_WEEKS = 4
FEATURE_MOM_LONG_WEEKS = 12
FEATURE_BRENT_RET_WEEKS = 12

# Walk-forward evaluation.
WALK_FORWARD_FIRST_TRAIN_END = "2007-12-31"
WALK_FORWARD_FIRST_TEST_YEAR = 2008
WALK_FORWARD_EMBARGO_WEEKS = 4
RANDOM_STATE = 42

LOGIT_PARAMS = {"class_weight": "balanced", "random_state": RANDOM_STATE}  # default penalty is L2
GBM_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 200,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Backtest.
BACKTEST_TRANSACTION_COST_BP = 5
BACKTEST_SIGNAL_THRESHOLD_TOP_PCT = 0.20
BACKTEST_VOL_TARGET_WEIGHT_CAP = 2.0
BACKTEST_NAMED_EPISODES = {
    "2008 GFC": ("2008-09-01", "2008-12-31"),
    "Aug 2011": ("2011-08-01", "2011-08-31"),
    "Aug 2015": ("2015-08-01", "2015-08-31"),
    "Mar 2020": ("2020-03-01", "2020-03-31"),
    "Jul-Aug 2024": ("2024-07-01", "2024-08-31"),
}

# Current reading / regime.
CURRENT_READING_STALE_INPUT_DAYS = 45
REGIME_ELEVATED_PERCENTILE = 50
REGIME_HIGH_PERCENTILE = 90

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
FIGURES_DIR = "outputs/figures"
TABLES_DIR = "outputs/tables"
