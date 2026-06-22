"""
Factor construction and scoring.

5 factors:
  - Momentum   : 12-1 month cumulative return (skip last month)
  - Value      : 1 / P/B  (higher = cheaper)
  - Quality    : ROE
  - Low Vol    : negative annualised vol (lower vol = higher score)
  - Size       : negative log market cap (smaller = higher score, classic SMB)

Pipeline:
  1. Compute raw factor for each asset
  2. Cross-sectional winsorize (1%-99%)
  3. Cross-sectional z-score
  4. Optional: sector-neutralise
  5. Composite score = weighted average of factor z-scores
"""

import numpy as np
import pandas as pd
from scipy import stats


# ─── Raw factor computation ─────────────────────────────────────────────────

def factor_momentum(prices: pd.DataFrame) -> pd.Series:
    """
    12-1 month momentum: return from t-252 to t-21 trading days.
    Skip last month to avoid short-term reversal contamination.
    """
    if len(prices) < 252:
        return pd.Series(dtype=float)
    ret = prices.iloc[-21] / prices.iloc[-252] - 1
    return ret


def factor_value(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Value = 1 / P/B. Higher score = cheaper stock.
    """
    pb = fundamentals["pb_ratio"].dropna()
    pb = pb[pb > 0]
    return (1 / pb).rename("value")


def factor_quality(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Quality = ROE. Higher = more profitable.
    """
    return fundamentals["roe"].dropna().rename("quality")


def factor_low_vol(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Low vol = negative annualised volatility.
    Lower vol → higher score.
    """
    vol = returns.tail(window).std() * np.sqrt(252)
    return (-vol).rename("low_vol")


def factor_size(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Size = negative log market cap (SMB: smaller = higher score).
    """
    mc = fundamentals["market_cap"].dropna()
    mc = mc[mc > 0]
    return (-np.log(mc)).rename("size")


# ─── Normalisation ───────────────────────────────────────────────────────────

def winsorize(series: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
    """Clip extreme values at 1%-99% cross-sectionally."""
    arr = stats.mstats.winsorize(series.dropna(), limits=limits)
    return pd.Series(arr, index=series.dropna().index)


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """Zero mean, unit std across assets at a given point in time."""
    s = winsorize(series)
    return (s - s.mean()) / s.std()


def sector_neutralise(scores: pd.Series, sectors: pd.Series) -> pd.Series:
    """
    Demean scores within each sector so factor exposure
    doesn't just reflect sector tilts.
    """
    neutralised = scores.copy()
    for sector in sectors.unique():
        mask = sectors == sector
        common = scores.index.intersection(sectors[mask].index)
        if len(common) < 2:
            continue
        neutralised[common] = scores[common] - scores[common].mean()
    return neutralised


# ─── Composite score ─────────────────────────────────────────────────────────

def compute_composite_score(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    fundamentals: pd.DataFrame,
    weights: dict,          # {"momentum": 0.2, "value": 0.2, ...}
    sector_neutral: bool = True,
) -> pd.DataFrame:
    """
    Build composite factor score for all assets.

    Returns DataFrame with columns:
        [momentum, value, quality, low_vol, size, composite]
    """
    factor_funcs = {
        "momentum": lambda: factor_momentum(prices),
        "value":    lambda: factor_value(fundamentals),
        "quality":  lambda: factor_quality(fundamentals),
        "low_vol":  lambda: factor_low_vol(returns),
        "size":     lambda: factor_size(fundamentals),
    }

    active_factors = {k: v for k, v in weights.items() if v > 0}

    raw_scores = {}
    for fname, w in active_factors.items():
        raw = factor_funcs[fname]()
        if raw.empty:
            continue
        z = cross_sectional_zscore(raw)
        if sector_neutral and "sector" in fundamentals.columns:
            sectors = fundamentals["sector"].dropna()
            common = z.index.intersection(sectors.index)
            z = sector_neutralise(z[common], sectors[common])
        raw_scores[fname] = z

    if not raw_scores:
        return pd.DataFrame()

    scores_df = pd.DataFrame(raw_scores)

    # Normalise weights to sum to 1 over available factors
    available = list(scores_df.columns)
    w_vec = np.array([active_factors[f] for f in available], dtype=float)
    w_vec /= w_vec.sum()

    scores_df["composite"] = scores_df[available].fillna(0) @ w_vec

    return scores_df


# ─── Stock selection ──────────────────────────────────────────────────────────

def select_top_stocks(
    scores_df: pd.DataFrame,
    n_stocks: int = 50,
    long_only: bool = True,
) -> pd.Index:
    """
    Return tickers of top-n stocks by composite score.
    """
    ranked = scores_df["composite"].dropna().sort_values(ascending=False)
    return ranked.head(n_stocks).index
