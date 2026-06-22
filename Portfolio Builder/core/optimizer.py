"""
Portfolio optimisation.

Methods:
  - Minimum variance  (default, no expected return estimate needed)
  - Equal weight      (benchmark)

Covariance estimation:
  - Sample covariance
  - Ledoit-Wolf shrinkage (default, more stable in high dimensions)
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf


def estimate_covariance(returns: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray:
    """
    Estimate covariance matrix of returns.

    Parameters
    ----------
    returns : pd.DataFrame  (T x N)
    method  : 'ledoit_wolf' | 'sample'
    """
    R = returns.dropna().values

    if method == "ledoit_wolf":
        lw = LedoitWolf()
        lw.fit(R)
        return lw.covariance_

    return np.cov(R.T)


def min_variance(
    returns: pd.DataFrame,
    cov_method: str = "ledoit_wolf",
    max_weight: float = 0.10,
    min_weight: float = 0.0,
) -> pd.Series:
    """
    Minimum variance portfolio via cvxpy.

    Constraints:
      - weights sum to 1
      - min_weight <= w_i <= max_weight
      - long only (w_i >= 0)

    Returns
    -------
    pd.Series : optimal weights indexed by ticker
    """
    cov = estimate_covariance(returns, method=cov_method)
    n = cov.shape[0]
    tickers = returns.columns

    w = cp.Variable(n)
    portfolio_variance = cp.quad_form(w, cov)

    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        # Fallback to equal weight if solver fails
        return equal_weight(returns)

    weights = pd.Series(w.value, index=tickers)
    weights = weights.clip(lower=0)
    weights /= weights.sum()
    return weights


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    """Naive 1/N benchmark."""
    n = len(returns.columns)
    return pd.Series(1 / n, index=returns.columns)


# ─── Portfolio metrics ────────────────────────────────────────────────────────

def compute_portfolio_metrics(
    weights: pd.Series,
    returns: pd.DataFrame,
    rf: float = 0.05,
) -> dict:
    """
    Compute main portfolio metrics given weights and return history.

    Parameters
    ----------
    weights  : pd.Series (tickers)
    returns  : pd.DataFrame daily log returns (T x N)
    rf       : annual risk-free rate
    """
    common = weights.index.intersection(returns.columns)
    w = weights[common]
    w /= w.sum()
    R = returns[common]

    port_ret = R @ w  # daily portfolio return

    ann_ret  = port_ret.mean() * 252
    ann_vol  = port_ret.std()  * np.sqrt(252)
    sharpe   = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

    cum      = (1 + port_ret).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd   = drawdown.min()

    return {
        "ann_return":  ann_ret,
        "ann_vol":     ann_vol,
        "sharpe":      sharpe,
        "max_drawdown": max_dd,
        "cum_returns": cum,
        "daily_returns": port_ret,
        "drawdown_series": drawdown,
    }
