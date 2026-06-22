"""
Data fetching and universe construction.
S&P 500 tickers via Wikipedia, price/fundamental data via yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import streamlit as st
from datetime import datetime, timedelta


import io

@st.cache_data(ttl=86400)
def get_sp500_tickers() -> list[str]:
    """Scrape S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return sorted(tickers)


@st.cache_data(ttl=3600)
def fetch_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers.
    Returns DataFrame: index=date, columns=tickers.
    Drops tickers with >20% missing data.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    # Drop tickers with too much missing data
    threshold = 0.80
    raw = raw.dropna(axis=1, thresh=int(len(raw) * threshold))
    raw = raw.ffill().bfill()
    return raw


@st.cache_data(ttl=3600)
def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch fundamental data for factor construction.
    Returns DataFrame: index=ticker, columns=[pb_ratio, roe, market_cap, beta]
    Uses yfinance .info — slow for large universes, cached aggressively.
    """
    records = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            records.append({
                "ticker": ticker,
                "pb_ratio":    info.get("priceToBook"),
                "roe":         info.get("returnOnEquity"),
                "market_cap":  info.get("marketCap"),
                "beta":        info.get("beta"),
                "sector":      info.get("sector", "Unknown"),
            })
        except Exception:
            records.append({"ticker": ticker})

    df = pd.DataFrame(records).set_index("ticker")
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_rolling_vol(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    """Annualised rolling volatility (last window days) per ticker."""
    return returns.tail(window).std() * np.sqrt(252)