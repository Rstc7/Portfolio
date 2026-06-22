"""
Multi-Factor Portfolio Builder
Bloomberg-inspired dark UI — same aesthetic as Derivatives Lab
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from core.data import (
    get_sp500_tickers, fetch_price_data,
    fetch_fundamentals, compute_returns, compute_rolling_vol
)
from core.factors import compute_composite_score, select_top_stocks
from core.optimizer import min_variance, equal_weight, compute_portfolio_metrics

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Factor Portfolio Builder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Bloomberg dark theme ─────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:       #0a0a0f;
    --surface:  #111118;
    --card:     #16161f;
    --border:   #25253a;
    --accent:   #f5a623;
    --accent2:  #4fc3f7;
    --green:    #00e676;
    --red:      #ff5252;
    --text:     #e8e8f0;
    --muted:    #6b6b8a;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;
  }

  html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }

  /* Cards */
  .bb-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }

  /* Header */
  .bb-header {
    border-bottom: 2px solid var(--accent);
    padding-bottom: 12px;
    margin-bottom: 28px;
  }
  .bb-title {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .bb-subtitle {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.08em;
    margin-top: 4px;
  }

  /* Section labels */
  .bb-label {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  /* KPI tiles */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }
  .kpi-tile {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 4px;
    padding: 14px 18px;
  }
  .kpi-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .kpi-value {
    font-family: var(--mono);
    font-size: 24px;
    font-weight: 600;
    color: var(--text);
  }
  .kpi-value.pos { color: var(--green); }
  .kpi-value.neg { color: var(--red); }

  /* Factor weight sliders label */
  .factor-tag {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid var(--border);
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--accent2);
    padding: 2px 8px;
    margin-bottom: 4px;
  }

  /* Streamlit widget overrides */
  .stSlider > div > div { background: var(--border) !important; }
  .stSelectbox > div, .stMultiselect > div { background: var(--card) !important; }
  label { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 11px !important; }
  .stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 10px 24px !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }
  h1,h2,h3 { color: var(--text) !important; font-family: var(--mono) !important; }
  .stMetric label { color: var(--muted) !important; }
  [data-testid="metric-container"] { background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="bb-header">
  <div class="bb-title">Multi-Factor Portfolio Builder</div>
  <div class="bb-subtitle">S&P 500 · Minimum Variance · Factor Scoring Engine</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="bb-label">Universe & Horizon</div>', unsafe_allow_html=True)

    lookback_years = st.slider("Lookback (years)", 1, 10, 3)
    n_stocks = st.slider("Stocks in portfolio", 10, 100, 30, step=5)
    max_weight = st.slider("Max weight per stock (%)", 2, 20, 10) / 100

    st.markdown("---")
    st.markdown('<div class="bb-label">Factor Weights</div>', unsafe_allow_html=True)
    st.caption("Weights are normalised automatically")

    w_momentum = st.slider("Momentum (12-1)",  0, 10, 2)
    w_value    = st.slider("Value (1/PB)",      0, 10, 2)
    w_quality  = st.slider("Quality (ROE)",     0, 10, 2)
    w_low_vol  = st.slider("Low Volatility",    0, 10, 2)
    w_size     = st.slider("Size (SMB)",        0, 10, 2)

    st.markdown("---")
    st.markdown('<div class="bb-label">Options</div>', unsafe_allow_html=True)
    sector_neutral = st.toggle("Sector neutralisation", value=True)
    cov_method = st.selectbox("Covariance estimator", ["ledoit_wolf", "sample"])

    st.markdown("---")
    run_btn = st.button("▶  BUILD PORTFOLIO", use_container_width=True)

# ─── Main logic ──────────────────────────────────────────────────────────────

if run_btn:

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365 * lookback_years + 30)).strftime("%Y-%m-%d")

    factor_weights = {
        "momentum": w_momentum,
        "value":    w_value,
        "quality":  w_quality,
        "low_vol":  w_low_vol,
        "size":     w_size,
    }

    if sum(factor_weights.values()) == 0:
        st.error("At least one factor weight must be > 0.")
        st.stop()

    # ── 1. Universe ──
    with st.spinner("Loading S&P 500 universe…"):
        tickers = get_sp500_tickers()

    # ── 2. Prices ──
    with st.spinner(f"Fetching price data for {len(tickers)} tickers…"):
        prices = fetch_price_data(tickers, start_date, end_date)
        returns = compute_returns(prices)

    available_tickers = prices.columns.tolist()

    # ── 3. Fundamentals ──
    with st.spinner("Fetching fundamental data (P/B, ROE, market cap…)"):
        # Use a subset to keep it fast — top 200 by data availability
        subset = available_tickers[:200]
        fundamentals = fetch_fundamentals(subset)

    # ── 4. Factor scores ──
    with st.spinner("Computing factor scores…"):
        scores_df = compute_composite_score(
            prices[subset],
            returns[subset],
            fundamentals,
            weights=factor_weights,
            sector_neutral=sector_neutral,
        )

    if scores_df.empty:
        st.error("Factor computation failed — not enough data.")
        st.stop()

    # ── 5. Stock selection ──
    selected = select_top_stocks(scores_df, n_stocks=n_stocks)
    sel_returns = returns[selected].dropna()

    # ── 6. Optimisation ──
    with st.spinner("Optimising portfolio (min variance)…"):
        weights = min_variance(sel_returns, cov_method=cov_method, max_weight=max_weight)
        ew_weights = equal_weight(sel_returns)

    # ── 7. Metrics ──
    metrics    = compute_portfolio_metrics(weights, sel_returns)
    ew_metrics = compute_portfolio_metrics(ew_weights, sel_returns)

    # ═══════════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════════

    # KPI tiles
    ann_ret  = metrics["ann_return"]
    ann_vol  = metrics["ann_vol"]
    sharpe   = metrics["sharpe"]
    max_dd   = metrics["max_drawdown"]

    ret_cls  = "pos" if ann_ret  >= 0 else "neg"
    dd_cls   = "neg"

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-tile">
        <div class="kpi-label">Ann. Return</div>
        <div class="kpi-value {ret_cls}">{ann_ret*100:+.1f}%</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-label">Ann. Volatility</div>
        <div class="kpi-value">{ann_vol*100:.1f}%</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value {'pos' if sharpe>1 else ''}">{sharpe:.2f}</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value {dd_cls}">{max_dd*100:.1f}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Performance", "🧱  Factor Scores", "⚖️  Weights", "📋  Holdings"
    ])

    PLOT_LAYOUT = dict(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#111118",
        font=dict(family="IBM Plex Mono", color="#e8e8f0", size=11),
        xaxis=dict(gridcolor="#25253a", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#25253a", showgrid=True, zeroline=False),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    # ── Tab 1: Performance ────────────────────────────────────
    with tab1:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("Cumulative Return", "Drawdown"),
        )

        fig.add_trace(go.Scatter(
            x=metrics["cum_returns"].index,
            y=metrics["cum_returns"].values,
            name="Min Variance",
            line=dict(color="#f5a623", width=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ew_metrics["cum_returns"].index,
            y=ew_metrics["cum_returns"].values,
            name="Equal Weight",
            line=dict(color="#4fc3f7", width=1.5, dash="dot"),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=metrics["drawdown_series"].index,
            y=metrics["drawdown_series"].values * 100,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#ff5252", width=1),
            fillcolor="rgba(255,82,82,0.15)",
        ), row=2, col=1)

        fig.update_layout(
            **PLOT_LAYOUT,
            height=520,
            showlegend=True,
            legend=dict(bgcolor="#111118", bordercolor="#25253a", borderwidth=1),
        )
        fig.update_annotations(font=dict(family="IBM Plex Mono", color="#6b6b8a", size=10))
        st.plotly_chart(fig, use_container_width=True)

        # vs EW comparison table
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="bb-label">Min Variance vs Equal Weight</div>', unsafe_allow_html=True)
            comp = pd.DataFrame({
                "Min Variance": [
                    f"{metrics['ann_return']*100:+.2f}%",
                    f"{metrics['ann_vol']*100:.2f}%",
                    f"{metrics['sharpe']:.2f}",
                    f"{metrics['max_drawdown']*100:.2f}%",
                ],
                "Equal Weight": [
                    f"{ew_metrics['ann_return']*100:+.2f}%",
                    f"{ew_metrics['ann_vol']*100:.2f}%",
                    f"{ew_metrics['sharpe']:.2f}%",
                    f"{ew_metrics['max_drawdown']*100:.2f}%",
                ],
            }, index=["Ann. Return", "Ann. Vol", "Sharpe", "Max DD"])
            st.dataframe(comp, use_container_width=True)

    # ── Tab 2: Factor scores ──────────────────────────────────
    with tab2:
        factor_cols = [c for c in scores_df.columns if c != "composite"]
        sel_scores  = scores_df.loc[selected, factor_cols + ["composite"]].copy()
        sel_scores  = sel_scores.sort_values("composite", ascending=False)

        st.markdown('<div class="bb-label">Factor Z-Scores — Selected Stocks</div>', unsafe_allow_html=True)

        FACTOR_COLORS = {
            "momentum": "#f5a623",
            "value":    "#4fc3f7",
            "quality":  "#00e676",
            "low_vol":  "#ce93d8",
            "size":     "#ff8a65",
        }

        fig2 = go.Figure()
        for fname in factor_cols:
            if fname in sel_scores.columns:
                fig2.add_trace(go.Bar(
                    x=sel_scores.index,
                    y=sel_scores[fname],
                    name=fname.replace("_", " ").title(),
                    marker_color=FACTOR_COLORS.get(fname, "#888"),
                    opacity=0.8,
                ))

        fig2.update_layout(
            **PLOT_LAYOUT,
            barmode="group",
            height=420,
            xaxis_tickangle=-45,
            showlegend=True,
            legend=dict(bgcolor="#111118", bordercolor="#25253a", borderwidth=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Heatmap
        st.markdown('<div class="bb-label">Factor Heatmap</div>', unsafe_allow_html=True)
        heatmap_data = sel_scores[factor_cols].head(30).T

        fig3 = go.Figure(go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[[0, "#ff5252"], [0.5, "#111118"], [1, "#00e676"]],
            zmid=0,
            colorbar=dict(
                tickfont=dict(family="IBM Plex Mono", color="#e8e8f0", size=10),
                outlinecolor="#25253a",
            ),
        ))
        fig3.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 3: Weights ────────────────────────────────────────
    with tab3:
        w_df = weights.reset_index()
        w_df.columns = ["ticker", "weight"]
        w_df = w_df[w_df["weight"] > 1e-4].sort_values("weight", ascending=False)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<div class="bb-label">Weight Distribution</div>', unsafe_allow_html=True)
            fig4 = go.Figure(go.Bar(
                x=w_df["ticker"],
                y=w_df["weight"] * 100,
                marker=dict(
                    color=w_df["weight"],
                    colorscale=[[0, "#25253a"], [1, "#f5a623"]],
                    showscale=False,
                ),
            ))
            fig4.update_layout(
                **PLOT_LAYOUT,
                height=380,
                yaxis_title="Weight (%)",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            st.markdown('<div class="bb-label">Concentration</div>', unsafe_allow_html=True)
            fig5 = go.Figure(go.Pie(
                labels=w_df["ticker"].head(15).tolist() + (["Other"] if len(w_df) > 15 else []),
                values=(w_df["weight"].head(15).tolist() +
                        ([w_df["weight"].iloc[15:].sum()] if len(w_df) > 15 else [])),
                hole=0.55,
                marker=dict(
                    colors=px.colors.sequential.Oranges[2:] + ["#25253a"],
                    line=dict(color="#0a0a0f", width=1),
                ),
                textfont=dict(family="IBM Plex Mono", size=9, color="#e8e8f0"),
            ))
            fig5.update_layout(**PLOT_LAYOUT, height=380, showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

        # Effective N
        hhi = (weights ** 2).sum()
        eff_n = 1 / hhi if hhi > 0 else 0
        st.markdown(f"""
        <div class="bb-card">
          <div class="bb-label">Diversification</div>
          <span style="font-family:var(--mono);color:var(--accent);font-size:18px;font-weight:600;">
            Effective N = {eff_n:.1f}
          </span>
          <span style="font-family:var(--mono);color:var(--muted);font-size:11px;margin-left:12px;">
            (1/HHI — higher = more diversified)
          </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Holdings ───────────────────────────────────────
    with tab4:
        st.markdown('<div class="bb-label">Portfolio Holdings</div>', unsafe_allow_html=True)

        holdings = pd.DataFrame({
            "Weight (%)": (weights * 100).round(2),
            "Composite Score": scores_df.loc[weights.index, "composite"].round(3),
        }).sort_values("Weight (%)", ascending=False)

        # Add factor scores per stock
        for fc in factor_cols:
            if fc in scores_df.columns:
                holdings[fc.replace("_", " ").title()] = scores_df.loc[weights.index, fc].round(3)

        st.dataframe(
            holdings.style
            .background_gradient(subset=["Weight (%)"], cmap="YlOrBr")
            .background_gradient(subset=["Composite Score"], cmap="RdYlGn")
            .format({"Weight (%)": "{:.2f}%"}),
            use_container_width=True,
            height=500,
        )

        csv = holdings.to_csv()
        st.download_button(
            "⬇  Export holdings CSV",
            data=csv,
            file_name="portfolio_holdings.csv",
            mime="text/csv",
        )

else:
    # Landing state
    st.markdown("""
    <div class="bb-card" style="text-align:center;padding:60px 40px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#6b6b8a;letter-spacing:0.08em;margin-bottom:16px;">
        CONFIGURE PARAMETERS IN SIDEBAR
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#25253a;">
        ─── Select factors · set weights · hit BUILD PORTFOLIO ───
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About this app"):
        st.markdown("""
        **Pipeline**
        1. Universe : S&P 500 via Wikipedia + yfinance
        2. Factors  : Momentum (12-1) · Value (1/PB) · Quality (ROE) · Low Vol · Size
        3. Scoring  : Cross-sectional z-score + optional sector neutralisation
        4. Selection: Top-N stocks by composite score
        5. Optimise : Minimum variance (Ledoit-Wolf covariance)
        6. Metrics  : Sharpe, Max DD, vs Equal Weight benchmark

        **Stack** : Python · Streamlit · yfinance · cvxpy · scikit-learn · Plotly
        """)
