"""
Derivatives Lab — Streamlit App
Bloomberg-inspired dark UI
Products: Vanilla, Binary, Barrier, Asian, Lookback,
           Standard Autocall, Athena, Phoenix, Reverse Convertible,
           Capital Protected, Bonus Certificate
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

#PAGE CONFIG
st.set_page_config(
    page_title="Derivatives Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

#CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"], .main, section.main > div {
    background-color: #0a0e14 !important;
}
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d !important;
}
h1,h2,h3,h4 { color: #e6edf3 !important; font-family: 'Inter', sans-serif !important; }
p, .stMarkdown p, label, .stRadio label { color: #c9d1d9 !important; }

/* Metric cards */
.mcard {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
    margin-bottom: 4px;
}
.mcard-val {
    font-size: 1.25rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.mcard-lbl {
    font-size: 0.68rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* Product header */
.prod-header {
    border-left: 3px solid #f0883e;
    padding: 6px 0 6px 14px;
    margin-bottom: 20px;
}
.prod-header h2 { margin: 0 0 4px 0 !important; font-size: 1.4rem !important; }
.prod-header p  { margin: 0 !important; color: #8b949e !important; font-size: 0.9rem !important; }

/* Info box */
.ibox {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #c9d1d9;
    line-height: 1.7;
}

/* Sidebar quick-ref */
.qref {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 12px;
    font-size: 0.8rem;
    color: #8b949e;
    line-height: 1.8;
}

/* Tabs */
.stTabs [data-baseweb="tab"] { color: #768390 !important; font-size: 0.88rem; }
.stTabs [aria-selected="true"] {
    color: #f0883e !important;
    border-bottom: 2px solid #f0883e !important;
}
.stTabs [data-baseweb="tab-list"] { background: transparent !important; }

/* Sliders & selects */
div[data-baseweb="slider"] [data-testid="stTickBar"] { color: #768390; }
div[data-baseweb="select"] > div { background: #161b22 !important; border-color: #30363d !important; color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)



# MATH HELPERS

def _d1d2(S, K, T, r, q, sig):
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(np.maximum(S, 1e-9) / np.maximum(K, 1e-9)) +
              (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T) + 1e-12)
    return d1, d1 - sig * np.sqrt(T)

def call_price(S, K, T, r, q, sig):
    d1, d2 = _d1d2(S, K, T, r, q, sig)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def put_price(S, K, T, r, q, sig):
    d1, d2 = _d1d2(S, K, T, r, q, sig)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def delta(S, K, T, r, q, sig, flag):
    d1, _ = _d1d2(S, K, T, r, q, sig)
    return np.exp(-q * T) * norm.cdf(d1) if flag == "c" else -np.exp(-q * T) * norm.cdf(-d1)

def gamma(S, K, T, r, q, sig):
    d1, _ = _d1d2(S, K, T, r, q, sig)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sig * np.sqrt(T) + 1e-12)

def vega(S, K, T, r, q, sig):
    d1, _ = _d1d2(S, K, T, r, q, sig)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100

def theta(S, K, T, r, q, sig, flag):
    d1, d2 = _d1d2(S, K, T, r, q, sig)
    t1 = -S * np.exp(-q * T) * norm.pdf(d1) * sig / (2 * np.sqrt(T) + 1e-12)
    if flag == "c":
        return (t1 - r * K * np.exp(-r * T) * norm.cdf(d2)  + q * S * np.exp(-q * T) * norm.cdf(d1))  / 365
    else:
        return (t1 + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365

def rho(S, K, T, r, q, sig, flag):
    _, d2 = _d1d2(S, K, T, r, q, sig)
    if flag == "c": return  K * T * np.exp(-r * T) * norm.cdf(d2)  / 100
    else:           return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


# Monte Carlo paths
@st.cache_data(show_spinner=False)
def mc_paths(S0, r, q, sig, T, n_steps, n_paths, seed=42):
    np.random.seed(seed)
    dt = T / n_steps
    Z  = np.random.standard_normal((n_paths, n_steps))
    lr = (r - q - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * Z
    log_S = np.log(S0) + np.hstack([np.zeros((n_paths, 1)), np.cumsum(lr, axis=1)])
    return np.exp(log_S)


# PLOTLY HELPERS

BG   = "#0a0e14"
BG2  = "#0d1117"
GRID = "#1c2128"
C    = ["#f0883e", "#58a6ff", "#3fb950", "#f85149", "#bc8cff", "#ffa657", "#79c0ff"]

def _base(**kw):
    lo = dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#c9d1d9", family="JetBrains Mono, monospace", size=11),
        xaxis=dict(gridcolor=GRID, zerolinecolor="#2d333b", tickcolor="#768390"),
        yaxis=dict(gridcolor=GRID, zerolinecolor="#2d333b", tickcolor="#768390"),
        legend=dict(bgcolor="#161b22", bordercolor="#444c56", borderwidth=1, font=dict(size=10)),
        margin=dict(l=52, r=24, t=46, b=44),
        hovermode="x unified",
    )
    lo.update(kw)
    return lo

def mcard(label, value, color=C[0]):
    return f"""<div class="mcard">
      <div class="mcard-val" style="color:{color}">{value}</div>
      <div class="mcard-lbl">{label}</div>
    </div>"""

def mcards_row(items):
    """items = list of (label, value, color)"""
    cols = st.columns(len(items))
    for col, (lbl, val, clr) in zip(cols, items):
        col.markdown(mcard(lbl, val, clr), unsafe_allow_html=True)

def prod_header(icon, title, subtitle):
    st.markdown(f"""<div class="prod-header">
      <h2>{icon} {title}</h2>
      <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def vline(fig, x, color="#768390", dash="dot", label="", row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_vline(x=x, line=dict(color=color, dash=dash, width=1),
                  annotation_text=label, annotation_font=dict(size=10, color=color), **kw)

def hline(fig, y, color="#768390", dash="dot", label=""):
    fig.add_hline(y=y, line=dict(color=color, dash=dash, width=1),
                  annotation_text=label, annotation_font=dict(size=10, color=color))


# 1 — VANILLA CALL / PUT

def render_vanilla(S, sig, r, q, T, pname):
    flag  = "c" if "Call" in pname else "p"
    label = "Call" if flag == "c" else "Put"
    col0  = C[0] if flag == "c" else C[3]

    prod_header(
        "📈" if flag == "c" else "📉", pname,
        f"Right to {'buy' if flag=='c' else 'sell'} the asset at strike K at maturity T — European Black-Scholes"
    )

    pc1, pc2 = st.columns([1, 2])
    with pc1:
        K    = st.slider("Strike K", 50.0, 200.0, S, 1.0, key="v_k")
        side = st.radio("Direction", ["Long", "Short"], horizontal=True, key="v_side")
    mult = 1 if side == "Long" else -1

    P  = call_price(S, K, T, r, q, sig) if flag == "c" else put_price(S, K, T, r, q, sig)
    d1, d2 = _d1d2(S, K, T, r, q, sig)
    D  = delta(S, K, T, r, q, sig, flag)
    G  = gamma(S, K, T, r, q, sig)
    V  = vega(S, K, T, r, q, sig)
    Th = theta(S, K, T, r, q, sig, flag)
    Rh = rho(S, K, T, r, q, sig, flag)
    intrin   = float(np.maximum(S - K, 0) if flag == "c" else np.maximum(K - S, 0))
    time_val = P - intrin

    with pc2:
        mcards_row([
            ("BS Price",        f"{P:.3f}",          C[0]),
            ("Delta Δ",         f"{D:+.4f}",         C[1]),
            ("Gamma Γ",         f"{G:.5f}",          C[2]),
            ("Vega ν / 1%vol",  f"{V:.4f}",          C[4]),
            ("Theta θ / day",   f"{Th:.4f}",         C[3]),
            ("Rho ρ / 1%r",     f"{Rh:.4f}",         C[5]),
            ("Intrinsic Val.",  f"{intrin:.3f}",     C[6]),
            ("Time Val.",       f"{time_val:.3f}",   "#8b949e"),
        ])

    t1, t2, t3 = st.tabs(["📈 Payoff & P&L", "🔢 Greeks vs Moneyness", "📋 Formulas"])

    # Payoff
    with t1:
        Sv = np.linspace(S * 0.4, S * 1.6, 500)
        pf = np.maximum(Sv - K, 0) if flag == "c" else np.maximum(K - Sv, 0)
        pl = mult * (pf - P)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Sv, y=mult * pf, name="Payoff at expiry",
                                  line=dict(color=col0, width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=Sv, y=pl, name="Net P&L (incl. premium)",
                                  line=dict(color=C[1], width=2.5),
                                  fill="tozeroy", fillcolor="rgba(88,166,255,0.06)"))
        vline(fig, K, C[3], label=f"K={K:.0f}")
        vline(fig, S, C[5], label=f"S₀={S:.0f}")
        hline(fig, 0)
        fig.update_layout(title=f"{side} {label} — Payoff & P&L",
                           xaxis_title="Spot at expiry  S_T",
                           yaxis_title="P&L", **_base())
        st.plotly_chart(fig, use_container_width=True)

    # Greeks 
    with t2:
        T_vals   = [0.083, 0.25, 0.5, 1.0, 2.0]
        T_labels = ["1m",  "3m", "6m", "1y", "2y"]
        T_colors = [C[0], C[1], C[2], C[4], C[5]]
        mn = np.linspace(0.5, 1.5, 300)
        Sv2 = mn * K  # K fixed, S varies → moneyness = S/K

        greek_fns = [
            ("Price",           lambda Sv,T_: call_price(Sv,K,T_,r,q,sig) if flag=="c" else put_price(Sv,K,T_,r,q,sig)),
            ("Delta Δ",         lambda Sv,T_: delta(Sv,K,T_,r,q,sig,flag)),
            ("Gamma Γ",         lambda Sv,T_: gamma(Sv,K,T_,r,q,sig)),
            ("Vega ν / 1%vol",  lambda Sv,T_: vega(Sv,K,T_,r,q,sig)),
            ("Theta θ / day",   lambda Sv,T_: theta(Sv,K,T_,r,q,sig,flag)),
            ("Rho ρ / 1%r",     lambda Sv,T_: rho(Sv,K,T_,r,q,sig,flag)),
        ]
        pos = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
        fig_g = make_subplots(rows=2, cols=3,
                               subplot_titles=[g[0] for g in greek_fns],
                               vertical_spacing=0.16, horizontal_spacing=0.08)
        for (gname, gfn), (ro, co) in zip(greek_fns, pos):
            for Tv, Tl, tc in zip(T_vals, T_labels, T_colors):
                y = gfn(Sv2, Tv)
                fig_g.add_trace(go.Scatter(x=mn, y=y, name=Tl,
                                            line=dict(color=tc, width=1.8),
                                            showlegend=(ro == 1 and co == 1),
                                            legendgroup=Tl),
                                row=ro, col=co)
            fig_g.add_vline(x=1.0, line=dict(color="#444c56", dash="dot", width=1), row=ro, col=co)
            fig_g.update_xaxes(gridcolor=GRID, row=ro, col=co)
            fig_g.update_yaxes(gridcolor=GRID, row=ro, col=co)

        fig_g.update_layout(
            height=540, title=f"Greeks — {side} {label}  (K={K:.0f}, σ={sig*100:.0f}%)",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color="#c9d1d9", size=10),
            legend=dict(bgcolor="#161b22", bordercolor="#444c56"),
            margin=dict(l=40, r=20, t=55, b=35),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    # Formulas
    with t3:
        st.markdown(f"""
**Black-Scholes Price (European {'Call' if flag=='c' else 'Put'})**

{"$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$" if flag=="c" else "$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$"}

$$d_1 = \\frac{{\\ln(S/K)+(r-q+\\tfrac{{\\sigma^2}}{{2}})T}}{{\\sigma\\sqrt{{T}}}}, \\quad d_2 = d_1 - \\sigma\\sqrt{{T}}$$

**Present Values** — d₁ = `{float(d1):.4f}`, d₂ = `{float(d2):.4f}`

**Economic decomposition:**
{"- Long forward weighted by N(d₁) → delta exposure \n- Short bond K·e⁻ʳᵀ·N(d₂) → financing" if flag=="c"
  else "- Long bond K·e⁻ʳᵀ·N(−d₂) → capital protection \n- Short forward weighted by N(−d₁) → negative exposure"}
        """)


# 2 — BINARY

def render_binary(S, sig, r, q, T):
    prod_header("⚡", "Binary Option Cash-or-Nothing",
                "Pays R if S_T > K (call) or S_T < K (put), otherwise 0. Payoff discontinuity = delta singularity.")

    c1, c2, c3 = st.columns(3)
    with c1: K    = st.slider("Strike K",  50.0, 200.0, S,    1.0, key="bi_k")
    with c2: R    = st.slider("Payout R",   1.0, 200.0, 100.0, 5.0, key="bi_r")
    with c3: btyp = st.radio("Type", ["Binary Call", "Binary Put"], horizontal=True, key="bi_t")
    is_call = btyp == "Binary Call"

    _, d2v = _d1d2(S, K, T, r, q, sig)
    pbin  = float(R * np.exp(-r * T) * (norm.cdf(d2v) if is_call else norm.cdf(-d2v)))
    dbin  = float(R * np.exp(-r * T) * norm.pdf(d2v) / (S * sig * np.sqrt(T) + 1e-9) * (1 if is_call else -1))
    gbin  = float(-R * np.exp(-r * T) * norm.pdf(d2v) * d2v / (S**2 * sig**2 * T + 1e-9) * (1 if is_call else -1))

    mcards_row([("Price", f"{pbin:.4f}", C[0]), ("Delta Δ", f"{dbin:.5f}", C[1]),
                ("Gamma Γ", f"{gbin:.5f}", C[2]),
                ("% of payout", f"{pbin/R*100:.1f}%", C[5])])

    t1, t2 = st.tabs(["📈 Price & Payoff", "🔢 Greeks"])
    with t1:
        Sv = np.linspace(S * 0.4, S * 1.6, 600)
        _, d2r = _d1d2(Sv, K, T, r, q, sig)
        pr = R * np.exp(-r * T) * (norm.cdf(d2r) if is_call else norm.cdf(-d2r))
        pf = np.where(Sv >= K, R, 0.0) if is_call else np.where(Sv < K, R, 0.0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Sv, y=pf, name="Payoff at expiry",
                                  line=dict(color=C[0], width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=Sv, y=pr, name=f"Price (T={T:.1f}y)",
                                  line=dict(color=C[1], width=2)))
        vline(fig, K, C[3], label=f"K={K:.0f}")
        vline(fig, S, C[5], label=f"S₀={S:.0f}")
        fig.update_layout(title="Binary — Price & Payoff",
                           xaxis_title="Spot", yaxis_title="Value", **_base())
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        Sv2 = np.linspace(S * 0.5, S * 1.5, 500)
        sign = 1 if is_call else -1
        _, d2r2 = _d1d2(Sv2, K, T, r, q, sig)
        dv = R * np.exp(-r*T) * norm.pdf(d2r2) / (Sv2*sig*np.sqrt(T)+1e-9) * sign
        gv = -R * np.exp(-r*T) * norm.pdf(d2r2) * d2r2 / (Sv2**2*sig**2*T+1e-9) * sign

        fig2 = make_subplots(1, 2, subplot_titles=["Delta Δ", "Gamma Γ"])
        for (y, nm, cc), co in zip([(dv,"Delta",C[1]),(gv,"Gamma",C[2])], [1,2]):
            fig2.add_trace(go.Scatter(x=Sv2/K, y=y, name=nm,
                                       line=dict(color=cc, width=2)), row=1, col=co)
            fig2.add_vline(x=1.0, line=dict(color="#444c56", dash="dot", width=1), row=1, col=co)
            fig2.update_xaxes(gridcolor=GRID, title_text="S / K", row=1, col=co)
            fig2.update_yaxes(gridcolor=GRID, row=1, col=co)
        fig2.update_layout(height=360, paper_bgcolor=BG, plot_bgcolor=BG,
                            font=dict(color="#c9d1d9"), margin=dict(l=40,r=20,t=50,b=35),
                            legend=dict(bgcolor="#161b22", bordercolor="#444c56"))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="ibox">⚠️ <b>Singularity at strike:</b> Delta → ∞ as T → 0 and S → K. Impossible to dynamically hedge near expiry. In practice, desks replicate with a tight call-spread.</div>', unsafe_allow_html=True)


# 3 — BARRIER

def render_barrier(S, sig, r, q, T):
    prod_header("🚧", "Barrier Option KO / KI",
                "Activated (Knock-In) or deactivated (Knock-Out) if the underlying breaches H during the option's life.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: K      = st.slider("Strike K",    50.0, 200.0, S,      1.0, key="ba_k")
    with c2: fl     = st.radio("Call/Put",     ["Call","Put"],  horizontal=True, key="ba_f")
    with c3: btyp   = st.radio("KI / KO",      ["Knock-Out","Knock-In"], horizontal=True, key="ba_t")
    with c4: bdir   = st.radio("Direction",    ["Down","Up"],   horizontal=True, key="ba_d")

    H_def = S * (0.83 if bdir=="Down" else 1.17)
    H = st.slider(f"Barrier H ({bdir})", S*0.4, S*1.6, H_def, 1.0, key="ba_H")

    flag = "c" if fl == "Call" else "p"
    code = ("U" if bdir=="Up" else "D") + ("O" if btyp=="Knock-Out" else "I")

    N_P, N_S = 18_000, int(T*252)
    paths = mc_paths(S, r, q, sig, T, N_S, N_P)
    S_T   = paths[:, -1]
    hit   = paths.max(axis=1) >= H if bdir=="Up" else paths.min(axis=1) <= H
    raw   = np.maximum(S_T - K, 0) if flag=="c" else np.maximum(K - S_T, 0)
    pf_mc = np.where(hit, 0, raw) if "O" in code else np.where(hit, raw, 0)
    bp    = float(np.exp(-r*T) * pf_mc.mean())
    van   = float(call_price(S,K,T,r,q,sig) if flag=="c" else put_price(S,K,T,r,q,sig))
    prob_hit = hit.mean()

    mcards_row([
        ("Barrier Price",        f"{bp:.3f}",                   C[0]),
        ("Vanilla Price",        f"{van:.3f}",                  "#8b949e"),
        ("Discount vs Vanilla",  f"{(1-bp/van)*100:.1f}%" if van>0 else "—", C[2]),
        ("P(barrier hit)",       f"{prob_hit*100:.1f}%",        C[3]),
    ])

    t1, t2 = st.tabs(["📊 Path Simulation", "🏗️ Composition"])
    with t1:
        t_ax = np.linspace(0, T, N_S+1)
        fig = go.Figure()
        for i in range(min(100, N_P)):
            col_path = "rgba(248,81,73,0.18)" if (paths[i].max()>=H if bdir=="Up" else paths[i].min()<=H) else "rgba(88,166,255,0.14)"
            fig.add_trace(go.Scatter(x=t_ax, y=paths[i], mode="lines",
                                      line=dict(color=col_path, width=0.8),
                                      showlegend=False, hoverinfo="skip"))
        fig.add_hline(y=H, line=dict(color=C[3], width=2.5, dash="dash"),
                      annotation_text=f"Barrier H={H:.0f}")
        fig.add_hline(y=K, line=dict(color=C[2], width=1.5, dash="dot"),
                      annotation_text=f"Strike K={K:.0f}")
        fig.add_hline(y=S, line=dict(color=C[5], width=1,  dash="dot"),
                      annotation_text=f"S₀={S:.0f}")
        fig.update_layout(title=f"{code} {fl} — MC Simulation (red = barrier hit)",
                           xaxis_title="Time (years)", yaxis_title="Spot", **_base())
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown(f"""
<div class="ibox">
<b>Fundamental relation:</b><br>
<code>KO + KI = Vanilla</code><br><br>
→ A <b>Knock-Out</b> is worth less than a vanilla because it can be deactivated.<br>
→ A <b>Knock-In</b> is worth the difference: it only has value if the barrier is hit.<br><br>

<b>Values:</b> KO = {bp:.3f}  |  KI ≈ {van-bp:.3f}  |  Vanilla = {van:.3f}<br><br>

<b>Vega Behavior:</b><br>
For a KO: an increase in σ brings the barrier closer → can make Vega <i>negative</i> (counter-intuitive!).<br>
For a KI: classic positive Vega — the higher the σ, the more likely to hit H.<br><br>

<b>Gamma:</b> Very high and potentially <i>negative</i> near H for a KO (explosion of rebalancing costs).
</div>
        """, unsafe_allow_html=True)


# 4 — ASIAN

def render_asian(S, sig, r, q, T):
    prod_header("🌏", "Asian Option",
                "Payoff based on the underlying's average over the period. Cheaper than a vanilla (effective vol ≈ σ/√3).")

    c1, c2, c3 = st.columns(3)
    with c1: K    = st.slider("Strike K",   50.0,200.0, S, 1.0, key="as_k")
    with c2: avg  = st.radio("Average",     ["Arithmetic","Geometric"], horizontal=True, key="as_a")
    with c3: fl   = st.radio("Call / Put",  ["Call","Put"], horizontal=True, key="as_f")
    flag = "c" if fl=="Call" else "p"

    N_P, N_S = 25_000, max(int(T*252), 50)
    paths = mc_paths(S, r, q, sig, T, N_S, N_P)
    A = paths.mean(axis=1) if avg=="Arithmetic" else np.exp(np.log(paths+1e-12).mean(axis=1))
    pf = np.maximum(A-K,0) if flag=="c" else np.maximum(K-A,0)
    ap = float(np.exp(-r*T)*pf.mean())
    vp = float(call_price(S,K,T,r,q,sig) if flag=="c" else put_price(S,K,T,r,q,sig))
    sig_eff = sig / np.sqrt(3)

    mcards_row([
        ("Asian Price",    f"{ap:.4f}", C[0]),
        ("Vanilla Price",  f"{vp:.4f}", "#8b949e"),
        ("Discount",       f"{(1-ap/vp)*100:.1f}%" if vp>0 else "—", C[2]),
        ("σ_eff (geom.)",  f"{sig_eff*100:.1f}%", C[4]),
    ])

    t1, t2 = st.tabs(["📊 Average Distribution", "📋 Intuition"])
    with t1:
        fig = make_subplots(1, 2, subplot_titles=["Distribution of Ā", "Asian vs Vanilla Payoff"])
        fig.add_trace(go.Histogram(x=A, nbinsx=80, marker_color="rgba(88,166,255,0.55)",
                                    showlegend=False), row=1, col=1)
        fig.add_vline(x=K, line=dict(color=C[3], dash="dash", width=2),
                      annotation_text=f"K={K:.0f}", row=1, col=1)

        Sv = np.linspace(S*0.4, S*1.6, 300)
        van_pf = np.maximum(Sv-K,0) if flag=="c" else np.maximum(K-Sv,0)
        # Asian payoff approximation: replace S_T by average → smoothed
        sig_a  = sig_eff
        _, d2a = _d1d2(Sv, K, T, r, q, sig_a)
        asian_approx = call_price(Sv,K,T,r,q,sig_a) if flag=="c" else put_price(Sv,K,T,r,q,sig_a)
        fig.add_trace(go.Scatter(x=Sv, y=van_pf,      name="Vanilla payoff",
                                  line=dict(color="#8b949e", dash="dash", width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=Sv, y=asian_approx, name="Asian (approx.)",
                                  line=dict(color=C[0], width=2)), row=1, col=2)
        for co in [1,2]:
            fig.update_xaxes(gridcolor=GRID, row=1, col=co)
            fig.update_yaxes(gridcolor=GRID, row=1, col=co)
        fig.update_layout(height=400, paper_bgcolor=BG, plot_bgcolor=BG,
                           font=dict(color="#c9d1d9"), margin=dict(l=40,r=20,t=50,b=35),
                           legend=dict(bgcolor="#161b22",bordercolor="#444c56"))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown(f"""
<div class="ibox">
<b>Why are Asians cheaper?</b><br>
The average of a log-normal process over [0,T] has an effective volatility of roughly <b>σ/√3 ≈ {sig_eff*100:.1f}%</b> (continuous geometric case).<br>
In arithmetic average, the approximation is similar. The longer T is, the stronger the smoothing effect.<br><br>

<b>Use Cases:</b><br>
• Energy / Commodities contracts (average monthly or quarterly price)<br>
• Protection against price manipulation on a specific spot date<br>
• Corporate FX (hedging average annual flows)<br><br>

<b>Pricing:</b><br>
• Continuous geometric → closed-form formula (Kemna-Vorst 1990)<br>
• Arithmetic → no closed-form formula, MC or Turnbull-Wakeman approximation
</div>
        """, unsafe_allow_html=True)


# 5 — LOOKBACK

def render_lookback(S, sig, r, q, T):
    prod_header("🔭", "Lookback Option (Floating Strike)",
                "Call: S_T − min(S)  |  Put: max(S) − S_T. Always in-the-money — captures the optimal trajectory.")

    is_call = st.radio("Type", ["Lookback Call  (S_T − min S)", "Lookback Put  (max S − S_T)"],
                        horizontal=True, key="lb_t") == "Lookback Call  (S_T − min S)"

    N_P, N_S = 25_000, max(int(T*252), 50)
    paths = mc_paths(S, r, q, sig, T, N_S, N_P)
    S_T   = paths[:, -1]
    pf    = S_T - paths.min(axis=1) if is_call else paths.max(axis=1) - S_T
    lbp   = float(np.exp(-r*T)*pf.mean())
    van   = float(call_price(S,S,T,r,q,sig) if is_call else put_price(S,S,T,r,q,sig))

    mcards_row([
        ("Lookback Price",      f"{lbp:.3f}", C[0]),
        ("ATM Vanilla",         f"{van:.3f}", "#8b949e"),
        ("Premium vs Vanilla",  f"{(lbp/van-1)*100:.1f}%" if van>0 else "—", C[2]),
        ("E[payoff]",           f"{pf.mean():.3f}", C[5]),
    ])

    n_show = 8
    t_ax   = np.linspace(0, T*252, N_S+1)
    fig    = go.Figure()
    for i in range(n_show):
        path = paths[i]
        ext  = path.argmin() if is_call else path.argmax()
        fig.add_trace(go.Scatter(x=t_ax, y=path, mode="lines",
                                  line=dict(color=C[1], width=1.4, ),
                                  opacity=0.65, showlegend=False))
        fig.add_trace(go.Scatter(x=[t_ax[ext]], y=[path[ext]],
                                  mode="markers",
                                  marker=dict(color=C[3], size=10, symbol="x-thin-open", line=dict(width=2)),
                                  showlegend=(i==0),
                                  name=f"Floating strike ({'min' if is_call else 'max'})"))
    fig.add_hline(y=S, line=dict(color=C[5], dash="dot", width=1),
                  annotation_text=f"S₀={S:.0f}")
    fig.update_layout(title=f"Lookback {'Call' if is_call else 'Put'} — 8 simulated paths",
                       xaxis_title="Days", yaxis_title="Price", **_base())
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="ibox">💡 The lookback is <b>always ITM</b> at expiry (payoff ≥ 0). Its price is typically <b>2–3× that of an ATM vanilla</b> because it buys retrospectively at the most favorable price of the entire trajectory. Very sensitive to the discrete monitoring step (daily vs continuous).</div>', unsafe_allow_html=True)


# 6 — AUTOCALL

@st.cache_data(show_spinner=False)
def _mc_autocall(S0, r, q, sig, obs_tuple, ac_bar, cap_bar, cpn, N=30_000):
    obs = list(obs_tuple)
    T_tot  = obs[-1]
    n_steps = max(int(T_tot*252), 252)
    dt = T_tot / n_steps
    idx_obs = [max(0, min(int(t*n_steps/T_tot), n_steps)) for t in obs]

    np.random.seed(42)
    Z  = np.random.standard_normal((N, n_steps))
    lr = (r-q-0.5*sig**2)*dt + sig*np.sqrt(dt)*Z
    paths = np.exp(np.log(S0) + np.hstack([np.zeros((N,1)), np.cumsum(lr,axis=1)]))

    pf   = np.zeros(N)
    done = np.zeros(N, dtype=bool)
    rtime= np.full(N, -1)

    for i,(t,ix) in enumerate(zip(obs, idx_obs)):
        fired = (~done) & (paths[:,ix] >= ac_bar*S0)
        pf[fired] = np.exp(-r*t) * (1 + (i+1)*cpn)
        rtime[fired] = i
        done |= fired

    S_f  = paths[:, idx_obs[-1]]
    loss = (~done) & (S_f < cap_bar*S0)
    ok   = (~done) & (~loss)
    pf[ok]   = np.exp(-r*T_tot)*1.0
    pf[loss] = np.exp(-r*T_tot)*(S_f[loss]/S0)

    red_p = [(rtime==i).mean() for i in range(len(obs))]
    return pf.mean(), red_p, loss.mean(), ok.mean(), pf

def render_autocall(S, sig, r, q, T):
    prod_header("🔄", "Standard Autocall",
                "Automatic early redemption + coupon if underlying ≥ autocall barrier at each observation date.")

    c1,c2,c3,c4 = st.columns(4)
    with c1: ac_bar = st.slider("Autocall Barrier (%)", 80,120,100,5, key="ac_a")/100
    with c2: cap_bar= st.slider("Capital Barrier (%)",  40,90,  60, 5, key="ac_c")/100
    with c3: cpn    = st.slider("Coupon / period (%)",   2,20,   8, 1, key="ac_k")/100
    with c4: freq   = st.radio("Frequency", ["Annual","Semi-Annual","Quarterly"], key="ac_f")

    per = {"Annual":1,"Semi-Annual":2,"Quarterly":4}[freq]
    obs = tuple(round((i+1)/per,4) for i in range(int(T*per)))

    with st.spinner("Monte Carlo…"):
        price, rp, p_loss, p_ok, pf = _mc_autocall(S, r, q, sig, obs, ac_bar, cap_bar, cpn)

    p_ac = sum(rp)
    mcards_row([
        ("Present Value",      f"{price:.3f}",                               C[0]),
        ("Expected Return",    f"{(price-1)*100:+.1f}%",                    C[2] if price>=1 else C[3]),
        ("P(Autocall)",        f"{p_ac*100:.1f}%",                           C[1]),
        ("P(Capital Loss)",    f"{p_loss*100:.1f}%",                         C[3]),
    ])

    t1,t2,t3 = st.tabs(["📊 Probabilities per Date","📈 Discounted Payoffs Distribution","🏗️ Composition"])

    with t1:
        labs = [f"T={t:.2f}y" for t in obs]
        cum  = np.cumsum(rp)
        fig = make_subplots(1,2, subplot_titles=["P(autocall) per date","Final Scenarios"],
                    specs=[[{"type": "xy"}, {"type": "domain"}]])
        fig.add_trace(go.Bar(x=labs, y=[p*100 for p in rp],
                              name="P(autocall Tᵢ)", marker_color=C[1],
                              text=[f"{p*100:.1f}%" for p in rp], textposition="outside"), row=1,col=1)
        fig.add_trace(go.Scatter(x=labs, y=cum*100, name="Cumulative P(autocall)",
                                  line=dict(color=C[0],width=2), mode="lines+markers"), row=1,col=1)
        fig.add_trace(go.Pie(labels=["Autocall","Capital OK","Capital Loss"],
                              values=[p_ac*100, p_ok*100, p_loss*100],
                              marker_colors=[C[2],C[1],C[3]], hole=0.45,
                              textfont=dict(color="white", size=11)), row=1,col=2)
        for co in [1]:
            fig.update_xaxes(gridcolor=GRID, row=1,col=co)
            fig.update_yaxes(gridcolor=GRID, row=1,col=co)
        fig.update_layout(height=420, paper_bgcolor=BG, plot_bgcolor=BG,
                           font=dict(color="#c9d1d9"), margin=dict(l=40,r=20,t=50,b=35),
                           legend=dict(bgcolor="#161b22",bordercolor="#444c56"))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig2=go.Figure()
        fig2.add_trace(go.Histogram(x=pf*100, nbinsx=100,
                                     marker_color="rgba(88,166,255,0.55)",
                                     marker_line=dict(color=C[1],width=0.4)))
        fig2.add_vline(x=100, line=dict(color=C[5],dash="dash",width=2), annotation_text="Nominal")
        fig2.add_vline(x=price*100, line=dict(color=C[0],dash="dash",width=2),
                       annotation_text=f"E={price*100:.1f}")
        fig2.update_layout(title="Discounted Payoffs Distribution",
                            xaxis_title="Payoff (base 100)", yaxis_title="Frequency", **_base())
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        zcb = 100*np.exp(-r*obs[-1])
        dig = sum((i+1)*cpn*100*p for i,p in enumerate(rp))
        dip = -(price*100 - zcb - dig + p_loss*100*0.5)
        st.markdown(f"""
<div class="ibox">
<b>Replicating Portfolio (approx. risk-neutral)</b><br><br>

| Component | Description | Approx. Value |
|------------|-------------|----------------|
| 🏦 ZCB | Nominal redemption at T={obs[-1]:.1f}y | {zcb:.1f}% |
| 📈 Digital Strip | Coupons if autocall at each Tᵢ | {max(0,dig):.1f}% |
| 📉 Down-and-In Put | Loss if S < {cap_bar*100:.0f}% at maturity | ≈{min(0,dip):.1f}% |
| **Total** | Issued at par | **≈ 100%** |

<br><b>Key Scenarios:</b><br>
📈 Upward → Autocall at T₁ : +{(1+cpn)*100:.0f}%<br>
➡️ Flat → Maturity: 100%<br>
📉 Moderate Drop (>{cap_bar*100:.0f}%) → 100%<br>
💥 Crash (<{cap_bar*100:.0f}%) → Proportional Loss
</div>
        """, unsafe_allow_html=True)


# 7 — ATHENA

@st.cache_data(show_spinner=False)
def _mc_athena(S0, r, q, sig, obs_tuple, ac_bar, cap_bar, cpn, N=30_000):
    obs = list(obs_tuple)
    T_tot  = obs[-1]
    n_steps = max(int(T_tot*252), 252)
    dt = T_tot / n_steps
    idx_obs = [max(0, min(int(t*n_steps/T_tot), n_steps)) for t in obs]

    np.random.seed(42)
    Z  = np.random.standard_normal((N, n_steps))
    lr = (r-q-0.5*sig**2)*dt + sig*np.sqrt(dt)*Z
    paths = np.exp(np.log(S0) + np.hstack([np.zeros((N,1)), np.cumsum(lr,axis=1)]))

    pf      = np.zeros(N)
    done    = np.zeros(N, dtype=bool)
    mem     = np.zeros(N)          # coupons mémorisés (en nombre de périodes)
    rtime   = np.full(N, -1)

    for i,(t,ix) in enumerate(zip(obs, idx_obs)):
        mem[~done] += 1            # chaque période non remboursée accumule
        fired = (~done) & (paths[:,ix] >= ac_bar*S0)
        pf[fired] = np.exp(-r*t) * (1 + mem[fired]*cpn)
        mem[fired] = 0
        rtime[fired] = i
        done |= fired

    S_f  = paths[:, idx_obs[-1]]
    loss = (~done) & (S_f < cap_bar*S0)
    ok   = (~done) & (~loss)
    pf[ok]   = np.exp(-r*T_tot)*(1 + mem[ok]*cpn)
    pf[loss] = np.exp(-r*T_tot)*(S_f[loss]/S0)

    red_p = [(rtime==i).mean() for i in range(len(obs))]
    return pf.mean(), red_p, loss.mean(), ok.mean(), pf

def render_athena(S, sig, r, q, T):
    prod_header("🏛️", "Athena (Memory Autocall)",
                "Autocall variant: unpaid coupons accumulate and are paid at the next autocall event or at maturity.")

    c1,c2,c3,c4 = st.columns(4)
    with c1: ac_bar  = st.slider("Autocall Barrier (%)", 80,120,100,5, key="at_a")/100
    with c2: cap_bar = st.slider("Capital Barrier (%)",  40,90,  60, 5, key="at_c")/100
    with c3: cpn     = st.slider("Coupon / period (%)",   2,15,   6, 1, key="at_k")/100
    with c4: freq    = st.radio("Frequency", ["Annual","Semi-Annual"], key="at_f")

    per  = {"Annual":1,"Semi-Annual":2}[freq]
    obs  = tuple(round((i+1)/per,4) for i in range(int(T*per)))

    with st.spinner("Monte Carlo Athena…"):
        price, rp, p_loss, p_ok, pf = _mc_athena(S, r, q, sig, obs, ac_bar, cap_bar, cpn)

    max_mem_cpn = len(obs)*cpn*100
    mcards_row([
        ("Present Value",      f"{price:.3f}",              C[0]),
        ("Expected Return",    f"{(price-1)*100:+.1f}%",   C[2] if price>=1 else C[3]),
        ("P(Autocall)",        f"{sum(rp)*100:.1f}%",       C[1]),
        ("P(Capital Loss)",    f"{p_loss*100:.1f}%",        C[3]),
    ])

    t1,t2,t3 = st.tabs(["📊 Memory Coupon per Date","📈 Distribution","🏗️ Composition"])

    with t1:
        labs = [f"T={t:.2f}y" for t in obs]
        mem_vals = [(i+1)*cpn*100 for i in range(len(obs))]
        fig = make_subplots(1,2, subplot_titles=["P(Autocall) + Memory Coupon","Final Scenarios"],
                             specs=[[{"secondary_y":True},{"type":"pie"}]])
        fig.add_trace(go.Bar(x=labs, y=[p*100 for p in rp], name="P(autocall)",
                              marker_color=C[1]), row=1, col=1)
        fig.add_trace(go.Scatter(x=labs, y=mem_vals, name="Memory coupon (%)",
                                  line=dict(color=C[0], width=2),
                                  mode="lines+markers+text",
                                  text=[f"{v:.0f}%" for v in mem_vals],
                                  textposition="top center"),
                      secondary_y=True, row=1, col=1)
        fig.add_trace(go.Pie(labels=["Autocall","Capital OK","Capital Loss"],
                              values=[sum(rp)*100, p_ok*100, p_loss*100],
                              marker_colors=[C[2],C[1],C[3]], hole=0.45,
                              textfont=dict(color="white")), row=1, col=2)
        fig.update_xaxes(gridcolor=GRID, row=1, col=1)
        fig.update_yaxes(gridcolor=GRID, row=1, col=1)
        fig.update_yaxes(showgrid=False, secondary_y=True, row=1, col=1)
        fig.update_layout(height=420, paper_bgcolor=BG, plot_bgcolor=BG,
                           font=dict(color="#c9d1d9"), margin=dict(l=40,r=20,t=50,b=35),
                           legend=dict(bgcolor="#161b22",bordercolor="#444c56"))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig2=go.Figure()
        fig2.add_trace(go.Histogram(x=pf*100, nbinsx=100,
                                     marker_color="rgba(63,185,80,0.55)",
                                     marker_line=dict(color=C[2],width=0.4)))
        fig2.add_vline(x=100, line=dict(color=C[5],dash="dash",width=2))
        fig2.add_vline(x=price*100, line=dict(color=C[2],dash="dash",width=2),
                       annotation_text=f"E={price*100:.1f}")
        fig2.update_layout(title="Payoff Distribution — Athena",
                            xaxis_title="Payoff (%)", **_base())
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        st.markdown(f"""
<div class="ibox">
<b>Athena vs Standard Autocall</b><br><br>

| Feature | Autocall | Athena |
|---------|----------|--------|
| Coupon on autocall | Period coupons | <b>All accumulated coupons</b> |
| Coupon at maturity without loss | ❌ | ✅ (all remaining coupons) |
| Capital barrier | ✅ | ✅ |

<br><b>The memory effect</b> is a strip of <b>conditional digital puts</b> on past dates: each missed coupon becomes an option on the next observation.<br><br>

<b>Maximum memory coupon</b>: {max_mem_cpn:.0f}%  (if autocall only at the last date)<br><br>

<b>Pricing implication:</b> Athena is <i>more expensive</i> to issue (memory premium) or, for a constant premium, offers a lower unitary coupon. The issuer hedges the memory with a strip of swaptions or OTC digital puts.
</div>
        """, unsafe_allow_html=True)


# 8 — PHOENIX

@st.cache_data(show_spinner=False)
def _mc_phoenix(S0, r, q, sig, obs_tuple, ac_bar, cpn_bar, cap_bar, cpn, memory, N=30_000):
    obs = list(obs_tuple)
    T_tot   = obs[-1]
    n_steps = max(int(T_tot*252), 252)
    dt = T_tot / n_steps
    idx_obs = [max(0, min(int(t*n_steps/T_tot), n_steps)) for t in obs]

    np.random.seed(42)
    Z  = np.random.standard_normal((N, n_steps))
    lr = (r-q-0.5*sig**2)*dt + sig*np.sqrt(dt)*Z
    paths = np.exp(np.log(S0) + np.hstack([np.zeros((N,1)), np.cumsum(lr,axis=1)]))

    pf        = np.zeros(N)
    done      = np.zeros(N, dtype=bool)
    mem       = np.zeros(N)
    cpn_paid  = np.zeros(N)
    rtime     = np.full(N, -1)

    for i,(t,ix) in enumerate(zip(obs, idx_obs)):
        S_obs = paths[:,ix]
        eligible = (~done) & (S_obs >= cpn_bar*S0)
        if memory:
            mem[~done] += 1
            payout = mem[eligible]*cpn
            pf[eligible] += np.exp(-r*t)*payout
            cpn_paid[eligible] += payout
            mem[eligible] = 0
        else:
            pf[eligible] += np.exp(-r*t)*cpn
            cpn_paid[eligible] += cpn

        fired = (~done) & (S_obs >= ac_bar*S0)
        pf[fired] += np.exp(-r*t)*1.0
        rtime[fired] = i
        done |= fired

    S_f  = paths[:, idx_obs[-1]]
    loss = (~done) & (S_f < cap_bar*S0)
    ok   = (~done) & (~loss)
    pf[ok]   += np.exp(-r*T_tot)*1.0
    pf[loss] += np.exp(-r*T_tot)*(S_f[loss]/S0)

    red_p    = [(rtime==i).mean() for i in range(len(obs))]
    return pf.mean(), red_p, loss.mean(), ok.mean(), pf, float(cpn_paid.mean()*100)

def render_phoenix(S, sig, r, q, T):
    prod_header("🐦", "Phoenix (Conditional Coupon + Autocall)",
                "Coupon paid if S > coupon barrier (< autocall barrier) at each date. Optional memory effect. Conditional capital at maturity.")

    c1,c2,c3 = st.columns(3)
    with c1:
        ac_bar  = st.slider("Autocall Barrier (%)", 90,120,100,5, key="px_a")/100
        cap_bar = st.slider("Capital Barrier (%)",  40,80,  60, 5, key="px_c")/100
    with c2:
        cpn_bar = st.slider("Coupon Barrier (%)",   50,100, 70, 5, key="px_b")/100
        cpn     = st.slider("Coupon / period (%)",   2,20,  10, 1, key="px_k")/100
    with c3:
        mem  = st.toggle("Memory Effect", value=True, key="px_m")
        freq = st.radio("Frequency", ["Quarterly","Semi-Annual","Annual"], key="px_f")

    per  = {"Quarterly":4,"Semi-Annual":2,"Annual":1}[freq]
    obs  = tuple(round((i+1)/per,4) for i in range(int(T*per)))

    with st.spinner("Monte Carlo Phoenix…"):
        price, rp, p_loss, p_ok, pf, avg_cpn = _mc_phoenix(
            S, r, q, sig, obs, ac_bar, cpn_bar, cap_bar, cpn, mem)

    mcards_row([
        ("Present Value",      f"{price:.3f}",         C[0]),
        ("Average coupon paid", f"{avg_cpn:.1f}%",       C[2]),
        ("P(Autocall)",        f"{sum(rp)*100:.1f}%",  C[1]),
        ("P(Capital Loss)",    f"{p_loss*100:.1f}%",   C[3]),
    ])

    t1,t2,t3 = st.tabs(["🗺️ Performance Zones","📈 Distribution","🏗️ Composition"])

    with t1:
        # Zone diagram (Spot performance over time)
        fig = make_subplots(1,2, subplot_titles=["Performance Zones","P(Autocall) per Date"])
        for (y0,y1,col_z,nm) in [
            (ac_bar*100,150,      "rgba(63,185,80,0.15)",  f"Autocall (>{ac_bar*100:.0f}%)"),
            (cpn_bar*100,ac_bar*100,"rgba(88,166,255,0.15)",f"Coupon ({cpn_bar*100:.0f}%–{ac_bar*100:.0f}%)"),
            (cap_bar*100,cpn_bar*100,"rgba(255,166,87,0.15)",f"Protection ({cap_bar*100:.0f}%–{cpn_bar*100:.0f}%)"),
            (0,cap_bar*100,       "rgba(248,81,73,0.15)",  f"Loss (<{cap_bar*100:.0f}%)"),
        ]:
            fig.add_trace(go.Scatter(
                x=[0,T,T,0,0], y=[y1,y1,y0,y0,y1],
                fill="toself", fillcolor=col_z,
                line=dict(color="rgba(0,0,0,0)"), name=nm), row=1,col=1)

        labs = [f"T={t:.2f}y" for t in obs]
        fig.add_trace(go.Bar(x=labs, y=[p*100 for p in rp],
                              name="P(autocall Tᵢ)", marker_color=C[4],
                              showlegend=False), row=1,col=2)
        for co in [1,2]:
            fig.update_xaxes(gridcolor=GRID, row=1,col=co)
            fig.update_yaxes(gridcolor=GRID, row=1,col=co)
        fig.update_layout(height=420, paper_bgcolor=BG, plot_bgcolor=BG,
                           font=dict(color="#c9d1d9"), margin=dict(l=40,r=20,t=50,b=35),
                           legend=dict(bgcolor="#161b22",bordercolor="#444c56",font=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig2=go.Figure()
        fig2.add_trace(go.Histogram(x=pf*100, nbinsx=100,
                                     marker_color="rgba(188,140,255,0.55)",
                                     marker_line=dict(color=C[4],width=0.4)))
        fig2.add_vline(x=100, line=dict(color=C[5],dash="dash",width=2))
        fig2.add_vline(x=price*100, line=dict(color=C[4],dash="dash",width=2),
                       annotation_text=f"E={price*100:.1f}")
        fig2.update_layout(title="Payoff Distribution — Phoenix",
                            xaxis_title="Payoff (%)", **_base())
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        st.markdown(f"""
<div class="ibox">
<b>4 replicating building blocks of Phoenix</b><br><br>

| Component | Description |
|------------|-------------|
| 🏦 ZCB | Nominal redemption (financing partial protection) |
| 📈 Coupon digital strip | Conditional payment if S > {cpn_bar*100:.0f}% at each Tᵢ {'+ memory' if mem else ''} |
| 🔄 Autocall digitals | Early redemption if S > {ac_bar*100:.0f}% |
| 📉 DIP | Downside exposure if S < {cap_bar*100:.0f}% at maturity |

<br><b>Key Difference Autocall / Athena / Phoenix:</b><br>
• <b>Autocall</b>: coupon barrier = autocall barrier = 100% — coupon only if redeemed<br>
• <b>Athena</b>  : same as autocall but memory on coupons<br>
• <b>Phoenix</b> : coupon barrier <b>decoupled</b> from autocall barrier (here {cpn_bar*100:.0f}% vs {ac_bar*100:.0f}%) — coupon paid even if not autocalled {'with memory' if mem else 'without memory'}<br><br>

Average coupon observed in simulation: <b>{avg_cpn:.1f}%</b>
</div>
        """, unsafe_allow_html=True)


# 9 — REVERSE CONVERTIBLE

def render_reverse_convertible(S, sig, r, q, T):
    prod_header("↩️", "Reverse Convertible",
                "High coupon bond with risk of physical delivery of shares if S_T < K. Composition: ZCB + short put.")

    c1,c2,c3 = st.columns(3)
    with c1: K_pct    = st.slider("Strike K (% of spot)", 70,110,100,5, key="rc_k")
    with c2: nominal  = st.slider("Notional (€)",  1000,100000,10000,1000, key="rc_n")
    with c3: has_bar  = st.toggle("Add DIP barrier", value=False, key="rc_b")

    K = K_pct/100*S
    H = st.slider("Barrier H (%K)", 50,99,70,1, key="rc_H")/100*K if has_bar else 0.0

    pp = float(put_price(S, K, T, r, q, sig))
    if has_bar and H > 0:
        N_P,N_S = 20_000,int(T*252)
        paths = mc_paths(S,r,q,sig,T,N_S,N_P)
        hit   = paths.min(axis=1) <= H
        raw   = np.maximum(K - paths[:,-1], 0)
        pp    = float(np.exp(-r*T)*np.where(hit,raw,0).mean())  # DIP put

    zcb  = np.exp(-r*T)
    cpn_ann = pp / (zcb*T)*100
    cpn_tot = pp / zcb*100
    be       = K - pp

    mcards_row([
        ("Put Premium received", f"{pp:.3f}",      C[0]),
        ("Annual coupon",        f"{cpn_ann:.2f}%", C[2]),
        ("Total coupon",         f"{cpn_tot:.2f}%", C[1]),
        ("Break-even",           f"{be:.2f}",      C[5]),
    ])

    t1,t2 = st.tabs(["📈 Payoff", "🏗️ Composition"])
    with t1:
        Sv  = np.linspace(S*0.3, S*1.5, 500)
        ca  = pp/zcb   
        pf_rc = np.where(Sv >= K, 1 + ca/S, Sv/K + ca/S)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Sv/S*100, y=pf_rc*nominal/nominal*100,
                                  name="Reverse Convertible", line=dict(color=C[0],width=2.5),
                                  fill="tozeroy", fillcolor="rgba(240,136,62,0.07)"))
        fig.add_hline(y=100, line=dict(color="#8b949e",dash="dash",width=1.5),
                      annotation_text="Initial capital")
        vline(fig, K/S*100, C[3], label=f"K={K:.0f}")
        vline(fig, 100,     C[5], label=f"S₀={S:.0f}")
        fig.update_layout(title="Payoff Reverse Convertible (base 100%)",
                           xaxis_title="Underlying performance (%)",
                           yaxis_title="Redeemed value (% of notional)", **_base())
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig2=go.Figure(go.Waterfall(
            orientation="v", measure=["absolute","relative","relative","total"],
            x=["Invested Capital","ZCB (−discount)","Short Put (+premium)","Product Price"],
            y=[100, -(100-zcb*100), pp/S*S, 0],
            connector=dict(line=dict(color="#30363d")),
            decreasing=dict(marker=dict(color=C[3])),
            increasing=dict(marker=dict(color=C[2])),
            totals=dict(marker=dict(color=C[0])),
            text=[f"{v:.1f}%" for v in [100,-(100-zcb*100),pp/S*S,100]],
            textposition="outside",
        ))
        fig2.update_layout(title="Decomposition Reverse Convertible (base 100)",
                            yaxis_title="Value (%)", **_base())
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"""
<div class="ibox">
<b>Mechanics:</b> The issuer <i>sells</i> this product at 100 → uses {zcb*100:.1f}% to buy a ZCB + receives {pp/S*100:.2f}% by selling a put → apparent coupon of <b>{cpn_ann:.2f}%/year</b>.<br><br>
<b>Real risk:</b> If S_T < K, the investor receives <b>shares</b> instead of cash — they implicitly <i>sold</i> a put on the underlying.<br>
Break-even: {be:.2f} (−{(S-be)/S*100:.1f}% from S₀)
</div>
        """, unsafe_allow_html=True)


# 10 — CAPITAL PROTECTED

def render_capital_protected(S, sig, r, q, T):
    prod_header("🛡️", "Capital Protected Note",
                "Capital 100% guaranteed at maturity + partial upside participation. Composition: ZCB + ATM call.")

    c1,c2,c3 = st.columns(3)
    with c1: prot  = st.slider("Capital protection (%)",  80,100,100, 5, key="cg_p")/100
    with c2: cap   = st.slider("Max gain cap (%, 0=∞)",    0,100,  0, 5, key="cg_c")/100
    with c3: lever = st.slider("Call leverage (% budget)",  50,200,100, 5, key="cg_l")/100

    zcb_cost  = prot * np.exp(-r*T)
    cp        = float(call_price(S, S, T, r, q, sig))
    budget_op = 1 - zcb_cost
    partic    = budget_op / cp * lever if cp > 0 else 0

    mcards_row([
        ("ZCB",            f"{zcb_cost*100:.1f}%", C[2]),
        ("Options Budget", f"{budget_op*100:.1f}%",C[1]),
        ("ATM Call cost",  f"{cp/S*100:.2f}%",     "#8b949e"),
        ("Participation",  f"{partic*100:.1f}%",   C[0]),
    ])

    t1,t2 = st.tabs(["📈 Payoff","🏗️ Composition"])
    with t1:
        Sv    = np.linspace(S*0.3, S*2.0, 500)
        up    = np.maximum(Sv-S, 0)*partic
        if cap > 0: up = np.minimum(up, cap*S)
        pf_cg = prot + up/S

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Sv/S*100, y=pf_cg*100, name="Capital Protected",
                                  line=dict(color=C[0],width=2.5),
                                  fill="tozeroy", fillcolor="rgba(240,136,62,0.07)"))
        fig.add_hline(y=100, line=dict(color="#8b949e",dash="dash",width=1.5),
                      annotation_text="Nominal")
        vline(fig, 100, C[5], label="ATM")
        if cap > 0:
            cap_spot = (1 + cap/partic)*100 if partic > 0 else 200
            vline(fig, cap_spot, C[3], label=f"Cap={cap*100:.0f}% gain")
        fig.update_layout(title="Payoff Capital Protected Note (base 100%)",
                           xaxis_title="Underlying performance (%)",
                           yaxis_title="Redemption (%)", **_base())
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig2=go.Figure()
        fig2.add_trace(go.Bar(
            x=["ZCB (protection)","ATM Call (participation)"],
            y=[zcb_cost*100, budget_op*100],
            marker_color=[C[2],C[1]],
            text=[f"{v:.1f}%" for v in [zcb_cost*100,budget_op*100]],
            textposition="outside", width=0.4,
        ))
        fig2.add_hline(y=100, line=dict(color=C[0],dash="dash",width=2),
                       annotation_text="Capital = 100%")
        fig2.update_layout(title="Decomposition Capital Protected Note",
                            yaxis_title="Value (%)", yaxis_range=[0,115], **_base())
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"""
<div class="ibox">
<b>Fundamental trade-off:</b><br>
• Rates ↑ → cheaper ZCB → more options budget → participation ↑<br>
• Volatility ↑ → more expensive call → participation ↓ (at constant budget)<br>
• Longer maturity → cheaper ZCB → participation ↑<br><br>
<b>Currently:</b> ZCB = {zcb_cost*100:.1f}%, budget = {budget_op*100:.1f}%, participation = <b>{partic*100:.1f}%</b><br>
In a low-rate market (r ≈ 0%), the ZCB costs almost 100% → near-zero participation — this is why these products disappeared after 2008.
</div>
        """, unsafe_allow_html=True)


# 11 — BONUS CERTIFICATE

def render_bonus_cert(S, sig, r, q, T):
    prod_header("🎯", "Bonus Certificate",
                "100% upside participation + guaranteed bonus level if the lower barrier is never breached. Composition: Forward + Down-and-In Put.")

    c1,c2 = st.columns(2)
    with c1: H = st.slider("Lower Barrier H (% spot)", 50,95,75,5, key="bc_h")/100*S
    with c2: B = st.slider("Bonus Level B (% spot)",  100,160,120,5, key="bc_b")/100*S

    N_P,N_S = 25_000, int(T*252)
    paths   = mc_paths(S, r, q, sig, T, N_S, N_P)
    S_T     = paths[:,-1]
    hit     = paths.min(axis=1) <= H
    pf      = np.where(hit, S_T, np.maximum(S_T, B))
    cert_p  = float(np.exp(-r*T)*pf.mean())
    p_hit   = hit.mean()
    p_bonus = ((~hit)&(S_T<=B)).mean()

    mcards_row([
        ("Certificate Price",    f"{cert_p:.2f}",          C[0]),
        ("Premium vs Spot",      f"{(cert_p/S-1)*100:+.1f}%", C[5]),
        ("P(barrier hit)",       f"{p_hit*100:.1f}%",      C[3]),
        ("P(bonus activated)",   f"{p_bonus*100:.1f}%",    C[2]),
    ])

    t1,t2 = st.tabs(["📈 Payoff & Simulation","🏗️ Composition"])
    with t1:
        Sv  = np.linspace(S*0.2, S*2.0, 600)
        pf_ok  = np.maximum(Sv, B)
        pf_hit = Sv
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Sv, y=pf_ok,  name=f"Barrier NOT hit (bonus {B:.0f})",
                                  line=dict(color=C[2],width=2.5)))
        fig.add_trace(go.Scatter(x=Sv, y=pf_hit, name="Barrier hit (tracker)",
                                  line=dict(color=C[3],width=2.5,dash="dash")))
        fig.add_trace(go.Scatter(x=Sv, y=Sv,     name="Pure tracker benchmark",
                                  line=dict(color="#8b949e",width=1,dash="dot")))
        vline(fig, H, C[3], label=f"H={H:.0f}")
        vline(fig, B, C[2], label=f"Bonus={B:.0f}")
        vline(fig, S, C[5], label=f"S₀={S:.0f}")
        fig.update_layout(title="Bonus Certificate Payoff at Maturity",
                           xaxis_title="Final Price", yaxis_title="Value", **_base())
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fwd   = S * np.exp(-q*T)
        dip_approx = cert_p - fwd * np.exp(-r*T) + S*np.exp(-r*T)  # rough
        st.markdown(f"""
<div class="ibox">
<b>Bonus Certificate = Forward + Down-and-In Put (DIP)</b><br><br>

| Component | Description | Approx. Value |
|------------|-------------|----------------|
| 📈 Financed Forward | 100% exposure to underlying | {fwd*np.exp(-r*T):.2f} |
| 📉 DIP Put (H={H:.0f}) | Floor at bonus level if barrier not hit | ~{max(0,cert_p - fwd*np.exp(-r*T) + S*np.exp(-r*T) - S*np.exp(-r*T)):.2f} |
| 🧮 Total | Certificate Price | <b>{cert_p:.2f}</b> |

<br><b>Intuition:</b> If H never breached → the DIP ensures max(S_T, B), yielding the bonus.<br>
If H breached → the DIP disappears (KI activated), the product becomes a simple tracker.<br><br>

<b>⚠️ Behavioral Risk:</b> If H is breached intraday then rebounds, the investor might falsely believe they still have the bonus — the underlying must be continuously monitored or verified with the issuer.
</div>
        """, unsafe_allow_html=True)


# 12 — WORST-OF AUTOCALL

@st.cache_data(show_spinner=False)
def _mc_worstof(spots_t, vols_t, r, q, rho_mat_t, obs_tuple, ac_bar, cap_bar, cpn, N=30_000):
    """Correlated GBM paths via Cholesky. Worst performer drives all barriers."""
    spots   = np.array(spots_t)
    vols    = np.array(vols_t)
    rho_mat = np.array(rho_mat_t)
    obs     = list(obs_tuple)
    n_a     = len(spots)
    T_tot   = obs[-1]
    n_steps = max(int(T_tot * 252), 252)
    dt      = T_tot / n_steps
    idx_obs = [max(0, min(int(t * n_steps / T_tot), n_steps)) for t in obs]

    np.random.seed(42)
    L     = np.linalg.cholesky(rho_mat)                              # (n_a, n_a)
    Z_raw = np.random.standard_normal((N, n_steps, n_a))             # iid
    Z     = Z_raw @ L.T                                              # correlated

    # Build paths for each asset: (N, n_a, n_steps+1)
    paths = np.zeros((N, n_a, n_steps + 1))
    paths[:, :, 0] = spots
    for i in range(n_a):
        lr = (r - q - 0.5 * vols[i]**2) * dt + vols[i] * np.sqrt(dt) * Z[:, :, i]
        log_S = np.log(spots[i]) + np.hstack([np.zeros((N, 1)), np.cumsum(lr, axis=1)])
        paths[:, i, :] = np.exp(log_S)

    # Normalise each asset to 1 (% of initial)
    paths_norm = paths / spots[np.newaxis, :, np.newaxis]            # (N, n_a, steps+1)

    pf    = np.zeros(N)
    done  = np.zeros(N, dtype=bool)
    rtime = np.full(N, -1)

    for k, (t, ix) in enumerate(zip(obs, idx_obs)):
        worst = paths_norm[:, :, ix].min(axis=1)                     # worst perf at obs date
        fired = (~done) & (worst >= ac_bar)
        pf[fired]    = np.exp(-r * t) * (1 + (k + 1) * cpn)
        rtime[fired] = k
        done |= fired

    worst_final = paths_norm[:, :, idx_obs[-1]].min(axis=1)
    loss = (~done) & (worst_final < cap_bar)
    ok   = (~done) & (~loss)
    pf[ok]   = np.exp(-r * T_tot) * 1.0
    pf[loss] = np.exp(-r * T_tot) * worst_final[loss]

    rp = [(rtime == k).mean() for k in range(len(obs))]
    return pf.mean(), rp, loss.mean(), ok.mean(), pf, paths_norm


def render_worstof(S, sig, r, q, T):
    prod_header("📉📈", "Worst-of Autocall (Multi-Asset)",
                "Autocall on the WORST performer of N assets. Correlation is the main driver: low ρ → cheaper product to issue but riskier.")

    # ── Paramètres produit ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_assets = st.radio("Num underlyings", [2, 3], horizontal=True, key="wo_n")
        ac_bar   = st.slider("Autocall Barrier (%)", 80, 120, 100, 5, key="wo_a") / 100
    with c2:
        cap_bar  = st.slider("Capital Barrier (%)",  40,  90,  60, 5, key="wo_c") / 100
        cpn      = st.slider("Coupon / period (%)",   2,  20,   8, 1, key="wo_k") / 100
    with c3:
        freq     = st.radio("Frequency", ["Annual", "Semi-Annual"], key="wo_f")
        rho12    = st.slider("Correlation ρ₁₂", -0.9, 0.99, 0.60, 0.05, key="wo_r12")
    with c4:
        sig1 = st.slider("σ asset 1 (%)", 5, 60, int(sig*100), 1, key="wo_s1") / 100
        sig2 = st.slider("σ asset 2 (%)", 5, 60, int(sig*100)+5, 1, key="wo_s2") / 100
        if n_assets == 3:
            sig3 = st.slider("σ asset 3 (%)", 5, 60, int(sig*100)+10, 1, key="wo_s3") / 100
            rho13 = st.slider("ρ₁₃", -0.9, 0.99, 0.50, 0.05, key="wo_r13")
            rho23 = st.slider("ρ₂₃", -0.9, 0.99, 0.55, 0.05, key="wo_r23")

    per  = {"Annual": 1, "Semi-Annual": 2}[freq]
    obs  = tuple(round((i + 1) / per, 4) for i in range(int(T * per)))

    if n_assets == 2:
        spots_t  = (100.0, 100.0)
        vols_t   = (sig1, sig2)
        rho_mat  = ((1.0, rho12), (rho12, 1.0))
    else:
        spots_t  = (100.0, 100.0, 100.0)
        vols_t   = (sig1, sig2, sig3)
        # Build PSD matrix — clip eigenvalues to ensure positive definiteness
        rho_mat_raw = np.array([[1.0, rho12, rho13],
                                 [rho12, 1.0, rho23],
                                 [rho13, rho23, 1.0]])
        vals, vecs = np.linalg.eigh(rho_mat_raw)
        vals = np.maximum(vals, 1e-6)
        rho_mat_np = vecs @ np.diag(vals) @ vecs.T
        d = np.sqrt(np.diag(rho_mat_np))
        rho_mat_np = rho_mat_np / np.outer(d, d)
        rho_mat = tuple(map(tuple, rho_mat_np.tolist()))

    with st.spinner("Monte Carlo Worst-of…"):
        price, rp, p_loss, p_ok, pf, paths_norm = _mc_worstof(
            spots_t, vols_t, r, q, rho_mat, obs, ac_bar, cap_bar, cpn)

    mcards_row([
        ("Present Value",       f"{price:.3f}",              C[0]),
        ("Expected Return",     f"{(price-1)*100:+.1f}%",   C[2] if price >= 1 else C[3]),
        ("P(Autocall)",         f"{sum(rp)*100:.1f}%",       C[1]),
        ("P(Capital Loss)",     f"{p_loss*100:.1f}%",        C[3]),
    ])

    t1, t2, t3, t4 = st.tabs([
        "🎯 Correlation Impact", "📊 Probabilities & Paths", "📈 Distribution", "🏗️ Composition"
    ])

    # Tab 1 : prix vs corrélation
    with t1:
        rho_range = np.linspace(-0.9, 0.99, 20)
        prices_rho, ploss_rho = [], []
        for rho_v in rho_range:
            rm = ((1.0, rho_v), (rho_v, 1.0))
            p_r, _, pl_r, _, _, _ = _mc_worstof(
                (100.0, 100.0), (sig1, sig2), r, q, rm, obs, ac_bar, cap_bar, cpn)
            prices_rho.append(p_r * 100)
            ploss_rho.append(pl_r * 100)

        fig1 = make_subplots(1, 2,
                              subplot_titles=["Product Price vs ρ",
                                              "P(Capital Loss) vs ρ"])
        fig1.add_trace(go.Scatter(x=rho_range, y=prices_rho, name="Price (base 100)",
                                   line=dict(color=C[0], width=2.5),
                                   fill="tozeroy", fillcolor="rgba(240,136,62,0.07)"),
                       row=1, col=1)
        fig1.add_hline(y=100, line=dict(color="#444c56", dash="dot", width=1), row=1, col=1)
        fig1.add_vline(x=rho12, line=dict(color=C[5], dash="dash", width=1.5),
                       annotation_text=f"Current ρ={rho12:.2f}", row=1, col=1)

        fig1.add_trace(go.Scatter(x=rho_range, y=ploss_rho, name="P(Capital Loss) %",
                                   line=dict(color=C[3], width=2.5),
                                   fill="tozeroy", fillcolor="rgba(248,81,73,0.07)"),
                       row=1, col=2)
        fig1.add_vline(x=rho12, line=dict(color=C[5], dash="dash", width=1.5),
                       annotation_text=f"Current ρ={rho12:.2f}", row=1, col=2)

        for co in [1, 2]:
            fig1.update_xaxes(gridcolor=GRID, title_text="Correlation ρ₁₂", row=1, col=co)
            fig1.update_yaxes(gridcolor=GRID, row=1, col=co)
        fig1.update_layout(height=400, paper_bgcolor=BG, plot_bgcolor=BG,
                            font=dict(color="#c9d1d9"), margin=dict(l=40, r=20, t=50, b=35),
                            showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f"""<div class="ibox">
<b>Key Intuition:</b> When ρ ↓, dispersion between assets ↑ → the worst performer is more severe → price ↓ and loss risk ↑.<br>
At ρ = 1: the worst-of is identical to a single-asset autocall.<br>
At ρ = 0 or negative: the issuer can offer a <b>much higher coupon</b> for the same issue capital — this is its commercial advantage.
</div>""", unsafe_allow_html=True)

    # Tab 2 : proba + chemins
    with t2:
        labs = [f"T={t:.2f}y" for t in obs]
        fig2 = make_subplots(1, 2,
                              subplot_titles=["P(Autocall) per date",
                                              "Worst performer — 60 simulated paths"])
        cum = np.cumsum(rp)
        fig2.add_trace(go.Bar(x=labs, y=[p * 100 for p in rp], name="P(autocall Tᵢ)",
                               marker_color=C[1],
                               text=[f"{p*100:.1f}%" for p in rp],
                               textposition="outside"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=labs, y=cum * 100, name="Cumulative",
                                   line=dict(color=C[0], width=2),
                                   mode="lines+markers"), row=1, col=1)

        n_steps = paths_norm.shape[2] - 1
        t_ax    = np.linspace(0, T, n_steps + 1)
        worst_paths = paths_norm[:, :, :].min(axis=1)  # (N, steps+1)
        for i in range(min(60, len(worst_paths))):
            col_p = "rgba(248,81,73,0.20)" if worst_paths[i].min() < cap_bar \
                    else "rgba(88,166,255,0.15)"
            fig2.add_trace(go.Scatter(x=t_ax, y=worst_paths[i] * 100,
                                       mode="lines",
                                       line=dict(color=col_p, width=0.8),
                                       showlegend=False, hoverinfo="skip"), row=1, col=2)
        fig2.add_hline(y=ac_bar * 100,  line=dict(color=C[2], width=2,   dash="dash"),
                       annotation_text=f"Autocall {ac_bar*100:.0f}%", row=1, col=2)
        fig2.add_hline(y=cap_bar * 100, line=dict(color=C[3], width=2,   dash="dot"),
                       annotation_text=f"Capital {cap_bar*100:.0f}%",  row=1, col=2)

        for co in [1, 2]:
            fig2.update_xaxes(gridcolor=GRID, row=1, col=co)
            fig2.update_yaxes(gridcolor=GRID, row=1, col=co)
        fig2.update_layout(height=430, paper_bgcolor=BG, plot_bgcolor=BG,
                            font=dict(color="#c9d1d9"), margin=dict(l=40, r=20, t=50, b=35),
                            legend=dict(bgcolor="#161b22", bordercolor="#444c56"))
        st.plotly_chart(fig2, use_container_width=True)

    # Tab 3 : distribution
    with t3:
        fig3 = make_subplots(1, 2,
                              subplot_titles=["Discounted payoffs distribution",
                                              "Final worst performer distribution (%)"])
        fig3.add_trace(go.Histogram(x=pf * 100, nbinsx=100,
                                     marker_color="rgba(88,166,255,0.55)",
                                     marker_line=dict(color=C[1], width=0.4),
                                     showlegend=False), row=1, col=1)
        fig3.add_vline(x=100, line=dict(color=C[5], dash="dash", width=2), row=1, col=1,
                       annotation_text="Nominal")
        fig3.add_vline(x=price * 100, line=dict(color=C[0], dash="dash", width=2),
                       annotation_text=f"E={price*100:.1f}", row=1, col=1)

        worst_final = paths_norm[:, :, -1].min(axis=1) * 100
        fig3.add_trace(go.Histogram(x=worst_final, nbinsx=80,
                                     marker_color="rgba(248,81,73,0.45)",
                                     marker_line=dict(color=C[3], width=0.4),
                                     showlegend=False), row=1, col=2)
        fig3.add_vline(x=cap_bar * 100, line=dict(color=C[3], dash="dash", width=2),
                       annotation_text=f"Capital {cap_bar*100:.0f}%", row=1, col=2)

        for co in [1, 2]:
            fig3.update_xaxes(gridcolor=GRID, row=1, col=co)
            fig3.update_yaxes(gridcolor=GRID, row=1, col=co)
        fig3.update_layout(height=400, paper_bgcolor=BG, plot_bgcolor=BG,
                            font=dict(color="#c9d1d9"), margin=dict(l=40, r=20, t=50, b=35))
        st.plotly_chart(fig3, use_container_width=True)

    # Tab 4 : composition 
    with t4:
        st.markdown(f"""<div class="ibox">
<b>Replicating Decomposition — Worst-of Autocall</b><br><br>

| Component | Description | Role |
|------------|-------------|------|
| 🏦 ZCB | Nominal redemption at T={obs[-1]:.1f}y | Protection financing |
| 📈 Digital autocall strip | Coupons if worst ≥ {ac_bar*100:.0f}% at each Tᵢ | Coupon source |
| 📉 Worst-of DI Put | Loss if worst < {cap_bar*100:.0f}% at maturity | Residual risk |
| ➕ <b>Correlation discount</b> | Sale of implied correlation between underlyings | Coupon financing lever |

<br><b>Why does worst-of allow a higher coupon?</b><br>
The issuer sells a basket option whose value depends on the <b>dispersion</b> between assets.<br>
• At ρ = 1: worst-of ≡ single-asset → no correlation discount<br>
• At ρ = 0: high divergence risk → issuer receives a large correlation premium ↑↑<br><br>

<b>Hedging (delta-hedging):</b><br>
• Delta on each asset: ∂V/∂Sᵢ depends on the probability that asset i is the worst performer<br>
• <b>Cross-gamma</b> ∂²V/∂Sᵢ∂Sⱼ: non-zero term, hedging one asset affects the others<br>
• The desk must manage <b>implied vs realized correlation</b> — this is the true risk of the eq exotic desk.<br><br>

<b>Assets used (ρ₁₂ = {rho12:.2f}):</b> σ₁={sig1*100:.0f}%  σ₂={sig2*100:.0f}%
</div>""", unsafe_allow_html=True)


# 13 — VARIANCE SWAP

def render_varswap(S, sig, r, q, T):
    prod_header("σ²", "Variance Swap",
                "Exchanges realized variance vs variance strike (Kvar). Pure exposure to σ², zero delta. Static replication via a strip of puts and calls (Carr-Madan).")

    c1, c2, c3 = st.columns(3)
    with c1:
        Kvar_pct = st.slider("Variance Strike Kvar (vol %)", 5.0, 60.0, sig * 100, 0.5,
                              key="vs_kv", help="Variance strike expressed in equiv. vol (Kvar = (vol%)²/100²)")
        Kvar     = (Kvar_pct / 100) ** 2
    with c2:
        notional = st.slider("Vega Notional (€)", 10_000, 1_000_000, 100_000, 10_000,
                              key="vs_N", help="Gain/loss of 1 vol point on the variance strike")
        var_not  = notional / (2 * Kvar_pct)     # variance notional = vega_notional / (2*Kvol)
    with c3:
        n_obs = st.slider("Observations (days)", 20, 504, int(T * 252), 10, key="vs_obs")

    # Fair strike = intégrale strips Carr-Madan
    # Discrete approximation: Kvar_CM = 2/T * sum over OTM options
    K_grid  = np.linspace(S * 0.4, S * 2.2, 160)
    dK      = K_grid[1] - K_grid[0]
    F       = S * np.exp((r - q) * T)
    weights = np.where(K_grid <= F,
                       put_price(S, K_grid, T, r, q, sig),
                       call_price(S, K_grid, T, r, q, sig)) / K_grid**2
    Kvar_CM = float(2 / T * np.sum(weights) * dK)
    Kvol_CM = np.sqrt(Kvar_CM) * 100   # en vol %

    # Monte Carlo realized variance
    N_P, N_S = 30_000, n_obs
    paths    = mc_paths(S, r, q, sig, T, N_S, N_P)
    log_rets = np.log(paths[:, 1:] / paths[:, :-1])
    var_ann  = log_rets.var(axis=1) * (252)          
    vol_ann  = np.sqrt(var_ann) * 100                 
    pnl      = var_not * (var_ann - Kvar)             

    mcards_row([
        ("Fair Strike (Carr-Madan)", f"{Kvol_CM:.2f}%",                    C[0]),
        ("Input Kvar strike",        f"{Kvar_pct:.2f}%",                  "#8b949e"),
        ("Vega Notional",            f"€{notional:,.0f}",                 C[1]),
        ("Var Notional",             f"€{var_not:,.0f}",                  C[4]),
        ("E[Realized vol]",          f"{vol_ann.mean():.2f}%",             C[2]),
        ("E[P&L long var]",          f"€{pnl.mean():+,.0f}",               C[2] if pnl.mean() >= 0 else C[3]),
        ("VaR 95% P&L",              f"€{np.percentile(pnl,5):+,.0f}",     C[3]),
        ("P(P&L > 0)",               f"{(pnl > 0).mean()*100:.1f}%",       C[5]),
    ])

    t1, t2, t3, t4 = st.tabs([
        "💰 Payoff & P&L", "🔬 Carr-Madan Replication",
        "📊 Realized Vol Distribution", "📋 Theory & Greeks"
    ])

    # Tab 1 : payoff 
    with t1:
        vol_range = np.linspace(1.0, 80.0, 400)
        var_range = (vol_range / 100) ** 2
        pnl_range = var_not * (var_range - Kvar)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=vol_range, y=pnl_range / 1000,
            name="P&L long variance swap",
            line=dict(color=C[0], width=2.5),
            fill="tozeroy", fillcolor="rgba(240,136,62,0.07)"
        ))
        # Break-even = Kvar en vol
        fig1.add_vline(x=Kvar_pct, line=dict(color="#444c56", dash="dot", width=1.5),
                       annotation_text=f"Strike = {Kvar_pct:.1f}%")
        fig1.add_vline(x=sig * 100, line=dict(color=C[5], dash="dash", width=1.5),
                       annotation_text=f"σ_impl = {sig*100:.0f}%")
        fig1.add_hline(y=0, line=dict(color="#444c56", width=1))

        # Convexity: vol swap P&L pour comparaison
        vol_not   = notional
        pnl_volsw = vol_not * (vol_range / 100 - Kvar_pct / 100)
        # Convexity adjustment ≈ var_swap > vol_swap for long
        fig1.add_trace(go.Scatter(
            x=vol_range, y=pnl_volsw / 1000,
            name="Vol swap P&L (linear)",
            line=dict(color=C[1], width=1.8, dash="dash")
        ))
        fig1.update_layout(
            title=f"P&L Long Variance Swap  (Vega notional = €{notional:,.0f})",
            xaxis_title="Realized volatility (%)",
            yaxis_title="P&L (k€)",
            **_base()
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f"""<div class="ibox">
<b>Variance swap convexity:</b> P&L = Var_notional × (Realized_σ² − Kvar)<br>
The curve is <b>convex in σ</b>: for the same vol move, long variance gains more on the upside than it loses on the downside.<br>
⟹ <b>E[Realized_σ²] &lt; √Kvar</b>  (Jensen's inequality) — the variance fair strike is always <i>above</i> the ATM forward vol.<br>
Convexity adjustment ≈ {((Kvol_CM - sig*100)):.2f} vpts (Carr-Madan vs implied ATM)
</div>""", unsafe_allow_html=True)

    # Tab 2 : réplication Carr-Madan
    with t2:
        K_puts  = K_grid[K_grid < F]
        K_calls = K_grid[K_grid >= F]
        w_puts  = 2 / T * put_price(S, K_puts,  T, r, q, sig) / K_puts**2  * dK * var_not
        w_calls = 2 / T * call_price(S, K_calls, T, r, q, sig) / K_calls**2 * dK * var_not

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=K_puts,  y=w_puts,  name="OTM Puts",
                               marker_color="rgba(248,81,73,0.65)",
                               marker_line=dict(width=0)))
        fig2.add_trace(go.Bar(x=K_calls, y=w_calls, name="OTM Calls",
                               marker_color="rgba(88,166,255,0.65)",
                               marker_line=dict(width=0)))
        fig2.add_vline(x=F, line=dict(color=C[5], dash="dash", width=1.5),
                       annotation_text=f"Forward F={F:.1f}")
        fig2.update_layout(
            title="Carr-Madan Replication Strip (weight in € / strike unit)",
            xaxis_title="Strike K", yaxis_title="Weight w(K) (€)",
            barmode="overlay", bargap=0,
            **_base()
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"""<div class="ibox">
<b>Replication Formula (Carr-Madan 1998):</b><br><br>
$$K_{{var}} = \\frac{{2}}{{T}} \\left[ \\int_0^F \\frac{{P(K)}}{{K^2}} dK + \\int_F^\\infty \\frac{{C(K)}}{{K^2}} dK \\right]$$<br><br>
• 1/K² Weight: Deep OTM options (very small K) have a <b>very high weight</b> — this is why the vol skew strongly impacts the var swap price.<br>
• In practice, the strip is <b>truncated</b> at liquid strikes → replication error (truncation bias)<br>
• Carr-Madan Fair strike: <b>{Kvol_CM:.2f}%</b>  (Implied ATM: {sig*100:.1f}%)
</div>""", unsafe_allow_html=True)

    # Tab 3 : distribution vol réalisée 
    with t3:
        fig3 = make_subplots(1, 2,
                              subplot_titles=[f"Realized vol distribution ({n_obs}d)",
                                              "Distribution P&L long variance (€)"])
        fig3.add_trace(go.Histogram(x=vol_ann, nbinsx=80,
                                     marker_color="rgba(63,185,80,0.55)",
                                     marker_line=dict(color=C[2], width=0.4),
                                     showlegend=False), row=1, col=1)
        fig3.add_vline(x=Kvar_pct, line=dict(color=C[0], dash="dash", width=2),
                       annotation_text=f"Kvar={Kvar_pct:.1f}%", row=1, col=1)
        fig3.add_vline(x=vol_ann.mean(), line=dict(color=C[2], dash="dot", width=1.5),
                       annotation_text=f"E[σ]={vol_ann.mean():.1f}%", row=1, col=1)

        fig3.add_trace(go.Histogram(x=pnl / 1000, nbinsx=80,
                                     marker_color="rgba(240,136,62,0.55)",
                                     marker_line=dict(color=C[0], width=0.4),
                                     showlegend=False), row=1, col=2)
        fig3.add_vline(x=0, line=dict(color="#444c56", dash="dash", width=1.5),
                       row=1, col=2, annotation_text="Break-even")
        fig3.add_vline(x=pnl.mean() / 1000, line=dict(color=C[0], dash="dot", width=1.5),
                       annotation_text=f"E={pnl.mean()/1000:+.1f}k€", row=1, col=2)

        for co in [1, 2]:
            fig3.update_xaxes(gridcolor=GRID, row=1, col=co)
            fig3.update_yaxes(gridcolor=GRID, row=1, col=co)
        fig3.update_layout(height=400, paper_bgcolor=BG, plot_bgcolor=BG,
                            font=dict(color="#c9d1d9"), margin=dict(l=40, r=20, t=50, b=35))
        st.plotly_chart(fig3, use_container_width=True)

        # Percentile table
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        df_pct = {
            "Percentile": [f"{p}%" for p in pcts],
            "Realized vol (%)": [f"{np.percentile(vol_ann, p):.2f}%" for p in pcts],
            "P&L long var (k€)": [f"{np.percentile(pnl, p)/1000:+.1f}" for p in pcts],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(df_pct), use_container_width=True, hide_index=True)

    # Tab 4 : théorie 
    with t4:
        st.markdown(f"""
**Payoff at maturity**

$$\\text{{P\\&L}} = N_{{var}} \\times \\left( \\sigma^2_{{realized}} - K_{{var}} \\right)$$

where $\\sigma^2_{{realized}} = \\frac{{252}}{{n}} \\sum_{{i=1}}^n \\left(\\ln\\frac{{S_i}}{{S_{{i-1}}}}\\right)^2$

**Variance swap Greeks**

| Greek | Value | Interpretation |
|-------|--------|----------------|
| **Delta** | 0 | No directional exposure |
| **Vega** | $2 N_{{var}} \\sigma$ | Grows with vol → vol swap concavity |
| **Gamma** | $2 N_{{var}}$ | Constant! The P&L gamma = $N_{{var}} \\times (\\delta\\sigma)^2$ |
| **Theta** | $-N_{{var}} \\sigma^2$ | Loses value if realized σ < strike |

**Variance swap vs Vol swap**

$$K_{{vol\\_swap}} \\approx \\sqrt{{K_{{var}}}} - \\frac{{\\text{{Convexity}}}}{{8 K_{{var}}^{{3/2}}}}$$

The convexity adjustment is roughly **{(Kvol_CM - sig*100):.2f} vpts** here.

**Link to the vol surface (Demeterfi-Derman-Kamal-Zou 1999)**

In the presence of skew, the variance fair strike is not equal to the implied ATM. The contribution of the wings (OTM puts) is amplified by the 1/K² weight. A negative vol skew (typical in equities) implies $K_{{var}} > \\sigma_{{ATM}}$.
        """)


# SIDEBAR

CATS = {
    "🎯 Vanilla Options":   ["European Call", "European Put"],
    "⚡ Exotic Options":  ["Binary Cash-or-Nothing", "Barrier (KO/KI)",
                               "Asian Option", "Lookback"],
    "🏗️ Structured Products": ["Standard Autocall", "Athena", "Phoenix (Memory)",
                                "Reverse Convertible", "Capital Protected Note",
                                "Bonus Certificate", "Worst-of Autocall"],
    "📐 Volatility Derivatives": ["Variance Swap"],
}

with st.sidebar:
    st.markdown("## 📊 Derivatives Lab")
    st.markdown("<p style='color:#8b949e;font-size:.83em;margin-top:-8px'>Interactive Pricer & Structurer</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    cat  = st.selectbox("**Category**", list(CATS.keys()))
    prod = st.selectbox("**Product**",   CATS[cat])

    st.markdown("---")
    st.markdown("**Market Parameters**")
    S   = st.slider("Spot  S₀",               50.0, 300.0, 100.0, 0.5)
    sig = st.slider("Volatility  σ (%)",        5.0,  80.0,  20.0, 0.5) / 100
    r   = st.slider("Risk-free rate  r (%)",  0.0,  10.0,   4.0, 0.1) / 100
    q   = st.slider("Div. yield  q (%)",         0.0,   8.0,   1.5, 0.1) / 100
    T   = st.slider("Maturity  T (years)",        0.1,   5.0,   1.0, 0.1)

    st.markdown("---")
    F     = S * np.exp((r-q)*T)
    atmc  = float(call_price(S,S,T,r,q,sig))
    atmp  = float(put_price(S,S,T,r,q,sig))
    svt   = sig*np.sqrt(T)*100
    st.markdown(f"""<div class="qref">
<b style="color:{C[0]}">Forward F</b>  :  {F:.2f}<br>
<b style="color:{C[1]}">ATM Call</b>  :  {atmc:.3f}  ({atmc/S*100:.2f}%)<br>
<b style="color:{C[2]}">ATM Put</b>   :  {atmp:.3f}  ({atmp/S*100:.2f}%)<br>
<b style="color:{C[4]}">σ√T</b>        :  {svt:.1f}%<br>
<b style="color:{C[6]}">ZCB(T)</b>    :  {np.exp(-r*T)*100:.2f}%
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Derivatives Lab · BS + risk-neutral MC")


# MAIN ROUTER

dispatch = {
    "European Call":           lambda: render_vanilla(S, sig, r, q, T, prod),
    "European Put":            lambda: render_vanilla(S, sig, r, q, T, prod),
    "Binary Cash-or-Nothing":  lambda: render_binary(S, sig, r, q, T),
    "Barrier (KO/KI)":         lambda: render_barrier(S, sig, r, q, T),
    "Asian Option":            lambda: render_asian(S, sig, r, q, T),
    "Lookback":                lambda: render_lookback(S, sig, r, q, T),
    "Standard Autocall":       lambda: render_autocall(S, sig, r, q, T),
    "Athena":                  lambda: render_athena(S, sig, r, q, T),
    "Phoenix (Memory)":        lambda: render_phoenix(S, sig, r, q, T),
    "Reverse Convertible":     lambda: render_reverse_convertible(S, sig, r, q, T),
    "Capital Protected Note":  lambda: render_capital_protected(S, sig, r, q, T),
    "Bonus Certificate":       lambda: render_bonus_cert(S, sig, r, q, T),
    "Worst-of Autocall":       lambda: render_worstof(S, sig, r, q, T),
    "Variance Swap":           lambda: render_varswap(S, sig, r, q, T),
}

dispatch[prod]()