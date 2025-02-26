import streamlit as st
import numpy as np
from scipy.stats import norm

### Définition des fonctions ###

def BlackScholes(r, S, K, T, sigma, type="c"):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if type == "c":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif type == "p":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        return None
    
    return price

def delta_calc(r, S, K, T, sigma, type="c"):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if type == "c" else -norm.cdf(-d1)

def gamma_calc(r, S, K, T, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega_calc(r, S, K, T, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) * 0.01

def theta_calc(r, S, K, T, sigma, type="c"):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2 if type == "c" else -d2)
    return (term1 - term2) / 365

def rho_calc(r, S, K, T, sigma, type="c"):
    d2 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T)) - sigma * np.sqrt(T)
    return (K * T * np.exp(-r * T) * norm.cdf(d2 if type == "c" else -d2)) * 0.01

### Interface StreamLit ###

st.title("Screener d'Options - Modèle Black-Scholes")

col1, col2, col3 = st.columns([5, 2, 5])

with col1:
    st.header("Saisie des variables")
    r = st.number_input("Taux d'intérêt (%)", value=1.0) / 100                  # Interest Rate
    S = st.number_input("Prix de l'actif sous-jacent", value=30.0)              # Underline Price
    K = st.number_input("Prix d'exercice", value=40.0)                          # Strike Price
    T = st.number_input("Temps avant expiration (jours)", value=240) / 365      # Time
    sigma = st.number_input("Volatilité (%)", value=30.0) / 100                 # Volatility
    option_type = st.selectbox("Type d'option", ["Call", "Put"])

with col3:
    st.header("Résultats de l'option")

    price = BlackScholes(r, S, K, T, sigma, type="c" if option_type == "Call" else "p")
    delta = delta_calc(r, S, K, T, sigma, type="c" if option_type == "Call" else "p")
    gamma = gamma_calc(r, S, K, T, sigma)
    vega = vega_calc(r, S, K, T, sigma)
    theta = theta_calc(r, S, K, T, sigma, type="c" if option_type == "Call" else "p")
    rho = rho_calc(r, S, K, T, sigma, type="c" if option_type == "Call" else "p")

    st.write(f"### Prix de l'option : {price:.2f}")
    st.write(f"**Delta :** {delta:.4f}")
    st.write(f"**Gamma :** {gamma:.4f}")
    st.write(f"**Vega :** {vega:.4f}")
    st.write(f"**Theta :** {theta:.4f}")
    st.write(f"**Rho :** {rho:.4f}")

