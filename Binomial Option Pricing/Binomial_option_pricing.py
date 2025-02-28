import numpy as np
from scipy.stats import norm
import streamlit as st

### Logique ###

def binomial_tree(K, S0, sigma, T, N, type = 'Call'):

    dt = T / N
    a = np.exp(r * dt)
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(- sigma * np.sqrt(dt))
    p = (a - d) / (u - d)

    disc = np.exp(-r * dt)

    S = S0 * d ** np.arange(N, -1, -1) * u ** np.arange(0, N+1, 1)

    if type == 'C':
        price = np.maximum(S - K, 0)
    else:
        price = np.maximum(K - S, 0)

    for i in np.arange(N-1, -1, -1):
        price = disc * (price[1:i+2] * p + price[:i+1] * (1-p))

    return price[0]

### Interface ###

st.title("Pricing d'option Européennes - Modèle Binomiale")

col1, col2, col3 = st.columns([5, 2, 5])

with col1:
    st.header("Saisie des variables")
    option_type = st.selectbox("Type d'option", ['Call', 'Put'])
    r = st.number_input("Taux sans risque", value = 1.0) / 100                  
    S0 = st.number_input("Prix de l'actif sous-jacent", value = 30.0)              
    K = st.number_input("Prix d'exercice", value = 30.0)
    sigma = st.number_input("Volatilité (%)", value = 30.0) / 100                          
    T = st.number_input("Temps avant expiration (jours)", value=240) / 365      
    N = st.number_input("Nombre de périodes", value = 300)
                     
with col3:
    st.header("Résultats de l'option")

    price = binomial_tree(K, S0, sigma, T, N, type="Call" if option_type == "Call" else "Put")
    
    st.write(f"### Prix de l'option : {price:.2f}")