import numpy as np
import streamlit as st

def option_pricing_tree(S0, u, d, p, n, K, r, option_type='Call', option_style='Européene'):

    stock_tree = np.zeros((n+1, n+1))
    stock_tree[0, 0] = S0
    
    for i in range(1, n+1):
        for j in range(i+1):
            stock_tree[j, i] = S0 * u**(i-j) * d**j
    
    option_tree = np.zeros((n+1, n+1))
    
    for j in range(n+1):
        if option_type == 'Call':
            option_tree[j, n] = max(stock_tree[j, n] - K, 0)
        else:  # put
            option_tree[j, n] = max(K - stock_tree[j, n], 0)
    
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            expected_value = np.exp(-r * (1/n)) * (p * option_tree[j, i+1] + (1-p) * option_tree[j+1, i+1])
            
            if option_style == 'Américaine':                                # Option américaine, possibilité d'exercer avant l'échéance
                if option_type == 'Call':
                    option_tree[j, i] = max(stock_tree[j, i] - K, expected_value)
                else:
                    option_tree[j, i] = max(K - stock_tree[j, i], expected_value)
            else:                                                                       # Option européenne
                option_tree[j, i] = expected_value
    
    return stock_tree, option_tree

### Interface Streamlit ###

st.title("Calcul du prix d'une option européenne ou américaine via un modèle binomiale")

col1, col2, col3, col4, col5 = st.columns([5, 2, 5, 2, 5])

### Paramètres ###

with col1:
    st.header("Saisie des variables")
    S0 = st.number_input("Prix initial de l'action", value=100.0)
    u = st.number_input("Facteur de hausse", value=1.3499)
    d = st.number_input("Facteur de baisse", value=0.7408)
    p = st.number_input("Probabilité de hausse", value=0.5097)
    n = st.number_input("Nombre d'étapes", value=3, min_value=1, step=1)
    K = st.number_input("Prix d'exercice", value=100.0)
    r = st.number_input("Taux sans risque", value=0.06)

with col3:
    
    st.header("Choix du type d'option")
    type_option = st.selectbox("Type d'option", ["Call", "Put"])
    style_option = st.selectbox("Style d'option", ["Européenne", "Américaine"])

with col5:

    if st.button("Calculer"):
        stock_tree, option_tree = option_pricing_tree(S0, u, d, p, n, K, r, type_option, style_option)
        
        st.subheader("Arbre des prix de l'action")
        st.write(stock_tree)
        
        st.subheader("Arbre des prix de l'option")
        st.write(option_tree)