import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

from Main import run_calculations
from Debug import DebugChecks


st.set_page_config(page_title="Pricer d'options", layout="wide")
st.title("Pricer d'options — Trinomial / Binomial (2026-2027)")


# ================= INPUTS =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Paramètres du marché")
    volatility = st.number_input("Volatility (%)", value=21.0, format="%.2f")
    risk_free = st.number_input("Riskfree (%)", value=3.0, format="%.2f")
    dividend = st.number_input("Dividende EUR", value=0.0, format="%.2f")
    spot_price = st.number_input("Spotprice", value=100.0, format="%.2f")

with col2:
    st.subheader("Paramètres de l'option")
    option_type = st.selectbox("Type d'option", ["Call", "Put"])
    exercise = st.selectbox("Type d'exercice", ["American", "European"])
    strike = st.number_input("Strike", value=101.0, format="%.2f")
    maturity = st.date_input("Maturity Date", value=date(2024, 12, 26))
    div_ex_date = st.date_input("Date ex-dividende", value=date(2024, 6, 15))

col3, col4 = st.columns(2)

with col3:
    st.subheader("Paramètres du pricer")
    pricing_date = st.date_input("Pricing Date", value=date(2024, 3, 1))
    ts = st.number_input("Time steps", value=100, min_value=1, step=1)
    tree_type = st.selectbox(
        "Type d'arbre",
        ["Auto (Binomial → Trinomial si dividende)", "Binomial", "Trinomial"],
    )
    pruning_threshold = st.number_input("Pruning threshold", value=1e-9, format="%.10f")
    spread_error = st.number_input("Spread error", value=0.0, format="%.6f")

with col4:
    st.subheader("Additional conditions")
    tree_bol = st.checkbox("Display the Tree")
    convergence = st.checkbox("Convergence")
    strike_study = st.checkbox("Compare the prices with the strike?")
    bs_condition = st.checkbox("Black-Scholes Pricing", value=True)
    mc_condition = st.checkbox("Monte Carlo Pricing")
    compute_greeks = st.checkbox(
        "Calculate the Greeks (Delta, Gamma, Vega, Vomma, Vanna, Theta)", value=True)
    debug_mode = st.checkbox("Debug mode (compteurs, vérifs probas)", value=False)


# ================= RUN =================
if st.button("Run the calculation", type="primary"):
    input_data = {
        "Volatility": volatility, "RiskFree": risk_free,
        "Dividend": dividend, "SpotPrice": spot_price,
        "Type": option_type, "Exercice": exercise,
        "Maturity": maturity, "Strike": strike,
        "DivExDate": div_ex_date, "PricingDate": pricing_date,
        "Ts": int(ts), "TreeType": tree_type,
        "PruningThreshold": pruning_threshold, "SpreadError": spread_error,
        "TreeBol": tree_bol, "Convergence": convergence,
        "StrikeStudy": strike_study, "BSCondition": bs_condition,
        "MCCondition": mc_condition, "ComputeGreeks": compute_greeks,
        "DebugMode": debug_mode,
    }
    try:
        with st.spinner("Calcul en cours..."):
            results = run_calculations(input_data)
        st.session_state["results"] = results
        st.session_state["input_data"] = input_data
    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")
        st.exception(e)


# ================= DISPLAY =================
if "results" in st.session_state:
    results = st.session_state["results"]
    input_data = st.session_state["input_data"]

    st.markdown("---")
    st.header("Results")

    ttm = (input_data["Maturity"] - input_data["PricingDate"]).days / 365
    msg = (f"{input_data['Exercice']} {input_data['Type']} strike {input_data['Strike']} "
           f"steps {input_data['Ts']} (r={input_data['RiskFree']}, "
           f"vol={input_data['Volatility']}%, div={input_data['Dividend']}, T={ttm:.2f}y) — "
           f"arbre={results.get('DebugInfo', {}).get('tree_type', input_data['TreeType'])}")
    st.markdown(f"**{msg}**")

    if "TreePrice" in results:
        c1, c2 = st.columns(2)
        c1.metric("Option Price", f"{results['TreePrice']:.6f}")
        c2.metric("Calculation Time", f"{results['TimePricing']:.4f} s")

    if "Tree Display Time" in results:
        st.write(f"Tree Display Time : {results['Tree Display Time']:.4f} s")
    if "New TS" in results:
        st.write(f"New Time Step : {results['New TS']:.0f}")

    if input_data["BSCondition"] and "BS Price" in results:
        st.subheader("Black-Scholes")
        c1, c2 = st.columns(2)
        c1.metric("BS Price", f"{results['BS Price']:.4f}")
        c2.metric("BS Time", f"{results['BS_Time']:.4f} s")

    if input_data["MCCondition"] and "MC Price" in results:
        st.subheader("Monte-Carlo")
        c1, c2 = st.columns(2)
        c1.metric("MC Price", f"{results['MC Price']:.4f}")
        c2.metric("MC Time", f"{results['MCTime']:.4f} s")

    if "ConvergenceFig" in results and results["ConvergenceFig"] is not None:
        st.subheader("Convergence")
        st.pyplot(results["ConvergenceFig"])

    if "StrikeStudyFig" in results and results["StrikeStudyFig"] is not None:
        st.subheader("Strike Study")
        st.pyplot(results["StrikeStudyFig"])

    # ================ TABS ================
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Greeks", "🧪 Debug / Vérifications", "🌡️ Heatmap", "📈 Arbre"])

    with tab1:
        if "GreeksTable" in results:
            st.dataframe(results["GreeksTable"].style.format("{:.6f}"))
            csv = results["GreeksTable"].to_csv().encode()
            st.download_button("Télécharger Greeks (CSV)", csv,
                               file_name="greeks.csv", mime="text/csv")
            c1, c2 = st.columns(2)
            c1.metric("Temps sans Greeks", f"{results.get('TimeNoGreeks', 0):.4f} s")
            c2.metric("Temps avec Greeks", f"{results.get('TimeGreeks', 0):.4f} s")
        else:
            st.info("Coche 'Calculate the Greeks'.")

    with tab2:
        st.subheader("Vérifications théoriques automatiques")
        st.caption("⚠️ Ces tests relancent le pricer plusieurs fois. Ça peut prendre du temps.")
        if st.button("Lancer les tests", key="run_debug"):
            dbg = DebugChecks(run_calculations, st.session_state["input_data"])
            with st.spinner("Tests en cours..."):
                df_tests = dbg.run_all()
            st.dataframe(df_tests)
            n_ok = (df_tests["OK"] == "✅").sum()
            n_bad = (df_tests["OK"] == "❌").sum()
            c1, c2 = st.columns(2)
            c1.metric("Tests OK", n_ok)
            c2.metric("Tests échoués", n_bad)

        st.markdown("---")
        st.subheader("Statistiques internes du dernier run")
        info = results.get("DebugInfo")
        if info:
            st.json(info)
        else:
            st.warning("Active 'Debug mode' pour voir les statistiques.")

    with tab3:
        st.subheader("Heatmap de l'arbre")
        df_tree = results.get("TreeDf")
        if df_tree is not None and not df_tree.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            data = df_tree.astype(float).values
            im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")
            plt.colorbar(im, ax=ax, label="Valeur")
            ax.set_xlabel("Time step"); ax.set_ylabel("Node index")
            ax.set_title("Arbre (couleur = prix sous-jacent)")
            st.pyplot(fig)
        else:
            st.info("Coche 'Display the Tree' pour voir la heatmap.")

    with tab4:
        if "TreeDf" in results:
            st.dataframe(results["TreeDf"])
        else:
            st.info("Coche 'Display the Tree'.")