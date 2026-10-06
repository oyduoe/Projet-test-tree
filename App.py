import streamlit as st
from datetime import date
from Main import run_calculations


st.set_page_config(page_title="Pricer d'options", layout="wide")
st.title("Pricer d'options")


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
    pruning_threshold = st.number_input("Pruning threshold", value=1e-9, format="%.10f")
    spread_error = st.number_input("Spread error", value=0.0, format="%.6f")

with col4:
    st.subheader("Additional conditions")
    tree_bol = st.checkbox("Display the Trinomial Tree")
    convergence = st.checkbox("Convergence")
    strike_study = st.checkbox("Compare the prices with the strike?")
    bs_condition = st.checkbox("Black-Scholes Pricing", value=True)
    mc_condition = st.checkbox("Monte Carlo Pricing")
    compute_delta_gamma = st.checkbox("Calculate the Delta and Gamma", value=True)
    compute_vega = st.checkbox("Calculate the Vega", value=True)


# ================= RUN =================
if st.button("Run the calculation", type="primary"):
    input_data = {
        "Volatility": volatility,
        "RiskFree": risk_free,
        "Dividend": dividend,
        "SpotPrice": spot_price,
        "Type": option_type,
        "Exercice": exercise,
        "Maturity": maturity,
        "Strike": strike,
        "DivExDate": div_ex_date,
        "PricingDate": pricing_date,
        "Ts": int(ts),
        "PruningThreshold": pruning_threshold,
        "SpreadError": spread_error,
        "TreeBol": tree_bol,
        "Convergence": convergence,
        "StrikeStudy": strike_study,
        "BSCondition": bs_condition,
        "MCCondition": mc_condition,
        "ComputeDeltaAndGammaTree": compute_delta_gamma,
        "ComputeVegaTree": compute_vega,
    }

    try:
        with st.spinner("Calcul en cours..."):
            results = run_calculations(input_data)
        st.session_state["results"] = results
        st.session_state["input_data"] = input_data
    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")


# ================= DISPLAY =================
if "results" in st.session_state:
    results = st.session_state["results"]
    input_data = st.session_state["input_data"]

    st.markdown("---")
    st.header("Results")

    ttm = (input_data["Maturity"] - input_data["PricingDate"]).days / 365
    message = (f"{input_data['Exercice']} {input_data['Type']} with strike at "
               f"{input_data['Strike']} and steps {input_data['Ts']} "
               f"(IR = {input_data['RiskFree']}, Vol = {input_data['Volatility']}%, "
               f"Div = {input_data['Dividend']} EUR and Time = {round(ttm, 1)})")
    st.markdown(f"**{message}**")

    if "TreePrice" in results:
        c1, c2 = st.columns(2)
        c1.metric("Option Price", f"{results['TreePrice']:.8f}")
        c2.metric("Calculation Time", f"{results['TimePricing']:.4f} s")

    if "Delta" in results or "Vega" in results:
        cols = st.columns(3)
        idx = 0
        if "Delta" in results:
            cols[idx].metric("Delta Tree", f"{results['Delta']:.8f}"); idx += 1
            cols[idx].metric("Gamma Tree", f"{results['Gamma']:.8f}"); idx += 1
        if "Vega" in results:
            cols[idx].metric("Vega Tree", f"{results['Vega']:.8f}")

    if "Tree Display Time" in results:
        st.write(f"Tree Display Time : {results['Tree Display Time']:.4f} s")
    if "New TS" in results:
        st.write(f"New Time Step (to Fix the Error) : {results['New TS']:.0f}")

    # --- Black-Scholes ---
    if input_data["BSCondition"] and "BS Price" in results:
        st.subheader("Black-Scholes Results")
        c1, c2 = st.columns(2)
        c1.metric("BS Price", f"{results['BS Price']:.2f}")
        c2.metric("BS Calculation Time", f"{results['BS_Time']:.4f} s")
        c1.metric("Tree - BS", f"{results['TreeGapBS']:.2f}")
        c2.metric("Time Diff (Tree - BS)", f"{results['TreeGapBSTime']:.4f} s")

    # --- Monte Carlo ---
    if input_data["MCCondition"] and "MC Price" in results:
        st.subheader("Monte Carlo Results")
        c1, c2 = st.columns(2)
        c1.metric("MC Price", f"{results['MC Price']:.2f}")
        c2.metric("MC Calculation Time", f"{results['MCTime']:.4f} s")
        c1.metric("Tree - MC", f"{results['MCGapTree']:.2f}")
        c2.metric("Time Diff (Tree - MC)", f"{results['MCGapTreeTime']:.4f} s")

    # --- Graphiques ---
    if "ConvergenceFig" in results and results["ConvergenceFig"] is not None:
        st.subheader("Graphique de Convergence")
        st.pyplot(results["ConvergenceFig"])

    if "StrikeStudyFig" in results and results["StrikeStudyFig"] is not None:
        st.subheader("Graphique d'étude du strike")
        st.pyplot(results["StrikeStudyFig"])

    # --- Arbre Trinomial ---
    if "TreeDf" in results and results["TreeDf"] is not None:
        st.subheader("Trinomial Tree (probabilités)")
        st.dataframe(results["TreeDf"])
        csv = results["TreeDf"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger l'arbre (CSV)",
            data=csv,
            file_name="trinomial_tree.csv",
            mime="text/csv",
        )