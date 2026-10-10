import streamlit as st
import pandas as pd
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
    bs_condition = st.checkbox("Black-Scholes Pricing", value=True)
    mc_condition = st.checkbox("Monte Carlo Pricing")
    binom_condition = st.checkbox("Binomial Pricing")
    compute_greeks = st.checkbox("Calculate the Greeks (Delta, Gamma, Vega, Vomma, Vanna, Theta)")

st.subheader("Comparaisons")
c1, c2, c3 = st.columns(3)
with c1:
    strike_study = st.checkbox("Prices & Greeks vs Strike")
with c2:
    compare_prices_steps = st.checkbox("Prices vs Steps")
with c3:
    compare_am_eur = st.checkbox("American vs European gaps")


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
        "BinomCondition": binom_condition,
        "ComputeGreeks": compute_greeks,
        "ComparePricesSteps": compare_prices_steps,
        "CompareAmEur": compare_am_eur,
        # Anciens flags conservés pour compat (non utilisés)
        "ComputeDeltaAndGammaTree": compute_greeks,
        "ComputeVegaTree": compute_greeks,
    }

    try:
        with st.spinner("Calcul en cours..."):
            results = run_calculations(input_data)
        st.session_state["results"] = results
        st.session_state["input_data"] = input_data
    except Exception as e:
        import traceback
        st.error(f"Erreur lors du calcul : {e}")
        st.code(traceback.format_exc())


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

    # --- Prix principaux ---
    cols = st.columns(4)
    cols[0].metric("Trinomial", f"{results.get('TreePrice', float('nan')):.6f}")
    cols[1].metric("Black-Scholes", f"{results.get('BS Price', float('nan')):.6f}"
                   if input_data["BSCondition"] else "—")
    cols[2].metric("Binomial", f"{results.get('Binom Price', float('nan')):.6f}"
                   if input_data["BinomCondition"] else "—")
    cols[3].metric("Monte Carlo", f"{results.get('MC Price', float('nan')):.6f}"
                   if input_data["MCCondition"] else "—")

    # --- Greeks Table ---
    if input_data["ComputeGreeks"]:
        st.subheader("Greeks par modèle")
        greeks_rows = ["Delta", "Gamma", "Vega", "Vomma", "Vanna", "Theta"]
        greeks_dict = {
            "Trinomial": results.get("greeks_tree", {}),
            "Binomial": results.get("greeks_binom", {}),
            "Black-Scholes": results.get("greeks_bs", {}),
            "Monte Carlo": results.get("greeks_mc", {}),
        }
        # On ne garde que les modèles actifs (dict non vide)
        greeks_dict = {k: v for k, v in greeks_dict.items() if v}
        if greeks_dict:
            data = {}
            for model, g in greeks_dict.items():
                data[model] = [g.get(r, float('nan')) for r in greeks_rows]
            df_g = pd.DataFrame(data, index=greeks_rows)
            st.dataframe(df_g.style.format("{:.6f}"))

    # --- Temps ---
    with st.expander("Détails des temps de calcul"):
        for k in ("TimePricing", "BS_Time", "Binom_Time", "MCTime", "Tree Display Time"):
            if k in results:
                st.write(f"{k} : {results[k]:.4f} s")
        if "New TS" in results:
            st.write(f"Nouveau Time Step (spread error) : {results['New TS']:.0f}")

    # --- Arbre Trinomial ---
    if "TreeDf" in results and results["TreeDf"] is not None:
        st.subheader("Trinomial Tree (probabilités)")
        st.dataframe(results["TreeDf"])
        csv = results["TreeDf"].to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger l'arbre (CSV)",
                           data=csv, file_name="trinomial_tree.csv",
                           mime="text/csv")

    # --- Convergence classique ---
    if "ConvergenceFig" in results and results["ConvergenceFig"] is not None:
        st.subheader("Convergence vers Black-Scholes")
        st.pyplot(results["ConvergenceFig"])

    # --- Strike Study ---
    if "StrikeStudyFig" in results and results["StrikeStudyFig"] is not None:
        st.subheader("Prix, Gaps et Slopes vs Strike")
        st.pyplot(results["StrikeStudyFig"])

    # --- Comparatifs : prix/gaps/temps vs steps ---
    if "ComparePricesSteps" in results:
        comp = results["ComparePricesSteps"]
        st.subheader("Convergence des prix vs Steps")
        st.pyplot(comp["prices_vs_steps"])
        st.subheader("Gaps de prix vs Steps")
        st.pyplot(comp["gaps_vs_steps"])
        st.subheader("Temps de calcul vs Steps")
        st.pyplot(comp["time_vs_steps"])

    # --- Comparatif American vs European ---
    if "CompareAmEur" in results and results["CompareAmEur"] is not None:
        st.subheader("Gap American - European par modèle")
        st.pyplot(results["CompareAmEur"])