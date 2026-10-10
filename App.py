import streamlit as st
import streamlit.components.v1 as components
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
    dividend_type = st.radio(
        "Type de dividende",
        ["Discrete", "Continuous"],
        horizontal=True,
        help=("Discrete : montant cash à la date ex-dividende (utilisé par "
              "Tree et Binomial via modèle escrowed). "
              "Continuous : rendement continu q = D/S (utilisé par BS).")
    )
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
    tree_bol = st.checkbox("Display the Trinomial Tree (SVG)")
    convergence = st.checkbox("Convergence")
    mc_condition = st.checkbox("Monte Carlo Pricing")
    binom_condition = st.checkbox("Binomial Pricing")
    compare_am_eur = st.checkbox("American vs European gap (Tree & Binomial)")

st.subheader("Comparaisons")
c1, c2 = st.columns(2)
with c1:
    strike_study = st.checkbox("Prices & Greeks vs Strike")
with c2:
    compare_prices_steps = st.checkbox("Prices vs Steps")


# ================= RUN =================
if st.button("Run the calculation", type="primary"):
    input_data = {
        "Volatility": volatility,
        "RiskFree": risk_free,
        "Dividend": dividend,
        "DividendType": dividend_type,
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
        "BSCondition": True,
        "MCCondition": mc_condition,
        "BinomCondition": binom_condition,
        "ComparePricesSteps": compare_prices_steps,
        "CompareAmEur": compare_am_eur,
        "ComputeDeltaAndGammaTree": True,
        "ComputeVegaTree": True,
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

    # --- Prix ---
    cols = st.columns(4)
    cols[0].metric("Trinomial", f"{results.get('TreePrice', float('nan')):.6f}")
    cols[1].metric("Black-Scholes", f"{results.get('BS Price', float('nan')):.6f}")
    cols[2].metric("Binomial",
                   f"{results.get('Binom Price', float('nan')):.6f}"
                   if input_data["BinomCondition"] else "—")
    cols[3].metric("Monte Carlo",
                   f"{results.get('MC Price', float('nan')):.6f}"
                   if input_data["MCCondition"] else "—")

    # --- Pricing times ---
    st.markdown("**Pricing times** *(time to compute the price only)*")
    pt = {"Trinomial": results.get("TimePricing"),
          "Black-Scholes": results.get("BS_Time")}
    if input_data["BinomCondition"]:
        pt["Binomial"] = results.get("Binom_Time")
    if input_data["MCCondition"]:
        pt["Monte Carlo"] = results.get("MCTime")
    pt = {k: v for k, v in pt.items() if v is not None}
    if pt:
        st.dataframe(
            pd.DataFrame({"Model": list(pt.keys()),
                          "Time (s)": [f"{v:.5f}" for v in pt.values()]}),
            hide_index=True, use_container_width=True)

    # --- Greeks ---
    st.subheader("Greeks par modèle")
    greeks_rows = ["Delta", "Gamma", "Vega", "Vomma", "Vanna", "Theta"]
    model_map = [("Tree", "greeks_tree"), ("Black-Scholes", "greeks_bs")]
    if input_data["BinomCondition"]:
        model_map.append(("Binomial", "greeks_binom"))
    if input_data["MCCondition"]:
        model_map.append(("Monte Carlo", "greeks_mc"))

    data = {}
    for name, key in model_map:
        g = results.get(key, {}) or {}
        if g:
            data[name] = [float(g.get(r, float('nan'))) for r in greeks_rows]
    if data:
        st.dataframe(pd.DataFrame(data, index=greeks_rows).round(6),
                     use_container_width=True)

    # --- Pricing + Greeks times ---
    st.markdown("**Pricing + Greeks times** *(price and all greeks)*")
    tt = {"Trinomial": results.get("Tree_Greeks_Time"),
          "Black-Scholes": results.get("BS_Greeks_Time")}
    if input_data["BinomCondition"]:
        tt["Binomial"] = results.get("Binom_Greeks_Time")
    if input_data["MCCondition"]:
        tt["Monte Carlo"] = results.get("MC_Greeks_Time")
    tt = {k: v for k, v in tt.items() if v is not None}
    if tt:
        st.dataframe(
            pd.DataFrame({"Model": list(tt.keys()),
                          "Time (s)": [f"{v:.5f}" for v in tt.values()]}),
            hide_index=True, use_container_width=True)
        st.caption("BS est analytique : prix et greeks calculés en un seul appel.")

    # --- Autres temps ---
    with st.expander("Autres temps de calcul"):
        if "Tree Display Time" in results:
            st.write(f"Génération SVG : {results['Tree Display Time']:.4f} s")
        if "New TS" in results:
            st.write(f"Nouveau Time Step (spread error) : {results['New TS']:.0f}")

    # --- Trinomial Tree SVG ---
    if input_data["TreeBol"]:
        st.subheader("Trinomial Tree")
        svg_str = results.get("TreeSvg")

        if svg_str:
            st.caption(
                f"Affichage limité aux 40 premières colonnes (sur {input_data['Ts']}). "
                f"SVG : {len(svg_str):,} caractères."
            )
            st.markdown(
                "- **Couleur du nœud = moneyness** : "
                "vert = ITM, ambre = ATM, **bleu = OTM** "
                "(inversé pour les Puts)\n"
                "- **Bordure rouge épaisse + point blanc central** : "
                "exercice anticipé optimal (options American uniquement)\n"
                "- **Arêtes** : bleu clair = Up, gris = Mid, orange = Down"
            )

            html = (
                '<div style="width:100%; height:100%; overflow:auto; '
                'border:1px solid #ddd; border-radius:6px; background:#fafafa;">'
                f'{svg_str}'
                '</div>'
            )
            components.html(html, height=900, scrolling=True)

            st.download_button(
                "Download SVG",
                data=svg_str.encode("utf-8"),
                file_name="trinomial_tree.svg",
                mime="image/svg+xml",
                key="dl_svg_tree",
            )
        else:
            st.warning(
                "SVG non généré. Vérifie que `svg-py` est installé : "
                "`pip install svg-py`. Regarde la console Python pour "
                "un éventuel message `[SVG] ...`."
            )

    # --- Convergence ---
    if "ConvergenceFig" in results and results["ConvergenceFig"] is not None:
        st.subheader("Convergence")
        st.pyplot(results["ConvergenceFig"], use_container_width=True)

    # --- Strike Study ---
    if "StrikeStudyFig" in results and results["StrikeStudyFig"] is not None:
        st.subheader("Prices, Gaps & Slopes vs Strike")
        st.pyplot(results["StrikeStudyFig"], use_container_width=True)
    if "StrikeStudyError" in results:
        st.error(f"Erreur Strike Study :\n{results['StrikeStudyError']}")

    # --- Comparatif vs Steps ---
    if "ComparePricesSteps" in results:
        comp = results["ComparePricesSteps"]
        st.subheader("Prix vs Steps")
        st.pyplot(comp["prices_vs_steps"], use_container_width=True)
        st.subheader("Gaps de prix vs Steps")
        st.pyplot(comp["gaps_vs_steps"], use_container_width=True)
        st.subheader("Temps de calcul vs Steps")
        st.pyplot(comp["time_vs_steps"], use_container_width=True)
    if "ComparePricesStepsError" in results:
        st.error(f"Erreur Prices vs Steps :\n{results['ComparePricesStepsError']}")

    # --- American vs European ---
    if "CompareAmEur" in results and results["CompareAmEur"] is not None:
        st.subheader("Gap American - European")
        st.pyplot(results["CompareAmEur"], use_container_width=True)
    if "CompareAmEurError" in results:
        st.error(f"Erreur American vs European :\n{results['CompareAmEurError']}")