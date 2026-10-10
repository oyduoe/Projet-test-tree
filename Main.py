import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from Market import Market
from OptionParameters import OptionParameters
from Tree import Tree
from Pricer import Pricer
from Extension import Extension
from Binomial import Binomial


# ----------------------------------------------------------------------
# Comparatifs
# ----------------------------------------------------------------------
def _compare_prices_vs_steps(input_data, step_list=(5, 10, 20, 40, 80, 160)):
    """Renvoie un dict avec 3 figures : prix vs steps, gaps vs steps, temps vs steps."""
    tree_prices, bs_prices, binom_prices, mc_prices = [], [], [], []
    tree_times, bs_times, binom_times, mc_times = [], [], [], []

    for T in step_list:
        local_input = dict(input_data)
        local_input["Ts"] = T

        mkt = Market(local_input)
        params = OptionParameters(local_input)
        pricer = Pricer(local_input)
        Ext = Extension(mkt, params, pricer)
        arbre = Tree()
        binom = Binomial(mkt, params, pricer)

        # BS
        pricer.Convergence = False
        t0 = time.time()
        bs_res = Ext.black_sholes()
        bs_times.append(time.time() - t0)
        bs_prices.append(bs_res["BS Price"])

        # Tree (sans affichage)
        pricer.display_tree_bool = False
        pricer.BS_condition = False
        pricer.MC_condition = False
        pricer.compute_DeltaAndGamma_tree_var = False
        t0 = time.time()
        arbre.init(mkt, params, pricer, True)
        tree_times.append(time.time() - t0)
        tree_prices.append(arbre.Root_Node.OptPrice)

        # Binomial
        t0 = time.time()
        bp = binom.price_option(T)
        binom_times.append(time.time() - t0)
        binom_prices.append(bp)

        # MC (steps limités pour rapidité)
        pricer.timeSteps = min(T, 50)
        t0 = time.time()
        np.random.seed(42)
        mp = Ext.monte_carlo_price(5000)
        mc_times.append(time.time() - t0)
        mc_prices.append(mp)

    # Figure 1 : prix vs steps
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(step_list, tree_prices, marker='o', label='Trinomial')
    ax1.plot(step_list, binom_prices, marker='s', label='Binomial')
    ax1.plot(step_list, bs_prices, marker='^', label='Black-Scholes', linestyle='--')
    ax1.plot(step_list, mc_prices, marker='x', label='Monte Carlo', linestyle=':')
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Option Price")
    ax1.set_title("Convergence des prix vs Steps")
    ax1.grid(True)
    ax1.legend()

    # Figure 2 : gaps vs steps
    gaps_bs = [t - b for t, b in zip(tree_prices, bs_prices)]
    gaps_binom = [t - b for t, b in zip(tree_prices, binom_prices)]
    gaps_mc = [t - m for t, m in zip(tree_prices, mc_prices)]
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(step_list, gaps_bs, marker='o', label='Tree - BS')
    ax2.plot(step_list, gaps_binom, marker='s', label='Tree - Binomial')
    ax2.plot(step_list, gaps_mc, marker='x', label='Tree - Monte Carlo')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Gap")
    ax2.set_title("Gaps de prix vs Steps")
    ax2.grid(True)
    ax2.legend()

    # Figure 3 : temps vs steps
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(step_list, tree_times, marker='o', label='Trinomial')
    ax3.plot(step_list, binom_times, marker='s', label='Binomial')
    ax3.plot(step_list, bs_times, marker='^', label='Black-Scholes', linestyle='--')
    ax3.plot(step_list, mc_times, marker='x', label='Monte Carlo', linestyle=':')
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Temps de calcul (s)")
    ax3.set_title("Temps de calcul vs Steps")
    ax3.grid(True)
    ax3.legend()

    return {"prices_vs_steps": fig1, "gaps_vs_steps": fig2, "time_vs_steps": fig3}


def _compare_american_european(input_data):
    """Bar chart des gaps American - European par modèle."""
    def price_model(model_name, exercise):
        local = dict(input_data)
        local["Exercice"] = exercise
        mkt = Market(local)
        params = OptionParameters(local)
        pricer = Pricer(local)
        Ext = Extension(mkt, params, pricer)
        arbre = Tree()
        binom = Binomial(mkt, params, pricer)

        pricer.Convergence = False
        pricer.display_tree_bool = False
        pricer.BS_condition = False
        pricer.MC_condition = False
        pricer.compute_DeltaAndGamma_tree_var = False

        arbre.init(mkt, params, pricer, True)
        tree_p = arbre.Root_Node.OptPrice
        binom_p = binom.price_option(pricer.timeSteps)

        Ext.black_sholes()
        bs_p = pricer.BS_price

        np.random.seed(42)
        mc_p = Ext.monte_carlo_price(5000)

        return {"Tree": tree_p, "Binomial": binom_p,
                "BS": bs_p, "MC": mc_p}

    eu = price_model("eu", "European")
    am = price_model("am", "American")

    models = ["Tree", "Binomial", "BS", "MC"]
    gaps = [am[m] - eu[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(models, gaps, color=['orange', 'blue', 'green', 'purple'])
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Price American - Price European")
    ax.set_title("Gap American vs European par modèle")
    ax.grid(True, axis='y')
    return fig


# ----------------------------------------------------------------------
# Point d'entrée principal
# ----------------------------------------------------------------------
def run_calculations(input_data: dict) -> dict:
    # Instancier
    mkt = Market(input_data)
    params = OptionParameters(input_data)
    arbre = Tree()
    pricer = Pricer(input_data)
    Ext = Extension(mkt, params, pricer)
    binom = Binomial(mkt, params, pricer)
    # Flag binom accessible dans Tree.display_results
    pricer.Binom_condition = input_data.get("BinomCondition", False)

    # --- Black & Scholes ---
    if input_data["BSCondition"]:
        t0 = time.time()
        bs_res = Ext.black_sholes()
        pricer.BS_Time = time.time() - t0
        pricer.greeks_bs = {k: bs_res[k] for k in
                            ("Delta", "Gamma", "Vega", "Vomma", "Vanna", "Theta")
                            if k in bs_res}

    # --- Monte Carlo ---
    if input_data["MCCondition"]:
        t0 = time.time()
        np.random.seed(42)
        Ext.monte_carlo_price(100000)
        pricer.MC_Time = time.time() - t0
        # Greeks MC (n plus faible)
        pricer.greeks_mc = Ext.monte_carlo_greeks(10000)

    # --- Binomial ---
    if input_data["BinomCondition"]:
        t0 = time.time()
        binom.price_option(pricer.timeSteps)
        pricer.Binom_price = binom.price
        pricer.Binom_Time = time.time() - t0

        def price_fn(m, p, pr):
            return binom.price_option(pr.timeSteps)
        from Greeks import finite_difference_greeks
        pricer.greeks_binom = finite_difference_greeks(price_fn, mkt, params, pricer)

    # --- Arbre Trinomial (prix principal) ---
    pricer.Convergence = input_data["Convergence"]
    arbre.init(mkt, params, pricer, False)
    results = dict(arbre.price_results)

    if input_data["Convergence"]:
        results["ConvergenceFig"] = arbre.Convergence_fig

    # --- Greeks Trinomiales ---
    if input_data["ComputeGreeks"]:
        # Sauvegarde de l'état pour ne pas casser le résultat principal
        saved_conv = pricer.Convergence
        pricer.Convergence = False
        pricer.greeks_tree = arbre.compute_greeks_tree(params, pricer, mkt)
        pricer.Convergence = saved_conv

    # --- Assemblage du dict de Greeks ---
    results["greeks_tree"] = pricer.greeks_tree
    results["greeks_bs"] = pricer.greeks_bs
    results["greeks_mc"] = pricer.greeks_mc
    results["greeks_binom"] = pricer.greeks_binom

    # --- Strike Study ---
    if input_data["StrikeStudy"]:
        fig = arbre.compute_StrikeStudy(
            10, mkt, pricer, params, Ext,
            binom=binom if input_data["BinomCondition"] else None,
            with_mc=input_data["MCCondition"])
        results["StrikeStudyFig"] = fig

    # --- Comparatifs ---
    if input_data.get("ComparePricesSteps", False):
        comp = _compare_prices_vs_steps(input_data)
        results["ComparePricesSteps"] = comp

    if input_data.get("CompareAmEur", False):
        results["CompareAmEur"] = _compare_american_european(input_data)

    # --- Arbre Trinomial (DataFrame) ---
    if input_data["TreeBol"] and arbre.df is not None:
        results["TreeDf"] = arbre.df

    return results