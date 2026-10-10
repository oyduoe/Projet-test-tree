import time
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Market import Market
from OptionParameters import OptionParameters
from Tree import Tree
from Pricer import Pricer
from Extension import Extension
from Binomial import Binomial
from Greeks import finite_difference_greeks


# ======================================================================
# COMPARATIFS
# ======================================================================
def _compare_prices_vs_steps(input_data, step_list=(5, 10, 20, 40, 80)):
    tree_prices, bs_prices, binom_prices, mc_prices = [], [], [], []
    tree_times, bs_times, binom_times, mc_times = [], [], [], []

    for T in step_list:
        local = dict(input_data)
        local["Ts"] = int(T)
        local["Convergence"] = False
        local["StrikeStudy"] = False
        local["TreeBol"] = False
        local["ComparePricesSteps"] = False
        local["CompareAmEur"] = False

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
        pricer.Binom_condition = False

        t0 = time.time(); bs_res = Ext.black_sholes(); bs_times.append(time.time() - t0)
        bs_prices.append(float(bs_res["BS Price"]))

        t0 = time.time(); arbre.init(mkt, params, pricer, True); tree_times.append(time.time() - t0)
        tree_prices.append(float(arbre.Root_Node.OptPrice))

        t0 = time.time(); bp = binom.price_option(int(T)); binom_times.append(time.time() - t0)
        binom_prices.append(float(bp))

        pricer.timeSteps = min(int(T), 50)
        np.random.seed(42)
        t0 = time.time(); mp = Ext.monte_carlo_price(5000); mc_times.append(time.time() - t0)
        mc_prices.append(float(mp))

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(step_list, tree_prices, marker='o', label='Trinomial')
    ax1.plot(step_list, binom_prices, marker='s', label='Binomial')
    ax1.plot(step_list, bs_prices, marker='^', linestyle='--', label='Black-Scholes')
    ax1.plot(step_list, mc_prices, marker='x', linestyle=':', label='Monte Carlo')
    ax1.set_xlabel("Time Steps"); ax1.set_ylabel("Option Price")
    ax1.set_title("Prix vs Steps"); ax1.grid(True); ax1.legend()

    gaps_bs = [t - b for t, b in zip(tree_prices, bs_prices)]
    gaps_bn = [t - b for t, b in zip(tree_prices, binom_prices)]
    gaps_mc = [t - m for t, m in zip(tree_prices, mc_prices)]
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(step_list, gaps_bs, marker='o', label='Tree - BS')
    ax2.plot(step_list, gaps_bn, marker='s', label='Tree - Binomial')
    ax2.plot(step_list, gaps_mc, marker='x', label='Tree - MC')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel("Time Steps"); ax2.set_ylabel("Gap")
    ax2.set_title("Gaps vs Steps"); ax2.grid(True); ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(step_list, tree_times, marker='o', label='Trinomial')
    ax3.plot(step_list, binom_times, marker='s', label='Binomial')
    ax3.plot(step_list, bs_times, marker='^', linestyle='--', label='Black-Scholes')
    ax3.plot(step_list, mc_times, marker='x', linestyle=':', label='Monte Carlo')
    ax3.set_xlabel("Time Steps"); ax3.set_ylabel("Temps (s)")
    ax3.set_title("Temps de calcul vs Steps"); ax3.grid(True); ax3.legend()

    return {"prices_vs_steps": fig1, "gaps_vs_steps": fig2, "time_vs_steps": fig3}


def _compare_american_european(input_data):
    """Gap American - European par modèle.

    Attention :
      - Tree et Binomial modélisent tous les deux l'exercice anticipé.
      - Black-Scholes est intrinsèquement européen (l'option "American" n'a
        aucun effet) → gap = 0.
      - Monte Carlo dans ce projet ne gère pas l'exercice anticipé → gap = 0.
    """
    def price_one(exercise):
        local = dict(input_data)
        local["Exercice"] = exercise
        local["Convergence"] = False
        local["StrikeStudy"] = False
        local["TreeBol"] = False
        local["ComparePricesSteps"] = False
        local["CompareAmEur"] = False

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
        tree_p = float(arbre.Root_Node.OptPrice)
        binom_p = float(binom.price_option(pricer.timeSteps))
        bs_p = float(Ext.black_sholes()["BS Price"])
        np.random.seed(42)
        mc_p = float(Ext.monte_carlo_price(5000))
        return {"Tree": tree_p, "Binomial": binom_p, "BS": bs_p, "MC": mc_p}

    eu = price_one("European")
    am = price_one("American")
    models = ["Tree", "Binomial", "BS", "MC"]
    gaps = [am[m] - eu[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, gaps, color=['orange', 'blue', 'green', 'purple'])
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Prix American - Prix European")
    ax.set_title("Gap American vs European par modèle\n"
                 "(BS et MC ne modélisent pas l'exercice anticipé → gap nul)")
    ax.grid(True, axis='y')

    # Annoter les valeurs
    for b, g in zip(bars, gaps):
        ax.text(b.get_x() + b.get_width() / 2,
                g, f"{g:.4f}",
                ha='center',
                va='bottom' if g >= 0 else 'top',
                fontsize=9)
    plt.tight_layout()
    return fig


# ======================================================================
# POINT D'ENTRÉE
# ======================================================================
def run_calculations(input_data: dict) -> dict:
    bs_checked = bool(input_data["BSCondition"])
    mc_checked = bool(input_data["MCCondition"])
    binom_checked = bool(input_data["BinomCondition"])

    mkt = Market(input_data)
    params = OptionParameters(input_data)
    pricer = Pricer(input_data)
    Ext = Extension(mkt, params, pricer)
    binom = Binomial(mkt, params, pricer)
    pricer.Binom_condition = binom_checked

    # ---- BS toujours calculé (présent dans tous les graphes) ----
    t0 = time.time()
    bs_res = Ext.black_sholes()
    pricer.BS_Time = time.time() - t0
    bs_greeks_all = {k: float(bs_res[k])
                     for k in ("Delta", "Gamma", "Vega", "Vomma", "Vanna", "Theta")
                     if k in bs_res}

    # ---- MC seulement si coché ----
    if mc_checked:
        np.random.seed(42)
        t0 = time.time()
        Ext.monte_carlo_price(50000)
        pricer.MC_Time = time.time() - t0

    # ---- Binomial seulement si coché ----
    if binom_checked:
        t0 = time.time()
        binom.price_option(pricer.timeSteps)
        pricer.Binom_price = float(binom.price)
        pricer.Binom_Time = time.time() - t0

    # ---- Arbre principal ----
    arbre = Tree()
    if input_data["Convergence"]:
        # On passe ext et binom pour que la convergence les calcule
        arbre.init(mkt, params, pricer, False, ext=Ext, binom=binom)
    else:
        arbre.init(mkt, params, pricer, False)

    results = dict(arbre.price_results)
    saved_df = arbre.df

    tp = results.get("TreePrice", 0.0)
    results["BS Price"] = pricer.BS_price
    results["BS_Time"] = pricer.BS_Time
    results["TreeGapBS"] = tp - pricer.BS_price
    if mc_checked:
        results["MC Price"] = pricer.MC_price
        results["MCTime"] = pricer.MC_Time
        results["MCGapTree"] = pricer.MC_price - tp
    if binom_checked:
        results["Binom Price"] = pricer.Binom_price
        results["Binom_Time"] = pricer.Binom_Time
        results["TreeGapBinom"] = tp - pricer.Binom_price

    if input_data["Convergence"]:
        results["ConvergenceFig"] = arbre.Convergence_fig

    # ---- Greeks ----
    pricer.greeks_tree = arbre.compute_greeks_tree(params, pricer, mkt)
    pricer.greeks_bs = bs_greeks_all if bs_checked else {}

    if mc_checked:
        try:
            pricer.greeks_mc = Ext.monte_carlo_greeks(10000)
        except Exception:
            pricer.greeks_mc = {}
    else:
        pricer.greeks_mc = {}

    if binom_checked:
        def binom_price_fn(m, p, pr):
            return binom.price_option(pr.timeSteps)
        pricer.greeks_binom = finite_difference_greeks(binom_price_fn, mkt, params, pricer)
    else:
        pricer.greeks_binom = {}

    results["greeks_tree"] = pricer.greeks_tree
    results["greeks_bs"] = pricer.greeks_bs
    results["greeks_mc"] = pricer.greeks_mc
    results["greeks_binom"] = pricer.greeks_binom

    # ---- Strike Study ----
    if input_data["StrikeStudy"]:
        try:
            fig = arbre.compute_StrikeStudy(
                10, mkt, pricer, params, Ext,
                binom=binom if binom_checked else None,
                with_mc=mc_checked)
            results["StrikeStudyFig"] = fig
        except Exception as e:
            results["StrikeStudyError"] = f"{e}\n{traceback.format_exc()}"

    # ---- Comparatifs ----
    if input_data.get("ComparePricesSteps", False):
        try:
            results["ComparePricesSteps"] = _compare_prices_vs_steps(input_data)
        except Exception as e:
            results["ComparePricesStepsError"] = f"{e}\n{traceback.format_exc()}"

    if input_data.get("CompareAmEur", False):
        try:
            results["CompareAmEur"] = _compare_american_european(input_data)
        except Exception as e:
            results["CompareAmEurError"] = f"{e}\n{traceback.format_exc()}"

    # ---- Arbre Trinomial DataFrame ----
    if input_data["TreeBol"] and saved_df is not None:
        results["TreeDf"] = saved_df

    return results