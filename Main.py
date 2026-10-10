import time
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import timedelta

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
    mc_checked = bool(input_data.get("MCCondition", False))
    binom_checked = bool(input_data.get("BinomCondition", False))

    tree_prices, bs_prices = [], []
    tree_times, bs_times = [], []
    binom_prices, binom_times = [], []
    mc_prices, mc_times = [], []

    for T in step_list:
        local = dict(input_data)
        local["Ts"] = int(T)
        local["Convergence"] = False
        local["StrikeStudy"] = False
        local["TreeBol"] = False
        local["ComparePricesSteps"] = False
        local["CompareAmEur"] = False
        local["CompareDivModes"] = False

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

        if binom_checked:
            t0 = time.time(); bp = binom.price_option(int(T)); binom_times.append(time.time() - t0)
            binom_prices.append(float(bp))

        if mc_checked:
            pricer.timeSteps = min(int(T), 50)
            np.random.seed(42)
            t0 = time.time(); mp = Ext.monte_carlo_price(5000); mc_times.append(time.time() - t0)
            mc_prices.append(float(mp))

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(step_list, tree_prices, marker='o', label='Trinomial')
    ax1.plot(step_list, bs_prices, marker='^', linestyle='--', label='Black-Scholes')
    if binom_checked:
        ax1.plot(step_list, binom_prices, marker='s', label='Binomial')
    if mc_checked:
        ax1.plot(step_list, mc_prices, marker='x', linestyle=':', label='Monte Carlo')
    ax1.set_xlabel("Time Steps"); ax1.set_ylabel("Option Price")
    ax1.set_title("Prix vs Steps"); ax1.grid(True); ax1.legend()

    gaps_bs = [t - b for t, b in zip(tree_prices, bs_prices)]
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(step_list, gaps_bs, marker='o', label='Tree - BS')
    if binom_checked:
        gaps_bn = [t - b for t, b in zip(tree_prices, binom_prices)]
        ax2.plot(step_list, gaps_bn, marker='s', label='Tree - Binomial')
    if mc_checked:
        gaps_mc = [t - m for t, m in zip(tree_prices, mc_prices)]
        ax2.plot(step_list, gaps_mc, marker='x', label='Tree - MC')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel("Time Steps"); ax2.set_ylabel("Gap")
    ax2.set_title("Gaps vs Steps"); ax2.grid(True); ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(step_list, tree_times, marker='o', label='Trinomial')
    ax3.plot(step_list, bs_times, marker='^', linestyle='--', label='Black-Scholes')
    if binom_checked:
        ax3.plot(step_list, binom_times, marker='s', label='Binomial')
    if mc_checked:
        ax3.plot(step_list, mc_times, marker='x', linestyle=':', label='Monte Carlo')
    ax3.set_xlabel("Time Steps"); ax3.set_ylabel("Temps (s)")
    ax3.set_title("Temps de calcul vs Steps"); ax3.grid(True); ax3.legend()

    return {"prices_vs_steps": fig1, "gaps_vs_steps": fig2, "time_vs_steps": fig3}


def _compare_american_european(input_data):
    """Gap American - European : uniquement Tree et Binomial."""
    def price_one(exercise):
        local = dict(input_data)
        local["Exercice"] = exercise
        local["Convergence"] = False
        local["StrikeStudy"] = False
        local["TreeBol"] = False
        local["ComparePricesSteps"] = False
        local["CompareAmEur"] = False
        local["CompareDivModes"] = False

        mkt = Market(local)
        params = OptionParameters(local)
        pricer = Pricer(local)
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
        return {"Tree": tree_p, "Binomial": binom_p}

    eu = price_one("European")
    am = price_one("American")
    models = ["Tree", "Binomial"]
    gaps = [am[m] - eu[m] for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, gaps, color=['orange', 'blue'])
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Prix American - Prix European")
    ax.set_title("Gap American vs European (modèles avec exercice anticipé)")
    ax.grid(True, axis='y')
    for b, g in zip(bars, gaps):
        ax.text(b.get_x() + b.get_width() / 2, g, f"{g:.4f}",
                ha='center', va='bottom' if g >= 0 else 'top', fontsize=10)
    plt.tight_layout()
    return fig


def _compare_dividend_modes(input_data, n_points=12):
    """Grille 2x2 : chaque modèle montre Discrete vs Continuous vs No-div.

    Renvoie None si Dividend <= 0 (aucune différence entre les modes).
    """
    D = float(input_data.get("Dividend", 0.0))
    if D <= 0:
        return None

    PD = input_data["PricingDate"]
    MD = input_data["Maturity"]
    total_days = (MD - PD).days
    if total_days <= 0:
        return None

    fractions = np.linspace(0.05, 0.95, n_points)
    ex_days = [max(1, int(round(f * total_days))) for f in fractions]

    # Pour chaque modèle on stocke deux listes : discrete / continuous
    prices = {
        "Binomial":      {"disc": [], "cont": []},
        "Trinomial":     {"disc": [], "cont": []},
        "Black-Scholes": {"disc": [], "cont": []},
        "Monte Carlo":   {"disc": [], "cont": []},
    }

    def _price_all(local):
        mkt = Market(local)
        params = OptionParameters(local)
        pricer = Pricer(local)
        Ext = Extension(mkt, params, pricer)
        binom = Binomial(mkt, params, pricer)
        arbre = Tree()

        # Neutralise tout ce qui n'est pas le prix
        pricer.Convergence = False
        pricer.display_tree_bool = False
        pricer.BS_condition = False
        pricer.MC_condition = False
        pricer.compute_DeltaAndGamma_tree_var = False
        pricer.Binom_condition = False

        # Binomial
        p_binom = float(binom.price_option(pricer.timeSteps))

        # Trinomial
        arbre.init(mkt, params, pricer, True)
        p_tree = float(arbre.Root_Node.OptPrice)

        # Black-Scholes
        bs = Ext.black_sholes()
        p_bs = float(bs["BS Price"])

        # Monte Carlo (modéré pour la rapidité)
        np.random.seed(42)
        p_mc = float(Ext.monte_carlo_price(3000))

        return p_binom, p_tree, p_bs, p_mc

    for d in ex_days:
        for mode in ("Discrete", "Continuous"):
            local = dict(input_data)
            local["DivExDate"] = PD + timedelta(days=d)
            local["DividendType"] = mode
            local["Convergence"] = False
            local["StrikeStudy"] = False
            local["TreeBol"] = False
            local["ComparePricesSteps"] = False
            local["CompareAmEur"] = False
            local["CompareDivModes"] = False

            b, t, s, m = _price_all(local)
            key = "disc" if mode == "Discrete" else "cont"
            prices["Binomial"][key].append(b)
            prices["Trinomial"][key].append(t)
            prices["Black-Scholes"][key].append(s)
            prices["Monte Carlo"][key].append(m)

    # Référence sans dividende
    local0 = dict(input_data)
    local0["Dividend"] = 0.0
    local0["Convergence"] = False
    local0["StrikeStudy"] = False
    local0["TreeBol"] = False
    local0["ComparePricesSteps"] = False
    local0["CompareAmEur"] = False
    local0["CompareDivModes"] = False
    ref_b, ref_t, ref_s, ref_m = _price_all(local0)
    refs = {"Binomial": ref_b, "Trinomial": ref_t,
            "Black-Scholes": ref_s, "Monte Carlo": ref_m}

    # ---- Grille 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    order = ["Binomial", "Trinomial", "Black-Scholes", "Monte Carlo"]

    for ax, name in zip(axes.flat, order):
        ax.plot(ex_days, prices[name]["disc"], marker='o',
                label='Discrete (cash)')
        ax.plot(ex_days, prices[name]["cont"], marker='s',
                label='Continuous (yield q=D/S)')
        ax.axhline(refs[name], color='gray', linestyle=':',
                   label=f'No dividend ({refs[name]:.4f})')
        ax.set_xlabel("Ex-dividend date (days from pricing date)")
        ax.set_ylabel("Option price")
        ax.set_title(name)
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.suptitle(f"Discrete vs Continuous dividend — D = {D:.2f} EUR",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    return fig


# ======================================================================
# POINT D'ENTRÉE
# ======================================================================
def run_calculations(input_data: dict) -> dict:
    mc_checked = bool(input_data["MCCondition"])
    binom_checked = bool(input_data["BinomCondition"])

    mkt = Market(input_data)
    params = OptionParameters(input_data)
    pricer = Pricer(input_data)
    Ext = Extension(mkt, params, pricer)
    binom = Binomial(mkt, params, pricer)
    pricer.Binom_condition = binom_checked

    # ---- BS (toujours) ----
    t0 = time.time()
    bs_res = Ext.black_sholes()
    pricer.BS_Time = time.time() - t0
    pricer.BS_Greeks_Time = pricer.BS_Time
    bs_greeks_all = {k: float(bs_res[k])
                     for k in ("Delta", "Gamma", "Vega", "Vomma", "Vanna", "Theta")
                     if k in bs_res}

    # ---- Monte Carlo ----
    if mc_checked:
        np.random.seed(42)
        t0 = time.time()
        Ext.monte_carlo_price(50000)
        pricer.MC_Time = time.time() - t0

    # ---- Binomial ----
    if binom_checked:
        t0 = time.time()
        binom.price_option(pricer.timeSteps)
        pricer.Binom_Time = time.time() - t0
        pricer.Binom_price = float(binom.price)

    # ---- Arbre principal ----
    arbre = Tree()
    if input_data["Convergence"]:
        arbre.init(mkt, params, pricer, False, ext=Ext, binom=binom)
    else:
        arbre.init(mkt, params, pricer, False)

    results = dict(arbre.price_results)
    pricing_time_tree = results.get("TimePricing", 0.0)
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
    t0 = time.time()
    pricer.greeks_tree = arbre.compute_greeks_tree(params, pricer, mkt)
    pricer.Tree_Greeks_Time = pricing_time_tree + (time.time() - t0)
    results["Tree_Greeks_Time"] = pricer.Tree_Greeks_Time

    pricer.greeks_bs = bs_greeks_all
    results["BS_Greeks_Time"] = pricer.BS_Greeks_Time

    if binom_checked:
        def binom_price_fn(m, p, pr):
            return binom.price_option(pr.timeSteps)
        t0 = time.time()
        pricer.greeks_binom = finite_difference_greeks(binom_price_fn, mkt, params, pricer)
        pricer.Binom_Greeks_Time = pricer.Binom_Time + (time.time() - t0)
        results["Binom_Greeks_Time"] = pricer.Binom_Greeks_Time
    else:
        pricer.greeks_binom = {}

    if mc_checked:
        try:
            t0 = time.time()
            pricer.greeks_mc = Ext.monte_carlo_greeks(10000)
            pricer.MC_Greeks_Time = pricer.MC_Time + (time.time() - t0)
            results["MC_Greeks_Time"] = pricer.MC_Greeks_Time
        except Exception:
            pricer.greeks_mc = {}
    else:
        pricer.greeks_mc = {}

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

    if input_data.get("CompareDivModes", False):
        try:
            fig = _compare_dividend_modes(input_data)
            results["CompareDivModesFig"] = fig  # peut être None si D=0
        except Exception as e:
            results["CompareDivModesError"] = f"{e}\n{traceback.format_exc()}"

    # ---- SVG tree ----
    if input_data["TreeBol"] and arbre.svg_str is not None:
        results["TreeSvg"] = arbre.svg_str

    return results