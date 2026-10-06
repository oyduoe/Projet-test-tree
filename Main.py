import time
from Market import Market
from OptionParameters import OptionParameters
from Tree import Tree
from Pricer import Pricer
from Extension import Extension


def run_calculations(input_data: dict) -> dict:
    """Point d'entrée appelé par App.py (Streamlit)."""
    # Instancier les objets
    mkt = Market(input_data)
    params = OptionParameters(input_data)
    arbre = Tree()
    pricer = Pricer(input_data)
    Ext = Extension(mkt, params, pricer)

    # Black & Scholes
    if input_data["BSCondition"]:
        start_time = time.time()
        Ext.black_sholes()
        end_time = time.time()
        pricer.BS_Time = end_time - start_time

    # Monte Carlo
    if input_data["MCCondition"]:
        start_timeMC = time.time()
        Ext.monte_carlo_price(100000)  # Nombre de simulations fixé à 100 000
        end_timeMC = time.time()
        pricer.MC_Time = end_timeMC - start_timeMC

    # Lancement du programme de l'arbre Trinomial
    arbre.init(mkt, params, pricer, False)
    results = dict(arbre.price_results)

    # Convergence vers Black & Scholes
    if input_data["Convergence"]:
        results["ConvergenceFig"] = arbre.Convergence_fig

    # ---- CORRECTION ----
    # On retire le "and not input_data['Convergence']" pour que Delta/Gamma
    # et Vega soient également calculés lorsque Convergence est coché.
    # Les attributs pricer.DeltaTree / pricer.GammaTree sont peuplés à chaque
    # itération dans Tree.build_tree et reflètent donc le dernier arbre calculé.
    if input_data["ComputeDeltaAndGammaTree"]:
        results["Delta"] = pricer.DeltaTree
        results["Gamma"] = pricer.GammaTree
    if input_data["ComputeVegaTree"]:
        results["Vega"] = arbre.compute_vega_tree(params, pricer, mkt)
    # --------------------

    # Étude du prix par rapport au strike
    if input_data["StrikeStudy"]:
        results["StrikeStudyFig"] = arbre.compute_StrikeStudy(10, mkt, pricer, params, Ext)

    # Récupération du DataFrame de l'arbre (au lieu de l'export Excel)
    if input_data["TreeBol"] and arbre.df is not None:
        results["TreeDf"] = arbre.df

    return results