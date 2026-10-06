import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
from Market import Market
from OptionParameters import OptionParameters


class Pricer:
    def __init__(self, input: dict):
        """Initialiser les paramètres du pricer à partir d'un dictionnaire d'input."""
        self.timeSteps = input["Ts"]  # Nombre de pas de temps
        self.PricingDate = input["PricingDate"]  # Date initiale
        self.display_tree_bool = input["TreeBol"]  # Condition d'affichage de l'arbre
        self.PruningTreshold = input["PruningThreshold"]  # Seuil pour le pruning
        self.Convergence = input["Convergence"]  # Condition pour la méthode de Convergence
        self.BS_condition = input["BSCondition"]  # Condition pour le pricing Black & Scholes
        self.MC_condition = input["MCCondition"]  # Condition pour le pricing Monte Carlo
        self.gap = input["SpreadError"]  # Erreur maximale demandée par l'utilisateur
        self.compute_DeltaAndGamma_tree_var = input["ComputeDeltaAndGammaTree"]  # Condition pour calculer le Delta et Gamma
        self.compute_Vega_tree_var = input["ComputeVegaTree"]  # Condition pour calculer le Vega
        self.StrikeStudy = input["StrikeStudy"]

        # ---- CORRECTION ----
        # Ajout de dtype=object pour éviter que pandas 2.x n'impose un StringDtype
        # sur les colonnes vides, ce qui provoquerait une erreur lors de
        # l'affectation de valeurs float dans les cellules.
        self.Convergence_df = pd.DataFrame(
            columns=["Time Steps", "Tree Price",
                     "Time to compute Trinomial Tree", "(Tree – BS) x NbSteps"],
            index=range(self.timeSteps),
            dtype=object,
        )
        # --------------------

        self.BS_price: float = 0.00  # Prix avec Black & Scholes
        self.BS_Time: float = 0.00   # Temps de calcul avec Black & Scholes
        self.MC_price: float = 0.00  # Prix avec Monte Carlo
        self.MC_Time: float = 0.00   # Temps de calcul avec Monte Carlo
        self.newTs: int = 0          # Nouveau pas de temps si l'on price avec le spread error
        self.DeltaTree: float
        self.GammaTree: float
        self.VegaTree: float

    def compute_perf(self, T: int, price: float, time_calculation: float, bs_price: float):
        """Affiche les résultats de l'arbre qui vient d'être modélisé."""
        # Ajout des résultats au DataFrame
        self.Convergence_df.loc[T, "Time Steps"] = T
        self.Convergence_df.loc[T, "Tree Price"] = price
        self.Convergence_df.loc[T, "Time to compute Trinomial Tree"] = time_calculation
        self.Convergence_df.loc[T, "(Tree – BS) x NbSteps"] = (price - bs_price) * T

    def create_plot_convergence(self) -> plt:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.Convergence_df["Time Steps"],
                self.Convergence_df["(Tree – BS) x NbSteps"], marker='o')

        # Personnalisation du graphique
        ax.set_title("Convergence vers Black-Scholes en fonction du pas de temps")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("(Tree – BS) x NbSteps")
        ax.grid(True)

        return fig  # Retourner l'objet Figure

    def create_plot_StrikeStudy(self, df_strike: DataFrame,
                                initial_strike: float, final_strike: float) -> plt:
        # Création d'une figure et d'axes partagés pour avoir les deux échelles
        fig, ax1 = plt.subplots()
        # Graphique des prix de Tree et Black-Scholes en fonction des strikes
        ax1.plot(df_strike["Strike"].astype(float),
                 df_strike["BS Price"].astype(float),
                 label="BS", color='green', linestyle='-')
        ax1.plot(df_strike["Strike"].astype(float),
                 df_strike["Tree Price"].astype(float),
                 label="Tree", color='orange', linestyle='-')
        ax1.plot(df_strike["Strike"].astype(float),
                 df_strike["Tree-BS"].astype(float),
                 label="Tree - BS", color='red', linestyle='-')

        ax1.set_xlim([initial_strike, final_strike])

        # étiquettes et titre pour le premier axe
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Prices', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Deuxième axe pour les pentes (slopes)
        ax2 = ax1.twinx()
        ax2.plot(df_strike["Strike"].iloc[1:-1].astype(float),
                 df_strike["Slope BS"].iloc[1:-1].astype(float),
                 label="Slope BS", color='green', linestyle='--')
        ax2.plot(df_strike["Strike"].iloc[1:-1].astype(float),
                 df_strike["Slope Tree"].iloc[1:-1].astype(float),
                 label="Slope Tree", color='orange', linestyle='--')

        # étiquettes et titre pour le deuxième axe
        ax2.set_ylabel('Slope', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)  # Légende combinée
        plt.title("Tree and Black-Scholes prices as functions of the strike")  # titre du graphique

        return fig

    def compute_time_steps(self, market: Market, params: OptionParameters) -> int:
        """Détermine le nombre de pas de temps nécessaire au respect de l'erreur maximale."""
        ttm = (params.DateMaturity - self.PricingDate).days / 365  # Time to Maturity
        # Formule fermée pour déterminer le nombre de pas de temps
        timeSteps = (3 / (8 * np.sqrt(2 * np.pi))) * (market.SpotPrice / self.gap) * (
                (market.volatility ** ttm) / np.sqrt(np.exp(market.volatility ** ttm) - 1))
        return round(timeSteps)

    def get_Convergence_df(self) -> DataFrame:
        """Retourne le DataFrame des Convergences."""
        return self.Convergence_df