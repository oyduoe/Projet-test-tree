import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
from Market import Market
from OptionParameters import OptionParameters


class Pricer:
    def __init__(self, input: dict):
        self.timeSteps = input["Ts"]
        self.PricingDate = input["PricingDate"]
        self.display_tree_bool = input["TreeBol"]
        self.PruningTreshold = input["PruningThreshold"]
        self.Convergence = input["Convergence"]
        self.BS_condition = input["BSCondition"]
        self.MC_condition = input["MCCondition"]
        self.gap = input["SpreadError"]
        self.compute_DeltaAndGamma_tree_var = input["ComputeDeltaAndGammaTree"]
        self.compute_Vega_tree_var = input["ComputeVegaTree"]
        self.StrikeStudy = input["StrikeStudy"]

        # Greeks calculées par modèle (remplis par Main)
        self.greeks_tree = {}
        self.greeks_bs = {}
        self.greeks_mc = {}
        self.greeks_binom = {}

        # Prix Binomial
        self.Binom_price = 0.0
        self.Binom_Time = 0.0

        # DataFrame de convergence (classique)
        self.Convergence_df = pd.DataFrame(
            columns=["Time Steps", "Tree Price",
                     "Time to compute Trinomial Tree", "(Tree – BS) x NbSteps"],
            index=range(self.timeSteps),
            dtype=object,
        )

        self.BS_price: float = 0.00
        self.BS_Time: float = 0.00
        self.MC_price: float = 0.00
        self.MC_Time: float = 0.00
        self.newTs: int = 0
        self.DeltaTree: float = 0.0
        self.GammaTree: float = 0.0
        self.VegaTree: float = 0.0

    # ------------------------------------------------------------------
    def compute_perf(self, T: int, price: float, time_calculation: float, bs_price: float):
        self.Convergence_df.loc[T, "Time Steps"] = T
        self.Convergence_df.loc[T, "Tree Price"] = price
        self.Convergence_df.loc[T, "Time to compute Trinomial Tree"] = time_calculation
        self.Convergence_df.loc[T, "(Tree – BS) x NbSteps"] = (price - bs_price) * T

    def create_plot_convergence(self) -> plt:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.Convergence_df["Time Steps"].astype(float),
                self.Convergence_df["(Tree – BS) x NbSteps"].astype(float), marker='o')
        ax.set_title("Convergence vers Black-Scholes en fonction du pas de temps")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("(Tree – BS) x NbSteps")
        ax.grid(True)
        return fig

    def create_plot_StrikeStudy(self, df_strike: DataFrame,
                                initial_strike: float, final_strike: float) -> plt:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ---- Sous-graphe gauche : prix ----
        ax1.plot(df_strike["Strike"].astype(float), df_strike["BS Price"].astype(float),
                 label="BS", color='green', linestyle='-')
        ax1.plot(df_strike["Strike"].astype(float), df_strike["Tree Price"].astype(float),
                 label="Tree", color='orange', linestyle='-')
        if "Binom Price" in df_strike.columns and df_strike["Binom Price"].astype(float).abs().sum() > 0:
            ax1.plot(df_strike["Strike"].astype(float), df_strike["Binom Price"].astype(float),
                     label="Binom", color='blue', linestyle='-')
        if "MC Price" in df_strike.columns and df_strike["MC Price"].astype(float).abs().sum() > 0:
            ax1.plot(df_strike["Strike"].astype(float), df_strike["MC Price"].astype(float),
                     label="Monte Carlo", color='purple', linestyle='-')
        ax1.plot(df_strike["Strike"].astype(float), df_strike["Tree-BS"].astype(float),
                 label="Tree - BS", color='red', linestyle='--')
        if "Tree-Binom" in df_strike.columns:
            ax1.plot(df_strike["Strike"].astype(float), df_strike["Tree-Binom"].astype(float),
                     label="Tree - Binom", color='darkorange', linestyle='--')
        if "Tree-MC" in df_strike.columns:
            ax1.plot(df_strike["Strike"].astype(float), df_strike["Tree-MC"].astype(float),
                     label="Tree - MC", color='magenta', linestyle='--')
        ax1.set_xlim([initial_strike, final_strike])
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Prices / Gaps')
        ax1.grid(True)
        ax1.legend(fontsize=8)
        ax1.set_title("Prices & Gaps vs Strike")

        # ---- Sous-graphe droit : slopes ----
        for col, lbl, c in [("Slope BS", "Slope BS", 'green'),
                            ("Slope Tree", "Slope Tree", 'orange'),
                            ("Slope Binom", "Slope Binom", 'blue'),
                            ("Slope MC", "Slope MC", 'purple')]:
            if col in df_strike.columns:
                vals = df_strike[col].iloc[1:-1].astype(float)
                if abs(vals).sum() > 0:
                    ax2.plot(df_strike["Strike"].iloc[1:-1].astype(float), vals,
                             label=lbl, color=c, linestyle='--')
        ax2.set_xlim([initial_strike, final_strike])
        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Slope')
        ax2.grid(True)
        ax2.legend(fontsize=8)
        ax2.set_title("Slopes vs Strike")

        plt.tight_layout()
        return fig

    def compute_time_steps(self, market: Market, params: OptionParameters) -> int:
        ttm = (params.DateMaturity - self.PricingDate).days / 365
        timeSteps = (3 / (8 * np.sqrt(2 * np.pi))) * (market.SpotPrice / self.gap) * (
                (market.volatility ** ttm) / np.sqrt(np.exp(market.volatility ** ttm) - 1))
        return round(timeSteps)

    def get_Convergence_df(self) -> DataFrame:
        return self.Convergence_df