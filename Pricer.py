import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Node import Node


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
        self.StrikeStudy = input["StrikeStudy"]

        self.TreeType = input.get("TreeType", "Auto (Binomial → Trinomial si dividende)")
        self.compute_Greeks = input.get("ComputeGreeks", False)
        # Aliases (compat avec l'ancien code)
        self.compute_DeltaAndGamma_tree_var = self.compute_Greeks
        self.compute_Vega_tree_var = self.compute_Greeks

        # Mode debug : bascule automatiquement les compteurs sur Node
        self.debug_mode = input.get("DebugMode", False)
        Node.COUNTING = self.debug_mode
        if self.debug_mode:
            Node.reset_counters()

        self.Convergence_df = pd.DataFrame(
            columns=["Time Steps", "Tree Price",
                     "Time to compute Trinomial Tree", "(Tree – BS) x NbSteps"],
            index=range(self.timeSteps), dtype=object,
        )

        self.BS_price = 0.0
        self.BS_Time = 0.0
        self.MC_price = 0.0
        self.MC_Time = 0.0
        self.newTs = 0
        self.DeltaTree = 0.0
        self.GammaTree = 0.0
        self.VegaTree = 0.0

    def compute_perf(self, T, price, time_calculation, bs_price):
        self.Convergence_df.loc[T, "Time Steps"] = T
        self.Convergence_df.loc[T, "Tree Price"] = price
        self.Convergence_df.loc[T, "Time to compute Trinomial Tree"] = time_calculation
        self.Convergence_df.loc[T, "(Tree – BS) x NbSteps"] = (price - bs_price) * T

    def create_plot_convergence(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.Convergence_df["Time Steps"],
                self.Convergence_df["(Tree – BS) x NbSteps"], marker='o')
        ax.set_title("Convergence vers Black-Scholes en fonction du pas de temps")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("(Tree – BS) x NbSteps")
        ax.grid(True)
        return fig

    def create_plot_StrikeStudy(self, df_strike, initial_strike, final_strike):
        fig, ax1 = plt.subplots()
        ax1.plot(df_strike["Strike"].astype(float), df_strike["BS Price"].astype(float),
                 label="BS", color='green', linestyle='-')
        ax1.plot(df_strike["Strike"].astype(float), df_strike["Tree Price"].astype(float),
                 label="Tree", color='orange', linestyle='-')
        ax1.plot(df_strike["Strike"].astype(float), df_strike["Tree-BS"].astype(float),
                 label="Tree - BS", color='red', linestyle='-')
        ax1.set_xlim([initial_strike, final_strike])
        ax1.set_xlabel('Strike'); ax1.set_ylabel('Prices', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(df_strike["Strike"].iloc[1:-1].astype(float),
                 df_strike["Slope BS"].iloc[1:-1].astype(float),
                 label="Slope BS", color='green', linestyle='--')
        ax2.plot(df_strike["Strike"].iloc[1:-1].astype(float),
                 df_strike["Slope Tree"].iloc[1:-1].astype(float),
                 label="Slope Tree", color='orange', linestyle='--')
        ax2.set_ylabel('Slope', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
        plt.title("Tree and Black-Scholes prices as functions of the strike")
        return fig

    def compute_time_steps(self, market, params) -> int:
        ttm = (params.DateMaturity - self.PricingDate).days / 365
        sigma = market.volatility
        timeSteps = (3 / (8 * np.sqrt(2 * np.pi))) * (market.SpotPrice / self.gap) * \
                    (sigma ** 2 * ttm) / np.sqrt(np.exp(sigma ** 2 * ttm) - 1)
        return max(1, round(timeSteps))

    def get_Convergence_df(self):
        return self.Convergence_df