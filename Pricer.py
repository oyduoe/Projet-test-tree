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

        self.greeks_tree = {}
        self.greeks_bs = {}
        self.greeks_mc = {}
        self.greeks_binom = {}

        self.Binom_price = 0.0
        self.Binom_Time = 0.0
        self.Binom_condition = False

        # Temps "prix + greeks" par modèle
        self.Tree_Greeks_Time: float = 0.0
        self.BS_Greeks_Time: float = 0.0
        self.Binom_Greeks_Time: float = 0.0
        self.MC_Greeks_Time: float = 0.0

        self.Convergence_df = pd.DataFrame(
            columns=["Time Steps", "Tree Price",
                     "Time to compute Trinomial Tree",
                     "(Tree – BS) x NbSteps",
                     "(Tree – MC) x NbSteps",
                     "(Tree – Binom) x NbSteps"],
            index=range(1, self.timeSteps + 1),
            dtype=object)

        self.BS_price: float = 0.00
        self.BS_Time: float = 0.00
        self.MC_price: float = 0.00
        self.MC_Time: float = 0.00
        self.newTs: int = 0
        self.DeltaTree: float = 0.0
        self.GammaTree: float = 0.0
        self.VegaTree: float = 0.0

    # ------------------------------------------------------------------
    def compute_perf(self, T, price, time_calculation, bs_price,
                     mc_price=None, binom_price=None):
        self.Convergence_df.loc[T, "Time Steps"] = T
        self.Convergence_df.loc[T, "Tree Price"] = price
        self.Convergence_df.loc[T, "Time to compute Trinomial Tree"] = time_calculation
        self.Convergence_df.loc[T, "(Tree – BS) x NbSteps"] = (price - bs_price) * T
        if mc_price is not None:
            self.Convergence_df.loc[T, "(Tree – MC) x NbSteps"] = (price - mc_price) * T
        if binom_price is not None:
            self.Convergence_df.loc[T, "(Tree – Binom) x NbSteps"] = (price - binom_price) * T

    def create_plot_convergence(self):
        df = self.Convergence_df.copy().dropna(subset=["Time Steps"])
        x = pd.to_numeric(df["Time Steps"], errors='coerce')
        y_bs = pd.to_numeric(df["(Tree – BS) x NbSteps"], errors='coerce')

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, y_bs, marker='o', label='Tree - BS')

        for col, lbl, marker in [("(Tree – Binom) x NbSteps", "Tree - Binomial", 's'),
                                 ("(Tree – MC) x NbSteps", "Tree - Monte Carlo", 'x')]:
            if col in df.columns:
                y = pd.to_numeric(df[col], errors='coerce')
                if y.notna().any():
                    ax.plot(x, y, marker=marker, label=lbl)

        ax.set_title("Convergence des modèles vs Black-Scholes")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("(Tree – Modèle) x NbSteps")
        ax.grid(True)
        ax.legend()
        return fig

    # ------------------------------------------------------------------
    def create_plot_StrikeStudy(self, df_strike, initial_strike, final_strike):
        """Un seul graphe : prix + gaps (axe gauche) et slopes (axe droit)."""
        fig, ax1 = plt.subplots(figsize=(12, 6))

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
        ax1.set_ylabel('Prices & Gaps', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        ax2 = ax1.twinx()
        for col, lbl, c in [("Slope BS", "Slope BS", 'green'),
                            ("Slope Tree", "Slope Tree", 'orange'),
                            ("Slope Binom", "Slope Binom", 'blue'),
                            ("Slope MC", "Slope MC", 'purple')]:
            if col in df_strike.columns:
                vals = df_strike[col].astype(float)
                if vals.abs().sum() > 0:
                    ax2.plot(df_strike["Strike"].astype(float), vals,
                             label=lbl, color=c, linestyle=':')
        ax2.set_ylabel('Slope', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8, ncol=2)
        plt.title("Prices, Gaps & Slopes vs Strike")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def compute_time_steps(self, market, params):
        ttm = (params.DateMaturity - self.PricingDate).days / 365
        timeSteps = (3 / (8 * np.sqrt(2 * np.pi))) * (market.SpotPrice / self.gap) * (
                (market.volatility ** ttm) / np.sqrt(np.exp(market.volatility ** ttm) - 1))
        return round(timeSteps)

    def get_Convergence_df(self):
        return self.Convergence_df