import math
import time
import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from Node import Node
from Market import Market
from OptionParameters import OptionParameters
from Pricer import Pricer
from Extension import Extension
from Greeks import finite_difference_greeks


MAX_SVG_DEPTH = 40


class Tree:
    def __init__(self):
        self.dt: float = 0.0
        self.Alpha: float = 0.0
        self.Root_Node: Node = None
        self.Market: Market = None
        self.candidateMid: Node = None
        self.Final_Node: Node = None
        self.tronc: Node = None
        self.ModifExMid = False
        self.isTrunc: bool = True
        self.Convergence_results: DataFrame = None
        self.price_results = {}
        self.nnext: Node = None
        self.nUp: Node = None
        self.nDown: Node = None
        self.df: DataFrame = None
        self.svg_str: str = None
        self._tree_snapshot: dict = None
        self.Convergence_fig = None
        self.time_calculation_tree = 0.00

    # ------------------------------------------------------------------
    def init(self, mkt, params, pricer, IsVega, ext=None, binom=None):
        if pricer.Convergence:
            for T in range(1, pricer.timeSteps + 1):
                print(f"Calcul du prix pour {T} Time Steps")
                self.build_tree(mkt, params, pricer, T, pricer.BS_price, True, IsVega,
                                ext=ext, binom=binom)
            self.Convergence_fig = pricer.create_plot_convergence()
        else:
            if pricer.gap != 0:
                pricer.newTs = pricer.compute_time_steps(mkt, params)
                self.build_tree(mkt, params, pricer, pricer.newTs, pricer.BS_price, False, IsVega)
            else:
                self.build_tree(mkt, params, pricer, pricer.timeSteps, pricer.BS_price, False, IsVega)

    def make_node(self, market_values):
        new_node = Node()
        new_node.init_node(market_values)
        return new_node

    def init_root_node_and_parameters(self, mkt, parameters, pricer, T):
        self.dt = ((parameters.DateMaturity - pricer.PricingDate).days / T) / 365
        self.Alpha = math.exp(mkt.volatility * math.sqrt(3 * self.dt))
        self.Root_Node = self.make_node(mkt.SpotPrice)
        if pricer.compute_DeltaAndGamma_tree_var:
            NodeCalcul = self.make_node(mkt.SpotPrice)
            NodeCalcul.Next_Mid = self.make_node(NodeCalcul.forward(mkt, self.dt, 0))
            self.compute_proba(NodeCalcul, NodeCalcul.Next_Mid, mkt, 0)
            self.Root_Node.UpNode = self.make_node(mkt.SpotPrice * self.Alpha)
            self.Root_Node.DownNode = self.make_node(mkt.SpotPrice / self.Alpha)
            self.Root_Node.Cum_Proba = NodeCalcul.Proba_Mid
            self.Root_Node.UpNode.Cum_Proba = NodeCalcul.Proba_Up
            self.Root_Node.DownNode.Cum_Proba = NodeCalcul.Proba_Down
        else:
            self.Root_Node.Cum_Proba = 1
        self.candidateMid = self.Root_Node

    # ------------------------------------------------------------------
    def _snapshot_tree(self, max_depth):
        """Capture la structure de l'arbre AVANT pricing.
        Renvoie {"columns": [[node, ...], ...], "edges": [(parent, attr, child), ...]}.
        Limité à max_depth + 1 colonnes.
        """
        root = self.Root_Node
        if root is None:
            return None

        columns = [[root]]
        edges = []
        current = [root]
        seen = {id(root)}

        for _ in range(max_depth):
            nxt = []
            for node in current:
                for attr in ("Next_Up", "Next_Mid", "Next_Down"):
                    child = getattr(node, attr, None)
                    if child is None:
                        continue
                    edges.append((node, attr, child))
                    if id(child) not in seen:
                        nxt.append(child)
                        seen.add(id(child))
            if not nxt:
                break
            nxt.sort(key=lambda x: x.UndPrice, reverse=True)
            columns.append(nxt)
            current = nxt

        return {"columns": columns, "edges": edges}

    # ------------------------------------------------------------------
    def build_tree(self, mkt, parameters, pricer, T, BS_Price,
                   condition_Convergence, isVega, ext=None, binom=None):
        start_time = time.time()
        self.init_root_node_and_parameters(mkt, parameters, pricer, T)
        for i in range(1, T + 1):
            self.build_columns(self.candidateMid, mkt, parameters, i, pricer)

        # --- Snapshot avant pricing (la structure Next_* sera détruite après) ---
        snapshot_needed = (pricer.display_tree_bool
                           and (not condition_Convergence or T == pricer.timeSteps))
        if snapshot_needed:
            self._tree_snapshot = self._snapshot_tree(
                min(MAX_SVG_DEPTH, T))

        # --- Pricing (backward) ---
        self.candidateMid.price(self.candidateMid, parameters, mkt, self)
        price = self.Root_Node.OptPrice
        end_time = time.time()
        time_calculation = end_time - start_time

        # --- SVG après pricing : OptPrice est rempli ---
        if snapshot_needed:
            self.time_calculation_tree = self.display_tree(pricer, parameters)

        if pricer.compute_DeltaAndGamma_tree_var:
            pricer.DeltaTree = self.Compute_DeltaTree(
                self.Root_Node.UpNode.OptPrice, self.Root_Node.DownNode.OptPrice, mkt.SpotPrice)
            pricer.GammaTree = self.Compute_GammaTree(
                price, self.Root_Node.UpNode.OptPrice,
                self.Root_Node.DownNode.OptPrice, mkt.SpotPrice)

        if condition_Convergence:
            mc_price = None
            binom_price = None
            if ext is not None and pricer.MC_condition:
                saved_ts = pricer.timeSteps
                saved_mc = pricer.MC_price
                pricer.timeSteps = T
                np.random.seed(42)
                try:
                    mc_price = float(ext.monte_carlo_price(1500))
                except Exception:
                    mc_price = None
                pricer.timeSteps = saved_ts
                pricer.MC_price = saved_mc
            if binom is not None and getattr(pricer, 'Binom_condition', False):
                try:
                    binom_price = float(binom.price_option(T))
                except Exception:
                    binom_price = None
            pricer.compute_perf(T, price, time_calculation, BS_Price,
                                mc_price=mc_price, binom_price=binom_price)

        if not isVega:
            self.display_results(pricer, time_calculation, price,
                                 self.time_calculation_tree, pricer.newTs)

    def display_results(self, pricer, time_calculation, price, time_calculation_tree, newTs):
        self.price_results = {'TreePrice': price, 'TimePricing': time_calculation}
        if newTs != 0:
            self.price_results["New TS"] = newTs
        if pricer.display_tree_bool:
            self.price_results["Tree Display Time"] = time_calculation_tree
        if pricer.BS_condition:
            self.price_results['BS Price'] = pricer.BS_price
            self.price_results['BS_Time'] = pricer.BS_Time
            self.price_results['TreeGapBS'] = price - pricer.BS_price
        if pricer.MC_condition:
            self.price_results['MC Price'] = pricer.MC_price
            self.price_results['MCTime'] = pricer.MC_Time
            self.price_results['MCGapTree'] = pricer.MC_price - price
        if getattr(pricer, 'Binom_condition', False):
            self.price_results['Binom Price'] = pricer.Binom_price
            self.price_results['Binom_Time'] = pricer.Binom_Time
            self.price_results['TreeGapBinom'] = price - pricer.Binom_price

    # ------------------------------------------------------------------
    def compute_greeks_tree(self, params, pricer, mkt):
        def price_fn(m, p, pr):
            saved = (pr.Convergence, pr.display_tree_bool, pr.BS_condition,
                     pr.MC_condition, pr.compute_DeltaAndGamma_tree_var,
                     getattr(pr, 'Binom_condition', False))
            pr.Convergence = False
            pr.display_tree_bool = False
            pr.BS_condition = False
            pr.MC_condition = False
            pr.compute_DeltaAndGamma_tree_var = False
            if hasattr(pr, 'Binom_condition'):
                pr.Binom_condition = False
            self.init(m, p, pr, True)
            price = self.Root_Node.OptPrice
            (pr.Convergence, pr.display_tree_bool, pr.BS_condition,
             pr.MC_condition, pr.compute_DeltaAndGamma_tree_var, bc) = saved
            if hasattr(pr, 'Binom_condition'):
                pr.Binom_condition = bc
            return price
        return finite_difference_greeks(price_fn, mkt, params, pricer)

    # ------------------------------------------------------------------
    def build_triple(self, candidateMid, mkt, div):
        prix_forward = candidateMid.forward(mkt, self.dt, div)
        self.build_mid(candidateMid, prix_forward, mkt, div)
        self.link_and_new_node(prix_forward * self.Alpha, candidateMid, self.nnext, True)
        self.link_and_new_node(prix_forward / self.Alpha, candidateMid, self.nnext, False)

    def build_mid(self, candidateMid, prix_forward, mkt, div):
        self.nnext = self.make_node(prix_forward)
        candidateMid.Next_Mid = self.nnext
        self.nnext.nprec = candidateMid
        self.compute_proba(candidateMid, self.nnext, mkt, div)
        candidateMid.Next_Mid.Cum_Proba = candidateMid.Cum_Proba * candidateMid.Proba_Mid

    def link_and_new_node(self, prix, candidateMid, nnext, upper):
        newnode = self.make_node(prix)
        if upper:
            newnode.Cum_Proba = candidateMid.Cum_Proba * candidateMid.Proba_Up
            newnode.DownNode = nnext
            nnext.UpNode = newnode
            candidateMid.Next_Up = newnode
            self.nUp = newnode
        else:
            newnode.Cum_Proba = candidateMid.Cum_Proba * candidateMid.Proba_Down
            newnode.UpNode = nnext
            nnext.DownNode = newnode
            candidateMid.Next_Down = newnode
            self.nDown = newnode

    def build_columns(self, candidateMid, mkt, parameters, i, pricer):
        div = mkt.compute_dividend(i, self.dt)
        self.isTrunc = True
        self.build_triple(candidateMid, mkt, div)
        self.tronc = candidateMid
        self.build_up_and_down(candidateMid, mkt, parameters, pricer, div, "up")
        if not self.isTrunc:
            self.nUp = candidateMid.Next_Up
            self.nnext = candidateMid.Next_Mid
            self.nDown = candidateMid.Next_Down
            self.build_up_and_down(candidateMid, mkt, parameters, pricer, div, "down")
        candidateMid = self.tronc
        self.candidateMid = candidateMid.Next_Mid

    def build_up_and_down(self, candidateMid, mkt, parameters, pricer, div, direction):
        if direction == "up":
            while candidateMid.UpNode is not None:
                self.isTrunc = False
                self.next_forward(candidateMid.UpNode, mkt, parameters, pricer, direction, div)
                candidateMid = candidateMid.UpNode
        elif direction == "down":
            while candidateMid.DownNode is not None:
                self.next_forward(candidateMid.DownNode, mkt, parameters, pricer, direction, div)
                candidateMid = candidateMid.DownNode

    def next_forward(self, candidateMid, mkt, parameters, pricer, direction, div):
        isUp = (direction == "up")
        if candidateMid.Cum_Proba > pricer.PruningTreshold:
            self.ModifExMid = False
            self.chek_and_link(candidateMid, mkt, direction, div, isUp, False)
            self.compute_and_allocating_probas(candidateMid, mkt, div, isUp)
            self.new_and_link(candidateMid, direction, False)
        else:
            self.chek_and_link(candidateMid, mkt, direction, div, isUp, True)

    def chek_and_link(self, candidateMid, mkt, direction, div, isUp, isprunne):
        if isUp:
            if div != 0:
                self.controle_next_mid(candidateMid, self.nUp, direction, mkt, div)
            if not self.ModifExMid and not isprunne:
                candidateMid.Next_Mid = self.nUp
                candidateMid.Next_Down = self.nnext
            elif not self.ModifExMid and isprunne:
                candidateMid.Next_Mid = self.nUp
                candidateMid.Proba_Mid = 1
        else:
            if div != 0:
                self.controle_next_mid(candidateMid, self.nDown, direction, mkt, div)
            if not self.ModifExMid and not isprunne:
                candidateMid.Next_Mid = self.nDown
                candidateMid.Next_Up = self.nnext
            elif not self.ModifExMid and isprunne:
                candidateMid.Next_Mid = self.nDown
                candidateMid.Proba_Mid = 1

    def compute_and_allocating_probas(self, candidateMid, mkt, div, isUp):
        self.compute_proba(candidateMid, candidateMid.Next_Mid, mkt, div)
        candidateMid.Next_Mid.Cum_Proba += candidateMid.Cum_Proba * candidateMid.Proba_Mid
        if isUp:
            candidateMid.Next_Down.Cum_Proba += candidateMid.Cum_Proba * candidateMid.Proba_Down
        else:
            candidateMid.Next_Up.Cum_Proba += candidateMid.Cum_Proba * candidateMid.Proba_Up

    def compute_transition_up_node(self, newnode, ModifExMid, IsDiveTime):
        if IsDiveTime:
            self.nnext = newnode; self.nDown = self.nUp; self.nUp = self.nnext
        elif not ModifExMid:
            self.nDown = self.nnext; self.nnext = self.nUp; self.nUp = newnode
        else:
            self.nUp = newnode

    def compute_transition_down_node(self, newnode, ModifExMid, IsDiveTime):
        if IsDiveTime:
            self.nnext = newnode; self.nUp = self.nDown; self.nDown = self.nnext
        elif not ModifExMid:
            self.nUp = self.nnext; self.nnext = self.nDown; self.nDown = newnode
        else:
            self.nDown = newnode

    def build_up(self, candidate_mid, is_dive_time):
        newnode = self.make_node(self.nUp.UndPrice * self.Alpha)
        if is_dive_time:
            candidate_mid.Next_Mid = newnode
            self.nUp.UpNode = newnode
            candidate_mid.Next_Down = self.nUp
            candidate_mid.Next_Down.UpNode = newnode
        else:
            candidate_mid.Next_Up = newnode
            candidate_mid.Next_Mid.UpNode = newnode
            candidate_mid.Next_Up.DownNode = self.nUp
        self.compute_transition_up_node(newnode, self.ModifExMid, is_dive_time)

    def build_down(self, candidate_mid, is_dive_time):
        newnode = self.make_node(self.nDown.UndPrice / self.Alpha)
        if is_dive_time:
            candidate_mid.Next_Mid = newnode
            candidate_mid.Next_Mid.UpNode = self.nDown
            candidate_mid.Next_Up = self.nDown
            candidate_mid.Next_Up.DownNode = newnode
        else:
            candidate_mid.Next_Down = newnode
            candidate_mid.Next_Mid.DownNode = newnode
            candidate_mid.Next_Down.UpNode = self.nDown
        self.compute_transition_down_node(newnode, self.ModifExMid, is_dive_time)

    def new_and_link(self, candidate_mid, direction, is_dive_time):
        is_up = (direction == "up")
        if is_up:
            self.build_up(candidate_mid, is_dive_time)
        else:
            self.build_down(candidate_mid, is_dive_time)
        if not is_dive_time:
            if is_up:
                candidate_mid.Next_Up.Cum_Proba = candidate_mid.Cum_Proba * candidate_mid.Proba_Up
            else:
                candidate_mid.Next_Down.Cum_Proba = candidate_mid.Cum_Proba * candidate_mid.Proba_Down

    def controle_next_mid(self, candidate_mid, potential_mid, direction, mkt, div):
        fwd = candidate_mid.forward(mkt, self.dt, div)
        if (fwd < (potential_mid.UndPrice * (1 + (1 / self.Alpha))) / 2
                or fwd > (potential_mid.UndPrice * (1 + self.Alpha)) / 2):
            self.new_and_link(candidate_mid, direction, True)
            self.ModifExMid = True

    def compute_proba(self, candidateMid, nnext, mkt, div):
        div_type = getattr(mkt, 'dividend_type', 'Discrete')

        if div_type == "Continuous":
            q = mkt.dividend_yield()
            drift = mkt.RiskFree - q
            esperance = candidateMid.UndPrice * np.exp(drift * self.dt)
            variance = (candidateMid.UndPrice ** 2) * np.exp(2 * drift * self.dt) * (
                    np.exp(mkt.volatility ** 2 * self.dt) - 1)
        else:
            esperance = candidateMid.UndPrice * np.exp(mkt.RiskFree * self.dt) - div
            variance = (candidateMid.UndPrice ** 2) * np.exp(2 * mkt.RiskFree * self.dt) * (
                    np.exp(mkt.volatility ** 2 * self.dt) - 1)

        candidateMid.Proba_Down = ((nnext.UndPrice ** -2 * (variance + esperance ** 2) - 1 -
                                    (self.Alpha + 1) * (nnext.UndPrice ** -1 * esperance - 1)) /
                                   ((1 - self.Alpha) * (self.Alpha ** -2 - 1)))
        candidateMid.Proba_Up = (nnext.UndPrice ** -1 * esperance - 1 -
                                 (self.Alpha ** -1 - 1) * candidateMid.Proba_Down) / (self.Alpha - 1)
        candidateMid.Proba_Mid = 1 - candidateMid.Proba_Down - candidateMid.Proba_Up
        if candidateMid.Proba_Down < 0 or candidateMid.Proba_Up < 0 or candidateMid.Proba_Mid < 0:
            print("Probabilités négatives détectées")

    def Compute_DeltaTree(self, P1, P_1, S0):
        return (P1 - P_1) / ((self.Alpha * S0) - (S0 / self.Alpha))

    def Compute_GammaTree(self, P, P1, P_1, S0):
        return ((((P1 - P) / ((self.Alpha * S0) - S0)) -
                 ((P - P_1) / (S0 - (S0 / self.Alpha))))
                / (((self.Alpha * S0) - (S0 / self.Alpha)) / 2))

    # ------------------------------------------------------------------
    def compute_StrikeStudy(self, StrikeSteps, mkt, pricer, params, Ext,
                            binom=None, with_mc=False):
        columns = ["Strike", "Tree Price", "BS Price", "Tree-BS",
                   "Slope BS", "Slope Tree"]
        if binom is not None:
            columns += ["Binom Price", "Tree-Binom", "Slope Binom"]
        if with_mc:
            columns += ["MC Price", "Tree-MC", "Slope MC"]

        df_strike = pd.DataFrame(0.0, index=range(StrikeSteps + 1), columns=columns)

        saved_conv = pricer.Convergence
        saved_tree = pricer.display_tree_bool
        saved_bs = pricer.BS_condition
        saved_mc = pricer.MC_condition
        saved_dg = pricer.compute_DeltaAndGamma_tree_var
        saved_binom = getattr(pricer, 'Binom_condition', False)

        pricer.Convergence = False
        pricer.display_tree_bool = False
        pricer.BS_condition = False
        pricer.MC_condition = False
        pricer.compute_DeltaAndGamma_tree_var = False
        pricer.Binom_condition = False

        original_strike = params.strike
        params.strike = original_strike - 5

        for T in range(1, StrikeSteps + 1):
            params.strike += 1
            self.build_tree(mkt, params, pricer, pricer.timeSteps,
                            pricer.BS_price, False, True)
            tree_price = self.Root_Node.OptPrice

            bs_res = Ext.black_sholes()
            bs_price = float(bs_res["BS Price"])

            df_strike.at[T, "Strike"] = params.strike
            df_strike.at[T, "Tree Price"] = tree_price
            df_strike.at[T, "BS Price"] = bs_price
            df_strike.at[T, "Tree-BS"] = tree_price - bs_price

            if binom is not None:
                bp = float(binom.price_option(pricer.timeSteps))
                df_strike.at[T, "Binom Price"] = bp
                df_strike.at[T, "Tree-Binom"] = tree_price - bp

            if with_mc:
                np.random.seed(42)
                mp = float(Ext.monte_carlo_price(3000))
                df_strike.at[T, "MC Price"] = mp
                df_strike.at[T, "Tree-MC"] = tree_price - mp

        params.strike = original_strike

        pricer.Convergence = saved_conv
        pricer.display_tree_bool = saved_tree
        pricer.BS_condition = saved_bs
        pricer.MC_condition = saved_mc
        pricer.compute_DeltaAndGamma_tree_var = saved_dg
        pricer.Binom_condition = saved_binom

        df_strike = self.compute_slope(df_strike, StrikeSteps)
        df_plot = df_strike.iloc[1:].copy()

        return pricer.create_plot_StrikeStudy(
            df_plot,
            df_plot["Strike"].iloc[0],
            df_plot["Strike"].iloc[-1])

    def compute_slope(self, df_strike, StrikeSteps):
        slope_pairs = [("BS Price", "Slope BS"),
                       ("Tree Price", "Slope Tree"),
                       ("Binom Price", "Slope Binom"),
                       ("MC Price", "Slope MC")]
        for T in range(2, StrikeSteps):
            for price_col, slope_col in slope_pairs:
                if price_col in df_strike.columns and slope_col in df_strike.columns:
                    denom = (df_strike.at[T + 1, "Strike"] - df_strike.at[T - 1, "Strike"])
                    if denom != 0:
                        df_strike.at[T, slope_col] = (
                            (df_strike.at[T + 1, price_col] - df_strike.at[T - 1, price_col]) / denom)
        return df_strike

    # ------------------------------------------------------------------
    # SVG tree — appelé APRÈS pricing pour disposer de OptPrice
    # ------------------------------------------------------------------
    def display_tree(self, pricer, params=None, max_display_depth=MAX_SVG_DEPTH):
        start_time_tree = time.time()
        actual_depth = min(max_display_depth, pricer.timeSteps)
        try:
            from TreeImage import TreeImage
            image = TreeImage(self, actual_depth, params=params)
            self.svg_str = image.as_str()
            print(f"[SVG] généré : {len(self.svg_str):,} caractères, "
                  f"{actual_depth} colonnes affichées (sur {pricer.timeSteps}).")
        except ImportError as e:
            print(f"[SVG] ImportError : {e}. Installe : pip install svg-py")
            self.svg_str = None
        except Exception as e:
            import traceback
            print(f"[SVG] Erreur : {e}")
            traceback.print_exc()
            self.svg_str = None
        return time.time() - start_time_tree