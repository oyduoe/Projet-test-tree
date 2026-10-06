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


class Tree:
    def __init__(self):
        """Initialise les paramètres de l'arbre."""
        self.dt: float             # Pas de temps
        self.Alpha: float          # Alpha
        self.Root_Node: Node       # Noeud de la racine
        self.Market: Market        # Marché
        self.candidateMid: Node    # Noeud candidate
        self.Final_Node: Node      # Noeud final
        self.tronc: Node           # Tronc de l'arbre
        self.ModifExMid: Node      # Indicateur de modification
        self.isTrunc: Node         # Indicateur de tronc
        self.Convergence_results: DataFrame  # Stocke les résultats de Convergence
        self.price_results = {}    # Stocke les résultats de prix
        self.nnext: Node           # Noeud mid tampon pour les prochains noeuds
        self.nUp: Node             # Noeud supérieur tampon pour les prochains noeuds
        self.nDown: Node           # Noeud inférieur tampon pour les prochains noeuds
        self.df: DataFrame = None
        self.Convergence_fig = None
        self.time_calculation_tree = 0.00  # Initialisation à 0 si on n'affiche pas l'arbre

    def init(self, mkt: Market, params: OptionParameters, pricer: Pricer, IsVega: bool):
        """Initialise l'arbre et construit les nœuds."""
        if pricer.Convergence:  # Si évaluation de Convergence demandée
            for T in range(1, pricer.timeSteps + 1):
                print(f"Calcul du prix pour {T} Time Steps")
                self.build_tree(mkt, params, pricer, T, pricer.BS_price, True, IsVega)
            self.Convergence_fig = pricer.create_plot_convergence()
        else:  # Appel simple si pas de Convergence
            if pricer.gap != 0:
                pricer.newTs = pricer.compute_time_steps(mkt, params)  # Déterminer les pas de temps
                self.build_tree(mkt, params, pricer, pricer.newTs, pricer.BS_price, False, IsVega)
            else:
                self.build_tree(mkt, params, pricer, pricer.timeSteps, pricer.BS_price, False, IsVega)

    def make_node(self, market_values: float) -> Node:
        """Crée un nouveau nœud avec la valeur du marché."""
        new_node = Node()  # Crée une instance de Node
        new_node.init_node(market_values)  # Initialise le nœud
        return new_node

    def init_root_node_and_parameters(self, mkt: Market, parameters: OptionParameters,
                                      pricer: Pricer, T) -> Node:
        """Initialise le nœud racine et les paramètres."""
        self.dt = ((parameters.DateMaturity - pricer.PricingDate).days / T) / 365  # Calcul de dt
        self.Alpha = math.exp(mkt.volatility * math.sqrt(3 * self.dt))  # Calcul d'alpha
        self.Root_Node = self.make_node(mkt.SpotPrice)  # Initialise le nœud racine avec le prix au comptant

        if pricer.compute_DeltaAndGamma_tree_var:  # Si on a demandé le calcul du delta et du gamma
            NodeCalcul = self.make_node(mkt.SpotPrice)  # Nœud sans lien avec l'arbre pour les probabilités de la racine
            NodeCalcul.Next_Mid = self.make_node(NodeCalcul.forward(mkt, self.dt, 0))
            self.compute_proba(NodeCalcul, NodeCalcul.Next_Mid, mkt, 0)
            self.Root_Node.UpNode = self.make_node(mkt.SpotPrice * self.Alpha)
            self.Root_Node.DownNode = self.make_node(mkt.SpotPrice / self.Alpha)
            # Attribution des probabilités
            self.Root_Node.Cum_Proba = NodeCalcul.Proba_Mid
            self.Root_Node.UpNode.Cum_Proba = NodeCalcul.Proba_Up
            self.Root_Node.DownNode.Cum_Proba = NodeCalcul.Proba_Down
        else:  # Si on ne veut pas calculer Delta et Gamma
            self.Root_Node.Cum_Proba = 1

        self.candidateMid = self.Root_Node  # Commence avec le nœud médian

    def build_tree(self, mkt: Market, parameters: OptionParameters, pricer: Pricer,
                   T: int, BS_Price: float, condition_Convergence: bool, isVega: bool) -> Node:
        """Construire l'arbre à partir des paramètres fournis."""
        start_time = time.time()  # Début du timer
        self.init_root_node_and_parameters(mkt, parameters, pricer, T)  # Initialisation

        for i in range(1, T + 1):  # Boucle sur les pas de temps
            self.build_columns(self.candidateMid, mkt, parameters, i, pricer)

        if pricer.display_tree_bool:
            self.time_calculation_tree = self.display_tree(pricer)

        # Calculer le prix de l'option
        self.candidateMid.price(self.candidateMid, parameters, mkt, self)
        price = self.Root_Node.OptPrice  # Sauvegarde du prix
        end_time = time.time()  # Fin du timer
        time_calculation = end_time - start_time  # Temps de calcul

        if pricer.compute_DeltaAndGamma_tree_var:  # Si on a demandé le calcul de Delta et Gamma
            pricer.DeltaTree = self.Compute_DeltaTree(self.Root_Node.UpNode.OptPrice,
                                                      self.Root_Node.DownNode.OptPrice, mkt.SpotPrice)
            pricer.GammaTree = self.Compute_GammaTree(price, self.Root_Node.UpNode.OptPrice,
                                                      self.Root_Node.DownNode.OptPrice, mkt.SpotPrice)

        # Stocke les résultats en fonction de si l'on a demandé la convergence ou non
        if condition_Convergence:  # Si on effectue les graphiques de convergence
            pricer.compute_perf(T, price, time_calculation, BS_Price)
        elif not isVega:  # On ne veut pas afficher les résultats des prix lorsqu'on calcule le vega non plus
            self.display_results(pricer, time_calculation, price, self.time_calculation_tree, pricer.newTs)

    def display_results(self, pricer: Pricer, time_calculation: float,
                        price: float, time_calculation_tree: float, newTs: int):
        """Affiche les résultats des prix."""
        self.price_results = {
            'TreePrice': price,
            'TimePricing': time_calculation,
        }
        if newTs != 0:
            self.price_results["New TS"] = newTs
        if pricer.display_tree_bool:
            self.price_results["Tree Display Time"] = time_calculation_tree

        if pricer.BS_condition:
            self.price_results['BS Price'] = pricer.BS_price
            self.price_results['BS_Time'] = pricer.BS_Time
            self.price_results['TreeGapBS'] = price - pricer.BS_price
            self.price_results['TreeGapBSTime'] = time_calculation - pricer.BS_Time

        if pricer.MC_condition:
            self.price_results['MC Price'] = pricer.MC_price
            self.price_results['MCTime'] = pricer.MC_Time
            self.price_results['MCGapTree'] = pricer.MC_price - price
            self.price_results['MCGapTreeTime'] = time_calculation - pricer.MC_Time

    def build_triple(self, candidateMid: Node, mkt: Market, div: float):
        """Méthode permettant la création du tronc à chaque pas de temps."""
        prix_forward = candidateMid.forward(mkt, self.dt, div)  # Calcul du prix forward
        self.build_mid(candidateMid, prix_forward, mkt, div)  # Création du next mid du tronc
        self.link_and_new_node(prix_forward * self.Alpha, candidateMid, self.nnext, True)  # Création du noeud Up
        self.link_and_new_node(prix_forward / self.Alpha, candidateMid, self.nnext, False)  # Création du noeud Down

    def build_mid(self, candidateMid: Node, prix_forward: float, mkt: Market, div: float):
        """Méthode créant le prochain milieu du tronc."""
        self.nnext = self.make_node(prix_forward)  # Création du noeud avec le prix forward
        candidateMid.Next_Mid = self.nnext  # Attribution du next_mid
        self.nnext.nprec = candidateMid  # Lien en arrière pour le calcul du prix
        self.compute_proba(candidateMid, self.nnext, mkt, div)
        candidateMid.Next_Mid.Cum_Proba = candidateMid.Cum_Proba * candidateMid.Proba_Mid

    def link_and_new_node(self, prix: float, candidateMid: Node, nnext: Node, upper: bool):
        """Méthode construisant les nouveaux up et down du tronc."""
        newnode = self.make_node(prix)  # Création du nouveau noeud

        if upper:  # Partie supérieure de l'arbre
            newnode.Cum_Proba = candidateMid.Cum_Proba * candidateMid.Proba_Up
            newnode.DownNode = nnext
            nnext.UpNode = newnode
            candidateMid.Next_Up = newnode
            self.nUp = newnode
        else:  # Partie inférieure de l'arbre
            newnode.Cum_Proba = candidateMid.Cum_Proba * candidateMid.Proba_Down
            newnode.UpNode = nnext
            nnext.DownNode = newnode
            candidateMid.Next_Down = newnode
            self.nDown = newnode

    def build_columns(self, candidateMid: Node, mkt: Market,
                      parameters: OptionParameters, i: int, pricer: Pricer):
        """Méthode construisant les colonnes pour chaque pas de temps."""
        div = mkt.compute_dividend(i, self.dt)  # Initialisation du dividende

        # Création du nœud médian pour ce pas de temps
        self.isTrunc = True
        self.build_triple(candidateMid, mkt, div)  # Création des trois nœuds
        self.tronc = candidateMid  # Garder un nœud pointé sur le tronc

        # Construction des noeuds Up
        self.build_up_and_down(candidateMid, mkt, parameters, pricer, div, "up")

        if not self.isTrunc:  # Vérification de la création de nœuds supérieurs
            self.nUp = candidateMid.Next_Up
            self.nnext = candidateMid.Next_Mid
            self.nDown = candidateMid.Next_Down
            self.build_up_and_down(candidateMid, mkt, parameters, pricer, div, "down")

        candidateMid = self.tronc  # On se replace au mid
        self.candidateMid = candidateMid.Next_Mid  # Décalage pour le pas de temps suivant

    def build_up_and_down(self, candidateMid: Node, mkt: Market, parameters: OptionParameters,
                          pricer: Pricer, div: float, direction: str):
        """Méthode qui gère l'appel des fonctions de création des nœuds up et down."""
        if direction == "up":
            while candidateMid.UpNode is not None:
                self.isTrunc = False
                self.next_forward(candidateMid.UpNode, mkt, parameters, pricer, direction, div)
                candidateMid = candidateMid.UpNode
        elif direction == "down":
            while candidateMid.DownNode is not None:
                self.next_forward(candidateMid.DownNode, mkt, parameters, pricer, direction, div)
                candidateMid = candidateMid.DownNode

    def next_forward(self, candidateMid: Node, mkt: Market, parameters: OptionParameters,
                     pricer: Pricer, direction: str, div: float):
        """Méthode calculant le forward pour les nœuds up et down en construction."""
        isUp = (direction == "up")

        if candidateMid.Cum_Proba > pricer.PruningTreshold:  # Prunning de l'arbre
            self.ModifExMid = False
            self.chek_and_link(candidateMid, mkt, direction, div, isUp, False)
            self.compute_and_allocating_probas(candidateMid, mkt, div, isUp)
            self.new_and_link(candidateMid, direction, False)
        else:
            self.chek_and_link(candidateMid, mkt, direction, div, isUp, True)

    def chek_and_link(self, candidateMid: Node, mkt: Market, direction: str, div: float,
                      isUp: bool, isprunne: bool):
        """Méthode qui vérifie le Mid en cas de dividendes, sinon lie les nœuds normalement."""
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

    def compute_and_allocating_probas(self, candidateMid: Node, mkt: Market,
                                      div: float, isUp: bool):
        """Méthode calculant les probabilités et allouant également pour les nœuds ayant plusieurs chemins."""
        self.compute_proba(candidateMid, candidateMid.Next_Mid, mkt, div)
        candidateMid.Next_Mid.Cum_Proba += candidateMid.Cum_Proba * candidateMid.Proba_Mid
        if isUp:
            candidateMid.Next_Down.Cum_Proba += candidateMid.Cum_Proba * candidateMid.Proba_Down
        else:
            candidateMid.Next_Up.Cum_Proba += candidateMid.Cum_Proba * candidateMid.Proba_Up

    def compute_transition_up_node(self, newnode: Node, ModifExMid: bool, IsDiveTime: bool):
        """Méthode modifiant les variables de transition pour le prochain nœud à créer."""
        if IsDiveTime:
            self.nnext = newnode
            self.nDown = self.nUp
            self.nUp = self.nnext
        elif not ModifExMid:
            self.nDown = self.nnext
            self.nnext = self.nUp
            self.nUp = newnode
        elif ModifExMid:
            self.nUp = newnode

    def compute_transition_down_node(self, newnode: Node, ModifExMid: bool, IsDiveTime: bool):
        """Méthode modifiant les variables de transition pour le prochain nœud à créer."""
        if IsDiveTime:
            self.nnext = newnode
            self.nUp = self.nDown
            self.nDown = self.nnext
        elif not ModifExMid:
            self.nUp = self.nnext
            self.nnext = self.nDown
            self.nDown = newnode
        elif ModifExMid:
            self.nDown = newnode

    def build_up(self, candidate_mid: Node, is_dive_time: bool):
        """Méthode qui gère la création de noeud up."""
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

    def build_down(self, candidate_mid: Node, is_dive_time: bool):
        """Méthode qui gère la création de noeud down."""
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

    def new_and_link(self, candidate_mid: Node, direction: str, is_dive_time: bool):
        """Méthode créant les nouveaux nœuds Up ou Down."""
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

    def controle_next_mid(self, candidate_mid: Node, potential_mid: Node,
                          direction: str, mkt: Market, div: float):
        """Méthode qui contrôle si le forward créé avec le tronc correspond pour le up ou le down."""
        fwd = candidate_mid.forward(mkt, self.dt, div)
        if (fwd < (potential_mid.UndPrice * (1 + (1 / self.Alpha))) / 2
                or fwd > (potential_mid.UndPrice * (1 + self.Alpha)) / 2):
            self.new_and_link(candidate_mid, direction, True)
            self.ModifExMid = True

    def compute_proba(self, candidateMid: Node, nnext: Node, mkt: Market, div: float):
        """Méthode de calcul des probabilités."""
        esperance = candidateMid.UndPrice * np.exp(mkt.RiskFree * self.dt) - div
        variance = (candidateMid.UndPrice ** 2) * np.exp(2 * mkt.RiskFree * self.dt) * (
                np.exp(mkt.volatility ** 2 * self.dt) - 1)

        candidateMid.Proba_Down = ((nnext.UndPrice ** -2 * (variance + esperance ** 2) - 1 -
                                   (self.Alpha + 1) * (nnext.UndPrice ** -1 * esperance - 1)) /
                                   ((1 - self.Alpha) * (self.Alpha ** -2 - 1)))
        candidateMid.Proba_Up = (nnext.UndPrice ** -1 * esperance - 1 - (self.Alpha ** -1 - 1) *
                                 candidateMid.Proba_Down) / (self.Alpha - 1)
        candidateMid.Proba_Mid = 1 - candidateMid.Proba_Down - candidateMid.Proba_Up
        if candidateMid.Proba_Down < 0 or candidateMid.Proba_Up < 0 or candidateMid.Proba_Mid < 0:
            print("Probabilités négatives détectées")

    def Compute_DeltaTree(self, P1: float, P_1: float, S0: float):
        """Méthode de calcul du delta de l'option avec le pricing trinomial."""
        return (P1 - P_1) / ((self.Alpha * S0) - (S0 / self.Alpha))

    def Compute_GammaTree(self, P: float, P1: float, P_1: float, S0: float):
        """Méthode de calcul du gamma de l'option avec le pricing trinomial."""
        return ((((P1 - P) / ((self.Alpha * S0) - S0)) - ((P - P_1) / (S0 - (S0 / self.Alpha))))
                / (((self.Alpha * S0) - (S0 / self.Alpha)) / 2))

    def compute_vega_tree(self, params: OptionParameters, pricer: Pricer, mkt: Market):
        """Méthode de calcul du vega de l'option avec le pricing trinomial."""
        pricer.Convergence = False
        pricer.display_tree_bool = False
        pricer.BS_condition = False
        pricer.MC_condition = False
        pricer.compute_DeltaAndGamma_tree_var = False
        original_volatility = mkt.volatility
        mkt.volatility = original_volatility + 0.01  # Augmenter la volatilité
        self.init(mkt, params, pricer, True)
        price_u = self.Root_Node.OptPrice

        mkt.volatility = original_volatility - 0.01  # Diminuer la volatilité
        self.init(mkt, params, pricer, True)
        price_d = self.Root_Node.OptPrice
        mkt.volatility = original_volatility

        return (price_u - price_d) / 2

    def compute_StrikeStudy(self, StrikeSteps: int, mkt: Market, pricer: Pricer,
                            params: OptionParameters, Extension: Extension):
        """Méthode de calcul de l'analyse du prix par rapport au strike."""
        df_strike = pd.DataFrame(0.0, index=range(StrikeSteps),
                                 columns=["Strike", "Tree Price", "BS Price", "Tree-BS", "Slope BS", "Slope Tree"])
        pricer.Convergence = False
        pricer.display_tree_bool = False
        params.strike = params.strike - 5
        for T in range(1, StrikeSteps + 1):
            params.strike += 1
            self.build_tree(mkt, params, pricer, pricer.timeSteps, pricer.BS_price, True, False)

            Tree_price = self.Root_Node.OptPrice
            BS_Result = Extension.black_sholes()
            BS_Price = BS_Result["BS Price"]
            df_strike.at[T, "Strike"] = params.strike
            df_strike.at[T, "Tree Price"] = Tree_price
            df_strike.at[T, "BS Price"] = BS_Price
            df_strike.at[T, "Tree-BS"] = Tree_price - BS_Price
        df_strike = self.compute_slope(df_strike, StrikeSteps)
        df_strike_fig = pricer.create_plot_StrikeStudy(
            df_strike, df_strike.at[2, "Strike"], df_strike.at[T, "Strike"])
        return df_strike_fig

    def compute_slope(self, df_strike: DataFrame, StrikeSteps: int) -> DataFrame:
        """Méthode de calcul de la slope."""
        for T in range(2, StrikeSteps):
            df_strike.at[T, "Slope BS"] = ((df_strike.at[T + 1, "BS Price"] - df_strike.at[T - 1, "BS Price"]) /
                                           (df_strike.at[T + 1, "Strike"] - df_strike.at[T - 1, "Strike"]))
            df_strike.at[T, "Slope Tree"] = ((df_strike.at[T + 1, "Tree Price"] - df_strike.at[T - 1, "Tree Price"]) /
                                             (df_strike.at[T + 1, "Strike"] - df_strike.at[T - 1, "Strike"]))
        return df_strike

    def display_tree(self, pricer: Pricer) -> float:
        """Construit le DataFrame de l'arbre (probabilités) et renvoie le temps de construction."""
        start_time_tree = time.time()  # Début du timer
        max_depth = pricer.timeSteps  # Nombre maximal de pas de temps
        width = max_depth + 1  # Largeur du DataFrame (nombre de colonnes)
        self.df = pd.DataFrame("", index=range(2 * max_depth + 1), columns=range(width))
        # Initialiser la position de la racine
        col_center = 0
        current_node = self.Root_Node

        for step in range(1, max_depth + 1):  # Boucle pour avancer dans les colonnes (par pas de temps)
            if current_node is None:
                break

            self.df.iloc[max_depth, col_center] = current_node.Proba_Mid  # On inscrit le mid
            midNode = current_node
            row = 1

            while current_node.UpNode is not None:  # Boucle pour inscrire toutes les valeurs up
                current_node = current_node.UpNode
                self.df.iloc[max_depth - row, col_center] = current_node.Proba_Mid
                row += 1

            current_node = midNode
            row = 1

            while current_node.DownNode is not None:  # Boucle pour inscrire toutes les valeurs down
                current_node = current_node.DownNode
                self.df.iloc[max_depth + row, col_center] = current_node.Proba_Mid
                row += 1

            current_node = midNode.Next_Mid  # On se replace au mid et on avance d'un pas de temps
            col_center += 1

        end_time_tree = time.time()  # Fin du timer
        return end_time_tree - start_time_tree  # Temps de construction