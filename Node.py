import math
from Market import Market
from OptionParameters import OptionParameters




class Node:
    def __init__(self):
        """Initialise un noeud de l'arbre."""
        self.UndPrice:float           # Valeur du sous-jacent (du noeud)
        self.Payoff :float             # Payoff du noeud
        self.Next_Up = None            # Noeud supérieur dans l'étape suivante
        self.Next_Down = None          # Noeud inférieur dans l'étape suivante
        self.Next_Mid = None           # Noeud médian dans l'étape suivante
        self.nprec = None              # Noeud Parent
        self.UpNode = None             # Noeud supérieur actuel
        self.DownNode = None           # Noeud inférieur actuel
        self.MidNode = None            # Dernier noeud médian
        self.Proba_Up:float =0.00         # Probabilité d'un mouvement vers le haut
        self.Proba_Down:float =0.00   # Probabilité d'un mouvement vers le bas
        self.Proba_Mid :float = 0.00       # Probabilité d'un mouvement médian
        self.Cum_Proba :float =0.00       # Probabilité cumulée du noeud
        self.OptPrice :float          # Prix de l'option à ce moment du noeud

    def forward(self, mkt:Market, dt:float, div:float):
        """Calcule la valeur forward du noeud."""
        return self.UndPrice * math.exp(mkt.RiskFree * dt) - div

    def init_node(self, market_value:float):
        """Initialise le noeud avec la valeur correspondante."""
        self.UndPrice = market_value

    def incremental_node(self, candidate_mid, is_up:bool, condition:bool):
        """
        Incrémente le noeud candidate_mid dans la bonne direction pour la prochaine itération.
        """
        if is_up:  # Si on incrémente sur la partie haute de l'arbre
            if candidate_mid.UpNode is not None:
                candidate_mid = candidate_mid.UpNode  # On pointe vers le noeud up
            else:
                condition = False  # Met à jour la condition si aucun noeud up n'existe
        else:  # Sinon on incrémente sur la partie inférieure de l'arbre
            if candidate_mid.DownNode is not None:
                candidate_mid = candidate_mid.DownNode  # On pointe vers le noeud down
            else:
                condition = False  # Met à jour la condition si aucun noeud down n'existe
        return candidate_mid, condition

    def price(self, candidate_mid, parameters:OptionParameters, mkt:Market, tree):
        """
        Calcule le prix de l'option avec la méthode Backward.
        """
        while candidate_mid.nprec is not None:  # Itère sur tout l'arbre à partir du lien du tronc
            condition = True  # Initialise la condition
            trunk = candidate_mid  # On garde une variable pointant vers le tronc pour y revenir facilement

            # Partie haute de l'arbre
            while condition:
                self.payoff_condition(candidate_mid, parameters, mkt, tree)  # Détermine si l'on effectue un calcul de payoff ou d'actualisation
                candidate_mid, condition = self.incremental_node(candidate_mid, True, condition)

            # Partie basse de l'arbre
            candidate_mid = trunk.DownNode  # On se replace au mid pour la partie basse
            condition = True  # Réinitialise la condition
            while condition:
                self.payoff_condition(candidate_mid, parameters, mkt, tree)  # Détermine si l'on effectue un calcul de payoff ou d'actualisation
                candidate_mid, condition = self.incremental_node(candidate_mid, False, condition)

            candidate_mid = trunk  # On se replace au mid
            candidate_mid = candidate_mid.nprec  # Puis on décale les next de manière à se placer correctement pour le pas de temps suivant.

        self.discounted_price(candidate_mid, parameters, mkt, tree)  # Rappel de la fonction de prix pour le dernier nœud
        if candidate_mid.UpNode is not None: # Si on a demandé le calcul des deltas et gammas, alors il faut calculer les prix des nœuds up et down de la racine
            self.discounted_price(candidate_mid.UpNode,parameters,mkt,tree) #Calcul du prix Up de la racine
            self.discounted_price(candidate_mid.DownNode, parameters, mkt, tree) #Calcul du prix Down de la racine


    def payoff_condition(self, candidate_mid, parameters:OptionParameters, mkt:Market, tree):
        """
        Calcule le payoff si l'on est à la fin de l'arbre, ou appelle la fonction d'actualisation des prix forward dans le cas contraire.
        """
        if candidate_mid.Next_Mid is None:  # Si l'on est au dernier pas de l'arbre
            candidate_mid.OptPrice = parameters.Vi(candidate_mid.UndPrice)  # Utilisation de la fonction de Payoff
        else:  # Si l'on n'est pas au dernier pas de l'arbre
            self.discounted_price(candidate_mid, parameters, mkt, tree)  # Utilisation de la fonction de prix actualisée

    def discounted_price(self, candidate_mid, parameters:OptionParameters, mkt:Market, tree):
        """
        Calcule la moyenne pondérée des probabilités des prix forward actualisés ou le payoff dans le cas d'une option américaine.
        """
        val_up = 0.0
        val_mid = 0.0
        val_down = 0.0

        if candidate_mid.Next_Up is not None:  # Vérifie qu'il existe bien un NextUp
            val_up = candidate_mid.Next_Up.OptPrice  # Récupère la valeur de l'option du nœud up suivant (déjà calculée)
            candidate_mid.Next_Up = None  # Détruit le lien pour libérer de la mémoire

        if candidate_mid.Next_Mid is not None:  # Vérifie qu'il existe bien un NextMid
            val_mid = candidate_mid.Next_Mid.OptPrice  # Récupère la valeur de l'option du nœud Mid suivant (déjà calculée)
            candidate_mid.Next_Mid = None  # Détruit le lien pour libérer de la mémoire

        if candidate_mid.Next_Down is not None:  # Vérifie qu'il existe bien un NextDown
            val_down = candidate_mid.Next_Down.OptPrice  # Récupère la valeur de l'option du nœud Down suivant (déjà calculée)
            candidate_mid.Next_Down = None  # Détruit le lien pour libérer de la mémoire

        if parameters.Exercice == "European":  # Si l'on a une option européenne
            candidate_mid.OptPrice = (val_up * candidate_mid.Proba_Up +
                                       val_mid * candidate_mid.Proba_Mid +
                                       val_down * candidate_mid.Proba_Down) * math.exp(-mkt.RiskFree * tree.dt)  # Actualisation de la moyenne pondérée des probabilités
        elif parameters.Exercice == "American":  # Si l'on a une option américaine
            candidate_mid.OptPrice = max(parameters.Vi(candidate_mid.UndPrice),
                                          (val_up * candidate_mid.Proba_Up +
                                           val_mid * candidate_mid.Proba_Mid +
                                           val_down * candidate_mid.Proba_Down) * math.exp(-mkt.RiskFree * tree.dt))  # Max entre l'actualisation et le payoff du noeud