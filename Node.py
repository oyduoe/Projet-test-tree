import math
from Market import Market
from OptionParameters import OptionParameters


class Node:
    def __init__(self):
        self.UndPrice: float
        self.Payoff: float
        self.Next_Up = None
        self.Next_Down = None
        self.Next_Mid = None
        self.nprec = None
        self.UpNode = None
        self.DownNode = None
        self.MidNode = None
        self.Proba_Up: float = 0.00
        self.Proba_Down: float = 0.00
        self.Proba_Mid: float = 0.00
        self.Cum_Proba: float = 0.00
        self.OptPrice: float

    def forward(self, mkt: Market, dt: float, div: float):
        """Valeur forward du nœud.

        - Mode Continuous : drift (r - q) où q = D / S.
        - Mode Discrete   : drift r, puis soustraction du cash `div`
                            au pas exact du dividende.
        """
        div_type = getattr(mkt, 'dividend_type', 'Discrete')
        if div_type == "Continuous":
            q = mkt.dividend_yield()
            return self.UndPrice * math.exp((mkt.RiskFree - q) * dt)
        return self.UndPrice * math.exp(mkt.RiskFree * dt) - div

    def init_node(self, market_value: float):
        self.UndPrice = market_value

    def incremental_node(self, candidate_mid, is_up: bool, condition: bool):
        if is_up:
            if candidate_mid.UpNode is not None:
                candidate_mid = candidate_mid.UpNode
            else:
                condition = False
        else:
            if candidate_mid.DownNode is not None:
                candidate_mid = candidate_mid.DownNode
            else:
                condition = False
        return candidate_mid, condition

    def price(self, candidate_mid, parameters, mkt, tree):
        while candidate_mid.nprec is not None:
            condition = True
            trunk = candidate_mid
            while condition:
                self.payoff_condition(candidate_mid, parameters, mkt, tree)
                candidate_mid, condition = self.incremental_node(candidate_mid, True, condition)
            candidate_mid = trunk.DownNode
            condition = True
            while condition:
                self.payoff_condition(candidate_mid, parameters, mkt, tree)
                candidate_mid, condition = self.incremental_node(candidate_mid, False, condition)
            candidate_mid = trunk
            candidate_mid = candidate_mid.nprec

        self.discounted_price(candidate_mid, parameters, mkt, tree)
        if candidate_mid.UpNode is not None:
            self.discounted_price(candidate_mid.UpNode, parameters, mkt, tree)
            self.discounted_price(candidate_mid.DownNode, parameters, mkt, tree)

    def payoff_condition(self, candidate_mid, parameters, mkt, tree):
        if candidate_mid.Next_Mid is None:
            candidate_mid.OptPrice = parameters.Vi(candidate_mid.UndPrice)
        else:
            self.discounted_price(candidate_mid, parameters, mkt, tree)

    def discounted_price(self, candidate_mid, parameters, mkt, tree):
        val_up = val_mid = val_down = 0.0
        if candidate_mid.Next_Up is not None:
            val_up = candidate_mid.Next_Up.OptPrice
            candidate_mid.Next_Up = None
        if candidate_mid.Next_Mid is not None:
            val_mid = candidate_mid.Next_Mid.OptPrice
            candidate_mid.Next_Mid = None
        if candidate_mid.Next_Down is not None:
            val_down = candidate_mid.Next_Down.OptPrice
            candidate_mid.Next_Down = None

        if parameters.Exercice == "European":
            candidate_mid.OptPrice = (val_up * candidate_mid.Proba_Up +
                                       val_mid * candidate_mid.Proba_Mid +
                                       val_down * candidate_mid.Proba_Down) * math.exp(-mkt.RiskFree * tree.dt)
        elif parameters.Exercice == "American":
            candidate_mid.OptPrice = max(parameters.Vi(candidate_mid.UndPrice),
                                          (val_up * candidate_mid.Proba_Up +
                                           val_mid * candidate_mid.Proba_Mid +
                                           val_down * candidate_mid.Proba_Down) * math.exp(-mkt.RiskFree * tree.dt))