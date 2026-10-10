import math
import numpy as np


class Binomial:
    """Pricer Binomial CRR (Cox-Ross-Rubinstein).

    Gère deux modes de dividende :
      - "Continuous" : dividende = rendement continu q = D / S0
      - "Discrete"   : dividende = montant cash à une date ex-dividende,
                       géré via le modèle « escrowed » (on retire la PV
                       du dividende du spot, puis on l'ajoute au spot
                       observable à chaque nœud selon la date).
    """

    def __init__(self, mkt, params, pricer):
        self.market = mkt
        self.params = params
        self.pricer = pricer
        self.price = 0.0

    # ------------------------------------------------------------------
    def price_option(self, n_steps=None) -> float:
        if n_steps is None:
            n_steps = self.pricer.timeSteps
        n_steps = max(int(n_steps), 1)

        S0 = self.market.SpotPrice
        K = self.params.strike
        r = self.market.RiskFree
        sigma = self.market.volatility
        T = (self.params.DateMaturity - self.pricer.PricingDate).days / 365.0

        if T <= 0:
            self.price = self.params.Vi(S0)
            return self.price

        dt = T / n_steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        is_american = (self.params.Exercice == "American")

        # ------- Paramétrage selon le type de dividende -------
        div_type = getattr(self.market, 'dividend_type', 'Discrete')

        if div_type == "Continuous":
            q = self.market.dividend_yield()
            p = (math.exp((r - q) * dt) - d) / (u - d)
            S0_tree = S0
            div_schedule = []              # rien à rajouter
        else:  # Discrete — modèle escrowed
            ex_date = self.market.div_date
            D = self.market.dividend
            if D > 0 and ex_date > 0:
                pv = D * math.exp(-r * ex_date)
                div_schedule = [(ex_date, D)]
            else:
                pv = 0.0
                div_schedule = []
            S0_tree = S0 - pv
            p = (math.exp(r * dt) - d) / (u - d)

        # ------- Valeurs terminales -------
        values = np.empty(n_steps + 1, dtype=float)
        for j in range(n_steps + 1):
            S_tree = S0_tree * (u ** (n_steps - j)) * (d ** j)
            S_obs = self._observable_spot(S_tree, T, div_schedule, r)
            values[j] = self.params.Vi(S_obs)

        # ------- Backward induction -------
        for i in range(n_steps - 1, -1, -1):
            t_i = i * dt
            for j in range(i + 1):
                values[j] = disc * (p * values[j] + (1 - p) * values[j + 1])
                if is_american:
                    S_tree = S0_tree * (u ** (i - j)) * (d ** j)
                    S_obs = self._observable_spot(S_tree, t_i, div_schedule, r)
                    values[j] = max(values[j], self.params.Vi(S_obs))

        self.price = float(values[0])
        return self.price

    # ------------------------------------------------------------------
    def _observable_spot(self, S_tree, t, div_schedule, r):
        """Ajoute la PV des dividendes futurs non encore payés à t."""
        S_obs = S_tree
        for (ex_date, amount) in div_schedule:
            if ex_date >= t:
                S_obs += amount * math.exp(-r * (ex_date - t))
        return S_obs