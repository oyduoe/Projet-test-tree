import numpy as np
from scipy.stats import norm


class Extension:
    def __init__(self, market, params, pricer):
        self.market = market
        self.params = params
        self.pricer = pricer

    # ==================================================================
    # BLACK-SCHOLES — supporte Discrete (Merton) et Continuous (yield)
    # ==================================================================
    def _bs_setup(self):
        """Renvoie (S_adj, q, ttm) selon le mode de dividende.

        - Continuous : S_adj = S, q = D / S
        - Discrete   : S_adj = S - PV(D), q = 0  (modèle de Merton)
        """
        ttm = (self.params.DateMaturity - self.pricer.PricingDate).days / 365.0
        div_type = getattr(self.market, 'dividend_type', 'Discrete')
        S = self.market.SpotPrice
        r = self.market.RiskFree
        D = self.market.dividend
        t_div = self.market.div_date

        if div_type == "Continuous":
            S_adj = S
            q = D / S if S > 0 else 0.0
        else:  # Discrete
            if D > 0 and 0 < t_div < ttm:
                S_adj = S - D * np.exp(-r * t_div)
            else:
                S_adj = S
            q = 0.0
        return S_adj, q, ttm

    def black_sholes(self):
        S, q, ttm = self._bs_setup()
        if ttm <= 0:
            self.pricer.BS_price = self.params.Vi(S)
            return {
                "BS Price": self.pricer.BS_price,
                "Delta": 0.0, "Gamma": 0.0, "Vega": 0.0,
                "Vomma": 0.0, "Vanna": 0.0, "Theta": 0.0, "Rho": 0.0
            }

        d1 = self.compute_d1(ttm, S, q)
        d2 = self.compute_d2(d1, ttm)

        if self.params.type_contrat == "Call":
            self.pricer.BS_price = self.compute_call_price(ttm, d1, d2, S, q)
            delta = norm.cdf(d1)
            rho = self.params.strike * ttm * np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2)
            theta = self.compute_theta_call(ttm, d1, d2, S, q)
        else:
            self.pricer.BS_price = self.compute_put_price(ttm, d1, d2, S, q)
            delta = norm.cdf(d1) - 1
            rho = -self.params.strike * ttm * np.exp(-self.market.RiskFree * ttm) * norm.cdf(-d2)
            theta = self.compute_theta_put(ttm, d1, d2, S, q)

        vega = S * np.exp(-q * ttm) * np.sqrt(ttm) * norm.pdf(d1)
        gamma = np.exp(-q * ttm) * norm.pdf(d1) / (S * self.market.volatility * np.sqrt(ttm))
        vomma = vega * d1 * d2 / self.market.volatility
        vanna = -np.exp(-q * ttm) * norm.pdf(d1) * d2 / self.market.volatility

        return {
            "BS Price": self.pricer.BS_price,
            "Delta": delta, "Gamma": gamma, "Vega": vega,
            "Vomma": vomma, "Vanna": vanna,
            "Theta": theta, "Rho": rho,
        }

    def compute_d1(self, ttm, S, q):
        return (np.log(S / self.params.strike) +
                (self.market.RiskFree - q + 0.5 * self.market.volatility ** 2) * ttm) / \
               (self.market.volatility * np.sqrt(ttm))

    def compute_d2(self, d1, ttm):
        return d1 - self.market.volatility * np.sqrt(ttm)

    def compute_call_price(self, ttm, d1, d2, S, q):
        return (S * np.exp(-q * ttm) * norm.cdf(d1) -
                self.params.strike * np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2))

    def compute_put_price(self, ttm, d1, d2, S, q):
        return (self.params.strike * np.exp(-self.market.RiskFree * ttm) * norm.cdf(-d2) -
                S * np.exp(-q * ttm) * norm.cdf(-d1))

    def compute_theta_call(self, ttm, d1, d2, S, q):
        return (-S * np.exp(-q * ttm) * norm.pdf(d1) * self.market.volatility / (2 * np.sqrt(ttm)) -
                self.market.RiskFree * self.params.strike *
                np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2) +
                q * S * np.exp(-q * ttm) * norm.cdf(d1)) / 365

    def compute_theta_put(self, ttm, d1, d2, S, q):
        return (-S * np.exp(-q * ttm) * norm.pdf(d1) * self.market.volatility / (2 * np.sqrt(ttm)) -
                self.market.RiskFree * self.params.strike *
                np.exp(-self.market.RiskFree * ttm) * (1 - norm.cdf(d2)) -
                q * S * np.exp(-q * ttm) * (1 - norm.cdf(d1))) / 365

    # ==================================================================
    # MONTE CARLO — supporte Discrete et Continuous
    # ==================================================================
    def monte_carlo_price(self, n_simulations: int) -> float:
        ttm = (self.params.DateMaturity - self.pricer.PricingDate).days / 365.0
        payoff_sum = 0.0

        for _ in range(n_simulations):
            s = self.simulate_asset_price(self.market.SpotPrice, ttm)
            if self.params.type_contrat == "Call":
                payoff = max(s - self.params.strike, 0)
            else:
                payoff = max(self.params.strike - s, 0)
            payoff_sum += payoff

        self.pricer.MC_price = np.exp(-self.market.RiskFree * ttm) * (payoff_sum / n_simulations)
        return self.pricer.MC_price

    def simulate_asset_price(self, S: float, ttm: float) -> float:
        sigma = self.market.volatility
        r = self.market.RiskFree
        dt = ttm / self.pricer.timeSteps
        div_type = getattr(self.market, 'dividend_type', 'Discrete')
        q = self.market.dividend_yield() if div_type == "Continuous" else 0.0

        for j in range(self.pricer.timeSteps):
            z = np.random.normal()
            S = S * np.exp((r - q - 0.5 * sigma ** 2) * dt +
                            sigma * np.sqrt(dt) * z)
            if div_type == "Discrete":
                t_prev = j * dt
                t_next = (j + 1) * dt
                if t_prev < self.market.div_date <= t_next:
                    S = max(S - self.market.dividend, 0.0)
            if S < 0:
                S = 0.0
        return S

    def monte_carlo_greeks(self, n_simulations: int = 20000) -> dict:
        def price_fn(m, p, pr):
            np.random.seed(42)
            return self.monte_carlo_price(n_simulations)
        from Greeks import finite_difference_greeks
        return finite_difference_greeks(price_fn, self.market, self.params, self.pricer)