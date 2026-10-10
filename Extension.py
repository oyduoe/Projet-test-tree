import numpy as np
from scipy.stats import norm
from datetime import timedelta


class Extension:
    def __init__(self, market, params, pricer):
        self.market = market
        self.params = params
        self.pricer = pricer

    # ------------------------------------------------------------------
    # BLACK-SCHOLES
    # ------------------------------------------------------------------
    def black_sholes(self):
        ttm = (self.params.DateMaturity - self.pricer.PricingDate).days / 365.0
        d1 = self.compute_d1(ttm)
        d2 = self.compute_d2(d1, ttm)

        if self.params.type_contrat == "Call":
            self.pricer.BS_price = self.compute_call_price(ttm, d1, d2)
            delta = self.compute_call_delta(d1)
            rho = self.compute_rho_call(ttm, d2)
            theta = self.compute_theta_call(ttm, d1, d2)
        else:
            self.pricer.BS_price = self.compute_put_price(ttm, d1, d2)
            delta = self.compute_put_delta(d1)
            rho = self.compute_rho_put(ttm, d2)
            theta = self.compute_theta_put(ttm, d1, d2)

        vega = self.compute_vega(ttm, d1)
        gamma = self.compute_gamma(ttm, d1)
        vomma = self.compute_vomma(ttm, d1, d2)
        vanna = self.compute_vanna(ttm, d1, d2)

        return {
            "BS Price": self.pricer.BS_price,
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Vomma": vomma,
            "Vanna": vanna,
            "Theta": theta,
            "Rho": rho,
        }

    def compute_d1(self, ttm: float) -> float:
        q = self.market.dividend / self.market.SpotPrice
        return (np.log(self.market.SpotPrice / self.params.strike) +
                (self.market.RiskFree - q + 0.5 * self.market.volatility ** 2) * ttm) / \
               (self.market.volatility * np.sqrt(ttm))

    def compute_d2(self, d1: float, ttm: float) -> float:
        return d1 - self.market.volatility * np.sqrt(ttm)

    def compute_call_price(self, ttm, d1, d2):
        q = self.market.dividend / self.market.SpotPrice
        return (self.market.SpotPrice * np.exp(-q * ttm) * norm.cdf(d1) -
                self.params.strike * np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2))

    def compute_put_price(self, ttm, d1, d2):
        q = self.market.dividend / self.market.SpotPrice
        return (self.params.strike * np.exp(-self.market.RiskFree * ttm) * norm.cdf(-d2) -
                self.market.SpotPrice * np.exp(-q * ttm) * norm.cdf(-d1))

    def compute_call_delta(self, d1):
        return norm.cdf(d1)

    def compute_put_delta(self, d1):
        return norm.cdf(d1) - 1

    def compute_vega(self, ttm, d1):
        q = self.market.dividend / self.market.SpotPrice
        return self.market.SpotPrice * np.exp(-q * ttm) * np.sqrt(ttm) * norm.pdf(d1)

    def compute_gamma(self, ttm, d1):
        q = self.market.dividend / self.market.SpotPrice
        return (np.exp(-q * ttm) * norm.pdf(d1)) / \
               (self.market.SpotPrice * self.market.volatility * np.sqrt(ttm))

    def compute_vomma(self, ttm, d1, d2):
        vega = self.compute_vega(ttm, d1)
        return vega * d1 * d2 / self.market.volatility

    def compute_vanna(self, ttm, d1, d2):
        q = self.market.dividend / self.market.SpotPrice
        return -np.exp(-q * ttm) * norm.pdf(d1) * d2 / self.market.volatility

    def compute_rho_call(self, ttm, d2):
        return self.params.strike * ttm * np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2)

    def compute_rho_put(self, ttm, d2):
        return -self.params.strike * ttm * np.exp(-self.market.RiskFree * ttm) * norm.cdf(-d2)

    def compute_theta_call(self, ttm, d1, d2):
        q = self.market.dividend / self.market.SpotPrice
        return (-self.market.SpotPrice * np.exp(-q * ttm) * norm.pdf(d1) *
                self.market.volatility / (2 * np.sqrt(ttm)) -
                self.market.RiskFree * self.params.strike *
                np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2) +
                q * self.market.SpotPrice * np.exp(-q * ttm) * norm.cdf(d1)) / 365

    def compute_theta_put(self, ttm, d1, d2):
        q = self.market.dividend / self.market.SpotPrice
        return (-self.market.SpotPrice * np.exp(-q * ttm) * norm.pdf(d1) *
                self.market.volatility / (2 * np.sqrt(ttm)) -
                self.market.RiskFree * self.params.strike *
                np.exp(-self.market.RiskFree * ttm) * (1 - norm.cdf(d2)) -
                q * self.market.SpotPrice * np.exp(-q * ttm) *
                (1 - norm.cdf(d1))) / 365

    # ------------------------------------------------------------------
    # MONTE CARLO
    # ------------------------------------------------------------------
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
        for j in range(self.pricer.timeSteps):
            z = np.random.normal()
            S = S * np.exp((self.market.RiskFree -
                            (self.market.dividend / self.market.SpotPrice) -
                            0.5 * self.market.volatility ** 2) *
                           (ttm / self.pricer.timeSteps) +
                           self.market.volatility *
                           np.sqrt(ttm / self.pricer.timeSteps) * z)
            if S < 0:
                S = 0
        return S

    def monte_carlo_greeks(self, n_simulations: int = 20000) -> dict:
        """Greeks MC par différences finies avec nombres aléatoires communs."""
        def price_fn(m, p, pr):
            np.random.seed(42)  # nombres aléatoires communs
            return self.monte_carlo_price(n_simulations)

        from Greeks import finite_difference_greeks
        return finite_difference_greeks(price_fn, self.market, self.params, self.pricer)