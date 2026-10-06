import numpy as np
from scipy.stats import norm


class Extension:
    def __init__(self, market, params, pricer):
        self.market = market
        self.params = params
        self.pricer = pricer

    def black_sholes(self): #'Calcul du prix de l'option avec Black & Scholes ainsi que des grecques
        ttm = (self.params.DateMaturity - self.pricer.PricingDate).days / 365.0  # Time to maturity
        d1 = self.compute_d1(ttm) #Détermination de d1 à partir d'une fonction pour éviter les calculs redondants
        d2 = self.compute_d2(d1, ttm) #Détermination de d2 à partir d'une fonction pour éviter les calculs redondants

        if self.params.type_contrat == "Call": #Si c'est un call, calcul du prix et des grecques dépendant du type de contrat
            self.pricer.BS_price = self.compute_call_price(ttm, d1, d2) #Pricing de Black&Scholes pour un call
            delta = self.compute_call_delta(d1) #Delta pour un call
            rho = self.compute_rho_call(ttm, d2) #Rho pour un call
            theta = self.compute_theta_call(ttm, d1, d2) #Theta pour un call
        else:  # Si c'est un Put, calcul du prix et des grecques dépendant du type de contrat
            self.pricer.BS_price = self.compute_put_price(ttm, d1, d2) #Pricing de Black&Scholes pour un put
            delta = self.compute_put_delta(d1) #Delta pour un put
            rho = self.compute_rho_put(ttm, d2) #Rho pour un put
            theta = self.compute_theta_put(ttm, d1, d2) #Theta pour un put
        #Grecques non dépendantes du type de contrat
        vega = self.compute_vega(ttm, d1) #Vega Call ou Put
        gamma = self.compute_gamma(ttm, d1) #Gamma Call ou Put

        return {
            "BS Price": self.pricer.BS_price,
            "Delta": delta,
            "Vega": vega,
            "Rho": rho,
            "Gamma": gamma,
            "Theta": theta
        }

    def compute_d1(self, ttm:float) -> float:#Fonction de calcul de d1
        q = self.market.dividend / self.market.SpotPrice
        return (np.log(self.market.SpotPrice / self.params.strike) +
                (self.market.RiskFree - q + 0.5 * self.market.volatility ** 2) * ttm) / \
               (self.market.volatility * np.sqrt(ttm))

    def compute_d2(self, d1:float, ttm:float) -> float: #Fonction de calcul de d2 à partir de d1
        return d1 - self.market.volatility * np.sqrt(ttm)

    def compute_call_price(self, ttm:float, d1:float, d2:float)-> float: #fonction de calcul prix d'un call avec Black&Scholes
        q = self.market.dividend / self.market.SpotPrice
        call_price = (self.market.SpotPrice * np.exp(-q * ttm) * norm.cdf(d1) -
                      self.params.strike * np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2))
        return call_price

    def compute_put_price(self, ttm:float, d1:float, d2:float)->float: #fonction de calcul prix d'un put avec Black&Scholes
        q = self.market.dividend / self.market.SpotPrice
        put_price = (self.params.strike * np.exp(-self.market.RiskFree * ttm) * norm.cdf(-d2) -
                     self.market.SpotPrice * np.exp(-q * ttm) * norm.cdf(-d1))
        return put_price

    def compute_call_delta(self, d1:float)-> float: #Fonction de calcul du delta d'un call
        return norm.cdf(d1)

    def compute_put_delta(self, d1:float)-> float: #fonction de calcul du delta d'un put
        return norm.cdf(d1) - 1

    def compute_vega(self, ttm:float, d1:float)-> float: #fonction de calcul du vega de l'option
        q = self.market.dividend / self.market.SpotPrice
        return (self.market.SpotPrice * np.exp(-q * ttm) * np.sqrt(ttm) * norm.pdf(d1))

    def compute_gamma(self, ttm:float, d1:float) -> float:# fonction de calcul du gamma de l'option
        q = self.market.dividend / self.market.SpotPrice
        return (np.exp(-q * ttm) * norm.pdf(d1)) / (self.market.SpotPrice * self.market.volatility * np.sqrt(ttm))

    def compute_rho_call(self, ttm:float, d2:float)-> float: #fonction de calcul du rho d'un call
        return (self.params.strike * ttm * np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2))

    def compute_rho_put(self, ttm:float, d2:float)-> float: #fonction de calcul du rho d'un put
        return - (self.params.strike * ttm * np.exp(-self.market.RiskFree * ttm) * norm.cdf(-d2))

    def compute_theta_call(self, ttm:float, d1:float, d2:float)-> float: #fonction de calcul du theta d'un call
        q = self.market.dividend / self.market.SpotPrice
        return (-self.market.SpotPrice * np.exp(-q * ttm) * norm.pdf(d1) * self.market.volatility / (2 * np.sqrt(ttm)) -
                self.market.RiskFree * self.params.strike * np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2) +
                q * self.market.SpotPrice * np.exp(-q * ttm) * norm.cdf(d1)) / 365

    def compute_theta_put(self, ttm:float, d1:float, d2:float) -> float: #fonction de calcul du theta d'un put
        q = self.market.dividend / self.market.SpotPrice
        return (-self.market.SpotPrice * np.exp(-q * ttm) * norm.pdf(d1) * self.market.volatility / (2 * np.sqrt(ttm)) -
                self.market.RiskFree * self.params.strike * np.exp(-self.market.RiskFree * ttm) * (1 - norm.cdf(d2)) -
                q * self.market.SpotPrice * np.exp(-q * ttm) * (1 - norm.cdf(d1))) / 365

    def monte_carlo_price(self, n_simulations:int)-> float: #Subroutine de pricing de Monte Carlo
        ttm = (self.params.DateMaturity - self.pricer.PricingDate).days / 365.0
        payoff_sum = 0.0

        for _ in range(n_simulations):
            s = self.simulate_asset_price(self.market.SpotPrice,ttm)
            if self.params.type_contrat == "Call":
                payoff = max(s - self.params.strike, 0)
            else:
                payoff = max(self.params.strike - s, 0)
            payoff_sum += payoff

        self.pricer.MC_price = np.exp(-self.market.RiskFree * ttm) * (payoff_sum / n_simulations)
        return self.pricer.MC_price

    def simulate_asset_price(self,S:float, ttm:float)-> float:#Méthode modélisant les simulations de Monte Carlo

        for j in range(self.pricer.timeSteps):
            z = np.random.normal()  # standard normal random variable
            S = S * np.exp((self.market.RiskFree - (self.market.dividend / self.market.SpotPrice)-0.5* self.market.volatility**2)*
                            (ttm/self.pricer.timeSteps)+self.market.volatility * np.sqrt(ttm/self.pricer.timeSteps)*z)
            if S < 0 : S = 0
        return  S
