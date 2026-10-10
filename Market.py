class Market:
    def __init__(self, input: dict):
        """Initialiser les paramètres du marché à partir d'un dictionnaire d'input."""
        self.SpotPrice = input["SpotPrice"]
        self.RiskFree = input["RiskFree"] / 100
        self.volatility = input["Volatility"] / 100
        self.dividend = input["Dividend"]

        # Type de dividende : "Continuous" (yield) ou "Discrete" (cash amount)
        self.dividend_type = input.get("DividendType", "Discrete")

        self.div_date = self.compute_date_div(input)

    def compute_date_div(self, input: dict) -> float:
        """Date ex-dividende en fraction d'année depuis PricingDate."""
        delta_days = (input["DivExDate"] - input["PricingDate"]).days
        return delta_days / 365

    def dividend_yield(self) -> float:
        """Rendement continu équivalent q = D / S (utilisé en mode Continuous)."""
        if self.SpotPrice <= 0:
            return 0.0
        return self.dividend / self.SpotPrice

    def compute_dividend(self, T: int, dt: float) -> float:
        """Valeur du dividende cash au pas T (utilisé par le Trinomial en mode Discrete)."""
        if self.dividend_type == "Continuous":
            return 0.0
        if (T * dt < self.div_date <= (T + 1) * dt) or (T == 1 and self.div_date <= dt):
            return self.dividend
        return 0.0