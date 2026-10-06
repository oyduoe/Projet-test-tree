

class Market:
    def __init__(self, input: dict):
        """Initialiser les paramètres du marché à partir d'un dictionnaire d'input."""
        self.SpotPrice = input["SpotPrice"]  # Prix spot du sous-jacent
        self.RiskFree = input["RiskFree"] /100  # Taux d'intérêt de marché
        self.volatility = input["Volatility"] /100  # Volatilité
        self.dividend = input["Dividend"]  # Montant du dividende
        self.div_date = self.compute_date_div(input)

    def compute_date_div(self, input: dict) -> float:
        """Calcule à quel pas de temps le dividende tombe."""
        delta_days = (input["DivExDate"] -input["PricingDate"] ).days
        return delta_days / 365  # Retourner la fraction d'année

    def compute_dividend(self, T: int, dt: float) -> float:
        """Prendre la valeur du dividende si l'on est à la date du dividende."""
        if (T * dt < self.div_date <= (T + 1) * dt) or (T == 1 and self.div_date <= dt):
            return self.dividend  # Si on est dans la période du dividende, on retourne sa valeur
        else:
            return 0.0  # Si on n'est pas à la date du dividende, on retourne 0