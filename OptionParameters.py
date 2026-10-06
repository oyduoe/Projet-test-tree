
class OptionParameters:
    def __init__(self, input: dict):
        """Initialiser les paramètres de l'option à partir d'un dictionnaire d'input."""
        self.strike = input["Strike"]  # Prix d'exercice de l'option
        self.type_contrat = input["Type"]  # Type d'option ("call" ou "put")
        self.Exercice = input["Exercice"]  # Origine de l'option ("American" ou "European")
        self.DateMaturity = input["Maturity"]  # Date de maturité de l'option
        self.date_dividende = input["DivExDate"]  # Date de dividende

    def Vi(self, S: float) -> float:
        """Calculer le payoff en fonction du type d'option."""
        if self.type_contrat == "Call":
            return max(S - self.strike, 0)  # Formule du payoff pour un call
        elif self.type_contrat == "Put":
            return max(self.strike - S, 0)  # Formule du payoff pour un put
        else:
            raise ValueError("Type de contrat non valide. Utilisez 'Call' ou 'Put'.")