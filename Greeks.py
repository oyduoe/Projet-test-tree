import numpy as np
from datetime import timedelta


def finite_difference_greeks(price_fn, mkt, params, pricer,
                             h_S_rel=0.01, h_vol=0.01, h_T_days=1):
    """
    Calcule Delta, Gamma, Vega, Vomma, Vanna, Theta par différences finies.

    price_fn(mkt, params, pricer) -> float :
        fonction qui renvoie le prix pour le modèle considéré.
        Les objets mkt/params/pricer sont mutés puis restaurés.

    Attention : les objets mkt et pricer sont mutés temporairement puis restaurés.
    """
    S0 = mkt.SpotPrice
    vol0 = mkt.volatility
    PD0 = pricer.PricingDate

    h_S = max(h_S_rel * S0, 1e-4)
    h_T = h_T_days / 365.0

    # Prix de référence
    P0 = price_fn(mkt, params, pricer)

    # Delta / Gamma
    mkt.SpotPrice = S0 + h_S
    Pu = price_fn(mkt, params, pricer)
    mkt.SpotPrice = S0 - h_S
    Pd = price_fn(mkt, params, pricer)
    mkt.SpotPrice = S0
    delta = (Pu - Pd) / (2 * h_S)
    gamma = (Pu - 2 * P0 + Pd) / (h_S ** 2)

    # Vega / Vomma
    mkt.volatility = vol0 + h_vol
    Pvu = price_fn(mkt, params, pricer)
    mkt.volatility = vol0 - h_vol
    Pvd = price_fn(mkt, params, pricer)
    mkt.volatility = vol0
    vega = (Pvu - Pvd) / (2 * h_vol)
    vomma = (Pvu - 2 * P0 + Pvd) / (h_vol ** 2)

    # Vanna
    mkt.SpotPrice = S0 + h_S; mkt.volatility = vol0 + h_vol
    Ppp = price_fn(mkt, params, pricer)
    mkt.SpotPrice = S0 + h_S; mkt.volatility = vol0 - h_vol
    Ppm = price_fn(mkt, params, pricer)
    mkt.SpotPrice = S0 - h_S; mkt.volatility = vol0 + h_vol
    Pmp = price_fn(mkt, params, pricer)
    mkt.SpotPrice = S0 - h_S; mkt.volatility = vol0 - h_vol
    Pmm = price_fn(mkt, params, pricer)
    mkt.SpotPrice = S0; mkt.volatility = vol0
    vanna = (Ppp - Ppm - Pmp + Pmm) / (4 * h_S * h_vol)

    # Theta (par jour calendaire)
    pricer.PricingDate = PD0 + timedelta(days=h_T_days)
    Ptp = price_fn(mkt, params, pricer)
    pricer.PricingDate = PD0 - timedelta(days=h_T_days)
    Ptm = price_fn(mkt, params, pricer)
    pricer.PricingDate = PD0
    theta = -(Ptp - Ptm) / (2 * h_T) / 365.0

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Vomma": vomma,
        "Vanna": vanna,
        "Theta": theta,
    }