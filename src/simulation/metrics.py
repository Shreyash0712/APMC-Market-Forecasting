"""
File: src/simulation/metrics.py
Role: Agent-Based Modeling Data Collection
Description: Contains the logic to compute macro-economic metrics during the simulation, 
such as the Price Dispersion Index (Coefficient of Variation).
Dependencies: numpy
"""

import numpy as np

def compute_price_dispersion_index(model):
    """
    Computes the Coefficient of Variation (CV) of prices across all mandis.
    CV = Standard Deviation / Mean.
    A lower CV indicates high spatial market integration (stable prices everywhere).
    A high CV indicates severe localized market crashes or spikes.
    """
    prices = [m["current_price"] for m in model.mandi_states.values()]
    mean_price = np.mean(prices)
    if mean_price == 0:
        return 0
    std_price = np.std(prices)
    return std_price / mean_price
