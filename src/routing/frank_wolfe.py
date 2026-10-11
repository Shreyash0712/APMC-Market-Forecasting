"""
File: src/routing/frank_wolfe.py
Role: Convex Optimization Engine
Description: Implements the Frank-Wolfe (Conditional Gradient) algorithm to find the 
Wardrop User Equilibrium. It uses a Linear Minimization Oracle (LMO) to iteratively 
balance the routing probability vector without ever requiring expensive projection operations.
Dependencies: numpy
"""

import numpy as np
from src.routing.payoff_functions import compute_net_payoff

def linear_minimization_oracle(current_payoffs):
    """
    The LMO for Wardrop Equilibrium. 
    It finds the 'steepest descent' direction by simply selecting the single mandi 
    that currently offers the highest absolute payoff, assuming everyone else stays put.
    Returns a one-hot vector 's' pointing to that best mandi.
    """
    best_mandi_idx = np.argmax(current_payoffs)
    s = np.zeros_like(current_payoffs)
    s[best_mandi_idx] = 1.0
    return s

def compute_wardrop_equilibrium(predicted_prices, distances, total_village_volume, baseline_capacities, apmc_fee_rate, max_iter=100, tol=1e-4, elasticity=None, diesel_price=None, truck_mileage=None):
    """
    Executes the Frank-Wolfe algorithm to find the optimal stochastic routing distribution.
    
    Args:
        predicted_prices (np.array): ST-GNN baseline price predictions for N candidate mandis.
        distances (np.array): PostGIS physical distances to the N mandis.
        total_village_volume (float): The total tonnes of crop waiting to be routed from the village.
        baseline_capacities (np.array): The organic daily capacity of the N mandis.
        apmc_fee_rate (float): The cached APMC fee decimal rate for the destination state.
        max_iter (int): Maximum Frank-Wolfe iterations.
        tol (float): Convergence tolerance.
        elasticity (float, optional): Custom price flexibility elasticity.
        diesel_price (float, optional): Custom diesel fuel price in INR/liter.
        truck_mileage (float, optional): Custom truck fuel economy in km/L.
        
    Returns:
        np.array: The optimal probability split vector (x) across the N mandis.
    """
    num_mandis = len(predicted_prices)
    
    # Initialize with a uniform distribution (naive state)
    x = np.ones(num_mandis) / num_mandis
    
    for k in range(max_iter):
        # 1. Evaluate Current State
        current_payoffs = np.zeros(num_mandis)
        for i in range(num_mandis):
            allocated_vol = x[i] * total_village_volume
            current_payoffs[i] = compute_net_payoff(
                predicted_price_per_tonne=predicted_prices[i],
                distance_km=distances[i],
                volume_tonnes=1.0, # Evaluate marginal payoff per tonne
                allocated_volume=allocated_vol,
                baseline_capacity=baseline_capacities[i],
                apmc_fee_rate=apmc_fee_rate,
                elasticity=elasticity,
                diesel_price=diesel_price,
                truck_mileage=truck_mileage
            )
            
        # 2. Linear Minimization Oracle (Find best alternative route)
        s = linear_minimization_oracle(current_payoffs)
        
        # 3. Calculate Frank-Wolfe Gap (Convergence Check)
        # The gap is the difference between the max possible payoff (if everyone switched to the best) 
        # and the current expected payoff. If it's near zero, we are at equilibrium.
        current_expected_payoff = np.sum(x * current_payoffs)
        max_possible_payoff = np.max(current_payoffs)
        fw_gap = max_possible_payoff - current_expected_payoff
        
        if fw_gap < tol:
            break
            
        # 4. Update Step (Standard Frank-Wolfe step size gamma = 2 / (k + 2))
        # Note: Because this is a convex combination of two valid probability vectors (x and s),
        # the new x is mathematically guaranteed to remain a valid probability vector (sum=1, x>=0).
        # ZERO projection operations required!
        gamma = 2.0 / (k + 2.0)
        x = (1.0 - gamma) * x + gamma * s
        
    return x
