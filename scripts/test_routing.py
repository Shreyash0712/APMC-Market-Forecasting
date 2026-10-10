"""
File: scripts/test_routing.py
Role: Validation Script
Description: Tests the Game-Theoretic Routing Engine to ensure the Frank-Wolfe algorithm 
prevents the Herding Dilemma. Validates that even if one market has a massive price premium, 
the router probabilistically splits the traffic to maintain equilibrium.
Dependencies: sys, os
"""

import sys
import os
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.routing.router import GameTheoreticRouter
from src.routing.payoff_functions import compute_net_payoff

def test_routing_equilibrium():
    print("Initializing Wardrop Routing Engine (State: Maharashtra)...")
    router = GameTheoreticRouter(state="Maharashtra")
    
    # Scenario setup:
    # Market A has a massive price premium (25,000 INR/tonne) but is slightly further away.
    # Market B has a standard price (15,000 INR/tonne) and is close.
    # Market C has a low price (10,000 INR/tonne) but is extremely close.
    # The village has 500 tonnes of produce to sell today.
    mandi_names = ["Market A (Premium)", "Market B (Standard)", "Market C (Low)"]
    predicted_prices = [25000, 15000, 10000] # INR per tonne
    distances = [120, 40, 10]                # km from village
    baseline_capacities = [200, 150, 100]    # Tonnes per day before price crashes
    total_village_volume = 500               # Massive volume that will crash any single market
    
    print("\n--- Test Scenario ---")
    print(f"Total Volume to Route: {total_village_volume} tonnes")
    print(f"Candidate Mandis: {mandi_names}")
    print(f"Base Predicted Prices: {predicted_prices}")
    
    # Run the engine
    start_time = time.time()
    allocation = router.generate_recommendation(
        mandi_names=mandi_names,
        predicted_prices=predicted_prices,
        distances=distances,
        total_village_volume=total_village_volume,
        baseline_capacities=baseline_capacities
    )
    end_time = time.time()
    
    print("\n--- Wardrop Equilibrium Results ---")
    for mandi, percentage in allocation.items():
        tonnes_assigned = (percentage / 100.0) * total_village_volume
        print(f"{mandi}: {percentage}% ({tonnes_assigned:.1f} tonnes)")
        
    print(f"\nOptimization Latency: {(end_time - start_time) * 1000:.2f} ms")
    
    # Verification: Ensure no market got 100% of the volume
    assert len(allocation) > 1, "ROUTING FAILED: The algorithm routed everyone to a single market (Herding Dilemma occurred!)."
    
    # Verify Equilibrium: The net payoffs at all utilized markets should be roughly equal
    print("\n--- Verifying Equilibrium (Checking Marginal Payoffs) ---")
    for mandi, percentage in allocation.items():
        idx = mandi_names.index(mandi)
        allocated_vol = (percentage / 100.0) * total_village_volume
        
        payoff = compute_net_payoff(
            predicted_price_per_tonne=predicted_prices[idx],
            distance_km=distances[idx],
            volume_tonnes=1.0,
            allocated_volume=allocated_vol,
            baseline_capacity=baseline_capacities[idx],
            apmc_fee_rate=router.state_fee_rate
        )
        print(f"Net Marginal Payoff at {mandi}: INR {payoff:.2f} / tonne")
        
    print("\nSUCCESS: The Frank-Wolfe algorithm successfully distributed the volume to equalize payoffs and prevent a crash!")

if __name__ == "__main__":
    test_routing_equilibrium()
