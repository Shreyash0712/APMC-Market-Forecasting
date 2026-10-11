"""
File: scripts/test_trust_simulation.py
Role: Behavioral Adherence & Disruption Routing Benchmark (Test 2)
Description: Evaluates how farmer compliance (driven by explainability trust) and 
real-time exogenous disruption awareness impact macroeconomic stability and farmer revenue.
Simulates 1,000 farmers across the compliance spectrum and under a flood stress test.
Consumes 0 API credits.
Dependencies: mesa, numpy, pandas
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.simulation.model import APMCMesaModel
from src.routing.payoff_functions import inverse_demand_function, compute_net_payoff

class TrustAwareMesaModel(APMCMesaModel):
    """Extends APMCMesaModel to introduce dynamic farmer compliance rates and real-time flood stress."""
    def __init__(self, num_agents, compliance_rate=0.85, state="Maharashtra", flood_active=False, disruption_aware=True):
        super().__init__(num_agents=num_agents, scenario="Intervention", state=state)
        self.compliance_rate = compliance_rate
        self.flood_active = flood_active
        self.disruption_aware = disruption_aware

    def step(self):
        # 1. Reset daily volumes
        for mandi in self.mandi_states.values():
            mandi["arriving_volume"] = 0.0
            
        mandi_names = list(self.mandi_states.keys())
        predicted_prices = np.array([m["current_price"] for m in self.mandi_states.values()])
        distances = np.array([m["distance_km"] for m in self.mandi_states.values()])
        baseline_capacities = np.array([m["capacity"] for m in self.mandi_states.values()])
        total_village_volume = sum([a.volume_tonnes for a in self.agents])
        
        # 2. Exogenous Disruption Context
        disruptions = {}
        if self.flood_active:
            # Market A's highway is severely flooded (accessibility = 0.15)
            disruptions["Market A (Premium)"] = {
                "road_accessibility": 0.15 if self.disruption_aware else 1.0,
                "supply_disruption": 0.85
            }
        
        # 3. Get Wardrop routing recommendation with anticipatory buffering
        self.current_routing_recommendation = self.router.generate_recommendation(
            mandi_names=mandi_names,
            predicted_prices=predicted_prices,
            distances=distances,
            total_village_volume=total_village_volume,
            baseline_capacities=baseline_capacities,
            adherence_rate=self.compliance_rate,
            disruptions=disruptions if self.disruption_aware else None
        )
        
        # 4. Agents decide based on compliance probability
        highest_price_mandi = max(self.mandi_states.keys(), key=lambda k: self.mandi_states[k]["current_price"])
        
        mandis = list(self.current_routing_recommendation.keys())
        probs = np.array([float(self.current_routing_recommendation[m]) / 100.0 for m in mandis])
        probs = probs / np.sum(probs)
        
        for agent in self.agents:
            if np.random.rand() < self.compliance_rate:
                # Compliant agent follows anticipatory Wardrop split
                agent.chosen_mandi = np.random.choice(mandis, p=probs)
            else:
                # Rogue agent herds to yesterday's highest price market
                agent.chosen_mandi = highest_price_mandi
                
            self.mandi_states[agent.chosen_mandi]["arriving_volume"] += agent.volume_tonnes
            
        # 5. Market clearing price adjustment (Inverse demand)
        for mandi_data in self.mandi_states.values():
            mandi_data["current_price"] = inverse_demand_function(
                base_price=mandi_data["baseline_price"],
                allocated_volume=mandi_data["arriving_volume"],
                baseline_capacity=mandi_data["capacity"]
            )
            
        # 6. Agent payout calculation (accounting for physical flood road penalties)
        for agent in self.agents:
            chosen = self.mandi_states[agent.chosen_mandi]
            dist = chosen["distance_km"]
            # If flood is active and agent chose Market A, road penalty causes 4x transit delay/spoilage
            if self.flood_active and agent.chosen_mandi == "Market A (Premium)":
                dist = dist * 4.0
                
            profit = compute_net_payoff(
                predicted_price_per_tonne=chosen["current_price"],
                distance_km=dist,
                volume_tonnes=agent.volume_tonnes,
                allocated_volume=chosen["arriving_volume"],
                baseline_capacity=chosen["capacity"],
                apmc_fee_rate=self.router.state_fee_rate
            )
            agent.daily_profit = profit * agent.volume_tonnes
            
        # 7. Collect metrics
        self.datacollector.collect(self)

def run_compliance_regime(name, compliance_rate, days=25, num_agents=800, flood_active=False, disruption_aware=True):
    print(f"Simulating: {name} (Compliance: {compliance_rate*100:.0f}%, Disruption Aware: {disruption_aware})...")
    np.random.seed(42)
    model = TrustAwareMesaModel(
        num_agents=num_agents, 
        compliance_rate=compliance_rate,
        flood_active=flood_active,
        disruption_aware=disruption_aware
    )
    for _ in range(days):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    avg_pdi = float(df["Price_Dispersion_Index"].mean())
    avg_profit = float(np.mean([a.daily_profit for a in model.agents]))
    return avg_pdi, avg_profit

def run_trust_benchmark():
    print("=== [TEST 2] Running Farmer Trust & Disruption-Aware Mesa Simulation ===")
    
    # 1. Normal Conditions: Compliance Spectrum
    print("\n--- Part A: The Compliance Trust Spectrum ---")
    pdi_zero, prof_zero = run_compliance_regime("1. Zero Compliance (Pure Herding)", compliance_rate=0.0)
    pdi_blackbox, prof_blackbox = run_compliance_regime("2. Black-Box AI (Low Trust / 35% Adherence)", compliance_rate=0.35)
    pdi_explainable, prof_explainable = run_compliance_regime("3. Explainable AI (High Trust / 85% Adherence)", compliance_rate=0.85)
    pdi_full, prof_full = run_compliance_regime("4. Optimal Full Equilibrium (100% Adherence)", compliance_rate=1.0)
    
    volatility_drop_full = ((pdi_zero - pdi_full) / pdi_zero) * 100
    profit_gain_full = ((prof_full - prof_zero) / max(1.0, prof_zero)) * 100
    
    print("\n--- Part B: Severe Flood Stress Test (July 2023 Shock) ---")
    pdi_flood_naive, prof_flood_naive = run_compliance_regime("Naive Router (Unaware of Flood)", compliance_rate=0.85, flood_active=True, disruption_aware=False)
    pdi_flood_aware, prof_flood_aware = run_compliance_regime("Disruption-Aware Router (Reroutes Traffic)", compliance_rate=0.85, flood_active=True, disruption_aware=True)
    profit_preservation = ((prof_flood_aware - prof_flood_naive) / max(1.0, prof_flood_naive)) * 100
    
    print("\n=======================================================")
    print("           BENCHMARK RESULTS & COMPARISONS")
    print("=======================================================")
    print(f"Normal Conditions Spectrum:")
    print(f"  * 0% Compliance (Herding)     : PDI = {pdi_zero:.4f} | Avg Profit = INR {prof_zero:.2f}")
    print(f"  * 35% Compliance (Black-Box)  : PDI = {pdi_blackbox:.4f} | Avg Profit = INR {prof_blackbox:.2f}")
    print(f"  * 85% Compliance (Explainable): PDI = {pdi_explainable:.4f} | Avg Profit = INR {prof_explainable:.2f}")
    print(f"  * 100% Full Equilibrium       : PDI = {pdi_full:.4f} | Avg Profit = INR {prof_full:.2f} (PDI down {volatility_drop_full:.2f}%!)")
    print()
    print(f"Severe Flood Stress Test (July 2023 Shock):")
    print(f"  * Naive Router (No Disruption Intel) : Avg Profit = INR {prof_flood_naive:.2f}")
    print(f"  * Disruption-Aware Router (Rerouted) : Avg Profit = INR {prof_flood_aware:.2f}")
    print(f"  * Farmer Profit Protected By AI     : +{profit_preservation:.2f}% gain (Shielded from flood spoilage!)")
    
    return {
        "pdi_zero": round(pdi_zero, 4),
        "pdi_blackbox": round(pdi_blackbox, 4),
        "pdi_explainable": round(pdi_explainable, 4),
        "pdi_full": round(pdi_full, 4),
        "volatility_drop_full": round(volatility_drop_full, 2),
        "pdi_flood_naive": round(pdi_flood_naive, 4),
        "pdi_flood_aware": round(pdi_flood_aware, 4),
        "profit_preservation": round(profit_preservation, 2)
    }

if __name__ == "__main__":
    run_trust_benchmark()
