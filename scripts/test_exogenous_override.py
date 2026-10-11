"""
File: scripts/test_exogenous_override.py
Role: Exogenous Disruption Override Benchmark (Test 3)
Description: Evaluates how the Bayesian Residual Gate uses the calibrated Scenario Bank
(including the real July 2023 tomato floods case) to overturn GNN false crash alarms.
Consumes 0 API credits.
Dependencies: json, numpy
"""

import os
import sys
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.explainability.exogenous_gate import BayesianExogenousGate

def run_override_benchmark():
    print("=== [TEST 3] Running Exogenous Disruption Override Benchmark ===")
    
    fixtures_path = os.path.join(BASE_DIR, "data", "fixtures", "exogenous_shocks.json")
    with open(fixtures_path, "r") as f:
        scenarios = json.load(f)
        
    gate = BayesianExogenousGate(sensitivity=1.5)
    
    # Baseline GNN prediction where historical seasonal patterns expect a heavy crash:
    # 85% Crash, 10% Stable, 5% Spike
    baseline_gnn_probs = {"Stable": 0.10, "Crash": 0.85, "Spike": 0.05}
    
    print(f"Baseline ST-GNN Forecast: Crash (Confidence: 85.0%)\n")
    results = []
    
    for sc in scenarios:
        s_id = sc["scenario_id"]
        name = sc["name"]
        disruption = sc["supply_disruption_index"]
        road_acc = sc["road_accessibility_factor"]
        
        update = gate.update_probabilities(
            gnn_probs=baseline_gnn_probs,
            supply_disruption_index=disruption,
            road_accessibility=road_acc
        )
        
        print(f"Scenario: {name}")
        print(f"  Headline: \"{sc['headline'][:95]}...\"")
        print(f"  Input Shock Parameters: Supply Disruption = {disruption:+.2f}, Road Access = {road_acc:.2f}")
        print(f"  Outcome: {update['original_prediction']} -> {update['updated_prediction']}")
        print(f"  New Probabilities: {update['updated_probabilities']}")
        print(f"  Rationale: {update['rationale']}\n")
        
        results.append({
            "scenario": name,
            "original_pred": update["original_prediction"],
            "updated_pred": update["updated_prediction"],
            "override_occurred": update["override_occurred"]
        })
        
    return results

if __name__ == "__main__":
    run_override_benchmark()
