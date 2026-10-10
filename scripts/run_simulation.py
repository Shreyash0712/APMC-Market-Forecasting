"""
File: scripts/run_simulation.py
Role: ML/Simulation Orchestration
Description: Instantiates the Mesa Agent-Based simulation for both the Baseline (Herding) 
and Intervention (Wardrop Routing) scenarios. Runs the simulation over a 30-day period 
and compares the Price Dispersion Index to prove the efficacy of the API.
Dependencies: pandas, mesa
"""

import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.model import APMCMesaModel

def run_scenario(scenario_name, num_agents=1000, days=30):
    print(f"\n--- Initializing {scenario_name} Simulation ---")
    print(f"Spawning {num_agents} autonomous FarmerAgents...")
    
    model = APMCMesaModel(num_agents=num_agents, scenario=scenario_name, state="Maharashtra")
    
    for day in range(days):
        if day % 5 == 0:
            print(f"Simulating Day {day}...")
        model.step()
        
    print(f"{scenario_name} Simulation Complete.")
    
    # Extract dataframe from DataCollector
    df = model.datacollector.get_model_vars_dataframe()
    return df

def main():
    print("Welcome to the APMC Market Forecasting Agent-Based Simulator.")
    print("Testing the impact of the Game-Theoretic Routing Engine on Macroeconomic Stability.")
    
    # Run Baseline (No API, Word of Mouth Herding)
    df_baseline = run_scenario("Baseline")
    
    # Run Intervention (Wardrop Equilibrium API)
    df_intervention = run_scenario("Intervention")
    
    print("\n=======================================================")
    print("                SIMULATION RESULTS")
    print("=======================================================\n")
    
    print("--- Average Market Prices over 30 Days ---")
    print(f"Baseline Market A (Premium) : INR {df_baseline['Market_A_Price'].mean():.2f}")
    print(f"Baseline Market B (Standard): INR {df_baseline['Market_B_Price'].mean():.2f}")
    print(f"Baseline Market C (Low)     : INR {df_baseline['Market_C_Price'].mean():.2f}")
    print()
    print(f"Intervention Market A (Premium) : INR {df_intervention['Market_A_Price'].mean():.2f}")
    print(f"Intervention Market B (Standard): INR {df_intervention['Market_B_Price'].mean():.2f}")
    print(f"Intervention Market C (Low)     : INR {df_intervention['Market_C_Price'].mean():.2f}")
    print()
    
    # Compare the Price Dispersion Index (Volatility / Market Integration)
    avg_pdi_baseline = df_baseline["Price_Dispersion_Index"].mean()
    avg_pdi_intervention = df_intervention["Price_Dispersion_Index"].mean()
    
    print("--- Macroeconomic Stability Metrics ---")
    print("Price Dispersion Index (Coefficient of Variation) [LOWER IS BETTER]")
    print(f"Baseline (Herding) PDI    : {avg_pdi_baseline:.4f}")
    print(f"Intervention (Wardrop) PDI: {avg_pdi_intervention:.4f}")
    
    if avg_pdi_intervention < avg_pdi_baseline:
        improvement = ((avg_pdi_baseline - avg_pdi_intervention) / avg_pdi_baseline) * 100
        print(f"\nSUCCESS: The Game-Theoretic Routing Engine reduced market volatility by {improvement:.2f}%!")
        print("By breaking the Herding Dilemma, the API successfully stabilized prices across the region.")
    else:
        print("\nFAILURE: The Routing Engine did not improve market stability.")

if __name__ == "__main__":
    main()
