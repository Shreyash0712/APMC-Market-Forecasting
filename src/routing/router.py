"""
File: src/routing/router.py
Role: Core Routing Interface
Description: The primary entry point for generating routing recommendations. It 
aggregates the ST-GNN predictions and physical constraints, passing them to the 
Frank-Wolfe algorithm to generate a stable, equilibrium-bound distribution of farmers.
Dependencies: numpy
"""

import numpy as np
import os
import psycopg2
from dotenv import load_dotenv
from src.routing.frank_wolfe import compute_wardrop_equilibrium

class GameTheoreticRouter:
    def __init__(self, state="DEFAULT"):
        self.state = state
        self.state_fee_rate = self._fetch_state_fee_from_db(state)
        
    def _fetch_state_fee_from_db(self, state_name):
        """Fetches the latest gazetted APMC tax rate for the state from TimescaleDB."""
        load_dotenv()
        db_url = os.environ.get("DB_URL")
        
        try:
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            # Fetch specific state, fallback to DEFAULT if not found in db
            cursor.execute("SELECT tax_rate_decimal FROM state_apmc_taxes WHERE state_name = %s", (state_name,))
            result = cursor.fetchone()
            
            if not result:
                cursor.execute("SELECT tax_rate_decimal FROM state_apmc_taxes WHERE state_name = 'DEFAULT'")
                result = cursor.fetchone()
                
            cursor.close()
            conn.close()
            
            return float(result[0]) if result else 0.030
        except Exception as e:
            print(f"Warning: Database connection failed, falling back to 3% default fee. Error: {e}")
            return 0.030
        
    def generate_recommendation(self, mandi_names, predicted_prices, distances, 
                              total_village_volume, baseline_capacities):
        """
        Calculates the Wardrop User Equilibrium for a group of farmers.
        
        Args:
            mandi_names (list): Names of the candidate destination mandis.
            predicted_prices (list/array): The AI's predicted price for each mandi (INR/tonne).
            distances (list/array): Geodesic driving distances to each mandi (km).
            total_village_volume (float): Total tonnes of produce to route.
            baseline_capacities (list/array): The organic daily capacity of each mandi.
            
        Returns:
            dict: A mapping of {mandi_name: percentage_allocation}
        """
        # Convert to numpy arrays for matrix math
        predicted_prices = np.array(predicted_prices)
        distances = np.array(distances)
        baseline_capacities = np.array(baseline_capacities)
        
        # Run Frank-Wolfe Optimization
        optimal_split = compute_wardrop_equilibrium(
            predicted_prices=predicted_prices,
            distances=distances,
            total_village_volume=total_village_volume,
            baseline_capacities=baseline_capacities,
            apmc_fee_rate=self.state_fee_rate,
            max_iter=200,
            tol=1e-5
        )
        
        # Format the output as a human-readable dictionary (rounding to 2 decimals)
        routing_allocation = {}
        for i, name in enumerate(mandi_names):
            if optimal_split[i] > 0.01: # Filter out mandis with near-zero allocations
                routing_allocation[name] = round(float(optimal_split[i]) * 100, 2)
                
        return routing_allocation
