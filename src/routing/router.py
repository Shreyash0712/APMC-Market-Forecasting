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
                              total_village_volume, baseline_capacities,
                              adherence_rate=1.0, disruptions=None):
        """
        Calculates the Wardrop User Equilibrium with anticipatory compliance buffering
        and real-time exogenous disruption impedance.
        
        Args:
            mandi_names (list): Names of candidate destination mandis.
            predicted_prices (list/array): Predicted price for each mandi (INR/tonne).
            distances (list/array): Geodesic driving distances to each mandi (km).
            total_village_volume (float): Total tonnes of produce to route.
            baseline_capacities (list/array): Organic daily capacity of each mandi.
            adherence_rate (float): Estimated fraction of farmers complying (0.0 to 1.0).
            disruptions (dict): Optional {mandi_name: {'road_accessibility': float, 'supply_disruption': float}}.
            
        Returns:
            dict: Mapping of {mandi_name: percentage_allocation} for compliant farmers.
        """
        # Convert to numpy arrays
        predicted_prices = np.array(predicted_prices, dtype=np.float64)
        distances = np.array(distances, dtype=np.float64)
        capacities = np.array(baseline_capacities, dtype=np.float64)
        
        # 1. Apply Exogenous Disruption Penalties (Road Accessibility & Floods)
        if disruptions:
            for i, name in enumerate(mandi_names):
                if name in disruptions:
                    d_info = disruptions[name]
                    road_acc = float(d_info.get("road_accessibility", 1.0))
                    # Scale distance inversely with road accessibility (submerged road = severe transit penalty)
                    if road_acc < 1.0:
                        distances[i] = distances[i] / max(0.05, road_acc)
                        
        # 2. Anticipatory Compliance Buffering (Shield against Rogue Herders)
        effective_volume = total_village_volume
        if adherence_rate < 1.0:
            rogue_volume = total_village_volume * (1.0 - adherence_rate)
            compliant_volume = total_village_volume * adherence_rate
            
            # Identify the target market that naive rogue herders will attack (highest raw price)
            rogue_target_idx = int(np.argmax(predicted_prices))
            
            # Pre-deduct the rogue volume from that market's capacity buffer
            capacities[rogue_target_idx] = max(10.0, capacities[rogue_target_idx] - rogue_volume)
            effective_volume = compliant_volume

        # 3. Run Frank-Wolfe Convex Optimization
        optimal_split = compute_wardrop_equilibrium(
            predicted_prices=predicted_prices,
            distances=distances,
            total_village_volume=effective_volume,
            baseline_capacities=capacities,
            apmc_fee_rate=self.state_fee_rate,
            max_iter=200,
            tol=1e-5
        )
        
        # Format the output as a percentage allocation for compliant farmers
        routing_allocation = {}
        for i, name in enumerate(mandi_names):
            if optimal_split[i] > 0.01:
                routing_allocation[name] = round(float(optimal_split[i]) * 100, 2)
                
        return routing_allocation
