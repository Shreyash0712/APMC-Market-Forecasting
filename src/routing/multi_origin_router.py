"""
File: src/routing/multi_origin_router.py
Role: Distributed Multi-Origin Coordinator
Description: High-level routing engine coordinating simultaneous agricultural dispatches
across multiple regional farming origins (e.g., Nashik, Sangli, Satara) to shared APMC terminals.
Dependencies: numpy
"""

import numpy as np
from src.routing.router import GameTheoreticRouter
from src.routing.multi_origin_frank_wolfe import compute_multi_origin_wardrop_equilibrium

class MultiOriginRouter(GameTheoreticRouter):
    def __init__(self, state="DEFAULT"):
        super().__init__(state=state)

    def route_multiple_origins(
        self,
        origins,
        destinations,
        distance_matrix_km,
        disruptions=None
    ):
        """
        Calculates joint Wardrop User Equilibrium across multiple origin hubs.
        
        Args:
            origins: List of dicts, e.g. [{'name': 'Nashik', 'volume_tonnes': 1200}, {'name': 'Sangli', 'volume_tonnes': 900}]
            destinations: List of dicts, e.g. [{'name': 'Pune', 'base_price': 25000, 'capacity': 1000}, ...]
            distance_matrix_km: 2D array [len(origins), len(destinations)] of driving distances.
            disruptions: Optional dict of destination disruptions {'Pune': {'road_accessibility': 0.20}}
            
        Returns:
            dict containing origin-specific percentage recommendations, destination clearing prices, and total arrivals.
        """
        origin_names = [o["name"] for o in origins]
        origin_volumes = [float(o["volume_tonnes"]) for o in origins]
        
        dest_names = [d["name"] for d in destinations]
        dest_prices = [float(d["base_price"]) for d in destinations]
        dest_capacities = [float(d["capacity"]) for d in destinations]
        
        dist_matrix = np.array(distance_matrix_km, dtype=np.float64).copy()
        
        # Apply real-time disruption impedance scaling
        if disruptions:
            for d_idx, d_name in enumerate(dest_names):
                if d_name in disruptions:
                    road_acc = float(disruptions[d_name].get("road_accessibility", 1.0))
                    if road_acc < 1.0:
                        dist_matrix[:, d_idx] = dist_matrix[:, d_idx] / max(0.05, road_acc)
                        
        # Solve Joint Multi-Origin Frank-Wolfe Optimization
        X_opt, iters = compute_multi_origin_wardrop_equilibrium(
            origin_volumes=origin_volumes,
            destination_base_prices=dest_prices,
            destination_capacities=dest_capacities,
            distance_matrix_km=dist_matrix,
            apmc_fee_rate=self.state_fee_rate
        )
        
        # Format human-readable dispatch recommendations for each origin hub
        recommendations = {}
        for o_idx, o_name in enumerate(origin_names):
            recommendations[o_name] = {}
            vol_total = origin_volumes[o_idx]
            for d_idx, d_name in enumerate(dest_names):
                tonnage = float(X_opt[o_idx, d_idx])
                pct = (tonnage / vol_total) * 100.0 if vol_total > 0 else 0.0
                if pct > 0.5:
                    recommendations[o_name][d_name] = {
                        "allocated_tonnes": round(tonnage, 1),
                        "percentage": round(pct, 1)
                    }
                    
        # Total arriving volume at each destination
        total_arrivals = np.sum(X_opt, axis=0)
        
        # Clearing prices under joint equilibrium
        from src.routing.payoff_functions import inverse_demand_function
        clearing_prices = {}
        for d_idx, d_name in enumerate(dest_names):
            price = inverse_demand_function(
                base_price=dest_prices[d_idx],
                allocated_volume=total_arrivals[d_idx],
                baseline_capacity=dest_capacities[d_idx]
            )
            clearing_prices[d_name] = {
                "clearing_price_per_tonne": round(price, 2),
                "total_arriving_tonnes": round(float(total_arrivals[d_idx]), 1),
                "capacity_tonnes": dest_capacities[d_idx],
                "utilization_pct": round((total_arrivals[d_idx] / dest_capacities[d_idx]) * 100, 1)
            }
            
        return {
            "origin_recommendations": recommendations,
            "destination_market_states": clearing_prices,
            "optimization_iterations": iters
        }
