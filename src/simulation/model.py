"""
File: src/simulation/model.py
Role: Agent-Based Modeling Environment
Description: The core Mesa Model orchestrating the agricultural ecosystem. It initializes 
the spatial environment, spawns the synthetic farmers, manages the passage of time (days), 
and updates market prices dynamically based on the inverse demand functions as volume arrives.
Dependencies: mesa, numpy
"""

from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np

from src.simulation.agents import FarmerAgent
from src.routing.router import GameTheoreticRouter
from src.routing.payoff_functions import inverse_demand_function, compute_net_payoff
from src.simulation.metrics import compute_price_dispersion_index

class APMCMesaModel(Model):
    """A model with some number of autonomous farmers choosing markets."""
    
    def __init__(self, num_agents, scenario="Baseline", state="Maharashtra"):
        super().__init__()
        self.num_agents = num_agents
        self.sim_scenario = scenario
        self.state = state
        
        # Initialize the API Router
        self.router = GameTheoreticRouter(state=self.state)
        self.current_routing_recommendation = {}
        
        # Initialize Market Nodes
        # For this simulation, we simulate 3 distinct mandis
        self.mandi_states = {
            "Market A (Premium)": {
                "baseline_price": 25000.0,
                "current_price": 25000.0,
                "capacity": 500.0,
                "distance_km": 120.0,
                "arriving_volume": 0.0
            },
            "Market B (Standard)": {
                "baseline_price": 15000.0,
                "current_price": 15000.0,
                "capacity": 300.0,
                "distance_km": 40.0,
                "arriving_volume": 0.0
            },
            "Market C (Low)": {
                "baseline_price": 10000.0,
                "current_price": 10000.0,
                "capacity": 200.0,
                "distance_km": 10.0,
                "arriving_volume": 0.0
            }
        }
        
        # Spawn Synthetic Farmer Agents
        # We assume each farmer has between 1 to 5 tonnes of produce to sell today
        for i in range(self.num_agents):
            volume = np.random.uniform(1.0, 5.0)
            a = FarmerAgent(self, volume_tonnes=volume)
            
        # Initialize DataCollector for tracking metrics over time
        self.datacollector = DataCollector(
            model_reporters={
                "Price_Dispersion_Index": compute_price_dispersion_index,
                "Market_A_Price": lambda m: m.mandi_states["Market A (Premium)"]["current_price"],
                "Market_B_Price": lambda m: m.mandi_states["Market B (Standard)"]["current_price"],
                "Market_C_Price": lambda m: m.mandi_states["Market C (Low)"]["current_price"]
            },
            agent_reporters={"Profit": "daily_profit", "Chosen_Mandi": "chosen_mandi"}
        )
        
    def step(self):
        """Advance the model by one step (one day)."""
        
        # 1. Reset daily arriving volumes
        for mandi in self.mandi_states.values():
            mandi["arriving_volume"] = 0.0
            
        # 2. If Intervention, query the Routing API before agents make decisions
        if self.sim_scenario == "Intervention":
            mandi_names = list(self.mandi_states.keys())
            predicted_prices = np.array([m["current_price"] for m in self.mandi_states.values()])
            distances = np.array([m["distance_km"] for m in self.mandi_states.values()])
            baseline_capacities = np.array([m["capacity"] for m in self.mandi_states.values()])
            
            # Sum up total volume for the village (for Wardrop equilibrium calculation)
            total_village_volume = sum([a.volume_tonnes for a in self.agents])
            
            self.current_routing_recommendation = self.router.generate_recommendation(
                mandi_names=mandi_names,
                predicted_prices=predicted_prices,
                distances=distances,
                total_village_volume=total_village_volume,
                baseline_capacities=baseline_capacities
            )
            
        # 3. Agents make their routing decisions
        self.agents.shuffle_do("step")
        
        # 4. End of Day: Calculate the new clearing prices based on the Herding Dilemma (Inverse Demand)
        for mandi_data in self.mandi_states.values():
            new_price = inverse_demand_function(
                base_price=mandi_data["baseline_price"], # Prices attempt to revert to mean if not overloaded
                allocated_volume=mandi_data["arriving_volume"],
                baseline_capacity=mandi_data["capacity"]
            )
            mandi_data["current_price"] = new_price
            
        # 5. Calculate final profits for each agent based on the new crashed/stable prices
        for agent in self.agents:
            chosen = self.mandi_states[agent.chosen_mandi]
            profit = compute_net_payoff(
                predicted_price_per_tonne=chosen["current_price"], # Using actual clearing price
                distance_km=chosen["distance_km"],
                volume_tonnes=agent.volume_tonnes,
                allocated_volume=chosen["arriving_volume"],
                baseline_capacity=chosen["capacity"],
                apmc_fee_rate=self.router.state_fee_rate
            )
            # Total profit for their load
            agent.daily_profit = profit * agent.volume_tonnes 
            
        # 6. Collect Data
        self.datacollector.collect(self)
