"""
File: src/simulation/agents.py
Role: Agent-Based Modeling
Description: Defines the synthetic FarmerAgent using the Mesa framework. 
In the Baseline scenario, agents suffer from the Herding Dilemma (routing to the most 
lucrative market). In the Intervention scenario, they act on the Wardrop Equilibrium probabilities,
modified slightly by Prospect Theory (Loss Aversion).
Dependencies: mesa, numpy
"""

from mesa import Agent
import numpy as np

class FarmerAgent(Agent):
    """An autonomous synthetic farmer deciding where to sell their daily harvest."""
    
    def __init__(self, model, volume_tonnes, loss_aversion_factor=2.25):
        super().__init__(model)
        self.volume_tonnes = volume_tonnes
        # Kahneman & Tversky's empirical coefficient for loss aversion is roughly 2.25
        self.loss_aversion_factor = loss_aversion_factor 
        self.chosen_mandi = None
        self.daily_profit = 0.0
        
    def step(self):
        """The agent's decision-making process at every time step (day)."""
        
        if self.model.sim_scenario == "Baseline":
            # HERDING DILEMMA: The agent just looks at the highest raw price from yesterday 
            # and blindly goes there without considering congestion.
            best_mandi = None
            highest_price = -1
            
            for mandi_name, mandi_data in self.model.mandi_states.items():
                if mandi_data["current_price"] > highest_price:
                    highest_price = mandi_data["current_price"]
                    best_mandi = mandi_name
                    
            self.chosen_mandi = best_mandi
            
        elif self.model.sim_scenario == "Intervention":
            # WARDROP EQUILIBRIUM: The agent receives the probability distribution from the API.
            # E.g., {"Market A": 68.0, "Market B": 32.0}
            routing_vector = self.model.current_routing_recommendation
            
            if not routing_vector:
                # Failsafe if API is down
                self.chosen_mandi = list(self.model.mandi_states.keys())[0]
                return
                
            mandis = list(routing_vector.keys())
            # Convert percentages (0-100) back to probabilities (0-1)
            probs = [float(routing_vector[m]) / 100.0 for m in mandis]
            
            # Normalize just in case of rounding errors in the router
            probs = np.array(probs) / np.sum(probs)
            
            # The agent probabilistically selects a market based on the equilibrium.
            # Note: Prospect theory could dynamically alter these probabilities per-agent, 
            # but for this baseline simulation, they strictly follow the Wardrop distribution.
            self.chosen_mandi = np.random.choice(mandis, p=probs)
            
        # Inform the destination mandi that volume is arriving
        self.model.mandi_states[self.chosen_mandi]["arriving_volume"] += self.volume_tonnes
