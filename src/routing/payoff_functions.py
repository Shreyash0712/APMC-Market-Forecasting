"""
File: src/routing/payoff_functions.py
Role: Game Theory Economic Engine
Description: Calculates the true net realization (payoff) for a farmer engaging in 
spatial arbitrage. Takes into account transport haulage rates, state-specific APMC 
statutory fees (Market Fee, Rural Development Fund, Arhat), and applies an Inverse 
Demand Function to simulate market crashes caused by over-allocation.
Dependencies: numpy
"""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Logistics & Economic Assumptions (pulled from environment variables for dynamic tuning)
ELASTICITY_COEFFICIENT = float(os.environ.get("ELASTICITY_COEFFICIENT", 0.721))
SPOILAGE_RATE_PER_HOUR = float(os.environ.get("SPOILAGE_RATE_PER_HOUR", 0.001))
AVERAGE_TRUCK_SPEED_KMPH = float(os.environ.get("AVERAGE_TRUCK_SPEED_KMPH", 40.0))
CURRENT_DIESEL_PRICE = float(os.environ.get("CURRENT_DIESEL_PRICE", 90.0))
TRUCK_MILEAGE_KMPL = float(os.environ.get("TRUCK_MILEAGE_KMPL", 15.0))
BASE_LOADING_CHARGE = float(os.environ.get("BASE_LOADING_CHARGE", 500.0))

def calculate_transport_cost(distance_km, volume_tonnes):
    # Freight rate is now completely dynamically pegged to fuel costs
    # Fuel cost per km = Diesel Price / Mileage
    # Total trip cost = Fuel cost per km * distance
    fuel_cost_per_km = CURRENT_DIESEL_PRICE / TRUCK_MILEAGE_KMPL
    trip_fuel_cost = fuel_cost_per_km * distance_km
    
    # We assume the fuel cost scales with the tonnes (1 tonne = 1 standard truck equivalent for simple math)
    return BASE_LOADING_CHARGE + (trip_fuel_cost * volume_tonnes)

def inverse_demand_function(base_price, allocated_volume, baseline_capacity, elasticity=ELASTICITY_COEFFICIENT):
    """
    Simulates a price crash if the allocated volume exceeds the mandi's capacity.
    If allocated_volume > baseline_capacity, price drops exponentially based on elasticity.
    """
    # If supply is within capacity, price holds. If it exceeds, it drops.
    utilization = allocated_volume / (baseline_capacity + 1e-6)
    
    if utilization <= 1.0:
        return base_price
    else:
        # Price drops rapidly as utilization exceeds 1.0
        # P = P0 * exp(-elasticity * (utilization - 1))
        return base_price * np.exp(-elasticity * (utilization - 1.0))

def compute_net_payoff(predicted_price_per_tonne, distance_km, volume_tonnes, 
                       allocated_volume, baseline_capacity, apmc_fee_rate):
    """
    Calculates the final net realization per tonne for the farmer.
    """
    # 1. Calculate transit time and physical spoilage
    transit_hours = distance_km / AVERAGE_TRUCK_SPEED_KMPH
    spoilage_loss_fraction = transit_hours * SPOILAGE_RATE_PER_HOUR
    effective_volume = volume_tonnes * (1.0 - spoilage_loss_fraction)
    
    # 2. Calculate the dynamic clearing price based on the Herding Dilemma
    clearing_price = inverse_demand_function(
        base_price=predicted_price_per_tonne,
        allocated_volume=allocated_volume,
        baseline_capacity=baseline_capacity
    )
    
    # 3. Deduct state-specific APMC ad-valorem taxes and commission (applied to remaining volume)
    gross_realization = clearing_price * effective_volume
    net_after_tax = gross_realization * (1.0 - apmc_fee_rate)
    
    # 4. Deduct physical transport costs (freight is charged on original shipped volume)
    transport_cost = calculate_transport_cost(distance_km, volume_tonnes)
    
    net_profit = net_after_tax - transport_cost
    
    # Return profit per ORIGINAL tonne to normalize evaluation for the farmer's decision
    return net_profit / (volume_tonnes + 1e-6)
