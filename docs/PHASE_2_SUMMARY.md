# Phase 2: Game-Theoretic Routing Engine

This document summarizes the architecture, empirical refinements, and successful baseline testing of Phase 2: The Wardrop User Equilibrium Routing Engine.

## 1. Architectural Setup
We built a highly optimized Convex Optimization engine to solve the *Performative Prediction* paradox (the Herding Dilemma), ensuring our AI price forecasts do not accidentally trigger market crashes when acted upon by thousands of farmers.

**Key Files & Components:**
- [src/routing/payoff_functions.py](../src/routing/payoff_functions.py): Calculates the true net realization (take-home profit) for a farmer based on distance, transport costs, state taxes, spoilage, and market capacity.
- [src/routing/frank_wolfe.py](../src/routing/frank_wolfe.py): Implements the projection-free Frank-Wolfe algorithm, utilizing a Linear Minimization Oracle to iteratively balance farmer distribution until no one can unilaterally improve their profit.
- [src/routing/router.py](../src/routing/router.py): The core API that wraps the optimization logic and outputs a stochastic probability routing vector.

## 2. Empirical Research & Realistic Variable Refinement
To ensure the mathematical equilibrium aligns with the physical reality of Indian agricultural logistics, we replaced generic assumptions with empirically backed research data sourced from the Reserve Bank of India (RBI) and NABCONS.

**Integrated Variables:**
- **Price Elasticity of Demand (`elasticity = 0.721`)**: Based on RBI ARDL models for Indian tomatoes. This high flexibility coefficient accurately simulates rapid price crashes when supply gluts occur.
- **Transit Spoilage Rate (`SPOILAGE_RATE_PER_HOUR = 0.1%`)**: Derived from the NABCONS post-harvest loss study (3.25% average transit loss). This actively penalizes the algorithm for routing farmers to distant markets.
- **Dynamic Transit Time (`AVERAGE_TRUCK_SPEED_KMPH = 40.0`)**: Combined with PostGIS geodesic distances to calculate the exact hours of transit, driving the spoilage rate.
- **Dynamic Diesel-Pegged Logistics (`CURRENT_DIESEL_PRICE`, `TRUCK_MILEAGE_KMPL`)**: The freight rate is dynamically calculated via environment variables based on the live price of diesel and generic truck mileage, replacing the static ₹2.5/km assumption.
- **PostgreSQL Tax Table (`state_apmc_taxes`)**: Eliminated hardcoded state taxes. The routing engine dynamically queries the TimescaleDB for the latest gazetted APMC statutory deductions during initialization and caches it for the optimization loop.

## 3. Baseline Equilibrium Testing
We executed a synthetic stress test (`scripts/test_routing.py`) where a village had 500 tonnes of produce to route between an extremely lucrative but distant market (Market A, 120km) and a standard nearby market (Market B, 40km).

**The Herding Dilemma Prevented:**
Instead of sending 100% of the volume to Market A (which would have crashed the price immediately), the Frank-Wolfe algorithm dynamically balanced the load:
- **Market A (Premium, Distant):** 68.17%
- **Market B (Standard, Close):** 31.83%

**Proof of Equilibrium:**
At this exact 68/32 split, the net marginal payoff at both markets perfectly equalized at **~INR 13,480 / tonne**. The algorithm successfully recognized that sending more volume to Market A would drop its net payoff below Market B due to the high `0.721` elasticity crash, the `0.1%` per-hour rot on the highway, and the updated diesel freight penalty.

**Performance Metrics:**
- **Optimization Latency:** 14.02 ms (Well under the strict 50ms API requirement, guaranteeing real-time advisory capabilities).
