# APMC Market Forecasting & Game-Theoretic Routing

An advanced operations research and machine learning platform designed to predict agricultural price crashes and actively prevent the **Herding Dilemma** (Performative Prediction) using stochastic convex optimization and neuro-symbolic explainability.

## The Problem: The Performative Prediction Paradox
If an AI simply predicts that "Tomatoes will be highly profitable in Pune tomorrow" and broadcasts this to 10,000 farmers via a mobile app, all 10,000 farmers will drive to Pune. This sudden supply glut mathematically guarantees the price will crash, rendering the AI's prediction performatively false and destroying farmer livelihoods. 

## The Solution: 4-Phase Architecture
This repository solves the paradox via a comprehensive four-phase architecture:

```mermaid
graph TD
    subgraph Data Layer
        A[TimescaleDB / PostGIS] --> B(Historical APMC Arrivals & Prices)
    end
    
    subgraph Phase 1: Machine Learning
        B --> C[Spatial Graph Construction]
        C --> D[Spatio-Temporal GNN]
        D -->|Forecasts Baseline Prices| E
    end
    
    subgraph Phase 2: Game Theory
        E[Frank-Wolfe Convex Optimizer]
        E -->|Calculates Wardrop Equilibrium| F[Stochastic Routing API]
    end
    
    subgraph Phase 3: Agent-Based Simulation
        F -->|Yields Route Probabilities| G[Mesa ABM]
        G -->|1,000 Synthetic Farmers acting on advice| H((Market Stabilizes))
    end

    subgraph Phase 4: Neuro-Symbolic Explainability
        D -->|GATv2 Attention & Saliency| I[Mathematical Attribution]
        I --> J[Tavily Real-Time Context Engine]
        J --> K[Groq Qwen3.8-27b Synthesizer]
        K -->|Grounded Farmer Advisory| L((Transparent Adoption))
        J -->|Bayesian Residual Gate| E
    end
```

---

## Project Milestones & Statistics

### Phase 1: Spatio-Temporal GNN (ST-GNN)
We built a PyTorch Geometric model combining **GATv2Conv** (Spatial Attention) and **Temporal Convolutional Networks** to detect early momentum indicators of impending price crashes across interconnected state markets.
- **Data Filtering:** Cleaned 40M+ rows of Indian APMC data down to memory-safe perishables.
- **Performance:** Achieved a **0.9256 PR-AUC** (Precision-Recall Area Under Curve), explicitly tuned to penalize False Negatives (missing a crash) via Class-Weighted Cross-Entropy.

### Phase 2: Game-Theoretic Routing Engine
Instead of deterministic "go here" routing, we built a probabilistic API. It applies the **Frank-Wolfe algorithm** to find the **Wardrop User Equilibrium** without needing computationally expensive geometrical projections.
- **Empirical Realism:** Integrated exact ARDL Elasticity coefficients (-0.721) for Indian Tomatoes (from the RBI) and NABCONS post-harvest transit spoilage rates.
- **Dynamic Math:** Automatically calculates real-time freight costs based on diesel prices, pulling live statutory APMC tax deductions from the PostgreSQL database.
- **Performance:** The algorithm balances probabilities and equalizes net farmer payoffs across a state in **~14 milliseconds**, operating well under standard 50ms API constraints.

### Phase 3: Agent-Based Simulation (Mesa)
To mathematically prove the routing engine stabilizes the macroeconomy, we instantiated 1,000 autonomous `FarmerAgents` programmed with psychological Loss Aversion (Prospect Theory).
- **Baseline Scenario (Herding):** Farmers routed via word-of-mouth (chasing yesterday's highest price).
- **Intervention Scenario:** Farmers routed probabilistically via our API.
- **Result:** The Price Dispersion Index (Volatility) dropped from `0.7875` to `0.4734`. **The API successfully reduced regional market volatility by ~40%!**

### Phase 4: Neuro-Symbolic Explainability & Disruption Routing
To overcome the "black box" adoption barrier and dynamically handle real-world crises (like floods or strikes), we built an explainability and adaptive rerouting tier:
- **Mathematical Attribution (`graph_explainer.py`):** Hooks into GATv2 attention weights ($\alpha_{ij}$) and temporal gradients to pinpoint causal neighbor mandis and lag days in **1,011 ms**.
- **Real-Time Context (`news_retriever.py`):** Uses Tavily to probe targeted market clusters for live weather and logistics disruptions.
- **Ultra-Fast Synthesis (`advisory_generator.py`):** Uses Groq's `qwen/qwen3.8-27b` to generate hallucination-resistant farmer advisories in **656 ms**!
- **Anticipatory Capacity Buffering:** The router anticipates non-compliant "rogue herders" and pre-deducts their volume from capacity buffers. Under full coordination, the Price Dispersion Index collapses to **0.2901 (a 60.95% volatility collapse)**!
- **Flood Disruption Shield:** During severe monsoon floods (e.g., July 2023 case), the router scales highway impedance and reroutes trucks, **protecting farmer incomes by +53.67%** (INR 6,233 $\rightarrow$ INR 9,578)!
- **Bayesian Override:** Overturns false crash warnings during extreme supply-destruction events.

---

## Quick Start

### 1. Requirements
- **Python 3.12+** (We strongly recommend using [uv](https://github.com/astral-sh/uv) for lightning-fast dependency resolution).
- **PostgreSQL 16+** with **TimescaleDB** and **PostGIS** extensions installed.

### 2. Environment Setup
Copy the example environment file and insert your local credentials:
```bash
cp .env.example .env
```

Configure your API keys and economic variables inside `.env`:
```env
# Database Credentials
DB_URL=postgres://apmc_admin:yourpassword@localhost:5432/apmc_forecasting

# Dynamic Economic & Logistics Configurations
CURRENT_DIESEL_PRICE=90.0
TRUCK_MILEAGE_KMPL=15.0

# Optional API Keys for Live Phase 4 Explainability
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Installation
Using `uv`, install the dependencies into a virtual environment:
```bash
uv venv
# On Windows:
.\.venv\Scripts\activate
# Install requirements:
uv pip install -r requirements.txt
```

### 4. Running the Pipelines

#### Run the Phase 3 Agent-Based Simulation:
```bash
python scripts/run_simulation.py
```

#### Run the Phase 4 Neuro-Symbolic Explainability Test:
```bash
python scripts/test_explainability.py
```

#### Run the Master Benchmark Suite (Fidelity, Trust Spectrum, & Flood Overrides):
```bash
python scripts/run_all_benchmarks.py
```

---

## 📚 Documentation
For an in-depth dive into the mathematical and economic theories powering this repo, see:
- [docs/KNOWLEDGE_CHART.md](docs/KNOWLEDGE_CHART.md) - Theoretical map of all ML, game theory, and economic concepts.
- [docs/PHASE_1_SUMMARY.md](docs/PHASE_1_SUMMARY.md) - Baseline ST-GNN architecture and chronological split validation.
- [docs/PHASE_2_SUMMARY.md](docs/PHASE_2_SUMMARY.md) - Frank-Wolfe Wardrop equilibrium and diesel logistics math.
- [docs/PHASE_3_SUMMARY.md](docs/PHASE_3_SUMMARY.md) - Initial Mesa Agent-Based Simulation results.
- [docs/PHASE_4_SUMMARY.md](docs/PHASE_4_SUMMARY.md) - Neuro-symbolic explainability engine details.
- [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) - Full quantitative scorecard across fidelity, trust, and disruption rerouting.
