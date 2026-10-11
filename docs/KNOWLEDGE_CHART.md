# APMC Project Knowledge Chart

This document serves as a high-level map of the complex technologies and theoretical concepts being used in this project. It explains *what* each component is and *where* it fits in our architecture, so you can explore the *why* and *how* in deeper detail on your own.

---

## 1. Technologies & Infrastructure

### TimescaleDB
- **What it is:** A time-series database built as an extension on top of PostgreSQL. It optimizes how data ordered by time is stored and queried.
- **Where we use it:** Our core database (`apmc_timescaledb`). We use it to store the massive historical market arrivals and prices. It allows us to rapidly query 14-day rolling averages (momentum) using its "Continuous Aggregates" feature, which is crucial for our ML model.

### PostGIS
- **What it is:** A spatial database extender for PostgreSQL. It allows the database to understand geometry (points, lines, polygons) and perform geographical calculations.
- **Where we use it:** Also in our core database. When we inserted the market data, PostGIS converted the raw latitude/longitude into a `GEOMETRY` column. This allows us to instantly query things like "find all markets within a 150km radius of Market A."

### uv
- **What it is:** An extremely fast Python package installer and resolver written in Rust. It replaces standard `pip`.
- **Where we use it:** We used it to create our isolated virtual environment (`.venv`) and install massive ML dependencies (like PyTorch) in a fraction of the usual time, keeping our system clean and adhering to our efficiency rules.

### PyTorch Geometric (PyG) & Temporal
- **What it is:** PyG is an extension for PyTorch specifically designed to build Graph Neural Networks. The "Temporal" variant adds layers for graphs that change over time.
- **Where we use it:** This is the engine for our Spatio-Temporal Graph Neural Network (ST-GNN). We will use it to build the actual AI model that forecasts price crashes.

### Mesa (Agent-Based Modeling)
- **What it is:** An open-source Python framework that allows you to create Agent-Based Models (ABMs). It manages the spatial grid, the agents, the scheduler, and data collection.
- **Where we use it:** In `src/simulation/`. We used it to build a synthetic agricultural ecosystem populated by 1,000 autonomous AI farmers to scientifically prove our routing engine works.

### Groq Cloud (LPUs & Qwen3.8-27b)
- **What it is:** Groq Language Processing Units (LPUs) are purpose-built deterministic hardware accelerators for LLMs, delivering extreme inference speeds (500+ tokens/sec). Qwen 3.8 27B is an open-weights dense reasoning model.
- **Where we use it:** In `src/explainability/advisory_generator.py`. It verbalizes the mathematical graph attributions into clear, 2-3 sentence farmer advisories with sub-700ms generation latency.

### Tavily Search API
- **What it is:** A specialized search engine built specifically for AI agents, filtering out web noise and returning clean factual snippets.
- **Where we use it:** In `src/explainability/news_retriever.py`. It probes target mandi clusters for real-world exogenous shocks (monsoon flooding, transport strikes, heatwaves) to ground the AI's explanations in reality.

---

## 2. Mathematical & ML Concepts

### Haversine Geodesic Math
- **What it is:** A formula used to calculate the great-circle distance between two points on a sphere given their longitudes and latitudes.
- **Where we use it:** In `scripts/build_graph.py`. The Earth is not flat, so simple Euclidean math (a² + b² = c²) is inaccurate over long distances. We use the Haversine formula to calculate the exact physical distance in kilometers between every single APMC market in India.

### Gaussian Kernel Adjacency Matrix ($W_{ij}$)
- **What it is:** A mathematical function that converts physical distance into a measure of "influence" or "similarity," ranging from 0 to 1. It decays exponentially.
- **Where we use it:** In `scripts/build_graph.py`. If two markets are 10km apart, their prices strongly influence each other (weight ~0.99). If they are 500km apart, they have almost zero influence (weight ~0.00). This matrix ($W_{ij}$) tells our Neural Network exactly how the markets are connected physically.

### Graph Attention Network (GATv2Conv)
- **What it is:** A specific type of neural network layer that allows a node (a market) to dynamically weigh the importance of its neighbors' features.
- **Where we use it:** This will be the spatial layer of our ST-GNN. If Market A is crashing, GATv2 allows Market B to "pay attention" to Market A's panic, ignoring other stable neighbors.

### Temporal Convolutional Network (TCN)
- **What it is:** A neural network architecture that uses 1D convolutions (looking back in time) to capture sequence patterns without suffering from the vanishing gradient problems of older models like LSTMs.
- **Where we use it:** This will be the temporal layer of our ST-GNN. It looks at the past 14 days of prices for a market to detect seasonal arrival cycles and momentum before a crash.

### Precision-Recall Area Under Curve (PR-AUC)
- **What it is:** A metric used to evaluate a model's performance when the classes are heavily imbalanced (i.e., normal days are common, market crashes are rare).
- **Where we use it:** In `src/models/trainer.py`. We explicitly avoid standard ROC-AUC because it can give overly optimistic scores when true negatives (normal days) dominate. PR-AUC focuses strictly on how well we detect the rare, catastrophic crashes.

### Class-Weighted Cross-Entropy Loss
- **What it is:** A modification to the standard loss function that penalizes the AI more heavily for making specific mistakes.
- **Where we use it:** In `scripts/train.py`. We assign a weight of 3.0 to the "Crash" class. If the AI misses a crash (false negative), it receives a massive error penalty, forcing it to learn early momentum indicators.

### ONNX Runtime
- **What it is:** Open Neural Network Exchange (ONNX) is an open format to represent deep learning models. The ONNX Runtime executes these models in highly optimized C++.
- **Where we use it:** (Proposed Optimization). We will use it to compile our trained PyTorch model (`st_gnn.py`) into a static graph, fusing kernels to drop our CPU inference latency below the critical 50ms threshold.

### Frank-Wolfe Algorithm (Convex Optimization)
- **What it is:** An iterative optimization algorithm that solves complex geometry problems without requiring computationally expensive "projections" back into the feasible space. It relies on a Linear Minimization Oracle.
- **Where we use it:** In `src/routing/frank_wolfe.py`. It is the engine that calculates the Wardrop Equilibrium probability split in under 15 milliseconds.

### Anticipatory Capacity Buffering (Rogue Herder Immunization)
- **What it is:** A game-theoretic technique where the router anticipates human non-compliance ($\theta < 1.0$), calculating the rogue volume $(1-\theta)V$ and pre-deducting it from the target market's available capacity before computing the equilibrium for compliant actors.
- **Where we use it:** In `src/routing/router.py`. It immunizes the market against uncoordinated free-riders dumping produce into high-price markets.

### Real-Time Disruption Impedance Scaling
- **What it is:** Scaling physical network geodesic distances by the inverse of corridor accessibility ($d_{\text{eff}} = d / R_{\text{access}}$) when verified weather or transport shocks occur.
- **Where we use it:** In `src/routing/router.py`. It automatically reroutes compliant trucks away from flooded or blocked highways, preserving farmer profits by +53.67%.

### Price Dispersion Index (Coefficient of Variation)
- **What it is:** A statistical measure of how spread out data is relative to its mean ($CV = \sigma / \mu$). 
- **Where we use it:** In `src/simulation/metrics.py`. We use it to measure macroeconomic stability. A high index means prices are volatile (some markets crashed, some skyrocketed). A low index means perfect spatial market integration.

### GATv2 Attention Attribution
- **What it is:** Extracting the dynamic attention weights $\alpha_{ij}$ computed by the Graph Attention Network during the forward pass to measure which neighbor nodes had the strongest message-passing influence.
- **Where we use it:** In `src/explainability/graph_explainer.py`. It tells us which surrounding mandis drove price contagion or price stability for the target market.

### Temporal Gradient Saliency
- **What it is:** Computing the partial derivative of the model's output score with respect to its temporal inputs ($\left|\frac{\partial \hat{y}}{\partial X}\right|$).
- **Where we use it:** In `src/explainability/graph_explainer.py`. It pinpoints which specific days in the 14-day historical rolling window triggered the forecast.

---

## 3. Economic Theories

### Neuro-Symbolic AI
- **What it is:** The fusion of deep neural networks (sub-symbolic pattern recognition) with symbolic reasoning and language models to produce explainable, grounded intelligence.
- **Where we use it:** Our entire Phase 4 explainability tier. The ST-GNN extracts non-linear numerical features, while the LLM verbalizes the causal drivers without hallucination.

### Wardrop User Equilibrium
- **What it is:** A principle from traffic/congestion game theory stating that in an equilibrium state, no individual can achieve a lower cost (or higher payout) by unilaterally changing their route. 
- **Where we use it:** In our final Routing Engine. Instead of telling all farmers to go to the single "best" market (which would cause a massive traffic jam and price crash there), the engine calculates an equilibrium spread so that every recommended market offers exactly the same net payout after travel costs.

### Performative Prediction
- **What it is:** A scenario where the act of making a prediction publicly changes the behavior of the agents, thereby altering the outcome and invalidating the prediction.
- **Where we use it:** This is the core problem we are solving. If we just predict "Tomato prices will spike in Pune," thousands of farmers will drive to Pune, causing prices to crash. Our entire architecture is designed to achieve *performative stability*—making a prediction that remains true even after farmers act on it.

### Prospect Theory (Loss Aversion)
- **What it is:** A behavioral economics theory (Kahneman & Tversky) stating that humans feel the pain of a loss (e.g., losing money) roughly twice as intensely as the joy of an equivalent gain. They are inherently risk-averse.
- **Where we use it:** In `src/simulation/agents.py`. We injected this psychological bias into our synthetic farmers to ensure our simulation accurately reflects irrational human behavior, rather than perfectly rational robotic behavior.

### Inverse Demand Function (Price Elasticity)
- **What it is:** A mathematical relationship mapping the quantity of a good supplied to the price consumers are willing to pay. 
- **Where we use it:** In `src/routing/payoff_functions.py`. We used the exact elasticity coefficient for Indian Tomatoes ($-0.721$) sourced from the Reserve Bank of India to simulate how fast the price crashes when a market exceeds its capacity.
