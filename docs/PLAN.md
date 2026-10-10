APMC Market Price Volatility Forecasting & Game-Theoretic Congestion-Aware Routing System
1. Executive Summary & Project Background
Agricultural markets in India (APMC mandis) experience acute localized price volatility for perishable horticultural crops (e.g., tomatoes, onions, green chillies). When harvesting gluts arrive simultaneously, local clearing prices crash below transport costs, causing distress sales and economic losses.

Existing predictive approaches focus solely on historical price regression or localized anomaly classification. However, publishing price forecasts or dispatch advisories creates a reflexive feedback loop: if thousands of farmers act on a recommendation to divert produce to a higher-priced neighboring market, that destination market immediately experiences an artificial glut and crashes.

This project addresses this fundamental limitation by combining a Spatio-Temporal Graph Neural Network (ST-GNN) for 3-to-7 day price crash forecasting with a Game-Theoretic Congestion-Aware Routing Engine based on Wardrop Equilibrium and stochastic routing allocations.
2. Core Economic & Algorithmic Foundations
To build a functional spatial arbitrage system, four theoretical and empirical pillars must be incorporated:

Performative Prediction & Reflexivity: Predictions alter the underlying data distribution once farmers act upon them. The recommendation engine must seek performative stability rather than static empirical risk minimization.
The Herding Dilemma (Minority Game / El Farol Bar Problem): Deterministic 'best mandi' recommendations guarantee failure because a market is only profitable when total arriving volume does not exceed local demand absorption capacity.
Spatial Arbitrage Realized Payoff: Arbitrage profitability depends strictly on net realized revenue after transit perishability decay, freight haulage distance costs, and APMC statutory market charges (mandi cess, weighment, and unloading charges).
Spatial Integration & Network Topology: Mandis do not operate as independent isolated markets. They form an interconnected spatial graph linked by national and state highway networks with lagged price transmission dynamics.
3. Mathematical Framework & Problem Formulation
3.1 Net Spatial Arbitrage Realized Payoff
For a farmer located at origin k deciding between local mandi i and alternative destination mandi j, the Payoff Equation is defined as:



Definitions:

E[P_j(t + τ_ij)]: Expected modal price at destination market j upon arrival at time t + τ_ij.
δ: Perishability spoilage and shrinkage rate per hour of road transit (0 ≤ δ ≤ 1).
τ_ij: Expected road transit time in hours accounting for highway infrastructure.
F_base + κ × d_kj: Logistics freight cost modeled as a base loading charge plus per-kilometer haulage rate κ over distance d_kj.
γ_j: Destination APMC statutory transaction costs (mandi cess, weighment fees, and hamali handling charges per quintal).
3.2 Endogenous Arrival Volume & Inverse Demand Function
Let a_t = (a_1, ..., a_N) denote the discrete routing choices of N independent farmers with batch volume q_n. The total arrival volume V_j(t) at market j becomes:


where I(·) is the binary indicator function.

Assuming an inverse demand curve with regional price elasticity of demand ε_d, the market clearing price is:
P_j(t) = P_0 × ( V_j(t) / V_0 )^(-1 / ε_d)
3.3 Game-Theoretic Resolution: Wardrop User Equilibrium
To avoid secondary market collapse, the advisory engine computes a Wardrop User Equilibrium (UE) where no farmer can unilaterally achieve higher net payout by deviating to another mandi. The optimization minimizes total travel and market congestion cost:



where C_j(V_j) = - Π_ij(V_j) is the congestion cost. The solution yields an optimal split probability vector p = (p_1, p_2, ..., p_M) across M candidate mandis.
4. Prior Academic & Empirical Research Review
Research Area
Key Citations
Primary Finding
Spatial Integration
Ghosh (2011), Sekhar (2012)
Perishable commodities exhibit asymmetric price transmission and high spatial friction.
IT & Market Access
Jensen (2007)
Mobile information eliminated price dispersion in Kerala sardine markets.
Informational Gains
Fafchamps & Minten (2012)
Purely informational SMS had muted gains due to credit ties and transport bottlenecks.
Performance Prediction
Perdomo et al. (2020)
Algorithms must optimize for performatively stable points to handle distribution shifts.
Congestion Games
Wardrop (1952), Rosenthal (1973)
Established the mathematical backbone for shared resource allocation and routing.

5. Spatio-Temporal Machine Learning Architecture
5.1 Graph Representation
Define the agricultural market graph as G = (V, E, W):

Vertices V: Set of M APMC mandis.
Edges E: Physical road connections and transport corridors between mandis.
Adjacency Weights W_ij: Spatial proximity matrix computed using Gaussian kernel over road distance: W_ij = exp( - d_ij^2 / σ^2 ).
5.2 Network Components
Spatial Convolution: Graph Attention Networks (GAT) capture price contagion and neighbor supply pressures.
Temporal Modeling: Temporal Convolutional Networks (TCN) capture seasonal arrival cycles and 14-day rolling momentum.
5.3 Target Classification States
The model classifies multi-horizon volatility states (horizon h = 3 to 7 days):

Crash: Relative price change ≤ -20% compared to 14-day SMA.
Spike: Relative price change ≥ +20% compared to 14-day SMA.
Stable: Within -20% to +20% bounds.
6. System Architecture & Engineering Pipeline
Data Ingestion: Daily commodity feeds via Agmarknet and e-NAM APIs; IMD localized precipitation alerts.
Storage Tier: PostgreSQL enhanced with TimescaleDB for time-series and PostGIS for spatial queries.
Inference Engine: Python FastAPI service serving ONNX compiled models for sub-50ms graph inference.
Routing Allocator: Solver module computing Wardrop equilibrium probability vectors using SciPy / OSQP.
API Gateway: Node.js / Express backend managing farmer authentication and WhatsApp/SMS push notifications.
7. Rigorous Evaluation Protocol
Precision-Recall AUC: Primary emphasis on minimizing false negatives in market crash detection.
Agent-Based Simulation (ABM): Synthetic simulation (using Mesa/SimPy with 1,000 farmer agents) comparing price volatility under No Advisory vs. Wardrop Congestion-Aware Advisory.
Price Dispersion Index: CV_t = σ_{P, t} / μ_{P, t}. Measures regional market price convergence.

8. Prior Academic & Empirical Research
Research Domain
Studies and Findings
Relevance
Spatial Market Integration in India
Ghosh (2011); Sekhar (2012); Bathla (2014): Used Cointegration and Vector Error Correction Models (VECM) across major APMCs. Found that while wholesale grain markets are reasonably integrated, perishable commodities (onion, tomato, potato) exhibit asymmetric price transmission and localized threshold friction due to perishability and high transport costs.
Proves that price divergence between neighboring mandis is frequent, but persistent spatial friction prevents natural equilibrium without active coordination.
Information Interventions & Price Dispersion
Jensen (2007) (Kerala Sardine Fishery, Quarterly Journal of Economics): Introduction of mobile phones allowed fishermen to arbitrage across beach markets, virtually eliminating price dispersion and waste without aggregate price crashes.
Fafchamps & Minten (2012) (Reuters Market Light in Maharashtra): Found that SMS price updates alone had limited income impact because farmers faced liquidity constraints and pre-committed relationships with middlemen (arhatiyas).
Information alone is insufficient; recommendations must account for farmer transport capacity and institutional frictions.
Performative Prediction & Congestion Games 
Perdomo, Hostetler, Steinhardt, & Hardt (2020): Established the theory of Performative Prediction, proving that models must optimize for performative stability rather than empirical risk minimization when predictions induce distribution shift.
Wardrop (1952); Rosenthal (1973): Congestion games and User Equilibrium (UE) vs. System Optimum (SO).



Dictates that your recommendation engine cannot emit deterministic "best mandi" advice; it must solve a stochastic routing allocation (mixed-strategy Nash equilibrium).




Research

https://www.data.gov.in/datasets_webservices/datasets/3670701 

https://journalajaees.com/index.php/AJAEES/article/view/2078/4131

https://arxiv.org/pdf/2408.05146

https://www.kaggle.com/datasets/arjunyadav99/indian-agricultural-mandi-prices-20232025 



