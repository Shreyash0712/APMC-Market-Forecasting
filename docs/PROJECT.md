Comprehensive Research Plan: APMC Price Volatility Forecasting and Congestion-Aware Routing System

Executive Summary and Theoretical Foundation

Agricultural markets in India, specifically the Agricultural Produce Market Committee (APMC) mandis, experience acute and highly localized price volatility. This phenomenon is especially devastating for perishable horticultural crops such as tomatoes, onions, and green chillies. When harvest seasons reach their peak, massive volumes of produce arrive simultaneously at local markets. This sudden, uncoordinated surge in supply overwhelms local demand absorption capacity, causing clearing prices to crash well below the cost of transportation and production1. The result is distress sales, massive economic losses for farmers, and high volumes of physical wastage.

Prior empirical research into spatial market integration in India reveals a complex landscape. Studies examining major APMCs utilizing Vector Error Correction Models (VECM) have demonstrated that while wholesale grain markets exhibit reasonable integration, perishable commodities face asymmetric price transmission and extreme spatial friction1. This friction persists due to the rapid decay of perishable goods and high transport costs, preventing natural market equilibrium without active, system-level coordination1. Historical attempts to solve this purely through information dissemination have yielded mixed results. The introduction of mobile phones in the Kerala sardine fishery famously eliminated price dispersion, but purely informational SMS interventions in Maharashtra had muted impacts because they failed to account for liquidity constraints, pre-committed intermediary relationships, and institutional frictions1.

Contemporary predictive approaches typically focus solely on historical price regression or localized anomaly classification. However, deploying a standard predictive model or a deterministic routing advisory system in the real world creates a dangerous phenomenon known as a reflexive feedback loop. If a mobile application broadcasts a deterministic recommendation—advising thousands of farmers to divert their produce to a neighboring market where prices are temporarily higher—the underlying data distribution changes the moment the farmers act on that advice. The destination market will immediately experience an artificial glut, causing its prices to crash, while the origin market may experience a sudden shortage1.

This dynamic is dictated by the theory of Performative Prediction, which proves that models deployed in social environments must optimize for performative stability rather than empirical risk minimization1. Furthermore, it perfectly mirrors the Herding Dilemma, or the El Farol Bar Problem in game theory: a shared resource is only profitable when the total arriving volume does not exceed its capacity1. Therefore, a deterministic "best mandi" recommendation system is mathematically guaranteed to fail in practice.

To resolve this, the proposed research project requires a sophisticated hybrid architecture. It combines a Spatio-Temporal Graph Neural Network (ST-GNN) designed to forecast price crashes three to seven days in advance, with a Game-Theoretic Congestion-Aware Routing Engine1. Instead of pointing all farmers to one market, the system computes a stochastic routing allocation based on the Wardrop User Equilibrium. This ensures that the recommended distribution of farmers across various markets stabilizes prices, perfectly saturates demand without exceeding it, and prevents secondary market collapses1. This document outlines a highly detailed, step-by-step architectural and research plan to build, train, deploy, and evaluate this system, prioritizing computational efficiency and accessible language to ensure the framework can be easily understood and replicated.

Core Economic and Mathematical Framework
To build a functional spatial arbitrage system, the underlying economics of a farmer transporting goods between markets must be mathematically formalized. The profitability of spatial arbitrage depends strictly on the net realized revenue after accounting for transit time, perishability, freight costs, and complex statutory market charges1.

The Spatial Arbitrage Payoff Equation
For an individual farmer located at an origin point deciding between their local mandi and an alternative destination mandi, the expected payoff is not simply the raw price difference observed on a dashboard. It must account for physical decay and institutional deductions. The payoff equation is defined to capture the true net realization. The expected payoff relies on the expected modal price at the destination market at the exact time of arrival, recognizing that prices change rapidly during transit1. Because perishable crops like tomatoes decay quickly, a perishability factor is applied based on the spoilage and shrinkage rate per hour of road transit1.

This destination revenue is compared against the expected current price at the local, default market, which serves as the baseline opportunity cost. From the potential profit, the system must subtract logistics and freight costs, modeled as a base loading and unloading charge plus a per-kilometer haulage rate multiplied by the distance traveled1. Finally, the system must deduct the destination APMC statutory transaction costs, which represents one of the most complex variables in the Indian agricultural landscape1.

Understanding Statutory Market Costs and Fees

The statutory transaction cost variable is highly complex because APMC fees are not uniform across India; agricultural marketing is a state subject under the Constitution of India6. For an accurate routing system, the algorithm cannot assume a flat national tax rate. The statutory costs include several distinct levies.

The primary levy is the Market Fee, often called Mandi Shulk or Market Cess. This is a statutory ad valorem levy on the gross sale value of notified produce, collected by the Market Committee to fund mandi infrastructure7. The incidence of this fee legally falls on the buyer in many states, but in practice, it is frequently passed back into the farmer's settlement receipt, known as the patti7. In some states, this is layered with a Rural Development Fund fee, earmarked for rural infrastructure7.

Beyond government taxes, the farmer must pay the Arhat, which is the commission earned by the commission agent, or Arhtiya, for facilitating the auction7. Commission agents operate as either Kacha Arhtiyas, who deal directly with farmers, or Pakka Arhtiyas, who deal between other agents9. Finally, the physical handling of the produce incurs labor charges, categorized as 
Hamali for loading and unloading, and Tulai for weighment at the mandi scale9.

State
Primary Market Fee
Rural Development Cess
Typical Commission (Arhat)
Total Approximate Deduction
Punjab
3%
3%
2.5%
8.5%
Haryana
2%
2%
2.5%
6.5%
Madhya Pradesh
1.5%
None
2.0%
3.5%
Maharashtra
1.0%
None
Variable
1.0% - 2.0%
Gujarat
0% (Deregulated)
None
Variable
0.8% - 2.0%

As illustrated in the table above, the variance in deductions is massive. In Punjab, the combined deductions can reach 8.5% for specific commodities, whereas in Gujarat, fruits and vegetables have been largely deregulated, resulting in a 0% market fee, though commission and weighment charges still apply7. The system's database must maintain an updated configuration of these fees per state and per commodity to ensure the payoff equation reflects the true net realization for the farmer, preventing the algorithm from incorrectly recommending a market where the tax burden negates the price premium10.

Endogenous Arrival Volume and Inverse Demand

The price at a destination market is not static; it reacts dynamically to the volume of produce arriving on any given day. This is modeled using an inverse demand function. The total arrival volume at a destination market is the sum of organic arrivals—farmers who would have gone there anyway—and the new arrivals induced by the routing system's recommendations1.
Assuming a regional price elasticity of demand, the market clearing price is mathematically forced to drop if the routing system directs too much volume to one location. As the total volume spikes relative to the baseline capacity, the inverse demand curve simulates a market crash1. The system's underlying algorithm must constantly anticipate this localized price drop before broadcasting recommendations, effectively solving a massive balancing act across the entire state's geography.

Step 1: Dataset Acquisition and Data Engineering

A robust research project requires high-quality, continuous, and diverse data. The proposed system relies on disparate data sources covering market prices, exogenous weather conditions, and spatial infrastructure routing, all integrated into a high-efficiency database architecture.

Primary Data Sources and API Ingestion

The foundational dataset is sourced from the Agricultural Marketing Information Network, commonly known as Agmarknet. Launched in 2000 as a central sector scheme, Agmarknet links thousands of agricultural produce markets spread across the country, providing daily minimum, maximum, and modal prices alongside arrival volumes12. This network is crucial because it ensures that farmers and systems have visibility into price disparities, serving as the basis for fairer price discovery12.

The system will programmatically ingest this data using REST APIs. While direct scraping is possible, it is inefficient. Instead, the project will leverage endpoints provided by the Centre for Economic Data and Analysis (CEDA), which serves clean Agmarknet data13. The API architecture allows the system to send POST requests specifying the commodity, state, and district, returning structured JSON payloads containing the historical and real-time market data13. Furthermore, integration with the newer National Agriculture Market (e-NAM) API endpoints provides additional transactional layers and visibility into pan-India electronic trading portals15.

API Endpoint Category
Primary Function
Update Frequency
System Utility
/agmarknet/prices
Fetches daily modal, min, and max prices.
Daily
Baseline historical truth for ST-GNN training.
/agmarknet/quantities
Retrieves daily arrival volumes by market.
Daily
Calculating supply pressure and market capacity.
IMD CAP Alerts
Provides localized severe weather warnings.
Real-time
Indicator for harvest disruptions or road blockages.
PostGIS Routing
Calculates actual road distances between nodes.
Static
Computing transit time and per-kilometer freight cost.

Exogenous shocks are a massive driver of agricultural volatility. The India Meteorological Department (IMD) API provides real-time localized precipitation alerts and seasonal forecasts16. Heavy rainfall disrupts harvesting operations and blocks rural transportation corridors, acting as an immediate constraint on market arrivals. By fusing IMD data with Agmarknet pricing, the model gains the context necessary to understand why a sudden supply drop occurred.

Database Architecture and Computational Efficiency

Storing and querying millions of rows of daily time-series data alongside complex spatial coordinates requires a highly specialized database architecture to ensure research efficiency. A standard relational database would quickly bottleneck during the training phase.

The foundational storage tier utilizes PostgreSQL, augmented with two critical extensions. First, PostGIS is implemented to handle all spatial queries. When a farmer opens the application, PostGIS allows the backend to instantly execute a radius query, identifying all candidate mandis within a 150-kilometer driving distance of the farmer's specific geolocation.

Second, the system utilizes TimescaleDB, an industry-leading time-series database extension for PostgreSQL. Efficiency is paramount for a production-grade research system, and TimescaleDB introduces a revolutionary feature known as Continuous Aggregates18. The machine learning model relies heavily on historical context, specifically requesting the 14-day Simple Moving Average (SMA) of prices and volumes for every market to establish momentum1. Calculating this 14-day rolling average on the fly for hundreds of markets across thousands of historical days would take minutes per query, bottlenecking the entire training loop.

Continuous aggregates solve this by automatically running in the background, pre-computing and incrementally updating these rolling averages as new daily data arrives via the Agmarknet API18. When the neural network requests the 14-day momentum feature, the database simply fetches a pre-calculated materialized view. This architectural choice reduces query latency from several seconds down to single-digit milliseconds, enabling rapid model iteration and real-time inference18.

Step 2: Spatio-Temporal Graph Neural Network (ST-GNN) Training

The predictive core of the routing system is a Spatio-Temporal Graph Neural Network (ST-GNN). Agricultural markets do not operate in isolation; they are deeply interconnected ecosystems where a shock in one location ripples outward1. For example, a severe price crash in a major hub like the Ahmedabad APMC will inevitably send waves to smaller surrounding sub-yards like Sanand or Dholka, because wholesale traders will immediately divert their procurement trucks to the cheaper hub21.

Standard time-series models like ARIMA or Prophet evaluate each market individually, completely ignoring this spatial contagion. ST-GNNs are perfectly suited for this domain because they combine graph-based spatial reasoning with temporal sequence modeling, allowing the network to learn how price pressures flow across the highway infrastructure over time4.

Graph Representation and Spatial Proximity

To process this data, the physical agricultural ecosystem is mathematically transformed into a graph. The vertices, or nodes, of this graph represent the individual APMC mandis1. The edges represent the physical highway connections and transport corridors linking these markets.
Crucially, the relationship between markets is defined by an adjacency weight matrix. This matrix represents spatial proximity, indicating how strongly two markets influence each other. Instead of a simple binary connection, the system computes the weights using a Gaussian kernel over the actual road distance. Markets that are physically close have high edge weights, meaning the neural network will assume a price shock in one strongly and rapidly influences the other, whereas markets separated by vast distances will have near-zero influence weights1.

Neural Network Architecture: GATv2 and TCN

The deep learning architecture is built utilizing PyTorch Geometric (PyG) and the specialized PyTorch Geometric Temporal library, which provides state-of-the-art tools for processing dynamic spatio-temporal signals24. The network features two primary computational blocks working in tandem.

The spatial convolution is handled by Graph Attention Networks (GAT). However, standard graph convolutions treat all neighboring nodes equally, and even the original GAT architecture suffers from a critical flaw known as the static attention problem. In standard GAT, the ranking of attended nodes is unconditioned on the query node, meaning the network struggles to dynamically shift its attention based on immediate context26. To solve this, the system implements GATv2Conv, an upgraded operator that fixes the static attention issue26. GATv2 allows the model to dynamically compute attention weights for every single prediction. If Ahmedabad is experiencing a massive supply glut, GATv2 allows neighboring nodes to temporarily pay maximum attention to Ahmedabad's price signals, ignoring other usually important neighbors26.

Once the spatial features are extracted for a given sequence of days, the temporal dependencies must be captured to understand momentum. This is achieved using Temporal Convolutional Networks (TCN). TCNs utilize dilated causal convolutions, allowing them to look back over long historical sequences, capturing the seasonal arrival cycles and the 14-day rolling momentum of prices1. Unlike traditional Recurrent Neural Networks (RNNs) or LSTMs, TCNs avoid the vanishing gradient problem and allow for massive parallelization during training, significantly reducing compute time27.

Multi-Horizon Target Classification

Attempting to predict the exact numerical price of a highly volatile commodity like tomatoes is largely an exercise in futility due to systemic noise. Therefore, this research models volatility forecasting as a multi-horizon state classification problem1.

The system predicts the categorical state of the market for a horizon of three to seven days into the future. The target states are categorized logically. A "Crash" is defined as a relative price drop of 20% or more compared to the 14-day Simple Moving Average. A "Spike" is a relative price increase of 20% or more. If the price remains within these bounds, it is classified as "Stable"1.

This creates a highly imbalanced dataset. Stable days are common, while crashes and spikes are rare extreme events. If trained naively, the neural network will simply learn to predict "Stable" every time to achieve high baseline accuracy. The training protocol mitigates this by utilizing weighted cross-entropy loss, heavily penalizing the model for missing a crash, ensuring the network learns the underlying spatial triggers for extreme volatility rather than taking the path of least mathematical resistance.

Step 3: Game-Theoretic Routing and the Frank-Wolfe Algorithm

The most critical phase of the research project occurs after the predictions are made. If the ST-GNN predicts a severe price spike in market A and a crash in market B, naive logic dictates advising all farmers at market B to immediately travel to market A. As established, this deterministic routing triggers the Herding Dilemma, causing market A to crash under the weight of the new arrivals1.

The core research novelty of this system is passing the ST-GNN predictions into a Game-Theoretic Routing Allocator. The system must find a state of performative stability—a recommendation that remains accurate even after thousands of independent human agents act upon it1.

Wardrop User Equilibrium

To achieve performative stability, the system treats agricultural logistics as a macro-level traffic assignment problem. It computes a Wardrop User Equilibrium. Formulated by John Glen Wardrop in 1952, this principle states that in an equilibrium state, no individual user can unilaterally achieve a lower cost (or in this case, a higher net payout) by deviating from their assigned route1.

In the context of the APMC market graph, this means the routing engine carefully balances the volumes sent to various candidate mandis until the expected net payoff at every utilized mandi is exactly identical. If one mandi offered a higher payout, more farmers would logically divert there, driving the price down until it equalized with the alternatives. The system pre-calculates this state of equilibrium before any farmer starts their engine. The optimization problem seeks to minimize the total travel and market congestion costs across all candidate mandis, subject to the constraint that all surplus volume in the region is assigned1.

Optimization via the Frank-Wolfe Algorithm

Solving this complex integral optimization over a multi-dimensional spatial network requires a specific and historically proven mathematical tool: the Frank-Wolfe algorithm5.
The Frank-Wolfe algorithm, originally published in 1956 and also known as the conditional gradient method, is an iterative, first-order optimization algorithm specifically tailored for constrained convex optimization5. It is the historical gold standard for computing Wardrop equilibria and solving traffic assignment problems because of its unique operational mechanics5.

The absolute necessity of the Frank-Wolfe algorithm in this architecture stems from its projection-free nature31. In standard machine learning optimization, such as projected gradient descent, taking a mathematical step toward the minimum might push the solution outside the "feasible set." For example, a standard algorithm might accidentally recommend sending 110% of farmers to a specific market, which is mathematically impossible. To correct this, the algorithm must perform a complex mathematical projection to pull the solution back into the feasible boundaries. Projection operations on complex polytopes are computationally massive and prohibitively slow for real-time systems31.

Frank-Wolfe avoids projections entirely through a highly efficient iterative process. In each iteration, it considers a linear approximation of the complex congestion objective function5. It then solves a simple direction-finding subproblem, essentially finding the theoretical best route based on that linear approximation, utilizing a Linear Minimization Oracle (LMO)5. The algorithm then determines an optimal step size through a line search, ensuring it moves toward the optimal solution without overshooting5.

Finally, the critical step: it updates the recommendation as a convex combination of the previous state and the newly found direction30. Because a convex combination of two valid probabilities is mathematically guaranteed to remain a valid probability, the solution never leaves the feasible set, completely eliminating the need for expensive projection operations31.
While the worst-case convergence rate of standard Frank-Wolfe is sublinear, modern variants utilized in this research, such as those employing away-steps or pairwise operations, guarantee much faster, linearly convergent rates for strongly convex objectives and specific problem classes, allowing the system to find the Wardrop equilibrium in fractions of a second5.

The ultimate output of the Frank-Wolfe algorithm is an optimal split probability vector across the candidate mandis1. Instead of generating a deterministic command to "Go to Ahmedabad," the system's API sends a probabilistic push notification. For a given village, 40% of the farmers might receive a recommendation for Ahmedabad, 35% for Padra, and 25% for Anand. This stochastic allocation guarantees that local demand at every destination is perfectly saturated but never overwhelmed, securing absolute performative stability.

Step 4: System Architecture and Inference Efficiency

For this project to function as a live, highly trusted advisory tool, extreme inference speed is mandatory. Agricultural price data is ingested daily, but farmer queries—likely submitted via a lightweight WhatsApp API gateway or mobile application—occur asynchronously in real-time. If the system takes several seconds to compute the ST-GNN forward pass and execute the Frank-Wolfe routing, the user experience degrades, and server costs scale unsustainably.
The system architecture involves a Python FastAPI service serving the models. To achieve extreme efficiency and the target of sub-50ms graph inference, the trained PyTorch models are not run natively in Python. Instead, they are compiled into the ONNX (Open Neural Network Exchange) format and executed using the ONNX Runtime engine1.

Deep CPU Performance Tuning via ONNX Runtime

While GPUs are excellent for the massive matrix multiplications required during the training phase, production inference for structured, tabular graph data often runs on CPU-based cloud instances to manage operational costs37. ONNX Runtime acts as the execution engine, taking the compiled .onnx file and figuring out how to execute the computational graph as fast as possible, providing extensive mechanisms for deep CPU performance tuning38.

To minimize latency, the system utilizes highly specific threading configurations via the Ort::SessionOptions API in the ONNX Runtime environment38. The runtime separates thread management into two distinct categories:
Intra-op Threads: This controls the parallelism inside a single mathematical operator. For example, when the system executes a massive matrix multiplication within the GATv2Conv layer, it splits that single task among multiple threads. The number of intra-op threads should generally be aligned with the number of physical CPU cores to maximize throughput without causing context-switching overhead38.
Inter-op Threads: This controls the parallelism between different operators. It attempts to run multiple independent tasks at once38.

For models like ST-GNNs where one layer's output generally feeds directly into the next in a sequential pipeline, attempting to parallelize operators often hurts performance due to the overhead of managing the threads. Therefore, the system sets the execution mode to ORT_SEQUENTIAL and strictly limits inter-op threads to 138.
Furthermore, ONNX Runtime allows control over thread spinning behavior. By default, threads may spin in a loop waiting for new work, which provides faster inference but consumes massive amounts of CPU cycles and power. For a cloud-deployed API handling burst traffic, the system configures the spin duration and exponential backoff parameters to reduce CPU usage during idle windows, ensuring the infrastructure remains cost-effective without sacrificing the sub-50ms latency target during active inference39.

Step 5: Rigorous Evaluation Protocol and Agent-Based Simulation

A standard machine learning train/test split based purely on accuracy metrics is woefully insufficient for a research project operating in a dynamic, performative economic environment. The evaluation protocol must simulate the real-world economic impact and psychological realities of the system1.

Metric 1: Precision-Recall Area Under Curve (PR-AUC)

In agricultural volatility forecasting, a "Stable" market day is the most common outcome, while a "Crash" is a rare but financially catastrophic event. If a model simply predicts "Stable" every single day, its overall baseline accuracy might look impressive at 85%, but it would be entirely useless for risk mitigation.

Therefore, standard accuracy metrics and Receiver Operating Characteristic (ROC) curves are explicitly discarded in favor of the Precision-Recall Area Under Curve (PR-AUC). The primary research emphasis is strictly on minimizing false negatives in market crash detection1. Missing a crash warning (a false negative) results in severe financial distress and physical waste for the farmer. Conversely, a false positive (warning of a crash that does not materialize) merely results in cautious routing to a safer, stable market. The PR-AUC ensures the model is evaluated on its ability to catch the rare events that destroy livelihoods.

Metric 2: Agent-Based Modeling (ABM) via Mesa

To evaluate the Game-Theoretic routing engine without risking actual farmers' capital, the research employs Agent-Based Modeling (ABM) using the Python Mesa framework1.
Mesa is an open-source, Apache2 licensed framework designed to simulate complex systems by creating autonomous "agents" that interact within a defined spatial environment40. Inspired by foundational models like Epstein and Axtell's "Sugarscape with Traders," the simulation instantiates 1,000 synthetic farmer agents, each initialized with a specific crop volume, geolocation, and base wealth1.

Mesa manages these agents through an advanced scheduling system and spatial grid. The BatchRunner class is utilized to run thousands of parallel simulations across all possible combinations of parameters, collecting data iteratively to ensure the results are statistically robust rather than an anomaly of a single run41. The DataCollector class records the state of every agent and market at every time step, exporting the results to Pandas DataFrames for deep analysis41.

A critical layer of behavioral economics is programmed into the logic of these Mesa agents: Prospect Theory and Loss Aversion47. Standard economic models falsely assume agents are perfectly rational actors maximizing utility. Prospect theory, developed by Nobel laureates Daniel Kahneman and Amos Tversky, proves that humans are fundamentally loss-averse; the psychological pain of losing money is far greater than the joy of gaining the exact same amount49.

In the Mesa simulation, the synthetic farmer agents are programmed to exhibit this loss aversion when evaluating the Wardrop equilibrium recommendations51. If the system recommends a highly volatile market with a massive potential payout but a high risk of a crash, the risk-averse agents will penalize that route, preferring a slightly lower but guaranteed payout at a stable market47. This psychological realism ensures the stochastic routing vectors are tested against human hesitation and bias.

The simulation compares two primary scenarios:
Baseline (No Advisory): Agents rely on word-of-mouth or recent memory, mimicking the current real-world herding behavior that leads to gluts.
Intervention (Wardrop Congestion-Aware Advisory): Agents receive and act upon the stochastic probability vectors generated by the Frank-Wolfe algorithm, adjusted by their internal loss aversion parameters.

Metric 3: Price Dispersion Index
The ultimate measure of the system's success is not just individual agent profit, but the stabilization of the macroeconomic environment. This is quantified continuously during the Mesa simulation using the Price Dispersion Index, calculated as the Coefficient of Variation () across the regional market1.

In this formula,  represents the standard deviation of prices across all mandis at time , and  is the mean price1. A high  indicates severe spatial friction and market failure—meaning tomatoes are rotting in one mandi for ₹2 per kilogram while simultaneously selling for ₹40 per kilogram in a mandi just 100 kilometers away. As the simulation progresses, the Mesa data collectors track this index. If the ST-GNN forecasting and the Frank-Wolfe routing engine function correctly together, the Price Dispersion Index will steadily approach zero, indicating perfect spatial integration, the eradication of localized gluts, and the successful resolution of the performative prediction paradox.
Works cited
APMC Price Volatility & Congestion-Aware Routing System.pdf
Analysing price volatility in agricultural value chains using systems, https://www.researchgate.net/publication/351894519_Analysing_price_volatility_in_agricultural_value_chains_using_systems_thinking_A_case_study_of_the_Indonesian_chilli_value_chain
Daily Price and Arrival Report - Agmarknet, https://agmarknet.gov.in/daily-price-and-arrival-report
ST-GNN: Spatio-Temporal Graph Neural Networks - Emergent Mind, https://www.emergentmind.com/topics/spatio-temporal-graph-neural-network-st-gnn
Frank–Wolfe algorithm - Wikipedia, https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm
Agricultural Produce Market Committee (APMC), https://ies.gov.in/arthapedia/concept/agricultural-produce-market-committee-apmc
Mandi Market Fee & Cess by State in India: Rates, Who Pays, and, https://www.mandigrow.com/blog/mandi-market-fee-cess-by-state-india
APMC Mandi Fee & Market Cess Rates — State-Wise ... - MandiGrow, https://www.mandigrow.com/blog/apmc-mandi-fee-calculation-india-2026
How APMC Mandi Works in India — Complete Guide for, https://www.mandigrow.com/blog/how-apmc-mandi-works-india-complete-guide
APMC and Mandi Cess Reconciliation Across Indian States, https://www.terra-insight.com/insights/apmc-mandi-cess-reconciliation-india/
The Gujarat Government Gazette - Prsindia.org, https://prsindia.org/files/bills_acts/bills_states/gujarat/2020/Gujarat%20APMC%20Ordinance%202020.pdf
AGMARKNET: Enhancing Market Information for Indian Farmers, https://agriculture.institute/institutional-support/agmarknet-market-information-indian-farmers/
Swagger UI - CEDA Data Portal API, https://api.ceda.ashoka.edu.in/documentation/
Prajwal-Shrimali/agmarknetAPI: API for agmarknet Website - GitHub, https://github.com/Prajwal-Shrimali/agmarknetAPI
eNam | Home, https://enam.gov.in/
IMD API Management - India Meteorological Department, https://api.imd.gov.in/
India Weather Forecast Bulletin - IMD, https://mausam.imd.gov.in/imd_latest/contents/all_india_forcast_bulletin.php
Continuous Aggregates: Incremental Materialized Views for Time, https://www.tigerdata.com/learn/continuous-aggregates-timescaledb
Aggregate Time Series Data with TimescaleDB - Mindee, https://www.mindee.com/blog/aggregate-time-series-data-with-timescaledb
TimescaleDB Data Retention Policies Tutorial - Binadit, https://binadit.com/tutorials/implement-timescaledb-data-retention-policies-for-time-series-optimization
Ahmedabad APMC Market Prices Today | Wholesale Rates - Fruits, https://www.commoditymarketlive.com/mandi-price-market/ahmedabad-apmc
Gujarat Mandi List by District | APMC Markets, https://mandipulse.com/mandi-bhav/gujarat
Gujarat APMC Mandi Bhav – Apps on Google Play, https://play.google.com/store/apps/details?id=com.envisiontechnolabs.GujaratAPMC&hl=en_IE
Introduction — PyTorch Geometric Temporal documentation, https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html
PyTorch Geometric Temporal Documentation — PyTorch Geometric, https://pytorch-geometric-temporal.readthedocs.io/
pytorch_geometric/torch_geometric/nn/conv/gatv2_conv.py at master, https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gatv2_conv.py
Temporal Edge Regression with PyTorch Geometric - Towards AI, https://pub.towardsai.net/temporal-edge-regression-with-pytorch-geometric-3267d79cfa03
torch_geometric.nn.conv.GATv2Conv - PyTorch Geometric, https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GATv2Conv.html
Linearly Convergent Frank-Wolfe Made Practical, https://opt-ml.org/papers/2019/paper_7.pdf
An Improved Frank–Wolfe Algorithm to Solve the Tactical Investment, https://www.mdpi.com/2227-7390/13/18/3038
Boosting Frank-Wolfe by Chasing Gradients, http://proceedings.mlr.press/v119/combettes20a/combettes20a-supp.pdf
Iterative Approach for Traffic Assignment Problem with Capacity Limits, https://www.ntmsci.com/AjaxTool/GetArticleByPublishedArticleId?PublishedArticleId=8730
Modeling the effect of autonomous vehicles (AVs) on the ... - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC11039750/
Linear Convergence of the Frank-Wolfe Algorithm over Product, https://arxiv.org/pdf/2505.11259
Frank-Wolfe Algorithms for Saddle Point problems, https://www.di.ens.fr/~gidel/slides/OPT2016_slides.pdf
Tune performance | onnxruntime, https://onnxruntime.ai/docs/performance/tune-performance/
ONNX Runtime Execution Providers, https://onnxruntime.ai/docs/execution-providers/
The Inference Engine Behind Your ONNX Models - Medium, https://isvidhi.medium.com/onnx-runtime-under-the-hood-how-onnx-models-actually-run-in-production-72290e8182c4
Thread management | onnxruntime, https://onnxruntime.ai/docs/performance/tune-performance/threading.html
Mesa: Agent-based modeling in Python — Mesa 4.0.0a0, https://mesa.readthedocs.io/
Utilizing Python for Agent-Based Modeling: The Mesa Framework, https://www.researchgate.net/publication/344675633_Utilizing_Python_for_Agent-Based_Modeling_The_Mesa_Framework
About Agent Based Modeling · mesa/mesa Wiki - GitHub, https://github.com/projectmesa/mesa/wiki/About-Agent-Based-Modeling
Agent-Based Models with Python: An Introduction to Mesa, https://www.complexityexplorer.org/courses/172-agent-based-models-with-python-an-introduction-to-mesa
Creating Your First Model — Mesa 4.0.0a0 documentation, https://mesa.readthedocs.io/latest/tutorials/0_first_model.html
Mesa 3: Agent-based modeling with Python in 2025 - Zenodo, https://zenodo.org/records/18651025
Python's Agent-Based Modelling for Agriculture - Medium, https://medium.com/@lazuardy.almuzaki/pythons-agent-based-modelling-for-agriculture-a6259081ab07
Prospect theory - Wikipedia, https://en.wikipedia.org/wiki/Prospect_theory
Prospect Theory - The Decision Lab, https://thedecisionlab.com/reference-guide/economics/prospect-theory
What is Prospect Theory? Evaluating Risk and Loss Aversion, https://www.ninety.io/founders-framework/articles/prospect-theory
Prospect Theory: An Analysis of Decision under Risk - MIT, https://web.mit.edu/curhan/www/docs/Articles/15341_Readings/Behavioral_Decision_Theory/Kahneman_Tversky_1979_Prospect_theory.pdf
Integrating Behavioral Theories in Agent-Based Models ... - Frontiers, https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2021.686329/full
Risk aversion, impatience, and adoption of conservation agriculture, https://pmc.ncbi.nlm.nih.gov/articles/PMC10901030/
Using Insights from Prospect Theory to Enhance Sustainable ... - MDPI, https://www.mdpi.com/2071-1050/10/8/2693
