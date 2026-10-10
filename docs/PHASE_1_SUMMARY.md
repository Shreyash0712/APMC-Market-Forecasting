# Phase 1: Data Engineering & AI Forecasting Baseline

This document summarizes the progress, file architecture, and baseline experimental results achieved during Phase 1 of the APMC Market Forecasting project. We will use this as a reference point to measure improvements in subsequent phases.

## 1. Architectural Setup
We successfully established a cutting-edge, containerized data pipeline and a highly optimized PyTorch ML environment.

**Key Files & Components:**
- [docker-compose.yml](../docker-compose.yml): Spins up the TimescaleDB and pgAdmin environments, using secure credentials loaded from our `.env` file.
- [01_enable_extensions.sql](../db_init/01_enable_extensions.sql) & [02_schema.sql](../db_init/02_schema.sql): Initializes the PostGIS spatial engine and the TimescaleDB hypertable for optimal time-series partitioning.
- [ingest_to_db.py](../scripts/ingest_to_db.py): Handled the rapid, memory-safe bulk ingestion of 480,000+ historical market records into the database.

## 2. Graph Construction
We successfully modeled the Indian agricultural market network as a Spatio-Temporal Graph.

**Key Files & Components:**
- [build_graph.py](../scripts/build_graph.py): Queried market coordinates, calculated exact Haversine geodesic distances, and applied a Gaussian Kernel to build the Adjacency Matrix ($W_{ij}$). 
- **Graph Topology:** The network resulted in 1,058 unique market nodes connected by 53,580 highly relevant spatial edges (pruned for sparsity), saved in `data/processed/market_spatial_graph.pt`.

## 3. Neural Network Architecture
We built a custom PyTorch model capable of capturing both temporal momentum and spatial price contagion.

**Key Files & Components:**
- [apmc_dataset.py](../src/data/apmc_dataset.py): Formulates 14-day rolling sequences and dynamically calculates the forward classification target (Crash $\le -20\%$, Spike $\ge +20\%$, Stable).
- [st_gnn.py](../src/models/st_gnn.py): The core architecture bridging 1D Temporal Convolutions (TCN) with Graph Attention v2 (GATv2Conv).
- [trainer.py](../src/models/trainer.py) & [train.py](../scripts/train.py): The orchestration and evaluation loops.

## 4. Initial Baseline Results
We ran a synthetic hardware benchmark followed by a 5-epoch training loop on the actual dataset. These are the baseline results we aim to improve upon.

### Hardware & Latency Benchmark (Batch Size = 1)
- **Target:** < 50ms CPU Inference
- **Achieved:** ~179ms CPU Inference
- **Status:** **Needs Optimization** (Proposed Solution: ONNX Runtime export before integration into the Wardrop Routing Engine).

### Training Metrics (Epoch 5 Validation - Chronological Split)
- **Crash PR-AUC:** 0.9256
- **Crash Recall:** 0.87
- **Crash Precision:** 0.84
- **Status:** **Exceptional Baseline**. After stripping out `random_split` to prevent time-series data leakage, the AI was forced to predict the final 20% of the timeline completely blind. Due to the Class-Weighted loss (3.0 for Crashes), it achieved an astonishing 87% recall on crashes, sacrificing its ability to predict Spikes (0.05 recall) in order to protect against catastrophic downside risk.

*Model weights are currently saved at `data/processed/st_gnn_weights.pt`.*
