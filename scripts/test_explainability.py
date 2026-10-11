"""
File: scripts/test_explainability.py
Role: Validation & Benchmark Script
Description: Tests the complete Neuro-Symbolic Explainability pipeline end-to-end:
GNN Attention Extraction -> Tavily News Retrieval -> Groq Qwen3.8-27b Grounded Synthesis.
Dependencies: torch, groq, tavily-python, dotenv
"""

import os
import sys
import time
import torch

# Add src to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# Ensure console handles UTF-8 (e.g. Rupee symbols in news snippets)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.st_gnn import ST_GNN
from src.explainability.graph_explainer import STGNNExplainer
from src.explainability.news_retriever import NewsRetriever
from src.explainability.advisory_generator import AdvisoryGenerator

def main():
    print("=== Testing Neuro-Symbolic Explainability Pipeline ===")
    
    # 1. Load Trained Model & Graph Topology
    graph_path = os.path.join(BASE_DIR, "data", "processed", "market_spatial_graph.pt")
    weights_path = os.path.join(BASE_DIR, "data", "processed", "st_gnn_weights.pt")
    
    if not os.path.exists(graph_path) or not os.path.exists(weights_path):
        print("Required model or graph files not found.")
        return

    print("Loading graph topology and model weights...")
    graph_data = torch.load(graph_path, weights_only=False)
    edge_index = graph_data["edge_index"]
    edge_attr = graph_data["edge_attr"]
    node_mapping = graph_data["node_mapping"]
    dist_matrix = graph_data.get("dist_matrix_km")
    num_nodes = len(node_mapping)
    
    model = ST_GNN(num_nodes=num_nodes, in_channels=1, hidden_channels=64, out_channels=3, seq_len=14)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    
    # 2. Pick a Target Mandi to Investigate (e.g., Pune or first available)
    target_name = "Pune" if "Pune" in node_mapping else list(node_mapping.keys())[0]
    target_idx = node_mapping[target_name]
    print(f"\nTarget Mandi: {target_name} (Index: {target_idx})")
    
    # Create synthetic input tensor matching real window [1, num_nodes, 14, 1]
    X_input = torch.randn((1, num_nodes, 14, 1))
    
    # 3. Mathematical Attribution via GNNExplainer
    print("\n--- Step 1: Extracting GNN Attention & Temporal Saliency ---")
    start_att = time.time()
    explainer = STGNNExplainer(model=model, node_mapping=node_mapping, dist_matrix_km=dist_matrix)
    attribution = explainer.explain_node(
        x=X_input,
        edge_index=edge_index,
        edge_attr=edge_attr,
        target_node_idx=target_idx,
        top_k_neighbors=3
    )
    att_time = (time.time() - start_att) * 1000
    print(f"Attribution extracted in {att_time:.1f} ms:")
    print(f"  Forecast: {attribution['predicted_class']} (Confidence: {attribution['confidence']*100:.1f}%)")
    print(f"  Probabilities: {attribution['class_probabilities']}")
    print(f"  Top Influential Neighbors: {attribution['top_influential_neighbors']}")
    print(f"  Salient Lag Days: {attribution['most_salient_days']}")
    
    # 4. Live News Retrieval via Tavily
    print("\n--- Step 2: Searching Regional Disruption Evidence (Tavily) ---")
    start_search = time.time()
    retriever = NewsRetriever()
    news_snippets = retriever.search_disruptions(
        target_mandi=target_name,
        neighbor_mandis=attribution["top_influential_neighbors"],
        commodity="Tomato",
        query_context="weather flood harvest arrival price"
    )
    search_time = (time.time() - start_search) * 1000
    print(f"Search completed in {search_time:.1f} ms. Found {len(news_snippets)} evidence sources:")
    for item in news_snippets:
        print(f"  - {item.get('title')}: {item.get('content')[:120]}...")
        
    # 5. Grounded Advisory Generation via Groq (Qwen3.8-27b)
    print("\n--- Step 3: Generating Grounded Advisory via Groq (Qwen3.8-27b) ---")
    start_gen = time.time()
    generator = AdvisoryGenerator(model="qwen/qwen3.8-27b")
    advisory = generator.generate_advisory(
        attribution=attribution,
        news_snippets=news_snippets,
        historical_trend="Historically, harvest peaks in mid-October trigger a 20%+ price crash in this region."
    )
    gen_time = (time.time() - start_gen) * 1000
    print(f"Advisory synthesized in {gen_time:.1f} ms.")
    
    print("\n=======================================================")
    print("                 FARMER EXPLANATION ADVISORY")
    print("=======================================================\n")
    print(advisory)
    print("\n=======================================================")
    print(f"Total Pipeline Latency: {att_time + search_time + gen_time:.1f} ms")

if __name__ == "__main__":
    main()
