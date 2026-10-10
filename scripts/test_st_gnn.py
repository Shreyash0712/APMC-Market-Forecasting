"""
File: scripts/test_st_gnn.py
Role: Validation Script
Description: Synthetically tests the ST-GNN forward pass to ensure tensor dimensionalities 
align perfectly between the TCN and GATv2 layers, and benchmarks inference latency to 
ensure we hit the project's sub-50ms target.
Dependencies: torch, time
"""

import time
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.st_gnn import ST_GNN

def run_synthetic_test():
    # Load the actual graph topology we built earlier
    graph_path = r"C:\Users\YASHVI Sharma\OneDrive\Desktop\APMC Market Forecasting\data\processed\market_spatial_graph.pt"
    
    print("Loading Graph Topology...")
    graph_data = torch.load(graph_path, weights_only=False)
    edge_index = graph_data['edge_index']
    edge_attr = graph_data['edge_attr']
    num_nodes = len(graph_data['node_mapping'])
    
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {edge_index.size(1)}")
    
    # Model Hyperparameters
    in_channels = 1   # Only Price initially
    hidden_channels = 64
    out_channels = 3  # Crash, Stable, Spike
    seq_len = 14      # 14 days of history
    batch_size = 1   # Inference batch size
    
    print("\nInitializing ST-GNN Model...")
    model = ST_GNN(
        num_nodes=num_nodes,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        seq_len=seq_len
    )
    
    # Generate synthetic input tensor matching the Dataset output
    # Shape: [batch_size, num_nodes, seq_len, in_channels]
    X_dummy = torch.randn((batch_size, num_nodes, seq_len, in_channels))
    
    print("\nExecuting Forward Pass...")
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(X_dummy, edge_index, edge_attr)
        
    # Latency Benchmark
    start_time = time.time()
    with torch.no_grad():
        out = model(X_dummy, edge_index, edge_attr)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    
    print("\n--- RESULTS ---")
    print(f"Input Shape:  {X_dummy.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Expected Output Shape: torch.Size([{batch_size}, {num_nodes}, {out_channels}])")
    
    assert out.shape == (batch_size, num_nodes, out_channels), "TENSOR SHAPE MISMATCH!"
    
    print(f"Inference Latency: {latency_ms:.2f} ms")
    if latency_ms < 50:
        print("Latency Target (sub-50ms): PASSED!")
    else:
        print("Latency Target (sub-50ms): FAILED!")

if __name__ == "__main__":
    run_synthetic_test()
