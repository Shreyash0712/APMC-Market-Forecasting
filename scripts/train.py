"""
File: scripts/train.py
Role: ML Orchestration
Description: Main script to instantiate the dataset, initialize the ST-GNN model, 
and execute the training loop over the historical APMC data.
Dependencies: torch, torch.utils.data
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.apmc_dataset import APMCDataset
from src.models.st_gnn import ST_GNN
from src.models.trainer import train_epoch, evaluate

def main():
    # File Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(BASE_DIR, "IDP_APMC_NEW", "filtered_perishables.csv")
    graph_path = os.path.join(BASE_DIR, "data", "processed", "market_spatial_graph.pt")
    model_save_path = os.path.join(BASE_DIR, "data", "processed", "st_gnn_weights.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device}")
    
    print("Initializing APMC Time-Series Dataset...")
    dataset = APMCDataset(data_path, graph_path, commodity="Tomato", seq_len=14, horizon=7)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check data filters or sequence length constraints.")
        return

    # Train/Val Split (Chronological 80/20 to prevent data leakage)
    # We MUST NOT use random_split for time-series, otherwise the model 'cheats' by seeing future data.
    train_size = int(0.8 * len(dataset))
    
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))
    
    # We do not shuffle the validation loader, but we can shuffle the train loader 
    # since each window is an independent sample.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print("Loading Static Spatial Graph Topology...")
    graph_data = torch.load(graph_path, weights_only=False)
    edge_index = graph_data['edge_index']
    
    # We need to reshape edge_attr for GATv2Conv edge_dim=1
    edge_attr = graph_data['edge_attr']
    
    num_nodes = len(graph_data['node_mapping'])
    
    print("Instantiating ST-GNN Architecture...")
    model = ST_GNN(
        num_nodes=num_nodes,
        in_channels=1,
        hidden_channels=64, # Keeping complexity high for maximum accuracy
        out_channels=3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Weighted CrossEntropy to heavily penalize missing a market crash (Class 1)
    # Weights for [Stable, Crash, Spike] -> Crash gets 3x penalty
    class_weights = torch.tensor([1.0, 3.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    epochs = 5 # Running 5 epochs for initial training test
    
    print("\n--- Initiating Training Loop ---")
    best_pr_auc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, edge_index, edge_attr, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_pr_auc, val_report = evaluate(model, val_loader, edge_index, edge_attr, device)
        print(f"Validation Crash PR-AUC: {val_pr_auc:.4f}")
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            torch.save(model.state_dict(), model_save_path)
            print(">>> New best model saved! <<<")
            
    print("\nFinal Validation Report:")
    print(val_report)

if __name__ == "__main__":
    main()
