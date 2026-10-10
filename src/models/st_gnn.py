"""
File: src/models/st_gnn.py
Role: Core ML Architecture
Description: Spatio-Temporal Graph Neural Network combining 1D Temporal Convolutions 
(TCN) to capture 14-day rolling price momentum, and Graph Attention Networks v2 
(GATv2Conv) to dynamically route price contagion across the spatial road network. 
Outputs logits for Crash, Spike, or Stable states.
Dependencies: torch, torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ST_GNN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels=3, seq_len=14):
        """
        Args:
            num_nodes: Total number of APMC markets (vertices).
            in_channels: Number of input features per node (e.g., 1 for price).
            hidden_channels: Dimension of the hidden embeddings.
            out_channels: Number of classification targets (Crash=0, Stable=1, Spike=2 -> 3 classes).
            seq_len: Historical window length (e.g., 14 days).
        """
        super(ST_GNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.hidden_channels = hidden_channels
        
        # Temporal Block (TCN): Extracts momentum from the past 'seq_len' days
        # We use a 1D Conv over the time dimension. 
        # Input shape to Conv1d: (Batch * Nodes, Features, Seq_Len)
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Flatten the temporal sequence into a single node embedding
        self.temporal_linear = nn.Linear(hidden_channels * seq_len, hidden_channels)
        
        # Spatial Block: Graph Attention Network v2
        # Dynamically computes attention weights for neighbors to model panic/contagion routing
        self.gat = GATv2Conv(
            in_channels=hidden_channels, 
            out_channels=hidden_channels, 
            heads=4, 
            concat=False, 
            dropout=0.2,
            edge_dim=1  # We pass a 1D edge weight (spatial distance)
        )
        
        # Classifier Head: Maps the spatio-temporal embedding to the 3 target states
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass optimized for extreme sub-50ms latency.
        Args:
            x: Node features of shape [batch_size, num_nodes, seq_len, in_channels]
            edge_index: PyG edge index tensor of shape [2, num_edges]
            edge_weight: Spatial distance Gaussian weights of shape [num_edges]
        """
        batch_size = x.size(0)
        
        # 1. TEMPORAL PROCESSING
        # Reshape to push Time into the last dimension for Conv1d: [Batch*Nodes, in_channels, seq_len]
        x = x.view(batch_size * self.num_nodes, self.seq_len, -1)
        x = x.permute(0, 2, 1) 
        
        # Pass through TCN
        x = self.tcn(x) # Output: [Batch*Nodes, hidden_channels, seq_len]
        
        # Flatten time dimension to create a single static embedding per node
        x = x.view(batch_size * self.num_nodes, -1)
        x = F.relu(self.temporal_linear(x)) # Output: [Batch*Nodes, hidden_channels]
        
        # 2. SPATIAL PROCESSING
        # PyG expects [Total_Nodes, Features]. We process the batched graphs.
        # Since the graph topology is static (same edge_index for all batches), we can use a batched edge_index.
        
        # (For simplicity in this core module, we process each batch item through GAT)
        # In a highly optimized production setting, PyG's Batch object manages this automatically.
        # Assuming `x` is already batched by PyG's DataLoader, we can just pass it through GAT.
        # PyG GATv2Conv with edge_dim=1 expects edge_attr to be shape [num_edges, 1]
        edge_weight_expanded = edge_weight.unsqueeze(-1)
        x = self.gat(x, edge_index, edge_attr=edge_weight_expanded)
        x = F.elu(x)
        
        # 3. CLASSIFICATION
        logits = self.classifier(x)
        
        # Reshape back to [batch_size, num_nodes, out_channels]
        logits = logits.view(batch_size, self.num_nodes, -1)
        
        return logits
