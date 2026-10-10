"""
File: src/data/apmc_dataset.py
Role: ML Data Pipeline
Description: PyTorch Dataset for Spatio-Temporal Graph Neural Network. Ingests raw data 
and constructs rolling 14-day sequence windows as node features, and calculates the 
target volatility classes (Crash, Spike, Stable) for a future horizon (e.g., 7 days).
Dependencies: torch, pandas, numpy
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class APMCDataset(Dataset):
    def __init__(self, data_path, graph_path, commodity="tomato", seq_len=14, horizon=7):
        """
        Initializes the Dataset by loading the sparse spatial graph and the time-series data.
        """
        self.seq_len = seq_len
        self.horizon = horizon
        self.commodity = commodity
        
        # Load spatial graph topology
        self.graph_data = torch.load(graph_path)
        self.node_mapping = self.graph_data['node_mapping']
        self.num_nodes = len(self.node_mapping)
        
        # Load time-series features
        self.features, self.targets = self._prepare_temporal_data(data_path)
        
    def _prepare_temporal_data(self, data_path):
        """
        Loads the filtered dataset, pivots into a dense [Time, Nodes] matrix, 
        and computes the target volatility classes.
        """
        # Load data efficiently using the apmc-data-access skill principles
        dtypes = {
            'market_center_name': 'category',
            'commodity_name': 'category',
            'arrival_quantity': 'float32',
            'modal_price': 'float32',
        }
        df = pd.read_csv(data_path, dtype=dtypes, parse_dates=['date'])
        df = df[df['commodity_name'] == self.commodity]
        
        # Pivot table to shape [Time, Nodes] for prices
        pivot_price = df.pivot_table(index='date', columns='market_center_name', values='modal_price', aggfunc='mean')
        # Forward fill missing days, then fill remaining with 0
        pivot_price = pivot_price.ffill().fillna(0)
        
        # Align columns with node_mapping indices
        aligned_columns = sorted(self.node_mapping.keys(), key=lambda x: self.node_mapping[x])
        # Add missing nodes as 0
        for col in aligned_columns:
            if col not in pivot_price.columns:
                pivot_price[col] = 0.0
                
        pivot_price = pivot_price[aligned_columns]
        price_matrix = pivot_price.values # Shape: [Total_Days, Num_Nodes]
        
        # Compute targets (h-day future prediction)
        # 14-day SMA for the target baseline
        sma_14 = pivot_price.rolling(window=14, min_periods=1).mean().values
        
        targets = np.zeros_like(price_matrix, dtype=np.int64)
        for t in range(len(price_matrix) - self.horizon):
            future_price = price_matrix[t + self.horizon]
            baseline = sma_14[t]
            
            # Avoid division by zero
            baseline_safe = np.where(baseline == 0, 1e-6, baseline)
            rel_change = (future_price - baseline_safe) / baseline_safe
            
            # 0: Stable, 1: Crash (<= -20%), 2: Spike (>= 20%)
            crash_mask = rel_change <= -0.20
            spike_mask = rel_change >= 0.20
            
            targets[t, crash_mask] = 1
            targets[t, spike_mask] = 2
            
        return torch.tensor(price_matrix, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        # We need at least seq_len days of history, plus the horizon in the future
        return max(0, len(self.features) - self.seq_len - self.horizon)

    def __getitem__(self, idx):
        """
        Returns:
        X: Tensor of shape [num_nodes, seq_len, num_features]
        Y: Tensor of shape [num_nodes] containing the classification targets
        """
        # Node features: Shape [num_nodes, seq_len, 1] (only price for now, can be extended to volume)
        X = self.features[idx : idx + self.seq_len].T.unsqueeze(-1)
        
        # Target classes at the end of the sequence (for the future horizon)
        Y = self.targets[idx + self.seq_len - 1]
        
        return X, Y
