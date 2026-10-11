"""
File: src/routing/adversarial_defense.py
Role: Adversarial Robustness & Cartel Defense
Description: Detects artificial auction price manipulation by unscrupulous mandi cartels
using spatial graph neighbor consensus (spatial z-score divergence). Sanitizes spoofed prices
to shield farmers from orchestrated supply glut traps.
Dependencies: numpy
"""

import numpy as np

class CartelAnomalyDetector:
    def __init__(self, divergence_threshold=0.35):
        """
        Args:
            divergence_threshold: Maximum allowable fractional premium above the 
                                  spatially-weighted neighbor price before flagging manipulation.
        """
        self.divergence_threshold = divergence_threshold

    def evaluate_spatial_consensus(self, candidate_prices, neighbor_graph_weights):
        """
        Compares each market's reported price against its geographic neighbors' consensus.
        
        Args:
            candidate_prices: Array of shape [M] with reported prices per mandi.
            neighbor_graph_weights: 2D array of shape [M, M] with spatial Gaussian adjacency weights.
            
        Returns:
            dict containing:
                - flags: Boolean array of shape [M] (True if cartel manipulation suspected).
                - spatial_consensus_prices: Array of expected neighbor-implied prices.
                - sanitized_prices: Price vector with spoofed artificial spikes damped to safe bounds.
                - divergence_scores: Percentage divergence from spatial neighbor consensus.
        """
        prices = np.array(candidate_prices, dtype=np.float64)
        W = np.array(neighbor_graph_weights, dtype=np.float64)
        num_mandis = len(prices)
        
        # Zero out self-loops to only aggregate actual neighbors
        np.fill_diagonal(W, 0.0)
        
        # Row-normalize adjacency weights
        row_sums = np.sum(W, axis=1, keepdims=True)
        # Avoid division by zero for isolated nodes
        W_norm = np.where(row_sums > 1e-6, W / row_sums, 0.0)
        
        # Compute expected neighbor consensus price: Shape [M]
        consensus_prices = np.dot(W_norm, prices)
        
        flags = np.zeros(num_mandis, dtype=bool)
        divergence_scores = np.zeros(num_mandis, dtype=np.float64)
        sanitized_prices = prices.copy()
        
        for i in range(num_mandis):
            exp_p = consensus_prices[i]
            if exp_p > 1e-4:
                divergence = (prices[i] - exp_p) / exp_p
                divergence_scores[i] = divergence
                
                # If a market reports a massive spike not corroborated by any spatial neighbor:
                if divergence > self.divergence_threshold:
                    flags[i] = True
                    # Sanitize price down to neighbor consensus + safe cap to prevent baiting
                    sanitized_prices[i] = exp_p * (1.0 + self.divergence_threshold * 0.5)
                    
        return {
            "manipulation_flagged": flags,
            "spatial_consensus_prices": np.round(consensus_prices, 2),
            "sanitized_prices": np.round(sanitized_prices, 2),
            "divergence_pct": np.round(divergence_scores * 100, 2)
        }
