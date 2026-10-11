"""
File: src/explainability/graph_explainer.py
Role: Mathematical Graph & Temporal Attribution Engine
Description: Extracts spatial attention weights from GATv2Conv and computes gradient-based
temporal saliency over the 14-day history to pinpoint exactly which neighboring mandis
and historical days drove the model's price forecast.
Dependencies: torch, torch_geometric, numpy
"""

import torch
import torch.nn.functional as F
import numpy as np

class STGNNExplainer:
    def __init__(self, model, node_mapping, dist_matrix_km=None):
        """
        Args:
            model: Trained ST_GNN instance.
            node_mapping: Dict mapping mandi_name -> node_idx.
            dist_matrix_km: Optional pairwise distance matrix for reporting physical proximity.
        """
        self.model = model
        self.node_mapping = node_mapping
        self.idx_to_node = {idx: name for name, idx in node_mapping.items()}
        self.dist_matrix_km = dist_matrix_km
        self.class_labels = {0: "Stable", 1: "Crash", 2: "Spike"}

    def explain_node(self, x, edge_index, edge_attr, target_node_idx, top_k_neighbors=3):
        """
        Mathematically interrogates the ST-GNN for a specific market node.
        
        Args:
            x: Tensor of shape [1, num_nodes, seq_len, in_channels]
            edge_index: PyG edge index tensor [2, num_edges]
            edge_attr: Spatial distance Gaussian weights [num_edges]
            target_node_idx: Integer index of the mandi being investigated
            top_k_neighbors: Number of top influencing neighbors to extract
            
        Returns:
            dict containing prediction probabilities, top influential neighbors, and salient lag days.
        """
        self.model.eval()
        x = x.clone().detach().requires_grad_(True)
        
        # 1. Forward Pass with Attention Hook
        # Prepare expanded edge attribute
        edge_attr_expanded = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        
        # Extract temporal embedding first
        batch_size = x.size(0)
        num_nodes = self.model.num_nodes
        seq_len = self.model.seq_len
        
        x_reshaped = x.view(batch_size * num_nodes, seq_len, -1).permute(0, 2, 1)
        tcn_out = self.model.tcn(x_reshaped)
        tcn_flat = tcn_out.view(batch_size * num_nodes, -1)
        node_embeddings = F.relu(self.model.temporal_linear(tcn_flat))
        
        # Call GATv2Conv with attention weights extraction
        gat_out, (att_edge_index, att_weights) = self.model.gat(
            node_embeddings, edge_index, edge_attr=edge_attr_expanded, return_attention_weights=True
        )
        gat_activated = F.elu(gat_out)
        logits = self.model.classifier(gat_activated).view(batch_size, num_nodes, -1)
        
        target_logits = logits[0, target_node_idx]
        probs = F.softmax(target_logits, dim=-1).detach().cpu().numpy()
        pred_class_idx = int(np.argmax(probs))
        pred_label = self.class_labels[pred_class_idx]
        
        # 2. Compute Temporal Feature Saliency via Backpropagation
        target_score = target_logits[pred_class_idx]
        self.model.zero_grad()
        target_score.backward(retain_graph=False)
        
        # Saliency for target node over the seq_len days: [seq_len]
        grad = x.grad[0, target_node_idx, :, 0].abs().cpu().numpy()
        grad_norm = grad / (np.sum(grad) + 1e-8)
        
        # Top 3 most salient historical days
        salient_day_indices = np.argsort(grad_norm)[::-1][:3]
        salient_days = [
            {"day_lag": int(d - seq_len), "saliency_percent": round(float(grad_norm[d]) * 100, 2)}
            for d in salient_day_indices
        ]
        
        # 3. Spatial Neighbor Attribution from GATv2 Attention
        # Find all edges pointing TO the target_node_idx: dest == target_node_idx
        dest_nodes = att_edge_index[1].cpu().numpy()
        source_nodes = att_edge_index[0].cpu().numpy()
        # Average across attention heads if multi-head
        weights_mean = att_weights.mean(dim=-1).detach().cpu().numpy()
        
        incoming_mask = (dest_nodes == target_node_idx) & (source_nodes != target_node_idx)
        src_neighbors = source_nodes[incoming_mask]
        weights_neighbors = weights_mean[incoming_mask]
        
        top_neighbors = []
        if len(weights_neighbors) > 0:
            sorted_indices = np.argsort(weights_neighbors)[::-1][:top_k_neighbors]
            for idx in sorted_indices:
                nbr_id = int(src_neighbors[idx])
                nbr_name = self.idx_to_node.get(nbr_id, f"Node_{nbr_id}")
                dist = None
                if self.dist_matrix_km is not None:
                    dist = round(float(self.dist_matrix_km[target_node_idx, nbr_id]), 1)
                    
                top_neighbors.append({
                    "mandi": nbr_name,
                    "attention_weight": round(float(weights_neighbors[idx]), 4),
                    "distance_km": dist
                })
                
        target_name = self.idx_to_node.get(target_node_idx, f"Node_{target_node_idx}")
        
        return {
            "target_mandi": target_name,
            "predicted_class": pred_label,
            "confidence": round(float(probs[pred_class_idx]), 4),
            "class_probabilities": {
                "Stable": round(float(probs[0]), 4),
                "Crash": round(float(probs[1]), 4),
                "Spike": round(float(probs[2]), 4)
            },
            "top_influential_neighbors": top_neighbors,
            "most_salient_days": salient_days
        }
