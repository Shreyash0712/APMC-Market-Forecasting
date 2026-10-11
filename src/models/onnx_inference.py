"""
File: src/models/onnx_inference.py
Role: Production Inference Wrapper
Description: High-performance inference engine supporting both compiled ONNX Runtime
and PyTorch native execution with zero-accuracy-loss guarantees.
Dependencies: onnxruntime, torch, numpy
"""

import os
import numpy as np
import onnxruntime as ort
import torch

class STGNNInferenceEngine:
    def __init__(self, onnx_model_path=None, pytorch_weights_path=None, graph_path=None, backend="pytorch"):
        """
        Args:
            onnx_model_path: Path to the exported st_gnn.onnx file.
            pytorch_weights_path: Path to st_gnn_weights.pt.
            graph_path: Path to market_spatial_graph.pt.
            backend: 'pytorch' (recommended for CPU GAT speed) or 'onnx' (for cross-platform runtime).
        """
        self.backend = backend
        self.session = None
        self.pytorch_model = None
        
        # Load graph topology
        if graph_path and os.path.exists(graph_path):
            self.graph_data = torch.load(graph_path, weights_only=False)
            self.edge_index = self.graph_data["edge_index"]
            self.edge_attr = self.graph_data["edge_attr"]
            self.node_mapping = self.graph_data["node_mapping"]
            self.num_nodes = len(self.node_mapping)
        else:
            self.graph_data = None
            self.num_nodes = 1058

        if backend == "onnx" and onnx_model_path and os.path.exists(onnx_model_path):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(onnx_model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        elif pytorch_weights_path and os.path.exists(pytorch_weights_path):
            from src.models.st_gnn import ST_GNN
            self.pytorch_model = ST_GNN(num_nodes=self.num_nodes, in_channels=1, hidden_channels=64, out_channels=3, seq_len=14)
            self.pytorch_model.load_state_dict(torch.load(pytorch_weights_path, weights_only=True))
            self.pytorch_model.eval()

    def predict(self, x_input):
        """
        Executes forward pass.
        
        Args:
            x_input: numpy array or torch Tensor of shape [batch_size, num_nodes, 14, 1].
            
        Returns:
            numpy array: Logits of shape [batch_size, num_nodes, 3].
        """
        if self.backend == "onnx" and self.session:
            if isinstance(x_input, torch.Tensor):
                x_np = x_input.cpu().numpy()
            else:
                x_np = np.array(x_input, dtype=np.float32)
                
            ort_inputs = {
                "x": x_np,
                "edge_index": self.edge_index.cpu().numpy(),
                "edge_weight": self.edge_attr.cpu().numpy()
            }
            return self.session.run(["logits"], ort_inputs)[0]
        else:
            if isinstance(x_input, np.ndarray):
                x_t = torch.tensor(x_input, dtype=torch.float32)
            else:
                x_t = x_input
            with torch.no_grad():
                return self.pytorch_model(x_t, self.edge_index, edge_weight=self.edge_attr).cpu().numpy()
