"""
File: scripts/build_graph.py
Role: ML Graph Construction
Description: Connects to the TimescaleDB/PostGIS instance to fetch unique market locations. 
Computes the pairwise geodesic distance matrix and applies a Gaussian kernel to construct 
the spatial Adjacency Matrix (W_ij). Creates a sparse PyTorch Geometric graph structure 
and saves it for ST-GNN training.
Dependencies: pandas, psycopg2, numpy, scikit-learn, torch
"""

import os
import psycopg2
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import haversine_distances
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.environ.get("DB_URL")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# Hyperparameters for the Spatial Graph
SIGMA_KM = 100.0  # Standard deviation for Gaussian kernel (determines spatial influence dropoff)
SPARSITY_THRESHOLD = 0.05  # Edges with weight below this are pruned for memory efficiency

def fetch_market_nodes():
    print("Fetching distinct market locations from database...")
    # We use AVG to ensure one coordinate pair per market center in case of minor data entry noise
    query = """
        SELECT 
            market_center_name, 
            AVG(latitude) as lat, 
            AVG(longitude) as lon
        FROM market_arrivals
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        GROUP BY market_center_name
        ORDER BY market_center_name;
    """
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def build_graph():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_nodes = fetch_market_nodes()
    num_nodes = len(df_nodes)
    print(f"Found {num_nodes} unique market nodes.")
    
    # 1. Compute Pairwise Geodesic Distances
    # sklearn haversine expects coordinates in radians (lat, lon)
    coords_rad = np.radians(df_nodes[['lat', 'lon']].values)
    dist_matrix_rad = haversine_distances(coords_rad, coords_rad)
    
    # Convert radians to kilometers (Earth radius ~6371 km)
    dist_matrix_km = dist_matrix_rad * 6371.0
    
    # 2. Apply Gaussian Kernel for Adjacency Weights W_ij = exp(- (d_ij^2) / sigma^2)
    print(f"Applying Gaussian Kernel (Sigma: {SIGMA_KM} km)...")
    W = np.exp(- (dist_matrix_km ** 2) / (SIGMA_KM ** 2))
    
    # 3. Sparsify the Graph to optimize for GATv2Conv
    print(f"Sparsifying graph (Threshold: {SPARSITY_THRESHOLD})...")
    # Keep edges where the adjacency weight is significant
    edge_indices = np.where(W > SPARSITY_THRESHOLD)
    
    src_nodes = edge_indices[0]
    dst_nodes = edge_indices[1]
    edge_weights = W[src_nodes, dst_nodes]
    
    # 4. Construct PyTorch Tensors
    edge_index = torch.tensor(np.vstack((src_nodes, dst_nodes)), dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
    
    print(f"Constructed sparse graph with {len(edge_weights)} edges out of {num_nodes**2} possible connections.")
    
    # 5. Save the Graph Data
    # Store the mapping from text name to integer ID so our ST-GNN knows which node is which
    node_mapping = {name: idx for idx, name in enumerate(df_nodes['market_center_name'])}
    
    graph_data = {
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'node_mapping': node_mapping,
        'dist_matrix_km': dist_matrix_km
    }
    
    out_file = os.path.join(OUTPUT_DIR, "market_spatial_graph.pt")
    torch.save(graph_data, out_file)
    print(f"Graph topology successfully saved to: {out_file}")

if __name__ == "__main__":
    build_graph()
