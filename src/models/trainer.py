"""
File: src/models/trainer.py
Role: ML Training Loop
Description: Implements the training and evaluation loop for the ST-GNN. 
Specifically focuses on calculating the Precision-Recall Area Under Curve (PR-AUC) 
for the 'Crash' class (Class 1) to aggressively minimize false negatives, avoiding 
misleading ROC curves in highly imbalanced datasets.
Dependencies: torch, scikit-learn, tqdm
"""

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc, classification_report
import numpy as np

def calculate_pr_auc(y_true, y_prob, target_class=1):
    """
    Computes PR-AUC for a specific class. 
    target_class=1 is 'Crash', which is the most economically catastrophic event.
    """
    # Binarize targets for the specific class
    y_true_binary = (y_true == target_class).astype(int)
    y_prob_class = y_prob[:, target_class]
    
    # If the class is not present in the batch, return 0.0
    if sum(y_true_binary) == 0:
        return 0.0
        
    precision, recall, _ = precision_recall_curve(y_true_binary, y_prob_class)
    return auc(recall, precision)

def train_epoch(model, dataloader, optimizer, criterion, edge_index, edge_attr, device):
    model.train()
    total_loss = 0.0
    
    for batch_X, batch_Y in dataloader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        # We pass the static graph topology alongside the dynamic temporal node features
        logits = model(batch_X, edge_index.to(device), edge_attr.to(device))
        
        # Reshape for Cross Entropy: [Batch * Nodes, Classes]
        logits = logits.view(-1, logits.size(-1))
        batch_Y = batch_Y.view(-1)
        
        loss = criterion(logits, batch_Y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, edge_index, edge_attr, device):
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_Y in dataloader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            logits = model(batch_X, edge_index.to(device), edge_attr.to(device))
            
            probs = torch.softmax(logits, dim=-1)
            
            all_targets.append(batch_Y.view(-1).cpu().numpy())
            all_probs.append(probs.view(-1, probs.size(-1)).cpu().numpy())
            
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    
    # Calculate PR-AUC specifically for the "Crash" class (Index 1)
    pr_auc_crash = calculate_pr_auc(y_true, y_prob, target_class=1)
    
    # Generate overall classification metrics
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(y_true, y_pred, target_names=["Stable", "Crash", "Spike"], zero_division=0)
    
    return pr_auc_crash, report
