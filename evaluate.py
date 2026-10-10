import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

from src.data_generator import DAOSybilDataGenerator
from src.graph_builder import DAOGraphBuilder
from src.models.gat_sybil import GATSybilDetector
from src.models.gcn_sybil import GCNSybilDetector
from src.trainer import GNNTrainer

def heuristic_sybil_detector(graph_data) -> np.ndarray:
    """
    Baseline Heuristic Detector:
    Flags accounts with low vote choice entropy (< 0.2) AND high participation rate (> 0.8)
    AND short account age (< 60 days) as Sybils.
    """
    raw_feats = graph_data["raw_features"]
    participation = raw_feats[:, 0]
    entropy = raw_feats[:, 1]
    age_days = raw_feats[:, 5] * 365.0

    scores = np.zeros(len(participation))
    for i in range(len(scores)):
        risk = 0.0
        if entropy[i] < 0.25: risk += 0.4
        if participation[i] > 0.75: risk += 0.3
        if age_days[i] < 45: risk += 0.3
        scores[i] = risk
    return scores

def run_evaluation_benchmark():
    print("=" * 70)
    print("BENCHMARK EVALUATION: GAT vs GCN vs HEURISTIC BASELINE")
    print("=" * 70)

    # 1. Generate Synthetic Test Dataset
    print("[*] Generating benchmark dataset (500 voters, 3 Sybil attack topologies)...")
    gen = DAOSybilDataGenerator(seed=123)
    df_votes, df_voters, df_proposals, gt = gen.generate_dao_dataset(num_legit=425, num_proposals=30)
    
    builder = DAOGraphBuilder(co_vote_threshold=0.38)
    graph_data = builder.build_graph(df_votes, df_voters, df_proposals, gt)

    x = graph_data["x"]
    edge_index = graph_data["edge_index"]
    edge_attr = graph_data["edge_attr"]
    y = graph_data["y"]

    in_dim = x.shape[1]

    # Create masks
    trainer_tmp = GNNTrainer()
    train_mask, val_mask, test_mask = trainer_tmp.create_masks(x.shape[0])
    test_labels = y[test_mask].numpy()

    # 2. Evaluate Multi-Head GAT Model
    print("[*] Training Graph Attention Network (GAT)...")
    gat_model = GATSybilDetector(in_channels=in_dim, hidden_channels=32, heads=4)
    gat_trainer = GNNTrainer(gat_model, lr=0.01)
    gat_results = gat_trainer.train_model(graph_data, epochs=100, patience=20)
    gat_scores = gat_model.get_sybil_scores(x, edge_index, edge_attr)[test_mask].numpy()

    # 3. Evaluate GCN Model
    print("[*] Training Graph Convolutional Network (GCN)...")
    gcn_model = GCNSybilDetector(in_channels=in_dim, hidden_channels=32)
    gcn_trainer = GNNTrainer(gcn_model, lr=0.01)
    gcn_results = gcn_trainer.train_model(graph_data, epochs=100, patience=20)
    gcn_scores = gcn_model.get_sybil_scores(x, edge_index, edge_attr)[test_mask].numpy()

    # 4. Evaluate Heuristic Baseline
    print("[*] Computing Heuristic Baseline Scores...")
    heuristic_scores = heuristic_sybil_detector(graph_data)[test_mask]

    def compute_metrics(probs, labels):
        auc = roc_auc_score(labels, probs)
        pr_auc = average_precision_score(labels, probs)
        
        # Optimal F1 threshold search over probabilities
        best_f1, best_prec, best_rec = 0.0, 0.0, 0.0
        for th in np.linspace(0.1, 0.9, 81):
            preds = (probs >= th).astype(int)
            p, r, f, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
            if f > best_f1:
                best_f1, best_prec, best_rec = f, p, r
                
        return {
            "ROC-AUC": round(float(auc), 4),
            "PR-AUC": round(float(pr_auc), 4),
            "Precision": round(float(best_prec), 4),
            "Recall": round(float(best_rec), 4),
            "F1-Score": round(float(best_f1), 4)
        }

    gat_metrics = compute_metrics(gat_scores, test_labels)
    gcn_metrics = compute_metrics(gcn_scores, test_labels)
    heuristic_metrics = compute_metrics(heuristic_scores, test_labels)

    df_results = pd.DataFrame([
        {"Model": "Graph Attention Network (GAT)", **gat_metrics},
        {"Model": "Graph Convolutional Network (GCN)", **gcn_metrics},
        {"Model": "Heuristic Rule-Based Baseline", **heuristic_metrics}
    ])

    print("\n" + "=" * 70)
    print("FINAL COMPARATIVE BENCHMARK RESULTS")
    print("=" * 70)
    print(df_results.to_string(index=False))

    return df_results

if __name__ == "__main__":
    run_evaluation_benchmark()
