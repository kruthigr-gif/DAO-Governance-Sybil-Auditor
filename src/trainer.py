import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GNNTrainer")

class GNNTrainer:
    """
    Manages model training, validation splits, metrics evaluation,
    and Sybil risk score generation.
    """
    def __init__(self, model: nn.Module = None, lr: float = 0.01, weight_decay: float = 1e-4):
        self.model = model
        if model is not None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = None
        # Class weighting to handle imbalanced Sybil nodes (Sybil class weight = 3.5)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.5]))

    def create_masks(self, num_nodes: int, train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Creates reproducible boolean train, validation, and test node masks."""
        np.random.seed(seed)
        indices = np.random.permutation(num_nodes)
        
        train_end = int(num_nodes * train_ratio)
        val_end = int(num_nodes * (train_ratio + val_ratio))

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True

        return train_mask, val_mask, test_mask

    def train_model(
        self,
        graph_data: Dict[str, Any],
        epochs: int = 120,
        patience: int = 30
    ) -> Dict[str, Any]:
        """Trains the GNN model and returns historical epoch metrics and best performance."""
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and Optimizer must be initialized before calling train_model.")

        x = graph_data["x"]
        edge_index = graph_data["edge_index"]
        edge_attr = graph_data["edge_attr"]
        y = graph_data["y"]
        num_nodes = x.size(0)

        train_mask, val_mask, test_mask = self.create_masks(num_nodes)

        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0

        history = {"train_loss": [], "val_auc": [], "val_f1": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            logits, _ = self.model(x, edge_index, edge_attr)
            loss = self.criterion(logits[train_mask], y[train_mask])
            loss.backward()
            self.optimizer.step()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_logits, _ = self.model(x, edge_index, edge_attr)
                val_probs = F.softmax(val_logits[val_mask], dim=1)[:, 1].numpy()
                val_labels = y[val_mask].numpy()

                val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
                # Optimal threshold search to avoid F1=0 with skewed score distributions
                best_f1_val = 0.0
                for th in np.linspace(0.05, 0.95, 37):
                    _preds = (val_probs >= th).astype(int)
                    _, _, _f1, _ = precision_recall_fscore_support(val_labels, _preds, average='binary', zero_division=0)
                    if _f1 > best_f1_val:
                        best_f1_val = _f1
                val_f1 = best_f1_val

            history["train_loss"].append(float(loss.item()))
            history["val_auc"].append(float(val_auc))
            history["val_f1"].append(float(val_f1))

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        test_metrics = self.evaluate(graph_data, test_mask)
        return {
            "history": history,
            "best_val_auc": best_val_auc,
            "test_metrics": test_metrics
        }

    def evaluate(self, graph_data: Dict[str, Any], mask: torch.Tensor = None) -> Dict[str, float]:
        """Evaluates model performance metrics on given node mask (or all nodes if mask is None)."""
        if self.model is None:
            raise ValueError("Model must be initialized before calling evaluate.")

        x = graph_data["x"]
        edge_index = graph_data["edge_index"]
        edge_attr = graph_data["edge_attr"]
        y = graph_data["y"]

        if mask is None:
            mask = torch.ones(x.size(0), dtype=torch.bool)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(x, edge_index, edge_attr)
            probs = F.softmax(logits[mask], dim=1)[:, 1].numpy()
            labels = y[mask].numpy()

            auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
            pr_auc = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
            # Optimal threshold search across [0.05, 0.95] to avoid F1=0 on skewed distributions
            best_f1, best_prec, best_rec = 0.0, 0.0, 0.0
            for th in np.linspace(0.05, 0.95, 91):
                _preds = (probs >= th).astype(int)
                _p, _r, _f, _ = precision_recall_fscore_support(labels, _preds, average='binary', zero_division=0)
                if _f > best_f1:
                    best_f1, best_prec, best_rec = _f, _p, _r
            prec, rec, f1 = best_prec, best_rec, best_f1

        return {
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4)
        }
