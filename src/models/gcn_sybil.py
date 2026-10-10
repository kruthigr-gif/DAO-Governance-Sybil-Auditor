import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer (Kipf & Welling, ICLR 2017).
    Performs normalized graph convolution with self-loops.
    """
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        N = x.size(0)
        E = edge_index.size(1)

        # Linear projection
        h = self.linear(x)

        # Create adjacency matrix with self-loops
        device = x.device
        src, dst = edge_index[0], edge_index[1]
        
        # Calculate node degrees
        deg = torch.zeros(N, device=device)
        if edge_attr is None:
            weights = torch.ones(E, device=device)
        else:
            weights = edge_attr.squeeze(-1)

        deg.index_add_(0, dst, weights)
        deg = deg + 1.0 # Self loops

        # Degree normalization d^{-1/2}
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        # Normalized edge weights
        norm_weights = deg_inv_sqrt[src] * weights * deg_inv_sqrt[dst]

        # Aggregate neighbor features
        out = torch.zeros((N, h.size(1)), device=device)
        messages = h[src] * norm_weights.unsqueeze(-1)
        out.index_add_(0, dst, messages)

        # Self-loop contribution
        self_norm = deg_inv_sqrt * deg_inv_sqrt
        out = out + h * self_norm.unsqueeze(-1)

        return out + self.bias


class GCNSybilDetector(nn.Module):
    """
    Graph Convolutional Network (GCN) baseline for Sybil attack detection.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 32, num_classes: int = 2, dropout: float = 0.3):
        super(GCNSybilDetector, self).__init__()
        self.conv1 = GraphConvLayer(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GraphConvLayer(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index, edge_attr)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)

        logits = self.fc(h)
        return logits, h

    def get_sybil_scores(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, edge_index, edge_attr)
            probs = F.softmax(logits, dim=1)
            sybil_scores = probs[:, 1]
        return sybil_scores


if __name__ == "__main__":
    x = torch.randn(10, 9)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    model = GCNSybilDetector(in_channels=9, hidden_channels=16)
    logits, emb = model(x, edge_index)
    print("GCN Output Logits Shape:", logits.shape)
