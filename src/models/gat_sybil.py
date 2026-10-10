import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Layer with edge feature incorporation.
    Computes learnable self-attention coefficients over node neighborhood.
    """
    def __init__(self, in_features: int, out_features: int, heads: int = 4, concat: bool = True, dropout: float = 0.2, negative_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Linear projection weight matrix per head
        self.W = nn.Parameter(torch.Tensor(heads, in_features, out_features))
        # Attention parameter vector: [heads, 2 * out_features + 1] (+1 for edge attribute weight)
        self.attn = nn.Parameter(torch.Tensor(heads, 2 * out_features + 1, 1))

        self.leaky_relu = nn.LeakyReLU(self.negative_slope)
        self.dropout_layer = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        x: [N, in_features]
        edge_index: [2, E]
        edge_attr: [E, 1] or None
        """
        N = x.size(0)
        E = edge_index.size(1)

        # 1. Project node features across heads: [N, heads, out_features]
        # x_proj shape: [N, heads, out_features]
        x_proj = torch.einsum('ni,hio->nho', x, self.W)

        # 2. Gather source and target node projections for edges
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        h_src = x_proj[src_nodes] # [E, heads, out_features]
        h_dst = x_proj[dst_nodes] # [E, heads, out_features]

        # Handle edge attributes
        if edge_attr is None:
            edge_attr_expanded = torch.ones((E, self.heads, 1), device=x.device)
        else:
            edge_attr_expanded = edge_attr.repeat(1, self.heads).unsqueeze(-1) # [E, heads, 1]

        # Concatenate src, dst representations and edge features: [E, heads, 2 * out_features + 1]
        h_cat = torch.cat([h_src, h_dst, edge_attr_expanded], dim=-1)

        # 3. Calculate unnormalized attention scores e_ij: [E, heads]
        # attn vector: [heads, 2 * out_features + 1, 1]
        attn_scores = torch.einsum('ehd,hdo->eho', h_cat, self.attn).squeeze(-1) # [E, heads]
        attn_scores = self.leaky_relu(attn_scores)

        # 4. Softmax over neighborhood using scatter_softmax / manual index normalization
        # Compute softmax per dst_node for each head
        attn_exp = torch.exp(attn_scores - attn_scores.max(dim=0, keepdim=True)[0])
        
        # Sum exponentiated scores for each destination node
        denom = torch.zeros((N, self.heads), device=x.device)
        denom.index_add_(0, dst_nodes, attn_exp)
        
        # Softmax attention coefficients alpha: [E, heads]
        alpha = attn_exp / (denom[dst_nodes] + 1e-12)
        alpha = self.dropout_layer(alpha)

        # Save last attention weights for explainability inspection
        self.last_alpha = alpha.detach()

        # 5. Aggregate neighborhood representations: [N, heads, out_features]
        out = torch.zeros((N, self.heads, self.out_features), device=x.device)
        # Weighted messages from src_nodes to dst_nodes
        messages = h_src * alpha.unsqueeze(-1) # [E, heads, out_features]
        
        # Scatter add messages to target destination nodes
        out.index_add_(0, dst_nodes, messages)

        if self.concat:
            # Flatten multi-head output: [N, heads * out_features]
            return out.view(N, self.heads * self.out_features)
        else:
            # Average multi-head output: [N, out_features]
            return out.mean(dim=1)


class GATSybilDetector(nn.Module):
    """
    Graph Attention Network (GAT) for DAO Sybil Attack Detection.
    Outputs a continuous Sybil suspicion score for every voter node.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 32, num_classes: int = 2, heads: int = 4, dropout: float = 0.3):
        super(GATSybilDetector, self).__init__()
        
        self.gat1 = GraphAttentionLayer(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        
        self.gat2 = GraphAttentionLayer(hidden_channels * heads, hidden_channels, heads=2, concat=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)

        # Classification Head
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        # Layer 1
        h = self.gat1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.dropout(h)

        # Layer 2
        h = self.gat2(h, edge_index, edge_attr)
        h = self.bn2(h)
        h = F.elu(h)
        h = self.dropout(h)

        # MLP Output
        feat_emb = F.relu(self.fc1(h))
        logits = self.classifier(feat_emb)
        return logits, feat_emb

    def get_sybil_scores(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """Returns continuous Sybil probability scores in range [0, 1]."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, edge_index, edge_attr)
            probs = F.softmax(logits, dim=1)
            sybil_scores = probs[:, 1] # Class 1 = Sybil
        return sybil_scores


if __name__ == "__main__":
    x = torch.randn(10, 9)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_attr = torch.tensor([[0.8], [0.8], [0.9], [0.9]])
    
    model = GATSybilDetector(in_channels=9, hidden_channels=16)
    logits, emb = model(x, edge_index, edge_attr)
    scores = model.get_sybil_scores(x, edge_index, edge_attr)
    print("GAT Output Logits Shape:", logits.shape)
    print("Sample Sybil Scores:", scores[:4])
