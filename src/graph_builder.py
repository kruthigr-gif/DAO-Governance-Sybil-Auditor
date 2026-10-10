import numpy as np
import pandas as pd
import networkx as nx
import torch
from typing import Dict, List, Tuple, Any

class DAOGraphBuilder:
    """
    Constructs a governance graph from DAO voting records.
    Computes topological and behavioral node features, edge co-voting similarities,
    and returns PyTorch-compatible graph structures.
    """

    def __init__(self, co_vote_threshold: float = 0.40, temporal_window_sec: float = 60.0):
        self.co_vote_threshold = co_vote_threshold
        self.temporal_window_sec = temporal_window_sec

    def build_graph(
        self,
        df_votes: pd.DataFrame,
        df_voters: pd.DataFrame,
        df_proposals: pd.DataFrame,
        ground_truth: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """
        Processes vote records into a PyTorch-compatible Graph Data Object dictionary.
        """
        # Unique voters and proposals
        voter_addresses = sorted(df_voters["address"].unique().tolist())
        num_voters = len(voter_addresses)
        address_to_idx = {addr: i for i, addr in enumerate(voter_addresses)}
        idx_to_address = {i: addr for i, addr in enumerate(voter_addresses)}

        proposal_ids = df_proposals["proposal_id"].tolist()
        num_proposals = len(proposal_ids)
        proposal_to_idx = {p_id: i for i, p_id in enumerate(proposal_ids)}
        proposal_starts = dict(zip(df_proposals["proposal_id"], df_proposals["start_time"]))

        # 1. Build Voting Choice Matrix (N x P) and Latency Matrix (N x P)
        # Choice matrix: 0 = Did not vote, 1 = FOR, 2 = AGAINST, 3 = ABSTAIN
        choice_matrix = np.zeros((num_voters, num_proposals), dtype=int)
        latency_matrix = np.zeros((num_voters, num_proposals), dtype=float)

        for _, row in df_votes.iterrows():
            v_idx = address_to_idx[row["voter_address"]]
            p_idx = proposal_to_idx[row["proposal_id"]]
            choice_matrix[v_idx, p_idx] = int(row["choice"])
            
            p_start = proposal_starts.get(row["proposal_id"], row["timestamp"])
            latency = max(0.0, float(row["timestamp"] - p_start))
            latency_matrix[v_idx, p_idx] = latency

        # 2. Compute Raw Node Features
        node_features = []
        for i, addr in enumerate(voter_addresses):
            v_info = df_voters[df_voters["address"] == addr].iloc[0]
            v_choices = choice_matrix[i]
            v_latencies = latency_matrix[i]

            # Feature 1: Participation Rate
            participated_mask = (v_choices > 0)
            num_votes_cast = np.sum(participated_mask)
            participation_rate = num_votes_cast / num_proposals if num_proposals > 0 else 0.0

            # Feature 2: Choice Entropy
            if num_votes_cast > 0:
                counts = np.bincount(v_choices[participated_mask], minlength=4)[1:]
                probs = counts / counts.sum()
                probs = probs[probs > 0]
                choice_entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
            else:
                choice_entropy = 0.0

            # Feature 3 & 4: Voting Latency Statistics
            if num_votes_cast > 0:
                active_latencies = v_latencies[participated_mask]
                mean_latency_hr = np.mean(active_latencies) / 3600.0
                std_latency_hr = np.std(active_latencies) / 3600.0
            else:
                mean_latency_hr = 48.0
                std_latency_hr = 0.0

            # Feature 5: Token Balance (log scale)
            log_balance = np.log10(max(1.0, float(v_info["token_balance"])))

            # Feature 6: Account Age (normalized days)
            norm_account_age = float(v_info["account_age_days"]) / 365.0

            node_features.append([
                participation_rate,
                choice_entropy,
                mean_latency_hr,
                std_latency_hr,
                log_balance,
                norm_account_age
            ])

        node_features_arr = np.array(node_features, dtype=np.float32)

        # Normalize feature columns (Z-score normalization)
        mean_feat = np.mean(node_features_arr, axis=0)
        std_feat = np.std(node_features_arr, axis=0) + 1e-6
        norm_node_features = (node_features_arr - mean_feat) / std_feat

        # 3. Construct Co-voting Graph Edges
        # Calculate pairwise voting agreement and synchronization
        nx_graph = nx.Graph()
        for i in range(num_voters):
            nx_graph.add_node(i)

        edges_src = []
        edges_dst = []
        edge_weights = []

        # Vectorized pairwise overlap check
        for i in range(num_voters):
            for j in range(i + 1, num_voters):
                c_i = choice_matrix[i]
                c_j = choice_matrix[j]
                
                # Shared proposals voted on
                shared_mask = (c_i > 0) & (c_j > 0)
                shared_count = np.sum(shared_mask)
                
                if shared_count < 3: # Need at least 3 shared proposals
                    continue

                # Identical choice agreement count
                agreed_count = np.sum((c_i[shared_mask] == c_j[shared_mask]))
                agreement_ratio = agreed_count / shared_count

                # Temporal synchronization (time delta between votes)
                time_deltas = np.abs(latency_matrix[i][shared_mask] - latency_matrix[j][shared_mask])
                sync_ratio = np.mean(time_deltas <= self.temporal_window_sec)

                # Composite Co-voting edge weight
                edge_w = 0.7 * agreement_ratio + 0.3 * sync_ratio

                if edge_w >= self.co_vote_threshold:
                    # Undirected graph (add both directions for PyTorch COO format)
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])
                    edge_weights.extend([edge_w, edge_w])
                    nx_graph.add_edge(i, j, weight=edge_w)

        # 4. Topological Graph Features (PageRank, Clustering Coefficient, Degree)
        pagerank_dict = nx.pagerank(nx_graph, weight="weight") if nx_graph.number_of_edges() > 0 else {i: 0.0 for i in range(num_voters)}
        clustering_dict = nx.clustering(nx_graph, weight="weight") if nx_graph.number_of_edges() > 0 else {i: 0.0 for i in range(num_voters)}
        degree_dict = nx.degree_centrality(nx_graph)

        topo_features = []
        for i in range(num_voters):
            topo_features.append([
                pagerank_dict.get(i, 0.0) * 100.0,
                clustering_dict.get(i, 0.0),
                degree_dict.get(i, 0.0)
            ])
        topo_features_arr = np.array(topo_features, dtype=np.float32)

        # Concatenate topological features to node feature tensor
        combined_features = np.hstack([norm_node_features, topo_features_arr])

        # 5. Build PyTorch Tensors
        x_tensor = torch.tensor(combined_features, dtype=torch.float32)
        edge_index_tensor = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

        # Ground truth labels tensor
        if ground_truth:
            labels = [ground_truth.get(addr, 0) for addr in voter_addresses]
            y_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            y_tensor = torch.zeros(num_voters, dtype=torch.long)

        return {
            "x": x_tensor,
            "edge_index": edge_index_tensor,
            "edge_attr": edge_attr_tensor,
            "y": y_tensor,
            "nx_graph": nx_graph,
            "address_to_idx": address_to_idx,
            "idx_to_address": idx_to_address,
            "voter_addresses": voter_addresses,
            "raw_features": node_features_arr,
            "feature_names": [
                "participation_rate", "choice_entropy", "mean_latency_hr",
                "std_latency_hr", "log_balance", "norm_account_age",
                "pagerank", "clustering_coeff", "degree_centrality"
            ]
        }


if __name__ == "__main__":
    from data_generator import DAOSybilDataGenerator
    gen = DAOSybilDataGenerator(42)
    df_votes, df_voters, df_proposals, ground_truth = gen.generate_dao_dataset()
    builder = DAOGraphBuilder(co_vote_threshold=0.40)
    graph_data = builder.build_graph(df_votes, df_voters, df_proposals, ground_truth)
    print("Graph Construction Complete:")
    print("Nodes:", graph_data["x"].shape)
    print("Edges:", graph_data["edge_index"].shape)
    print("Sybil Nodes Count:", (graph_data["y"] == 1).sum().item())
