import sys
import io

# Force UTF-8 output to prevent cp1252 UnicodeEncodeError on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import numpy as np
import os
import torch

from src.data_generator import DAOSybilDataGenerator
from src.graph_builder import DAOGraphBuilder
from src.models.gat_sybil import GATSybilDetector
from src.snapshot_fetcher import SnapshotFetcher
from src.trainer import GNNTrainer


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_vote_choice(choice):
    """Normalize Snapshot vote values to the DAO voting enum used by the graph builder."""
    if choice is None:
        return 0
    if isinstance(choice, (int, float)):
        value = int(choice)
        if value in (1, 2, 3):
            return value
        return 0
    if isinstance(choice, str):
        cleaned = choice.strip().lower()
        if cleaned in {"for", "yes", "1"}:
            return 1
        if cleaned in {"against", "no", "2"}:
            return 2
        if cleaned in {"abstain", "3"}:
            return 3
        try:
            numeric = int(float(cleaned))
            if numeric in (1, 2, 3):
                return numeric
        except ValueError:
            return 0
        return 0
    if isinstance(choice, dict):
        if not choice:
            return 0
        ranked = []
        for key, val in choice.items():
            try:
                ranked.append((float(val), int(key)))
            except (TypeError, ValueError):
                continue
        if not ranked:
            return 0
        return max(ranked, key=lambda item: item[0])[1]
    if isinstance(choice, (list, tuple)):
        for item in choice:
            if isinstance(item, dict):
                normalized = _normalize_vote_choice(item)
                if normalized:
                    return normalized
        return 0
    return 0


def _build_heuristic_ground_truth(df_votes: pd.DataFrame) -> dict:
    """Create pseudo-labels from real vote synchronization patterns when no labeled data exists."""
    labels = {}
    grouped = df_votes.groupby("voter_address")
    for voter, rows in grouped:
        if len(rows) < 3:
            continue
        dominant_choice_ratio = rows["choice"].value_counts(normalize=True).max()
        timestamps = rows["timestamp"].sort_values().to_numpy(dtype=float)
        if len(timestamps) > 1:
            deltas = np.diff(timestamps)
            near_sync_ratio = float(np.mean(deltas <= 180.0)) if len(deltas) else 0.0
        else:
            near_sync_ratio = 0.0

        suspicious = (
            dominant_choice_ratio >= 0.75 and rows["voting_power"].mean() <= 5000
        ) or near_sync_ratio >= 0.7 or (
            len(rows) >= 5 and dominant_choice_ratio >= 0.65 and rows["voting_power"].mean() <= 10000
        )

        if suspicious:
            labels[voter] = 1
        else:
            labels[voter] = 0
    return labels


def load_snapshot_dataset(space_id: str = "ens.eth", proposal_limit: int = 10):
    """Load real Snapshot governance data when available, else return None for synthetic fallback."""
    fetcher = SnapshotFetcher()
    proposals = fetcher.fetch_space_proposals(space_id=space_id, limit=max(5, proposal_limit))
    if not proposals:
        return None

    vote_rows = []
    voter_meta = {}

    for proposal in proposals:
        proposal_id = proposal.get("id")
        if not proposal_id:
            continue
        raw_votes = fetcher.fetch_proposal_votes(proposal_id, limit=1000)
        for vote in raw_votes:
            voter = vote.get("voter") or vote.get("voterAddress")
            if not voter:
                continue
            timestamp = int(vote.get("created") or proposal.get("start") or 0)
            choice = _normalize_vote_choice(vote.get("choice"))
            if choice == 0:
                continue
            voting_power = _to_float(vote.get("vp"), 0.0)

            vote_rows.append({
                "proposal_id": proposal_id,
                "voter_address": voter,
                "choice": int(choice),
                "voting_power": max(0.0, voting_power),
                "timestamp": timestamp,
            })

            current = voter_meta.setdefault(voter, {"token_balance": 0.0, "account_age_days": 1})
            current["token_balance"] = max(current.get("token_balance", 0.0), voting_power)
            current["account_age_days"] = max(current.get("account_age_days", 1), 1)

    if not vote_rows:
        return None

    df_votes = pd.DataFrame(vote_rows)
    df_proposals = pd.DataFrame([
        {
            "proposal_id": p.get("id"),
            "title": p.get("title", f"Snapshot Proposal {idx + 1}"),
            "start_time": int(p.get("start") or 0),
            "end_time": int(p.get("end") or 0),
            "sentiment_prob": np.array([0.6, 0.25, 0.15], dtype=np.float32)
        }
        for idx, p in enumerate(proposals)
        if p.get("id")
    ])

    if df_proposals.empty:
        return None

    for voter, meta in voter_meta.items():
        votes_for_voter = df_votes[df_votes["voter_address"] == voter]
        if votes_for_voter.empty:
            continue
        min_time = int(votes_for_voter["timestamp"].min())
        max_time = int(votes_for_voter["timestamp"].max())
        account_age_days = max(1, int((max_time - min_time) / 86400)) + 1
        meta["account_age_days"] = account_age_days

    df_voters = pd.DataFrame([
        {
            "address": voter,
            "is_sybil": 0,
            "cluster_id": 0,
            "token_balance": float(meta["token_balance"]),
            "account_age_days": int(meta["account_age_days"]),
            "participation_rate": 0.0,
            "archetype": "real_voter",
            "user_pref": np.array([0.5, 0.3, 0.2], dtype=np.float32)
        }
        for voter, meta in voter_meta.items()
    ])

    heuristic_labels = _build_heuristic_ground_truth(df_votes)
    df_voters["is_sybil"] = df_voters["address"].map(heuristic_labels).fillna(0).astype(int)
    ground_truth = df_voters.set_index("address")["is_sybil"].to_dict()

    return df_votes, df_voters, df_proposals, ground_truth


def _fallback_synthetic_dataset(num_legit: int = 425, num_proposals: int = 25):
    gen = DAOSybilDataGenerator(seed=42)
    return gen.generate_dao_dataset(num_legit=num_legit, num_proposals=num_proposals)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler — runs initialization on startup."""
    initialize_audit_data()
    yield

app = FastAPI(
    title="GNN DAO Sybil Detection Dashboard",
    description="Interactive Web3 Security & Audit Tool powered by Graph Attention Networks",
    lifespan=lifespan
)

# Global Cached Audit State
CACHE = {
    "graph_data": None,
    "df_voters": None,
    "df_votes": None,
    "df_proposals": None,
    "sybil_scores": None,
    "gat_model": None,
    "metrics": None
}

def initialize_audit_data(space_id: str = "ens.eth", proposal_limit: int = 12):
    """Load real Snapshot data when available, with a synthetic fallback for offline runs."""
    print("[*] Initializing DAO Sybil Audit Engine...")

    dataset = load_snapshot_dataset(space_id=space_id, proposal_limit=proposal_limit)
    if dataset is None:
        print("[*] No live Snapshot data available; using fallback synthetic DAO dataset.")
        df_votes, df_voters, df_proposals, ground_truth = _fallback_synthetic_dataset(num_legit=425, num_proposals=25)
    else:
        df_votes, df_voters, df_proposals, ground_truth = dataset
        print(f"[*] Live Snapshot data loaded for '{space_id}' with {len(df_voters)} voters across {len(df_proposals)} proposals.")

    builder = DAOGraphBuilder(co_vote_threshold=0.38)
    graph_data = builder.build_graph(df_votes, df_voters, df_proposals, ground_truth)

    x = graph_data["x"]
    edge_index = graph_data["edge_index"]
    edge_attr = graph_data["edge_attr"]

    gat_model = GATSybilDetector(in_channels=x.shape[1], hidden_channels=32, heads=4)
    trainer = GNNTrainer(gat_model, lr=0.008)
    audit_results = trainer.train_model(graph_data, epochs=75, patience=20)

    sybil_scores = gat_model.get_sybil_scores(x, edge_index, edge_attr).numpy()

    CACHE["graph_data"] = graph_data
    CACHE["df_voters"] = df_voters
    CACHE["df_votes"] = df_votes
    CACHE["df_proposals"] = df_proposals
    CACHE["sybil_scores"] = sybil_scores
    CACHE["gat_model"] = gat_model
    CACHE["metrics"] = audit_results["test_metrics"]
    print("[OK] Audit Engine Ready!")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Startup is now handled by the lifespan context manager above

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/api/summary")
def get_summary(space_id: str = Query("ens.eth", min_length=2)):
    """Returns high-level governance health metrics and risk breakdown."""
    if CACHE["graph_data"] is None:
        initialize_audit_data(space_id=space_id)
        
    df_voters = CACHE["df_voters"]
    scores = CACHE["sybil_scores"]
    metrics = CACHE["metrics"]

    total_voters = len(df_voters)
    total_vp = float(df_voters["token_balance"].sum())
    
    flagged_mask = scores >= 0.60
    flagged_count = int(np.sum(flagged_mask))
    
    voter_addrs = CACHE["graph_data"]["voter_addresses"]
    flagged_vp = float(df_voters[df_voters["address"].isin([voter_addrs[i] for i in range(len(voter_addrs)) if scores[i] >= 0.60])]["token_balance"].sum())
    
    risk_breakdown = {
        "Low": int(np.sum((scores < 0.30))),
        "Medium": int(np.sum((scores >= 0.30) & (scores < 0.60))),
        "High": int(np.sum((scores >= 0.60) & (scores < 0.85))),
        "Critical": int(np.sum((scores >= 0.85)))
    }
    
    # Calculate Governance Health Index (0-100%)
    health_index = round(max(0.0, 100.0 - (flagged_vp / total_vp) * 100.0), 1)

    return {
        "total_voters": total_voters,
        "total_voting_power": round(total_vp, 2),
        "flagged_sybils": flagged_count,
        "compromised_voting_power": round(flagged_vp, 2),
        "compromised_vp_percent": round((flagged_vp / total_vp) * 100, 2),
        "health_index": health_index,
        "risk_breakdown": risk_breakdown,
        "model_performance": metrics
    }

@app.get("/api/graph")
def get_graph_data(min_risk: float = Query(0.0, ge=0.0, le=1.0)):
    """Returns network nodes and edges filtered by minimum risk score for visualization."""
    graph_data = CACHE["graph_data"]
    df_voters = CACHE["df_voters"]
    scores = CACHE["sybil_scores"]
    voter_addrs = graph_data["voter_addresses"]

    nodes = []
    node_id_map = {}
    valid_node_ids = set()

    for idx, addr in enumerate(voter_addrs):
        score = float(scores[idx])
        if score >= min_risk:
            v_info = df_voters[df_voters["address"] == addr].iloc[0]
            
            if score < 0.30:
                risk_category = "Low"
                color = "#10b981" # Emerald Green
            elif score < 0.60:
                risk_category = "Medium"
                color = "#f59e0b" # Amber
            elif score < 0.85:
                risk_category = "High"
                color = "#f97316" # Coral Orange
            else:
                risk_category = "Critical"
                color = "#ef4444" # Crimson Red

            node_dict = {
                "id": idx,
                "label": f"{addr[:6]}...{addr[-4:]}",
                "address": addr,
                "sybil_score": round(score, 4),
                "risk_category": risk_category,
                "color": color,
                "voting_power": round(float(v_info["token_balance"]), 2),
                "account_age_days": int(v_info["account_age_days"]),
                "is_sybil": int(v_info["is_sybil"])
            }
            nodes.append(node_dict)
            valid_node_ids.add(idx)

    # Filter edges connecting valid nodes
    edge_index = graph_data["edge_index"]
    if edge_index.numel() == 0:
        edges = []
    else:
        edges_src = edge_index[0].detach().cpu().numpy()
        edges_dst = edge_index[1].detach().cpu().numpy()
        edge_attr = graph_data["edge_attr"].detach().cpu().numpy().reshape(-1)

        edges = []
        seen_pairs = set()

        for s, d, w in zip(edges_src, edges_dst, edge_attr):
            if s in valid_node_ids and d in valid_node_ids:
                pair = tuple(sorted([int(s), int(d)]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges.append({
                        "from": int(s),
                        "to": int(d),
                        "value": round(float(w), 3),
                        "title": f"Co-voting Similarity: {float(w):.2f}"
                    })

    return {"nodes": nodes, "edges": edges}

@app.get("/api/node/{node_id}")
def get_node_details(node_id: int):
    """Returns detailed voting history, risk factors, and neighbors for a specific voter."""
    graph_data = CACHE["graph_data"]
    df_voters = CACHE["df_voters"]
    df_votes = CACHE["df_votes"]
    scores = CACHE["sybil_scores"]
    voter_addrs = graph_data["voter_addresses"]

    if node_id < 0 or node_id >= len(voter_addrs):
        return JSONResponse(status_code=404, content={"message": "Node ID not found"})

    addr = voter_addrs[node_id]
    score = float(scores[node_id])
    voter_row = df_voters[df_voters["address"] == addr]
    if voter_row.empty:
        return JSONResponse(status_code=404, content={"message": "Node voter record missing"})
    v_info = voter_row.iloc[0]

    # Voter's votes
    v_votes = df_votes[df_votes["voter_address"] == addr].to_dict(orient="records")

    # Feature metrics
    raw_feats = graph_data["raw_features"][node_id]
    feat_names = graph_data["feature_names"]
    feat_dict = {name: round(float(val), 4) for name, val in zip(feat_names, raw_feats)}

    return {
        "node_id": node_id,
        "address": addr,
        "sybil_score": round(score, 4),
        "token_balance": round(float(v_info["token_balance"]), 2),
        "account_age_days": int(v_info["account_age_days"]),
        "features": feat_dict,
        "votes": v_votes[:10]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
