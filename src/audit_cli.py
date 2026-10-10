import sys
import os

# Set UTF-8 stdout encoding for Windows console compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import json
import pandas as pd
import numpy as np
import torch

from src.data_generator import DAOSybilDataGenerator
from src.graph_builder import DAOGraphBuilder
from src.models.gat_sybil import GATSybilDetector
from src.trainer import GNNTrainer

def run_security_audit(
    num_voters: int = 500,
    num_proposals: int = 25,
    co_vote_threshold: float = 0.40,
    epochs: int = 80,
    output_format: str = "markdown"
) -> str:
    """Executes a full DAO governance Sybil security audit."""
    print("=" * 70)
    print("DAO GOVERNANCE HEALTH & SYBIL SECURITY AUDITOR")
    print("=" * 70)

    # 1. Generate / Load Data
    print(f"[*] Simulating DAO governance graph ({num_voters} voters, {num_proposals} proposals)...")
    gen = DAOSybilDataGenerator(seed=42)
    df_votes, df_voters, df_proposals, ground_truth = gen.generate_dao_dataset(
        num_legit=num_voters - 75,
        num_proposals=num_proposals
    )

    # 2. Build Graph
    print("[*] Constructing co-voting interaction graph & extracting node features...")
    builder = DAOGraphBuilder(co_vote_threshold=co_vote_threshold)
    graph_data = builder.build_graph(df_votes, df_voters, df_proposals, ground_truth)

    x = graph_data["x"]
    edge_index = graph_data["edge_index"]
    voter_addresses = graph_data["voter_addresses"]
    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]

    print(f"    Graph metrics: {num_nodes} Nodes, {num_edges} Edges")

    # 3. Model Training & Sybil Scoring
    print("[*] Training Multi-Head Graph Attention Network (GAT)...")
    gat_model = GATSybilDetector(in_channels=x.shape[1], hidden_channels=32, heads=4)
    trainer = GNNTrainer(gat_model, lr=0.008)
    audit_results = trainer.train_model(graph_data, epochs=epochs, patience=20)
    metrics = audit_results["test_metrics"]

    print(f"    GAT Model Accuracy - ROC-AUC: {metrics['roc_auc']}, F1-Score: {metrics['f1_score']}")

    # 4. Generate Suspicion Scores for All Nodes
    sybil_scores = gat_model.get_sybil_scores(x, edge_index, graph_data["edge_attr"]).numpy()

    # 5. Risk Categorization & Summary Stats
    total_voting_power = df_voters["token_balance"].sum()
    flagged_sybils = []
    compromised_vp = 0.0

    risk_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}

    voter_audit_table = []
    for idx, addr in enumerate(voter_addresses):
        score = float(sybil_scores[idx])
        v_info = df_voters[df_voters["address"] == addr].iloc[0]
        vp = float(v_info["token_balance"])

        if score < 0.30:
            risk_tier = "Low"
        elif score < 0.60:
            risk_tier = "Medium"
        elif score < 0.85:
            risk_tier = "High"
        else:
            risk_tier = "Critical"

        risk_counts[risk_tier] += 1

        if score >= 0.60:
            flagged_sybils.append(addr)
            compromised_vp += vp

        voter_audit_table.append({
            "address": addr,
            "sybil_score": round(score, 4),
            "risk_tier": risk_tier,
            "voting_power": round(vp, 2),
            "account_age_days": int(v_info["account_age_days"])
        })

    flagged_percent_vp = round((compromised_vp / total_voting_power) * 100, 2)
    voter_audit_table.sort(key=lambda item: item["sybil_score"], reverse=True)

    report_content = f"""# DAO Governance Health & Sybil Audit Report

## 1. Executive Summary
- **Total Voters Audited**: {num_nodes}
- **Co-voting Graph Connections**: {num_edges} edges
- **Total Voting Power Analyzed**: {total_voting_power:,.2f} tokens
- **Flagged Sybil Accounts (High / Critical Risk)**: {len(flagged_sybils)} accounts
- **Governance Voting Power Compromised**: {compromised_vp:,.2f} tokens ({flagged_percent_vp}% of total DAO voting power)
- **Model Detection Performance (ROC-AUC)**: {metrics['roc_auc']}

## 2. Sybil Risk Breakdown
- Low Risk (< 0.30): {risk_counts['Low']} voters
- Medium Risk (0.30 - 0.60): {risk_counts['Medium']} voters
- High Risk (0.60 - 0.85): {risk_counts['High']} voters
- Critical Sybil (> 0.85): {risk_counts['Critical']} voters

## 3. Top Suspicious Accounts Requiring Disqualification
| Rank | Voter Address | Sybil Suspicion Score | Risk Level | Voting Power | Account Age |
|------|---------------|----------------------|------------|--------------|-------------|
"""

    for rank, item in enumerate(voter_audit_table[:15], start=1):
        report_content += f"| {rank} | `{item['address']}` | **{item['sybil_score']:.4f}** | {item['risk_tier']} | {item['voting_power']:,.2f} | {item['account_age_days']} days |\n"

    report_content += """
## 4. Recommendations for DAO Governance Council
1. **Disqualify Flagged Critical Accounts**: Immediately remove voting power from top flagged clusters before active proposal settlement.
2. **Implement Proof-of-Humanity (PoH) Verification**: Require Gitcoin Passport or WorldID verification for accounts with high voting frequency but zero choice entropy.
3. **Quadratic Voting Implementation**: Implement quadratic voting to diminish the governance impact of synchronized low-balance bot networks.
"""

    if output_format.lower() == "markdown":
        output_file = "dao_sybil_audit_report.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)
    else:
        output_file = "dao_sybil_audit_report.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_voters": num_nodes,
                    "total_edges": num_edges,
                    "flagged_sybils": len(flagged_sybils),
                    "compromised_voting_power": compromised_vp,
                    "compromised_vp_percentage": flagged_percent_vp,
                    "model_roc_auc": metrics['roc_auc']
                },
                "risk_counts": risk_counts,
                "audited_voters": voter_audit_table
            }, f, indent=2)

    print(f"\n[+] Audit Report successfully exported to '{output_file}'!")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAO Sybil Security Auditor CLI")
    parser.add_argument("--voters", type=int, default=500, help="Number of simulated voters")
    parser.add_argument("--proposals", type=int, default=25, help="Number of simulated proposals")
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "json"], help="Output report format")
    args = parser.parse_args()

    run_security_audit(num_voters=args.voters, num_proposals=args.proposals, output_format=args.format)
