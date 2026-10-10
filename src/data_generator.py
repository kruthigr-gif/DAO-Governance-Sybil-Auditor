import numpy as np
import pandas as pd
import random
import time
import secrets
from typing import Dict, List, Tuple, Any

def generate_eth_address() -> str:
    """Generates a random Ethereum-style hex address."""
    return "0x" + secrets.token_hex(20)

class DAOSybilDataGenerator:
    """
    Generates synthetic DAO voting data with realistic legitimate user behaviors
    and sophisticated Sybil attack patterns (Synchronized Botnets, Star Delegations,
    and Stealth Jittered Rings).
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    def generate_dao_dataset(
        self,
        num_legit: int = 450,
        num_proposals: int = 30,
        sybil_clusters: List[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        """
        Generates full DAO voting records, voter metadata, funding transfers, and ground truth labels.
        
        Default sybil_clusters configuration creates:
        1. Synchronized Botnet (30 nodes, 100% vote alignment, 0.98 time concurrence)
        2. Star Delegation Ring (20 nodes, echoes master voter)
        3. Stealth Split-Voting Ring (25 nodes, 85% alignment with subtle time jitter)
        """
        if sybil_clusters is None:
            sybil_clusters = [
                {"type": "botnet", "size": 30, "cohesion": 0.98, "time_window_sec": 5},
                {"type": "star_delegation", "size": 20, "cohesion": 0.95, "time_window_sec": 12},
                {"type": "stealth_ring", "size": 25, "cohesion": 0.85, "time_window_sec": 45}
            ]

        # 1. Generate Proposal Metadata
        proposals = []
        base_timestamp = 1700000000 # Benchmark epoch timestamp
        for i in range(num_proposals):
            prop_id = f"QmProp_{i+1:03d}"
            duration_hours = random.randint(48, 168)
            prop_start = base_timestamp + i * 86400 + random.randint(0, 3600)
            prop_end = prop_start + duration_hours * 3600
            # Natural baseline sentiment for proposal (0.7 FOR, 0.2 AGAINST, 0.1 ABSTAIN)
            prop_sentiment = np.random.dirichlet([7, 2, 1])
            proposals.append({
                "proposal_id": prop_id,
                "title": f"DAO Proposal #{i+1}: Governance Optimization & Treasury Allocation",
                "start_time": prop_start,
                "end_time": prop_end,
                "sentiment_prob": prop_sentiment
            })
        df_proposals = pd.DataFrame(proposals)

        # 2. Generate Legitimate Voters
        voters = []
        ground_truth = {} # address -> 0 (legit) or 1 (sybil)
        sybil_cluster_ids = {} # address -> cluster_id

        # Archetypes for legitimate voters:
        # - Core Delegates (high participation, high tokens)
        # - Active Community (medium participation, medium tokens)
        # - Casual Holders (low participation, low tokens)
        legit_addresses = []
        for i in range(num_legit):
            addr = generate_eth_address()
            legit_addresses.append(addr)
            ground_truth[addr] = 0
            sybil_cluster_ids[addr] = 0 # 0 means legitimate user
            
            archetype = random.choices(["core", "active", "casual"], weights=[0.15, 0.50, 0.35])[0]
            if archetype == "core":
                participation_rate = np.random.uniform(0.75, 1.0)
                token_balance = np.random.exponential(50000) + 5000
                account_age_days = random.randint(300, 1000)
            elif archetype == "active":
                participation_rate = np.random.uniform(0.35, 0.75)
                token_balance = np.random.exponential(10000) + 500
                account_age_days = random.randint(100, 600)
            else:
                participation_rate = np.random.uniform(0.05, 0.35)
                token_balance = np.random.exponential(2000) + 50
                account_age_days = random.randint(30, 300)

            # Legitimate user voting preferences (personalized vector)
            user_pref = np.random.dirichlet([5, 3, 2])

            voters.append({
                "address": addr,
                "is_sybil": 0,
                "cluster_id": 0,
                "archetype": archetype,
                "token_balance": round(token_balance, 2),
                "account_age_days": account_age_days,
                "participation_rate": participation_rate,
                "user_pref": user_pref
            })

        # 3. Generate Sybil Clusters & Accounts
        transfers = [] # Funding transfer graph (funder -> sybil)
        sybil_addresses_by_cluster = {}

        for c_idx, cluster in enumerate(sybil_clusters, start=1):
            c_type = cluster["type"]
            c_size = cluster["size"]
            dispenser_addr = generate_eth_address() # Shared funding source
            
            cluster_members = []
            for j in range(c_size):
                s_addr = generate_eth_address()
                ground_truth[s_addr] = 1
                sybil_cluster_ids[s_addr] = c_idx
                cluster_members.append(s_addr)

                # Sybil account profiles (often created recently with uniform small tokens)
                token_balance = round(np.random.normal(500, 50), 2)
                account_age_days = random.randint(5, 45)
                
                voters.append({
                    "address": s_addr,
                    "is_sybil": 1,
                    "cluster_id": c_idx,
                    "archetype": f"sybil_{c_type}",
                    "token_balance": max(10.0, token_balance),
                    "account_age_days": account_age_days,
                    "participation_rate": np.random.uniform(0.85, 1.0), # Sybils vote on almost everything
                    "user_pref": None # Sybils follow cluster policy
                })

                # Record funding transfer from dispenser
                transfers.append({
                    "from_address": dispenser_addr,
                    "to_address": s_addr,
                    "amount_eth": round(random.uniform(0.01, 0.05), 4),
                    "timestamp": base_timestamp - random.randint(86400, 86400 * 10)
                })

            sybil_addresses_by_cluster[c_idx] = (c_type, cluster_members, cluster)

        df_voters = pd.DataFrame(voters)

        # 4. Generate Votes (Co-voting interactions)
        votes = []

        for _, prop in df_proposals.iterrows():
            p_id = prop["proposal_id"]
            p_start = prop["start_time"]
            p_end = prop["end_time"]
            p_duration = p_end - p_start
            p_sentiment = prop["sentiment_prob"]

            # Legitimate votes
            for _, v_row in df_voters[df_voters["is_sybil"] == 0].iterrows():
                if random.random() <= v_row["participation_rate"]:
                    # Legitimate vote choice based on proposal sentiment and user pref
                    combined_prob = 0.6 * p_sentiment + 0.4 * v_row["user_pref"]
                    combined_prob /= combined_prob.sum()
                    choice = np.random.choice([1, 2, 3], p=combined_prob) # 1: FOR, 2: AGAINST, 3: ABSTAIN

                    # Vote timestamp naturally spread across proposal window
                    vote_time = p_start + int(np.random.beta(1.5, 3.0) * p_duration)

                    votes.append({
                        "proposal_id": p_id,
                        "voter_address": v_row["address"],
                        "choice": int(choice),
                        "voting_power": v_row["token_balance"],
                        "timestamp": vote_time
                    })

            # Sybil cluster votes (Coordinated behavior)
            for c_idx, (c_type, members, c_config) in sybil_addresses_by_cluster.items():
                # Attacker decides target choice for this proposal
                cluster_target_choice = np.random.choice([1, 2, 3], p=[0.75, 0.20, 0.05])
                
                # Base cluster trigger time
                cluster_base_time = p_start + random.randint(300, int(p_duration * 0.4))
                cohesion = c_config["cohesion"]
                time_window = c_config["time_window_sec"]

                for m_addr in members:
                    m_voter = df_voters[df_voters["address"] == m_addr].iloc[0]
                    # Check if this member votes on this proposal
                    if random.random() <= m_voter["participation_rate"]:
                        # Choice determination
                        if random.random() <= cohesion:
                            m_choice = cluster_target_choice
                        else:
                            m_choice = random.choice([1, 2, 3]) # Evasion noise

                        # Timestamp determination (synchronized cluster window)
                        m_time = cluster_base_time + int(np.random.normal(0, time_window / 2))
                        m_time = max(p_start, min(p_end, m_time))

                        votes.append({
                            "proposal_id": p_id,
                            "voter_address": m_addr,
                            "choice": int(m_choice),
                            "voting_power": m_voter["token_balance"],
                            "timestamp": m_time
                        })

        df_votes = pd.DataFrame(votes)
        df_transfers = pd.DataFrame(transfers)

        return df_votes, df_voters, df_proposals, ground_truth


if __name__ == "__main__":
    generator = DAOSybilDataGenerator(seed=42)
    df_votes, df_voters, df_proposals, ground_truth = generator.generate_dao_dataset()
    print(f"Generated {len(df_voters)} voters ({sum(ground_truth.values())} Sybils).")
    print(f"Generated {len(df_votes)} total votes across {len(df_proposals)} proposals.")
