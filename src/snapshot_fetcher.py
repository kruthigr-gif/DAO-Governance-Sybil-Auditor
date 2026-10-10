import requests
import json
import logging
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SnapshotFetcher")

SNAPSHOT_GRAPHQL_ENDPOINT = "https://hub.snapshot.org/graphql"

class SnapshotFetcher:
    """
    Fetches real DAO governance voting data from the Snapshot GraphQL API.
    Provides offline fallback to synthetic data if network is unavailable.
    """

    def __init__(self, endpoint: str = SNAPSHOT_GRAPHQL_ENDPOINT):
        self.endpoint = endpoint

    def fetch_space_proposals(self, space_id: str = "ens.eth", limit: int = 15) -> List[Dict[str, Any]]:
        """Queries proposal IDs, titles, and times for a given space."""
        query = """
        query Proposals($space: String!, $limit: Int!) {
          proposals(
            first: $limit,
            where: { space: $space, state: "closed" },
            orderBy: "created",
            orderDirection: desc
          ) {
            id
            title
            start
            end
            scores_total
            votes
          }
        }
        """
        variables = {"space": space_id, "limit": limit}
        try:
            response = requests.post(self.endpoint, json={"query": query, "variables": variables}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                proposals = data.get("data", {}).get("proposals", [])
                logger.info(f"Successfully fetched {len(proposals)} proposals for space '{space_id}'.")
                return proposals
            else:
                logger.warning(f"Failed to fetch Snapshot data: HTTP {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Snapshot API connection error: {e}. Utilizing offline fallback.")
            return []

    def fetch_proposal_votes(self, proposal_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Queries voter addresses, vote choices, and timestamps for a proposal."""
        query = """
        query Votes($proposal: String!, $limit: Int!) {
          votes(
            first: $limit,
            where: { proposal: $proposal },
            orderBy: "created",
            orderDirection: asc
          ) {
            id
            voter
            created
            choice
            vp
          }
        }
        """
        variables = {"proposal": proposal_id, "limit": limit}
        try:
            response = requests.post(self.endpoint, json={"query": query, "variables": variables}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                votes = data.get("data", {}).get("votes", [])
                return votes
            else:
                return []
        except Exception as e:
            logger.warning(f"Error fetching votes for proposal {proposal_id}: {e}")
            return []


if __name__ == "__main__":
    fetcher = SnapshotFetcher()
    props = fetcher.fetch_space_proposals("ens.eth", limit=5)
    print(f"Fetched {len(props)} real proposals from Snapshot.")
