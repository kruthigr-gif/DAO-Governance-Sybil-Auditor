let network = null;
let currentGraphData = { nodes: [], edges: [] };
let selectedNodeId = null;

// Initialize Web Dashboard on load
document.addEventListener("DOMContentLoaded", () => {
    loadAuditSummary();
    loadGraphData(0.0);
});

// 1. Fetch & Render Summary KPI Metrics
async function loadAuditSummary() {
    try {
        const response = await fetch("/api/summary");
        if (!response.ok) return;
        const data = await response.json();

        // Update KPIs
        document.getElementById("health-value").innerText = `${data.health_index}%`;
        document.getElementById("kpi-voters").innerText = data.total_voters;
        document.getElementById("kpi-vp").innerText = `${data.total_voting_power.toLocaleString()} Tokens`;
        
        document.getElementById("kpi-sybils").innerText = data.flagged_sybils;
        document.getElementById("kpi-compromised-vp").innerText = `${data.compromised_voting_power.toLocaleString()} Tokens Compromised`;
        
        document.getElementById("kpi-vp-percent").innerText = `${data.compromised_vp_percent}%`;
        document.getElementById("kpi-roc-auc").innerText = data.model_performance.roc_auc;

        // Update Risk Distribution Bars
        const total = data.total_voters;
        const rb = data.risk_breakdown;

        updateRiskBar("low", rb.Low, total);
        updateRiskBar("medium", rb.Medium, total);
        updateRiskBar("high", rb.High, total);
        updateRiskBar("critical", rb.Critical, total);
    } catch (err) {
        console.error("Failed to load audit summary:", err);
    }
}

function updateRiskBar(tier, count, total) {
    document.getElementById(`cnt-${tier}`).innerText = count;
    const pct = total > 0 ? (count / total) * 100 : 0;
    document.getElementById(`fill-${tier}`).style.width = `${pct}%`;
}

// 2. Fetch & Render Vis.js Network Graph
async function loadGraphData(minRisk = 0.0) {
    try {
        const response = await fetch(`/api/graph?min_risk=${minRisk}`);
        if (!response.ok) return;
        currentGraphData = await response.json();

        renderNetworkGraph(currentGraphData);
    } catch (err) {
        console.error("Failed to load graph data:", err);
    }
}

function renderNetworkGraph(graphData) {
    const container = document.getElementById("network-canvas");

    // Format Vis.js nodes
    const visNodes = graphData.nodes.map(n => ({
        id: n.id,
        label: n.label,
        color: {
            background: n.color,
            border: "#ffffff",
            highlight: { background: n.color, border: "#3b82f6" }
        },
        size: Math.max(12, Math.min(30, Math.log10(n.voting_power + 1) * 6)),
        font: { color: "#ffffff", face: "Inter", size: 12 },
        shape: n.risk_category === "Critical" ? "square" : "dot",
        borderWidth: n.risk_category === "Critical" ? 2 : 1
    }));

    // Format Vis.js edges
    const visEdges = graphData.edges.map(e => ({
        from: e.from,
        to: e.to,
        value: e.value,
        color: { color: "rgba(255, 255, 255, 0.15)", highlight: "#3b82f6" },
        smooth: { type: "continuous" }
    }));

    const data = { nodes: visNodes, edges: visEdges };

    const options = {
        nodes: {
            shadow: true
        },
        edges: {
            width: 1,
            selectionWidth: 3
        },
        physics: {
            solver: "forceAtlas2Based",
            forceAtlas2Based: {
                gravitationalConstant: -35,
                centralGravity: 0.015,
                springLength: 80,
                springConstant: 0.08
            },
            maxVelocity: 50,
            timestep: 0.4,
            stabilization: { iterations: 100 }
        },
        interaction: {
            hover: true,
            zoomView: true,
            dragView: true
        }
    };

    if (network) {
        network.destroy();
    }

    network = new vis.Network(container, data, options);

    // Event listener: Click on node to inspect
    network.on("selectNode", (params) => {
        if (params.nodes.length > 0) {
            const nodeIdx = params.nodes[0];
            inspectNode(nodeIdx);
        }
    });

    network.on("deselectNode", () => {
        clearInspector();
    });
}

// 3. Risk Filter Handler
function filterRisk(minRisk, btn) {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    loadGraphData(minRisk);
}

// 4. Node Details Inspection
async function inspectNode(nodeId) {
    selectedNodeId = nodeId;
    try {
        const response = await fetch(`/api/node/${nodeId}`);
        if (!response.ok) return;
        const details = await response.json();

        // UI Drawer elements
        document.getElementById("inspect-placeholder").classList.add("hidden");
        document.getElementById("inspect-details").classList.remove("hidden");

        document.getElementById("inspect-node-id").innerText = `Node #${details.node_id}`;
        document.getElementById("inspect-address").innerText = details.address;
        
        const score = details.sybil_score;
        const scorePill = document.getElementById("inspect-score-pill");
        const scoreNum = document.getElementById("inspect-score");
        scoreNum.innerText = score.toFixed(4);

        if (score < 0.30) {
            scorePill.style.borderColor = "var(--risk-low)";
            scorePill.style.background = "rgba(16, 185, 129, 0.12)";
            scoreNum.style.color = "var(--risk-low)";
        } else if (score < 0.60) {
            scorePill.style.borderColor = "var(--risk-medium)";
            scorePill.style.background = "rgba(245, 158, 11, 0.12)";
            scoreNum.style.color = "var(--risk-medium)";
        } else if (score < 0.85) {
            scorePill.style.borderColor = "var(--risk-high)";
            scorePill.style.background = "rgba(249, 115, 22, 0.12)";
            scoreNum.style.color = "var(--risk-high)";
        } else {
            scorePill.style.borderColor = "var(--risk-critical)";
            scorePill.style.background = "rgba(239, 68, 68, 0.12)";
            scoreNum.style.color = "var(--risk-critical)";
        }

        document.getElementById("inspect-vp").innerText = details.token_balance.toLocaleString();
        document.getElementById("inspect-age").innerText = `${details.account_age_days} days`;
        document.getElementById("inspect-entropy").innerText = details.features.choice_entropy.toFixed(2);
        document.getElementById("inspect-latency").innerText = `${details.features.mean_latency_hr.toFixed(1)} hrs`;

        // Render Vote List
        const votesContainer = document.getElementById("inspect-votes-list");
        votesContainer.innerHTML = "";
        
        if (details.votes && details.votes.length > 0) {
            details.votes.forEach(v => {
                const choiceMap = { 1: "FOR", 2: "AGAINST", 3: "ABSTAIN" };
                const choiceStr = choiceMap[v.choice] || "VOTED";
                const div = document.createElement("div");
                div.className = "vote-item";
                div.innerHTML = `
                    <span class="mono">${v.proposal_id}</span>
                    <span class="choice-${v.choice} font-bold">${choiceStr}</span>
                `;
                votesContainer.appendChild(div);
            });
        } else {
            votesContainer.innerHTML = "<p class='text-dim text-sm'>No recent votes recorded.</p>";
        }

    } catch (err) {
        console.error("Failed to inspect node:", err);
    }
}

function clearInspector() {
    document.getElementById("inspect-placeholder").classList.remove("hidden");
    document.getElementById("inspect-details").classList.add("hidden");
    document.getElementById("inspect-node-id").innerText = "Click a node on graph";
}
