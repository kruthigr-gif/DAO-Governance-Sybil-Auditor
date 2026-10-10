<div align="center">

# 🛡️ DAO Governance Sybil Auditor

### Graph-based analysis for identifying potentially coordinated DAO voting behavior

**Powered by Graph Attention Networks (GAT)**

[Overview](#-overview) · [Features](#-features) · [Screenshots](#-dashboard-screenshots) · [Setup](#-installation) · [Usage](#-usage)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Graph ML](https://img.shields.io/badge/Graph%20ML-GAT-7C3AED)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📌 Overview

**DAO Governance Sybil Auditor** is an educational governance-auditing project that uses graph machine learning to identify voter accounts whose voting behavior may be coordinated.

Instead of examining each wallet in isolation, the project models voters as a **co-voting network**. A Graph Attention Network (GAT) then produces a Sybil suspicion score for each voter, helping auditors explore patterns that may warrant further investigation.

The project includes an interactive web dashboard, a synthetic data generator, a model benchmark, and a command-line audit report generator.

> ⚠️ **Responsible use:** A high suspicion score is a signal for review—not proof that an account is malicious. Scores depend on the quality of the data, features, labels, and model. Review the underlying evidence before taking any governance action.

## ✨ Features

- **Graph-based voter analysis** — represent voters as nodes and co-voting relationships as edges.
- **GAT scoring model** — estimate a suspicion score for each voter.
- **Interactive dashboard** — inspect summary metrics, risk tiers, and voter-level details.
- **Risk filtering** — explore Low, Medium, High, and Critical risk groups.
- **Voter node inspector** — view account metrics and recent proposal votes.
- **Snapshot data integration** — attempt to retrieve proposal and vote data for the `ens.eth` space.
- **Synthetic-data fallback** — keep the dashboard usable when live data cannot be retrieved.
- **Model benchmarking** — compare GAT, GCN, and a heuristic baseline.
- **Exportable audit reports** — generate Markdown or JSON reports from the command line.

## 🖥️ Dashboard Screenshots

### 1. Dashboard summary

The overview displays the number of audited voters, flagged accounts, compromised voting power, and the GAT model's ROC-AUC metric.

![DAO Governance Sybil Auditor dashboard summary](Screenshots/Screenshot%202026-10-10%20140943.png)

### 2. Governance co-voting network

Each node represents a voter. The graph visualizes co-voting relationships, and the risk filters help focus the view on selected risk tiers.

![Interactive governance co-voting network](Screenshots/Screenshot%202026-10-10%20141032.png)

### 3. Voter node inspector

Inspect a selected wallet's suspicion score, token balance, account age, vote-choice entropy, average voting latency, and recent proposal votes.

![Voter node inspector](Screenshots/Screenshot%202026-10-10%20141108.png)

### 4. Risk distribution

The risk distribution panel summarizes the number of voters assigned to each risk tier.

![Voter risk distribution](Screenshots/Screenshot%202026-10-10%20141214.png)

### 5. Server startup

Example PowerShell output showing the FastAPI application starting and the dashboard API responding successfully.

![PowerShell server startup output](Screenshots/Screenshot%202026-10-10%20141154.png)

## 🧠 How It Works

1. **Load voting data.** The dashboard attempts to retrieve closed proposals and votes from the Snapshot GraphQL API for `ens.eth`. If live data is unavailable, it falls back to synthetic data. The benchmark and CLI audit use synthetic data.
2. **Build the graph.** Voters become nodes; edges connect voters whose choices and voting times are sufficiently similar across shared proposals.
3. **Engineer features.** The graph builder calculates participation rate, vote-choice entropy, voting-latency statistics, log token balance, account age, PageRank, clustering coefficient, and degree centrality.
4. **Train and score.** A GAT is trained using available labels and produces suspicion scores for voter nodes.
5. **Explore and report.** Review audit metrics and voter details in the dashboard, benchmark models, or export a CLI report.

### Risk score thresholds

| Risk tier | Suspicion score |
|---|---:|
| 🟢 Low | Below `0.30` |
| 🟡 Medium | `0.30` to below `0.60` |
| 🟠 High | `0.60` to below `0.85` |
| 🔴 Critical Sybil | `0.85` and above |

These tiers are review aids, not definitive labels of malicious behavior.

## 🧰 Tech Stack

| Technology | Role |
|---|---|
| Python | Application and machine-learning code |
| PyTorch | Graph model tensors and training |
| Graph Attention Network (GAT) | Main voter-scoring model |
| Graph Convolutional Network (GCN) | Benchmark comparison model |
| FastAPI + Uvicorn | API and local web server |
| NetworkX | Co-voting graph and topology features |
| Pandas + NumPy | Data processing and numerical features |
| scikit-learn | Evaluation metrics |
| Snapshot GraphQL API + Requests | Live governance-data retrieval |
| HTML, CSS, JavaScript + vis-network | Browser dashboard and interactive graph |

## ⚙️ Installation

### Requirements

- Python **3.11 or newer**
- Dependencies listed in `requirements.txt`
- Internet access for live Snapshot data; synthetic data is used if live data cannot be loaded

> **Note:** PyTorch installation can vary by operating system and hardware. If installation from `requirements.txt` fails, install the appropriate PyTorch build for your system first, then install the remaining requirements.

### 1. Clone the repository

```powershell
git clone https://github.com/kruthigr-gif/DAO-Governance-Sybil-Auditor.git
cd DAO-Governance-Sybil-Auditor
```

If you already have the project folder locally, open PowerShell in that folder instead.

### 2. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 🚀 Usage

### Run the dashboard

From the project root, start the server:

```powershell
python -m uvicorn src.app:app --reload --port 8000
```

When the server reports that application startup is complete, open:

**http://127.0.0.1:8000**

To stop the server, press `Ctrl+C` in the terminal.

Alternatively, use the included launcher:

```powershell
python .\run_dashboard.py
```

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/summary` | Audit totals, health index, risk breakdown, and model metrics |
| `GET /api/graph` | Voter network nodes and edges; supports the optional `min_risk` query parameter |
| `GET /api/node/{node_id}` | Details and recent votes for a voter node |

### Run the model benchmark

```powershell
python .\evaluate.py
```

This generates synthetic data and compares GAT, GCN, and a heuristic baseline using ROC-AUC, PR-AUC, precision, recall, and F1-score. Results can vary with the generated data and training run.

### Generate an audit report

Generate a Markdown report:

```powershell
python -m src.audit_cli --voters 500 --proposals 25 --format markdown
```

Generate a JSON report:

```powershell
python -m src.audit_cli --voters 500 --proposals 25 --format json
```

The report is written to the current working directory as `dao_sybil_audit_report.md` or `dao_sybil_audit_report.json`. To see all available options:

```powershell
python -m src.audit_cli --help
```

## 📁 Project Structure

```text
.
├── evaluate.py                 # GAT, GCN, and heuristic benchmark
├── run_dashboard.py            # Dashboard launcher
├── requirements.txt            # Python dependencies
├── README.md
├── Screenshots/
│   ├── Screenshot 2026-10-10 140943.png
│   ├── Screenshot 2026-10-10 141032.png
│   ├── Screenshot 2026-10-10 141108.png
│   ├── Screenshot 2026-10-10 141154.png
│   └── Screenshot 2026-10-10 141214.png
└── src/
    ├── app.py                  # FastAPI application and dashboard API
    ├── audit_cli.py            # Synthetic-data audit and report export
    ├── data_generator.py       # Synthetic DAO voting data
    ├── graph_builder.py        # Graph construction and voter features
    ├── snapshot_fetcher.py     # Snapshot GraphQL requests
    ├── trainer.py              # Model training
    ├── models/
    │   ├── gat_sybil.py        # Graph Attention Network
    │   └── gcn_sybil.py        # Graph Convolutional Network
    └── static/                 # Dashboard HTML, CSS, and JavaScript
```

## 🔬 Evaluation Notes

- The benchmark compares GAT, GCN, and a heuristic baseline on generated synthetic data.
- Metrics from synthetic data should not be interpreted as proof of real-world Sybil-detection performance.
- Live-data availability and the dashboard's current metrics depend on the data successfully loaded at runtime.
- A model score is best treated as one input to human review, not as an automatic enforcement decision.

## 🛣️ Possible Future Improvements

- Evaluate against carefully documented, independently labeled governance datasets.
- Add calibration and threshold analysis for suspicion scores.
- Provide more detail on why a wallet received a particular score.
- Add repeatable experiments across random seeds and report variability.
- Improve accessibility and graph exploration for large DAO communities.

## 📄 License

This project is described as using the **MIT License**. Include a `LICENSE` file in the repository if you intend to distribute it under that license.

---

<div align="center">

**Built for transparent, graph-based DAO governance research.**

</div>
