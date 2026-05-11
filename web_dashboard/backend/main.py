import sys
import os
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

# Add project root to sys.path to import from src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from src.handover_gnn_dqn.topology.topology import REGIONS, latlon_to_xy, build_adjacency_from_positions

app = FastAPI(title="SON-GNN-DQN Visualization API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "opencellid" / "pokhara_cells.csv"

def load_pokhara_cells():
    if not DATA_PATH.exists():
        return []
    
    df = pd.read_csv(DATA_PATH)
    # Filter for LTE and within Pokhara bounds
    reg = REGIONS["pokhara"]
    df = df[df["radio"] == "LTE"]
    df = df[(df["lat"] >= reg.lat_min) & (df["lat"] <= reg.lat_max)]
    df = df[(df["lon"] >= reg.lon_min) & (df["lon"] <= reg.lon_max)]
    
    # Ensure unique cell IDs (OpenCellID can have duplicates for different sectors)
    df = df.drop_duplicates(subset=["cellid"])
    
    # Return lat, lon, and cellid
    cells = []
    for _, row in df.iterrows():
        cells.append({
            "id": int(row["cellid"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "radio": row["radio"]
        })
    return cells

@app.get("/api/topology")
async def get_topology():
    cells = load_pokhara_cells()
    if not cells:
        return {"nodes": [], "edges": []}
    
    # Extract positions for adjacency calculation
    lats = np.array([c["lat"] for c in cells])
    lons = np.array([c["lon"] for c in cells])
    x, y = latlon_to_xy(lats, lons)
    positions = np.column_stack((x, y))
    
    # Build adjacency (k=3 for cleaner visualization)
    adj = build_adjacency_from_positions(positions, k_neighbors=3)
    
    edges = []
    n = len(cells)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                edges.append({
                    "source": cells[i]["id"],
                    "target": cells[j]["id"],
                    "weight": float(adj[i, j])
                })
                
    return {
        "nodes": cells,
        "edges": edges,
        "region": {
            "name": "Pokhara Valley",
            "center": [float(lats.mean()), float(lons.mean())]
        }
    }

@app.get("/api/simulation")
async def get_simulation():
    # Stub: Generate a simple trajectory through Pokhara
    cells = load_pokhara_cells()
    if not cells:
        return []
    
    lats = [c["lat"] for c in cells]
    lons = [c["lon"] for c in cells]
    
    # Start at center
    lat_start, lon_start = lats[0], lons[0]
    lat_end, lon_end = lats[-1], lons[-1]
    
    num_steps = 50
    trajectory = []
    for i in range(num_steps):
        alpha = i / num_steps
        curr_lat = lat_start + alpha * (lat_end - lat_start)
        curr_lon = lon_start + alpha * (lon_end - lon_start)
        
        # Simple simulated metrics
        trajectory.append({
            "step": i,
            "lat": float(curr_lat),
            "lon": float(curr_lon),
            "serving_cell": cells[0]["id"] if i < 25 else cells[-1]["id"],
            "rsrp": -90 + np.random.normal(0, 2),
            "rsrq": -12 + np.random.normal(0, 1),
            "throughput": 5.0 + np.random.normal(0, 0.5),
            "q_values": {str(c["id"]): float(np.random.random()) for c in cells[:5]}
        })
    
    return trajectory

@app.get("/api/comparison")
async def get_comparison():
    return {
        "metrics": ["Avg Throughput", "P5 Throughput", "Ping-pong Rate", "Jain Fairness"],
        "a3_ttt": [5.29, 2.47, 0.0, 0.63],
        "son_gnn_dqn": [5.34, 2.52, 0.0, 0.65],
        "load_aware": [5.28, 2.47, 11.27, 0.61]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
