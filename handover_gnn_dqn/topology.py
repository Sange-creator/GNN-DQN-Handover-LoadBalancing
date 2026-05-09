"""Topology loader: real cell positions from OpenCellID or realistic synthetic.

Supports:
1. Loading real eNodeB positions from OpenCellID CSV (if available)
2. Generating realistic topologies based on known city parameters
3. Converting lat/lon to local XY coordinates (meters)

Regions supported:
- Pokhara Valley (28.17-28.27N, 83.93-84.03E)
- Kathmandu Valley (27.65-27.75N, 85.28-85.38E)
- Dharan (26.78-26.83N, 87.27-87.32E)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "opencellid"


@dataclass
class RegionConfig:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    estimated_lte_cells: int
    urban_isd_m: float
    suburban_isd_m: float


REGIONS = {
    "pokhara": RegionConfig(
        name="Pokhara Valley",
        lat_min=28.17, lat_max=28.27,
        lon_min=83.93, lon_max=84.03,
        estimated_lte_cells=112,
        urban_isd_m=400.0,
        suburban_isd_m=1000.0,
    ),
    "kathmandu": RegionConfig(
        name="Kathmandu Valley",
        lat_min=27.65, lat_max=27.75,
        lon_min=85.28, lon_max=85.38,
        estimated_lte_cells=387,
        urban_isd_m=300.0,
        suburban_isd_m=800.0,
    ),
    "dharan": RegionConfig(
        name="Dharan",
        lat_min=26.78, lat_max=26.83,
        lon_min=87.27, lon_max=87.32,
        estimated_lte_cells=48,
        urban_isd_m=450.0,
        suburban_isd_m=1200.0,
    ),
    "biratnagar": RegionConfig(
        name="Biratnagar",
        lat_min=26.42, lat_max=26.50,
        lon_min=87.25, lon_max=87.30,
        estimated_lte_cells=65,
        urban_isd_m=400.0,
        suburban_isd_m=1000.0,
    ),
}


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon arrays to local XY meters (simple equirectangular)."""
    lat_ref = lat.mean()
    lon_ref = lon.mean()
    x = (lon - lon_ref) * 111_320.0 * np.cos(np.radians(lat_ref))
    y = (lat - lat_ref) * 110_540.0
    return x, y


def load_opencellid_csv(csv_path: Path, region: str = "pokhara") -> np.ndarray:
    """Load real eNodeB positions from OpenCellID CSV.

    Expected CSV columns: radio,mcc,net,area,cell,unit,lon,lat,range,samples,changeable,created,updated,averageSignal

    Returns: (N, 2) array of XY positions in meters.
    """
    reg = REGIONS[region]

    lats = []
    lons = []
    with open(csv_path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            radio = parts[0]
            if radio != "LTE":
                continue
            try:
                lon = float(parts[6])
                lat = float(parts[7])
            except (ValueError, IndexError):
                continue
            if reg.lat_min <= lat <= reg.lat_max and reg.lon_min <= lon <= reg.lon_max:
                lats.append(lat)
                lons.append(lon)

    if not lats:
        raise ValueError(f"No LTE cells found in {csv_path} for region {region}")

    lat_arr = np.array(lats)
    lon_arr = np.array(lons)
    x, y = latlon_to_xy(lat_arr, lon_arr)
    return np.column_stack((x, y))


def generate_realistic_topology(
    region: str = "pokhara",
    num_cells: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate realistic cell positions mimicking a Nepal city deployment.

    Uses a clustered Poisson point process:
    - Dense cluster in city center (urban ISD)
    - Sparser cells in suburbs (suburban ISD)
    - Randomized with minimum distance constraint

    Returns: (N, 2) array of XY positions in meters.
    """
    reg = REGIONS[region]
    rng = np.random.default_rng(seed)

    if num_cells is None:
        num_cells = reg.estimated_lte_cells

    area_x = (reg.lon_max - reg.lon_min) * 111_320.0 * np.cos(np.radians((reg.lat_min + reg.lat_max) / 2))
    area_y = (reg.lat_max - reg.lat_min) * 110_540.0

    positions = []
    center = np.array([area_x / 2, area_y / 2])

    urban_radius = min(area_x, area_y) * 0.3
    n_urban = int(num_cells * 0.6)
    n_suburban = num_cells - n_urban

    for _ in range(n_urban):
        while True:
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.exponential(urban_radius * 0.5)
            r = min(r, urban_radius)
            pos = center + r * np.array([np.cos(angle), np.sin(angle)])
            if 0 <= pos[0] <= area_x and 0 <= pos[1] <= area_y:
                if not positions or _min_dist(pos, positions) > reg.urban_isd_m * 0.4:
                    positions.append(pos)
                    break

    for _ in range(n_suburban):
        attempts = 0
        while attempts < 100:
            pos = rng.uniform([0, 0], [area_x, area_y])
            dist_to_center = np.linalg.norm(pos - center)
            if dist_to_center > urban_radius * 0.7:
                if not positions or _min_dist(pos, positions) > reg.suburban_isd_m * 0.3:
                    positions.append(pos)
                    break
            attempts += 1
        else:
            pos = rng.uniform([0, 0], [area_x, area_y])
            positions.append(pos)

    positions = np.array(positions[:num_cells])
    positions -= positions.mean(axis=0)
    return positions


def _min_dist(pos: np.ndarray, existing: list) -> float:
    dists = [np.linalg.norm(pos - p) for p in existing]
    return min(dists) if dists else float("inf")


def build_adjacency_from_positions(
    positions: np.ndarray,
    k_neighbors: int = 4,
) -> np.ndarray:
    """Build normalized adjacency from cell positions.

    Edges connect each cell to its k nearest neighbors with
    distance-weighted connections. Returns symmetric normalized adjacency.
    """
    n = len(positions)
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    scale = max(np.median(d[d > 0]) / 2.0, 1.0)
    w = np.exp(-d / scale)
    np.fill_diagonal(w, 0.0)

    adj = np.eye(n)
    for i in range(n):
        nn_idx = np.argsort(d[i])[1: k_neighbors + 1]
        adj[i, nn_idx] = w[i, nn_idx]
    adj = np.maximum(adj, adj.T)

    degree = np.sum(adj, axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-9))
    return inv_sqrt[:, None] * adj * inv_sqrt[None, :]


def load_topology(
    region: str = "pokhara",
    num_cells: int | None = None,
    opencellid_csv: Path | None = None,
    seed: int = 42,
    k_neighbors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or generate cell topology for a region.

    Returns:
        positions: (N, 2) cell positions in meters
        adjacency: (N, N) normalized adjacency matrix
    """
    if opencellid_csv is not None and opencellid_csv.exists():
        positions = load_opencellid_csv(opencellid_csv, region)
        if num_cells is not None and len(positions) > num_cells:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(positions), size=num_cells, replace=False)
            positions = positions[idx]
    else:
        positions = generate_realistic_topology(region, num_cells, seed)

    adjacency = build_adjacency_from_positions(positions, k_neighbors)
    return positions, adjacency


def get_area_size(positions: np.ndarray) -> float:
    """Get the bounding area size for configuring LTEConfig."""
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    return max(x_range, y_range) * 1.2
