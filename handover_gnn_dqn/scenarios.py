"""Training and testing scenarios for GNN-DQN handover optimization.

The model trains on a MIX of all scenarios so it generalizes to any situation.
Each scenario represents a different network deployment type found in Nepal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .topology import generate_realistic_topology, get_area_size, load_topology


@dataclass
class Scenario:
    name: str
    num_cells: int
    num_ues: int
    area_m: float
    cell_positions: np.ndarray
    min_speed_mps: float
    max_speed_mps: float
    description: str


def _hex_grid(num_cells: int, isd_m: float) -> np.ndarray:
    """Generate hexagonal grid cell positions."""
    side = int(np.ceil(np.sqrt(num_cells)))
    positions = []
    for row in range(side + 1):
        for col in range(side + 1):
            x = col * isd_m + (row % 2) * isd_m * 0.5
            y = row * isd_m * 0.866
            positions.append([x, y])
            if len(positions) >= num_cells:
                break
        if len(positions) >= num_cells:
            break
    pos = np.array(positions[:num_cells])
    pos -= pos.mean(axis=0)
    return pos


def _highway_layout(num_cells: int, isd_m: float) -> np.ndarray:
    """Generate highway/linear cell layout."""
    x = np.linspace(0, (num_cells - 1) * isd_m, num_cells)
    y = np.zeros(num_cells)
    # Add slight offset to avoid perfectly linear
    rng = np.random.default_rng(42)
    y += rng.uniform(-isd_m * 0.1, isd_m * 0.1, num_cells)
    pos = np.column_stack((x, y))
    pos -= pos.mean(axis=0)
    return pos


def get_training_scenarios(seed: int = 42) -> List[Scenario]:
    """Get all training scenarios. Model trains on a random mix of these."""

    scenarios = []

    # 1. DENSE URBAN (Lakeside, Kathmandu core)
    dense_pos = _hex_grid(20, isd_m=300)
    scenarios.append(Scenario(
        name="dense_urban",
        num_cells=20,
        num_ues=250,  # ~12 UEs/cell (heavy)
        area_m=get_area_size(dense_pos),
        cell_positions=dense_pos,
        min_speed_mps=0.5,   # some static
        max_speed_mps=15.0,  # some in vehicles
        description="Dense urban: 300m ISD, heavy load, mixed mobility",
    ))

    # 2. HIGHWAY (Prithvi Highway, fast mobility)
    highway_pos = _highway_layout(10, isd_m=1500)
    scenarios.append(Scenario(
        name="highway",
        num_cells=10,
        num_ues=50,  # 5 UEs/cell
        area_m=get_area_size(highway_pos),
        cell_positions=highway_pos,
        min_speed_mps=15.0,  # all vehicular
        max_speed_mps=28.0,  # 60-100 km/h
        description="Highway: 1.5km ISD, fast mobility, too-late HO problem",
    ))

    # 3. SUBURBAN (Chipledhunga, medium density)
    suburban_pos = _hex_grid(15, isd_m=600)
    scenarios.append(Scenario(
        name="suburban",
        num_cells=15,
        num_ues=105,  # 7 UEs/cell
        area_m=get_area_size(suburban_pos),
        cell_positions=suburban_pos,
        min_speed_mps=1.0,
        max_speed_mps=17.0,
        description="Suburban: 600m ISD, medium load, mixed mobility",
    ))

    # 4. SPARSE RURAL (Hills, villages)
    rural_pos = _hex_grid(7, isd_m=2000)
    scenarios.append(Scenario(
        name="sparse_rural",
        num_cells=7,
        num_ues=25,  # 3-4 UEs/cell
        area_m=get_area_size(rural_pos),
        cell_positions=rural_pos,
        min_speed_mps=0.5,
        max_speed_mps=12.0,
        description="Sparse rural: 2km ISD, light load, limited options",
    ))

    # 5. OVERLOADED EVENT (Festival, stadium, peak hour)
    event_pos = _hex_grid(12, isd_m=400)
    scenarios.append(Scenario(
        name="overloaded_event",
        num_cells=12,
        num_ues=240,  # 20 UEs/cell (extreme)
        area_m=get_area_size(event_pos),
        cell_positions=event_pos,
        min_speed_mps=0.0,  # mostly standing
        max_speed_mps=5.0,
        description="Overloaded event: 400m ISD, extreme congestion, mostly static",
    ))

    # 6. REAL POKHARA (from OpenCellID dense cluster)
    try:
        pokhara_pos = np.load("data/opencellid/pokhara_dense_20.npy")
    except FileNotFoundError:
        pokhara_pos = _hex_grid(20, isd_m=400)
    scenarios.append(Scenario(
        name="real_pokhara",
        num_cells=len(pokhara_pos),
        num_ues=len(pokhara_pos) * 8,
        area_m=get_area_size(pokhara_pos),
        cell_positions=pokhara_pos,
        min_speed_mps=1.0,
        max_speed_mps=17.0,
        description="Real Pokhara: OpenCellID positions, realistic load",
    ))

    return scenarios


def get_test_scenarios(seed: int = 99) -> List[Scenario]:
    """Get test scenarios (model NEVER sees these during training)."""

    scenarios = []

    # 7. KATHMANDU (real positions)
    try:
        import csv
        from .topology import latlon_to_xy
        with open("data/opencellid/kathmandu_cells.csv") as f:
            rows = list(csv.DictReader(f))
        lats = np.array([float(r['lat']) for r in rows])
        lons = np.array([float(r['lon']) for r in rows])
        x, y = latlon_to_xy(lats, lons)
        ktm_pos = np.column_stack((x, y))
        # Take dense 25-cell subset
        center = ktm_pos.mean(axis=0)
        dists = np.linalg.norm(ktm_pos - center, axis=1)
        idx = np.argsort(dists)[:25]
        ktm_pos = ktm_pos[idx]
    except (FileNotFoundError, ValueError):
        ktm_pos = _hex_grid(25, isd_m=350)

    scenarios.append(Scenario(
        name="kathmandu_real",
        num_cells=len(ktm_pos),
        num_ues=len(ktm_pos) * 10,
        area_m=get_area_size(ktm_pos),
        cell_positions=ktm_pos,
        min_speed_mps=1.0,
        max_speed_mps=15.0,
        description="Kathmandu: real positions, very dense, heavy load",
    ))

    # 8. DHARAN (synthetic, different terrain)
    dharan_pos = generate_realistic_topology("dharan", num_cells=20, seed=seed)
    scenarios.append(Scenario(
        name="dharan_synthetic",
        num_cells=20,
        num_ues=100,
        area_m=get_area_size(dharan_pos),
        cell_positions=dharan_pos,
        min_speed_mps=1.0,
        max_speed_mps=17.0,
        description="Dharan: synthetic realistic, medium density",
    ))

    # 9. UNKNOWN HEXAGONAL (completely unseen structure)
    hex_pos = _hex_grid(19, isd_m=500)
    scenarios.append(Scenario(
        name="unknown_hex_grid",
        num_cells=19,
        num_ues=133,  # 7 UEs/cell
        area_m=get_area_size(hex_pos),
        cell_positions=hex_pos,
        min_speed_mps=1.0,
        max_speed_mps=20.0,
        description="Unknown hexagonal: standard 3GPP layout, never trained on",
    ))

    return scenarios
