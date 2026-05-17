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
    mobility_model: str = "random"
    shadow_sigma_db: float = 6.0
    min_demand_mbps: float = 2.0
    max_demand_mbps: float = 8.0


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
    return np.array(positions[:num_cells], dtype=float)


def _highway_layout(num_cells: int, isd_m: float) -> np.ndarray:
    """Generate highway/linear cell layout."""
    x = np.linspace(0, (num_cells - 1) * isd_m, num_cells)
    y = np.zeros(num_cells)
    # Add slight offset to avoid perfectly linear
    rng = np.random.default_rng(42)
    y += rng.uniform(-isd_m * 0.1, isd_m * 0.1, num_cells)
    return np.column_stack((x, y)).astype(float)


def _ring_with_hole_layout(num_cells: int = 8, radius_m: float = 500.0) -> np.ndarray:
    """Cells arranged in a ring with an empty center → deliberate coverage hole."""
    angles = np.linspace(0.0, 2.0 * np.pi, num_cells, endpoint=False)
    x = radius_m + radius_m * np.cos(angles)
    y = radius_m + radius_m * np.sin(angles)
    return np.column_stack((x, y)).astype(float)


def get_training_scenarios(seed: int = 42) -> List[Scenario]:
    """Get all training scenarios. Model trains on a random mix of these."""

    scenarios = []

    # 1. DENSE URBAN (Lakeside, Kathmandu core)
    dense_pos = _hex_grid(20, isd_m=300)
    scenarios.append(Scenario(
        name="dense_urban",
        num_cells=20,
        num_ues=160,  # 8 UEs/cell (heavy but tractable for training)
        area_m=get_area_size(dense_pos),
        cell_positions=dense_pos,
        min_speed_mps=0.5,   # some static
        max_speed_mps=15.0,  # some in vehicles
        description="Dense urban: 300m ISD, heavy load, mixed mobility",
        mobility_model="random",
    ))

    # 2. HIGHWAY (Prithvi Highway, fast mobility)
    # Use 800m ISD so total span ~7.2km fits in a square area where UEs
    # remain within coverage. Original 1500m ISD caused 70% outage because
    # UEs scattered in 16km×16km but cells only covered a thin line.
    highway_pos = _highway_layout(10, isd_m=800)
    scenarios.append(Scenario(
        name="highway",
        num_cells=10,
        num_ues=50,  # 5 UEs/cell
        area_m=get_area_size(highway_pos),
        cell_positions=highway_pos,
        min_speed_mps=15.0,  # all vehicular
        max_speed_mps=28.0,  # 60-100 km/h
        description="Highway: 800m ISD, fast mobility, too-late HO problem",
        mobility_model="highway",
        shadow_sigma_db=7.0,
    ))

    # 2b. HIGHWAY FAST (Mugling-Narayanghat, very fast mobility + voice calls)
    # Targets the "call drop at highway speed" problem. Higher speed range
    # forces the model to learn proactive handover before signal death.
    highway_fast_pos = _highway_layout(12, isd_m=700)
    scenarios.append(Scenario(
        name="highway_fast",
        num_cells=12,
        num_ues=60,  # 5 UEs/cell
        area_m=get_area_size(highway_fast_pos),
        cell_positions=highway_fast_pos,
        min_speed_mps=22.0,   # 80 km/h minimum
        max_speed_mps=33.0,   # up to 120 km/h
        description="Highway fast: 700m ISD, 80-120 km/h, proactive HO required",
        mobility_model="highway",
        shadow_sigma_db=7.0,
        min_demand_mbps=1.5,  # voice/video call minimum
        max_demand_mbps=6.0,
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
        mobility_model="random",
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
        mobility_model="random",
        shadow_sigma_db=8.0,
        min_demand_mbps=1.0,
        max_demand_mbps=5.0,
    ))

    # 5. OVERLOADED EVENT (Festival, stadium, peak hour)
    event_pos = _hex_grid(12, isd_m=400)
    scenarios.append(Scenario(
        name="overloaded_event",
        num_cells=12,
        num_ues=156,  # 13 UEs/cell (heavy congestion)
        area_m=get_area_size(event_pos),
        cell_positions=event_pos,
        min_speed_mps=0.0,  # mostly standing
        max_speed_mps=5.0,
        description="Overloaded event: 400m ISD, extreme congestion, mostly static",
        mobility_model="event",
        min_demand_mbps=4.0,
        max_demand_mbps=10.0,
    ))

    # 6. REAL POKHARA (from OpenCellID dense cluster)
    try:
        pokhara_pos = np.load("data/raw/opencellid/pokhara_dense_20.npy")
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
        mobility_model="random",
        shadow_sigma_db=8.33,
    ))

    # 7. POKHARA DENSE PEAK HOUR (Track-B stress test on real positions)
    # 15 active UEs/cell for training speed (GNN generalizes to any count).
    # Higher demand (video-dominant traffic) simulates congestion pressure.
    try:
        pokhara_peak_pos = np.load("data/raw/opencellid/pokhara_dense_20.npy")
    except FileNotFoundError:
        pokhara_peak_pos = _hex_grid(20, isd_m=400)
    scenarios.append(Scenario(
        name="pokhara_dense_peakhour",
        num_cells=len(pokhara_peak_pos),
        num_ues=len(pokhara_peak_pos) * 15,
        area_m=get_area_size(pokhara_peak_pos),
        cell_positions=pokhara_peak_pos,
        min_speed_mps=0.5,
        max_speed_mps=14.0,
        description="Pokhara peak hour: real OpenCellID positions, 15 UEs/cell, video-heavy demand",
        mobility_model="random",
        shadow_sigma_db=8.33,
        min_demand_mbps=3.0,
        max_demand_mbps=10.0,
    ))

    return scenarios


def get_test_scenarios(seed: int = 99) -> List[Scenario]:
    """Get test scenarios (model NEVER sees these during training)."""

    scenarios = []

    # 7. KATHMANDU (real positions)
    try:
        import csv
        from .topology import latlon_to_xy
        with open("data/raw/opencellid/kathmandu_cells.csv") as f:
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
        mobility_model="random",
        shadow_sigma_db=8.33,
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
        mobility_model="random",
        shadow_sigma_db=8.33,
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
        mobility_model="random",
    ))

    # 10. COVERAGE HOLE (deliberate gap in the middle of the layout)
    # Eight cells arranged in a ring around an empty center; UE trajectories
    # that traverse the center see all neighbors weak. Tests "no valid
    # handover" behavior and policy stability under outage.
    hole_pos = _ring_with_hole_layout(num_cells=8, radius_m=500.0)
    scenarios.append(Scenario(
        name="coverage_hole",
        num_cells=8,
        num_ues=40,  # 5 UEs/cell
        area_m=get_area_size(hole_pos),
        cell_positions=hole_pos,
        min_speed_mps=1.0,
        max_speed_mps=12.0,
        description="Coverage hole: 8-cell ring with empty center, tests outage behavior",
        mobility_model="random",
        shadow_sigma_db=6.5,
    ))

    return scenarios


def get_stress_scenarios(seed: int = 137) -> List[Scenario]:
    """Get stress evaluation scenarios for *evaluation only* (model is NOT trained on these).

    The training scenarios use 5-15 UEs/cell, which keeps cell utilization at 30-80%.
    At that load level, the demand-capped throughput model makes most handover policies
    look identical because cells aren't actually congested.

    These stress scenarios push 20-30 UEs/cell with higher per-UE demand, driving
    aggregate demand to 1.5-1.7× cell capacity. That's the regime where load-aware
    handover decisions actually matter, and where son_gnn_dqn should differentiate
    from strongest_rsrp and a3_ttt.

    Use these for the headline "congestion stress" results table in the thesis.
    """
    scenarios: List[Scenario] = []

    # S1. DENSE URBAN STRESS — 20 cells × 20 UEs = 400 UEs, demand 4-20 Mbps
    # Aggregate per-cell demand: 20 × 12 = 240 Mbps → 1.6× cell capacity (150 Mbps)
    dense_pos = _hex_grid(20, isd_m=300)
    scenarios.append(Scenario(
        name="stress_dense_urban",
        num_cells=20,
        num_ues=400,
        area_m=get_area_size(dense_pos),
        cell_positions=dense_pos,
        min_speed_mps=0.5,
        max_speed_mps=15.0,
        description="Stress: dense urban with 20 UEs/cell, 1.6× capacity demand",
        mobility_model="random",
        shadow_sigma_db=6.0,
        min_demand_mbps=4.0,
        max_demand_mbps=20.0,
    ))

    # S2. OVERLOAD EVENT STRESS — 12 cells × 25 UEs = 300 UEs, demand 3-17 Mbps
    # Aggregate per-cell demand: 25 × 10 = 250 Mbps → 1.67× cell capacity
    event_pos = _hex_grid(12, isd_m=400)
    scenarios.append(Scenario(
        name="stress_overload_event",
        num_cells=12,
        num_ues=300,
        area_m=get_area_size(event_pos),
        cell_positions=event_pos,
        min_speed_mps=0.0,
        max_speed_mps=5.0,
        description="Stress: stadium/festival 25 UEs/cell, 1.67× capacity demand",
        mobility_model="event",
        shadow_sigma_db=6.0,
        min_demand_mbps=3.0,
        max_demand_mbps=17.0,
    ))

    # S3. POKHARA PEAK HOUR STRESS — real positions, 30 UEs/cell, demand 3-13 Mbps
    # Aggregate per-cell demand: 30 × 8 = 240 Mbps → 1.6× cell capacity
    # Higher UE count than training (15/cell) tests generalization to load density.
    try:
        pokhara_pos = np.load("data/raw/opencellid/pokhara_dense_20.npy")
    except FileNotFoundError:
        pokhara_pos = _hex_grid(20, isd_m=400)
    scenarios.append(Scenario(
        name="stress_pokhara_peakhour",
        num_cells=len(pokhara_pos),
        num_ues=len(pokhara_pos) * 30,
        area_m=get_area_size(pokhara_pos),
        cell_positions=pokhara_pos,
        min_speed_mps=0.5,
        max_speed_mps=14.0,
        description="Stress: real Pokhara positions, 30 UEs/cell, 1.6× capacity demand",
        mobility_model="random",
        shadow_sigma_db=8.33,
        min_demand_mbps=3.0,
        max_demand_mbps=13.0,
    ))

    # S4. HIGHWAY JAM — 10 cells linear, 15 UEs/cell with traffic-jam slow mobility
    # Different stress profile: moderate congestion + mobility constraints +
    # linear topology (limited handover options).
    # Aggregate per-cell demand: 15 × 12 = 180 Mbps → 1.2× cell capacity
    highway_pos = _highway_layout(10, isd_m=800)
    scenarios.append(Scenario(
        name="stress_highway_jam",
        num_cells=10,
        num_ues=150,
        area_m=get_area_size(highway_pos),
        cell_positions=highway_pos,
        min_speed_mps=1.0,    # traffic jam — slow
        max_speed_mps=10.0,
        description="Stress: highway traffic jam, 15 UEs/cell, linear topology",
        mobility_model="highway",
        shadow_sigma_db=7.0,
        min_demand_mbps=4.0,
        max_demand_mbps=20.0,
    ))

    return scenarios
