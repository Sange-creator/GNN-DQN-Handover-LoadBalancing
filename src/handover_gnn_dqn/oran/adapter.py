from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OranDecision:
    ue_id: int
    serving_cell: int
    target_cell: int
    load_proxy_rsrq: float
    prb_utilization: float
    prb_available: bool
    cio_delta_db: float


def target_to_cio_delta(serving_cell: int, target_cell: int, step_db: float = 1.0) -> float:
    """Translate a target-cell recommendation into a simple CIO demo action."""
    return 0.0 if serving_cell == target_cell else float(step_db)


def build_oran_decision(
    *,
    ue_id: int,
    serving_cell: int,
    target_cell: int,
    state: np.ndarray,
) -> OranDecision:
    """Extract demo-friendly telemetry from an ORAN_E2 state matrix.

    Column contract from `FeatureProfile.ORAN_E2`:
    10 = RSRQ load proxy, 11 = true PRB utilization, 12 = PRB availability.
    """
    target = int(target_cell)
    return OranDecision(
        ue_id=int(ue_id),
        serving_cell=int(serving_cell),
        target_cell=target,
        load_proxy_rsrq=float(state[target, 10]),
        prb_utilization=float(state[target, 11]),
        prb_available=bool(state[target, 12] >= 0.5),
        cio_delta_db=target_to_cio_delta(int(serving_cell), target),
    )
