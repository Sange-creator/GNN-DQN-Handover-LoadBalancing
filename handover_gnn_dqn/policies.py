from __future__ import annotations

import numpy as np
import torch

from .simulator import CellularNetworkEnv


class NoHandoverPolicy:
    name = "no_handover"

    def reset(self, env: CellularNetworkEnv) -> None:
        pass

    def select(self, env: CellularNetworkEnv, ue_idx: int) -> int:
        return int(env.serving[ue_idx])


class StrongestRsrpPolicy:
    name = "strongest_rsrp"

    def __init__(self, hysteresis_db: float = 2.0):
        self.hysteresis_db = hysteresis_db

    def reset(self, env: CellularNetworkEnv) -> None:
        pass

    def select(self, env: CellularNetworkEnv, ue_idx: int) -> int:
        rsrp = env.rsrp_matrix()[ue_idx]
        current = int(env.serving[ue_idx])
        best = int(np.argmax(rsrp))
        if rsrp[best] > rsrp[current] + self.hysteresis_db:
            return best
        return current


class A3HandoverPolicy:
    name = "a3_ttt"

    def __init__(self, offset_db: float = 3.0, time_to_trigger: int = 3):
        self.offset_db = offset_db
        self.time_to_trigger = time_to_trigger
        self.candidate = None
        self.counter = None

    def reset(self, env: CellularNetworkEnv) -> None:
        self.candidate = np.full(env.cfg.num_ues, -1, dtype=int)
        self.counter = np.zeros(env.cfg.num_ues, dtype=int)

    def select(self, env: CellularNetworkEnv, ue_idx: int) -> int:
        rsrp = env.rsrp_matrix()[ue_idx]
        current = int(env.serving[ue_idx])
        best = int(np.argmax(rsrp))

        if best != current and rsrp[best] > rsrp[current] + self.offset_db:
            if self.candidate[ue_idx] == best:
                self.counter[ue_idx] += 1
            else:
                self.candidate[ue_idx] = best
                self.counter[ue_idx] = 1
            if self.counter[ue_idx] >= self.time_to_trigger:
                self.counter[ue_idx] = 0
                return best
        else:
            self.candidate[ue_idx] = -1
            self.counter[ue_idx] = 0

        return current


class LoadAwarePolicy:
    """ReBuHa-like rule: combines signal quality with RSRQ-derived load estimate.

    Uses only UE-observable measurements (RSRP + RSRQ) so the comparison
    against GNN-DQN is fair — neither method has access to true eNB-side
    PRB utilisation.  Load is estimated from RSRQ using the same
    ``load_from_rsrq`` proxy as the GNN-DQN feature builder.
    """
    name = "load_aware"

    def __init__(self, load_weight: float = 0.48, handover_cost: float = 0.04):
        self.load_weight = load_weight
        self.handover_cost = handover_cost

    def reset(self, env: CellularNetworkEnv) -> None:
        pass

    def select(self, env: CellularNetworkEnv, ue_idx: int) -> int:
        current = int(env.serving[ue_idx])
        rsrp = env.rsrp_matrix()[ue_idx]
        quality = env.quality_from_rsrp(rsrp)
        # Estimate per-cell load from this UE's RSRQ measurement
        # (UE-observable, no eNB cooperation needed).
        rsrq = env._compute_rsrq()[ue_idx]           # (num_cells,)
        estimated_loads = env.load_from_rsrq(rsrq)    # [0, 1]
        score = quality - self.load_weight * np.clip(estimated_loads, 0.0, 1.8)
        score -= self.handover_cost * (np.arange(env.cfg.num_cells) != current)
        score[~env.valid_actions(ue_idx)] = -1e9
        return int(np.argmax(score))


class GnnDqnPolicy:
    name = "gnn_dqn"

    def __init__(self, agent, epsilon: float = 0.0):
        self.agent = agent
        self.epsilon = epsilon

    def reset(self, env: CellularNetworkEnv) -> None:
        pass

    def select(self, env: CellularNetworkEnv, ue_idx: int) -> int:
        state_t = torch.from_numpy(env.build_state(ue_idx)).float()
        edge_index, edge_weight = env.edge_data
        return self.agent.act(
            state_t, edge_index, edge_weight,
            epsilon=self.epsilon,
            valid_mask=env.valid_actions(ue_idx),
        )
