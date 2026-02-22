# mcts/scenario_updater.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from .types import WorldState, ObservationEvent

def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

@dataclass
class ScenarioUpdater:
    scenarios: np.ndarray      # [M,N,K,2]
    # 观测一致性门限（可取平台覆盖半径的一部分）
    gate_km: float

    def apply_observation(self, s: WorldState, obs: ObservationEvent) -> WorldState:
        """
        仅使用观测事件（命中/未命中 + 命中位置）更新 scen_mask，不用未来真值。
        """
        mask = s.scen_mask.copy()
        M, N, K, _ = self.scenarios.shape
        t = obs.t_idx
        vid = obs.vessel_id

        if obs.hit and obs.y_pos is not None:
            # 命中：保留在该时刻位置与观测位置足够接近的情景
            y = obs.y_pos
            for m in range(M):
                if not mask[m]:
                    continue
                pos_m = self.scenarios[m, vid, min(t, K - 1)]
                if _euclid(pos_m, y) > self.gate_km:
                    mask[m] = False
        else:
            # 未命中：仅有“未命中”信息时，无法用位置约束太死；
            # 这里保守处理：不做过滤（或者你也可以结合“观测中心+覆盖半径”来过滤）
            pass

        # 若过滤后全空：回退为全保留（并可记录一次失配）
        if not mask.any():
            mask[:] = True

        return WorldState(t_idx=s.t_idx, uavs=s.uavs, served=s.served, scen_mask=mask)