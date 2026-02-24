# mcts/types.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

Vec2 = Tuple[float, float]

@dataclass
class ObservationEvent:
    t_idx: int              # observation time index (discrete)
    vessel_id: int
    hit: bool
    y_pos: Optional[np.ndarray]  # [2,] if hit else None

@dataclass
class UAV:
    pos: int         # 不是经纬度坐标，而是基站 ID 或者正在监测的船 ID（-1表示在起点）
    battery_level: int   # the remaining battery level
    free_t_idx: int         # when it becomes idle

@dataclass
class WorldState:
    t_idx: int
    uavs: List[UAV]
    served: np.ndarray      # [N] bool, whether vessel already monitored (reward collected)
    vessel_traj: np.ndarray   # [M] bool, which scenarios remain feasible in simulation env

@dataclass(frozen=True)
class Action:
    uav_id: int
    vessel_id: int
    ob_time_idx: int