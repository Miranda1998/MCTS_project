# mcts/env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import math
from .types import WorldState, UAV, Action, ObservationEvent

def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

@dataclass
class OnlineEnvConfig:
    slot_gap_min: float
    horizon_steps: int
    info_delay_steps: int
    uav_speed_kmph: float
    uav_obs_dwell_steps: int
    hit_radius_km_aeos: float
    hit_radius_km_airship: float

@dataclass
class OnlineEnv:
    """
    - 真值轨迹 traj_true 用于：执行、命中判定、收益结算
    - 预测/情景用于：rollout 仿真（在 mcts_core 里调用 clone_sim / step_sim）
    """
    traj_true: np.ndarray      # [N,K,2]
    traj_mean: np.ndarray      # [N,K,2]
    scenarios: np.ndarray      # [M,N,K,2]
    reward: np.ndarray         # [N]
    cfg: OnlineEnvConfig
    # 常态化监测层计划（可选）：用于产生观测事件
    # 形式：list of tuples (t_idx, platform_type, vessel_id, plan_center[2])
    normal_plan: Optional[List[Tuple[int, str, int, np.ndarray]]] = None

    def reset(self, uav_init_pos: Optional[np.ndarray] = None) -> WorldState:
        N = self.traj_true.shape[0]
        M = self.scenarios.shape[0]
        if uav_init_pos is None:
            # 默认：所有 UAV 从 (0,0) 出发
            uav_init_pos = np.zeros(2, dtype=float)

        uavs = [UAV(pos=uav_init_pos.copy(), free_t_idx=0) for _ in range(self.cfg_uav_num())]
        served = np.zeros(N, dtype=bool)
        scen_mask = np.ones(M, dtype=bool)
        return WorldState(t_idx=0, uavs=uavs, served=served, scen_mask=scen_mask)

    def cfg_uav_num(self) -> int:
        # 从外部传入的 UAV 初始列表长度决定，这里用 reward/轨迹无法得出，默认在 main 里设置一致
        # 为简单起见：用一个约定，把 UAV 数存放在 cfg 中：由 main 构造 env 时注入属性
        return getattr(self.cfg, "uav_num", 1)

    def set_uav_num(self, uav_num: int) -> None:
        setattr(self.cfg, "uav_num", uav_num)

    def available_actions(self, s: WorldState) -> List[Action]:
        """当前时刻 t_idx，所有已空闲 UAV 可选的派单动作集合（给任意未服务船）"""
        t = s.t_idx
        idle_uavs = [i for i, u in enumerate(s.uavs) if u.free_t_idx <= t]
        unserved = np.where(~s.served)[0].tolist()
        acts: List[Action] = []
        for ui in idle_uavs:
            for vid in unserved:
                acts.append(Action(uav_id=ui, vessel_id=vid))
        return acts

    def _travel_steps(self, a: np.ndarray, b: np.ndarray) -> int:
        # km/h -> km/min -> steps
        dist_km = _euclid(a, b)
        speed_km_min = self.cfg.uav_speed_kmph / 60.0
        t_min = dist_km / max(speed_km_min, 1e-9)
        steps = int(math.ceil(t_min / self.cfg.slot_gap_min))
        return max(0, steps)

    def step_real(self, s: WorldState, a: Action) -> Tuple[WorldState, float]:
        """
        在真实环境中执行 UAV 观测动作并结算即时收益：
        - UAV 飞到船舶在到达时刻的真值位置
        - 若到达后仍在同一位置（定义为命中），则获得 reward[v]
        你可以按你的 SMDP 奖励改写：这里给最简可用版本（一次性收益）。
        """
        t0 = s.t_idx
        u = s.uavs[a.uav_id]
        if u.free_t_idx > t0:
            # UAV 还没空闲，动作无效：返回原状态、0收益
            return s, 0.0

        N, K, _ = self.traj_true.shape
        vid = a.vessel_id

        # 目标位置用“当前时刻的预测/真值”都可以，这里用当前真值位置决定航行目标
        # 如果你希望 UAV 以“仿真环境预测”去飞，再用真值结算，也只需把 dest 改成 sim 的位置
        dest = self.traj_true[vid, min(t0, K - 1)]

        fly_steps = self._travel_steps(u.pos, dest)
        arrive_t = min(t0 + fly_steps, self.cfg.horizon_steps - 1)
        dwell = self.cfg.uav_obs_dwell_steps
        finish_t = min(arrive_t + dwell, self.cfg.horizon_steps - 1)

        # 到达时刻真实位置
        true_pos = self.traj_true[vid, min(arrive_t, K - 1)]
        hit = (_euclid(true_pos, dest) <= 1e-6)  # 最简命中；你可以改为 UAV 传感半径判定

        r = 0.0
        served2 = s.served.copy()
        if hit and (not served2[vid]):
            r = float(self.reward[vid])
            served2[vid] = True

        # 更新 UAV 状态
        uavs2 = [UAV(pos=x.pos.copy(), free_t_idx=x.free_t_idx) for x in s.uavs]
        uavs2[a.uav_id].pos = true_pos.copy()     # 到达后 UAV 在真值位置
        uavs2[a.uav_id].free_t_idx = finish_t

        # 下一事件：推进到最近一个 UAV 空闲时刻 或者 t+1
        next_t = min(min([uu.free_t_idx for uu in uavs2]), self.cfg.horizon_steps - 1)
        next_t = max(next_t, t0 + 1)

        s2 = WorldState(t_idx=next_t, uavs=uavs2, served=served2, scen_mask=s.scen_mask.copy())
        return s2, r

    def generate_observations(self, t_idx: int) -> List[ObservationEvent]:
        """
        由常态化监测层计划在真实环境中产生观测事件（命中/未命中及命中位置）。
        这些观测应当在 t_idx + tau 才注入仿真环境：这部分由 scenario_updater 处理。
        """
        if not self.normal_plan:
            return []

        N, K, _ = self.traj_true.shape
        obs: List[ObservationEvent] = []
        for (tt, ptype, vid, center) in self.normal_plan:
            if tt != t_idx:
                continue
            true_pos = self.traj_true[vid, min(tt, K - 1)]
            if ptype.lower() == "aeos":
                R = self.cfg.hit_radius_km_aeos
            else:
                R = self.cfg.hit_radius_km_airship
            hit = (_euclid(true_pos, center) <= R)
            y = true_pos.copy() if hit else None
            obs.append(ObservationEvent(t_idx=tt, vessel_id=vid, hit=hit, y_pos=y))
        return obs