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
    vessels_num: int
    uav_num: int
    uav_battery_capacity: int
    uav_speed_kmph: float
    feasible_pair: set
    uav_obs_dwell_steps: int
    hit_radius_km_aeos: float
    hit_radius_km_airship: float
    hit_radius_km_drones: float
    base_set: List[int]  # 可选的 UAV 基站列表

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

    def reset(self) -> WorldState:
        '''
        状态初始化，所有无人机处在基站，且空闲，所有船舶待监测"
        '''
        N = self.traj_true.shape[0]
        M = self.scenarios.shape[0]

        # 每个无人机第一个动作是选择基站位置，因此初始位置设为-1
        uavs = [UAV(pos=-1, battery_level=self.cfg.uav_battery_capacity, free_t_idx=0)
                for _ in range(self.cfg.uav_num)]
        served = np.zeros(N, dtype=bool)
        return WorldState(t_idx=-1, uavs=uavs, served=served, vessel_traj=self.traj_mean)

    def _clone_state(self, s: WorldState) -> WorldState:
        uavs2 = [UAV(pos=u.pos, battery_level=u.battery_level, free_t_idx=u.free_t_idx) for u in s.uavs]
        served2 = s.served.copy()
        # vessel_traj 如果你后面会改它（更新情景/后验），这里也要 copy；否则可以共享引用
        vessel_traj2 = s.vessel_traj  # or s.vessel_traj.copy()
        return WorldState(t_idx=s.t_idx, uavs=uavs2, served=served2, vessel_traj=vessel_traj2)

    def _plus_base(self, unserved):
        unserved_plus_base_set = [base_id for base_id in self.cfg.base_set]
        for unserved_vid in unserved:
            unserved_plus_base_set.append(unserved_vid + len(self.cfg.base_set))
        return unserved_plus_base_set

    def _can_return_to_base(self, from_vid: int, start_t: int) -> bool:
        """
        只通过 feasible_pair 判断：从 (from_vid, start_t) 出发，是否存在可返回任一基站的可行弧
        """
        # start_t 之后的任意返回时刻都可以（找到了就 True）
        for b in self.cfg.base_set:
            # 如果你 feasible_pair 是“最早可达 obt”那种编码，也可以像下面一样从 start_t 往后找
            for t_ret in range(start_t, self.cfg.horizon_steps):
                if (from_vid, start_t, b, t_ret) in self.cfg.feasible_pair:
                    return True
        return False

    def available_actions_old(self, s: WorldState) -> List[Action]:
        """当前时刻 t_idx，所有已空闲 UAV 可选的派单动作集合（给任意未服务船）"""
        t = s.t_idx
        idle_uavs = [i for i, u in enumerate(s.uavs) if u.free_t_idx <=0]
        if t == -1:  # 如果是初始时刻，允许UAV从任意一个基站出发
            acts: List[Action] = []
            for ui in idle_uavs:
                for vid in self.cfg.base_set:
                # for vid in [0, 1, 2, 3]:
                    acts.append(Action(uav_id=ui, vessel_id=vid, ob_time_idx=0))
        else:
            unserved = np.where(~s.served)[0].tolist()
            acts: List[Action] = []
            unserved_plus_base = self._plus_base(unserved)
            for ui in idle_uavs:
                for vid in unserved_plus_base:
                    for obt in range(t, self.cfg.horizon_steps):  # 允许 UAV 选择未来某个时刻的观测
                        if (s.uavs[ui].pos, t, vid, obt) in self.cfg.feasible_pair:
                            acts.append(Action(uav_id=ui, vessel_id=vid, ob_time_idx=obt))
                            break  # 每艘船选唯一最早访问时刻
        return acts

    def available_actions(self, s: WorldState) -> List[Action]:
        """当前时刻 t_idx，所有已空闲 UAV 可选的派单动作集合（并保证执行后可返回基站）"""
        t = s.t_idx
        idle_uavs = [i for i, u in enumerate(s.uavs) if u.free_t_idx <= 0]

        acts: List[Action] = []

        if t == -1:
            uavs_not_at_base = [i for i, u in enumerate(s.uavs) if u.pos == -1]
            # 初始：选择基站（这里不做返航约束）
            for ui in idle_uavs:
                if ui not in uavs_not_at_base:
                    continue
                for b in self.cfg.base_set:
                # for b in [2]:
                    for start_t in [0, 1, 2, 3, 4]:
                        # start_t = 1
                        acts.append(Action(uav_id=ui, vessel_id=b, ob_time_idx=start_t))
            return acts

        unserved = np.where(~s.served)[0].tolist()
        uav_pos = [ui.pos for ui in s.uavs]
        unarrived = [vid for vid in unserved if vid not in uav_pos]
        if t > 0:
            unserved_plus_base = self._plus_base(unarrived)
        else:
            unserved_plus_base = [vid + len(self.cfg.base_set) for vid in unarrived]

        # unserved_plus_base = self._plus_base(unserved)
        for ui in idle_uavs:
            u_from = s.uavs[ui].pos
            for vid in unserved_plus_base:
                # 先找“最早可行的观测时刻 obt”
                # if u_from == 10:
                #     vid = 14
                chosen_obt = None
                for obt in range(t, self.cfg.horizon_steps):
                    if (u_from, t, vid, obt) in self.cfg.feasible_pair:
                        chosen_obt = obt
                        break

                if chosen_obt is None:
                    continue

                # 任务完成时刻：考虑 dwell（更严谨）
                # t_finish = min(chosen_obt + self.cfg.uav_obs_dwell_steps, self.cfg.horizon_steps - 1)
                t_finish = chosen_obt + self.cfg.uav_obs_dwell_steps
                # 关键过滤：必须存在从 vid 在 t_finish 返回某个基站的可行弧
                if vid >= len(self.cfg.base_set):  # 没选基站当目标：需要返航约束
                    if not self._can_return_to_base(vid, t_finish):
                        continue

                acts.append(Action(uav_id=ui, vessel_id=vid, ob_time_idx=t_finish))

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
        s = self._clone_state(s)  # 新状态防止混乱！

        t0 = s.t_idx
        u = s.uavs[a.uav_id]
        if u.free_t_idx > 0:
            # UAV 还没空闲，动作无效：返回原状态、0收益
            print('出错了，给没有空闲的无人机安排任务了')
            return s, 0.0

        vid = a.vessel_id
        finish_t = a.ob_time_idx

        if t0 == -1:
            uavs_act = UAV(pos=vid, battery_level=u.battery_level, free_t_idx=finish_t)
            s.uavs[a.uav_id] = uavs_act
            # 判断是否所有无人机都安排好了基站
            base_uavs = [i for i, u in enumerate(s.uavs) if u.pos != -1]
            if len(base_uavs) == self.cfg.uav_num:  # 当所有无人机都安排好了基站时，推进时间到最早的观测时刻
                next_t_derta = min([u.free_t_idx for u in s.uavs])
                uavs2 = [UAV(pos=x.pos, battery_level=x.battery_level,
                             free_t_idx=x.free_t_idx-next_t_derta) for x in s.uavs]
                served2 = s.served.copy()
                s2 = WorldState(t_idx=next_t_derta, uavs=uavs2, served=served2, vessel_traj=self.traj_mean)
                return s2, 0.0
            else:
                return s, 0.0

        else:
            remainning_free = finish_t - t0
            # 更新 UAV 状态
            uavs_act = UAV(pos=vid, battery_level=u.battery_level, free_t_idx=remainning_free)
            s.uavs[a.uav_id] = uavs_act

            idle_uavs = [i for i, u in enumerate(s.uavs) if u.free_t_idx == 0]
            if len(idle_uavs) != 0:  # 当有空闲的UAV时，固定时间继续给下一个空闲的无人机安排任务
                return s, 0
            else:  # 当没有空闲的UAV时，才可以开始推进时间
                next_t_derta = min([u.free_t_idx for u in s.uavs])
                # next_t = min(s.t_idx + next_t_derta, self.cfg.horizon_steps - 1)
                next_t = s.t_idx + next_t_derta
                if next_t > self.cfg.horizon_steps - 1:
                    return s, -10

                uavs2 = [UAV(pos=x.pos, battery_level=x.battery_level-next_t_derta,
                             free_t_idx=x.free_t_idx-next_t_derta) for x in s.uavs]

                which_uav_free = [i for i, u in enumerate(uavs2) if u.free_t_idx == 0]
                r = 0.0
                served2 = s.served.copy()
                for free_uav in which_uav_free:
                    served_vessel = uavs2[free_uav].pos - len(self.cfg.base_set)
                    # 到达时刻真实位置
                    dest = self.traj_mean[served_vessel, next_t]
                    true_pos = self.traj_true[served_vessel, next_t]
                    hit = (_euclid(true_pos, dest) <= self.cfg.hit_radius_km_drones)  # 最简命中；你可以改为 UAV 传感半径判定
                    # print('hit', hit)

                    if served_vessel < 0:
                        continue

                    if hit and (not served2[served_vessel]):
                        r += float(self.reward[served_vessel])
                        served2[served_vessel] = True

                s2 = WorldState(t_idx=next_t, uavs=uavs2, served=served2, vessel_traj=self.traj_mean)
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