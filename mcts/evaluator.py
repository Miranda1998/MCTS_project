# mcts/evaluator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from .env import OnlineEnv
from .mcts_core import MCTSPlanner
from .scenario_updater import ScenarioUpdater

@dataclass
class EvalResult:
    total_reward: float
    served_cnt: int
    steps: int
    decision_cnt: int

def run_episode(env: OnlineEnv,
                planner: MCTSPlanner,
                updater: ScenarioUpdater,
                budget_ms: int,
                iters: int = 0) -> EvalResult:
    s = env.reset()
    total = 0.0
    decision_cnt = 0

    # 用队列模拟“观测在 t+tau 到达”
    delayed_obs: Dict[int, List] = {}

    while s.t_idx < env.cfg.horizon_steps - 1:
        # 1) 真实环境在此刻执行常态化观测，产生 obs（但延迟到达）
        obs_list = env.generate_observations(s.t_idx)
        if obs_list:
            arrive_t = s.t_idx + env.cfg.info_delay_steps
            delayed_obs.setdefault(arrive_t, []).extend(obs_list)

        # 2) 到达的观测注入仿真环境（更新 scen_mask）
        if s.t_idx in delayed_obs:
            for obs in delayed_obs[s.t_idx]:
                s = updater.apply_observation(s, obs)

        # 3) 事件触发决策：如果有空闲 UAV，则 MCTS 选动作，否则推进到下一事件
        acts = env.available_actions(s)
        if not acts:
            # 没有可动 UAV：推进到下一步
            s = type(s)(t_idx=min(s.t_idx + 1, env.cfg.horizon_steps - 1),
                        uavs=s.uavs, served=s.served, vessel_traj=s.vessel_traj)
            continue

        a = planner.plan(s, budget_ms=budget_ms, iters=iters)
        # if a is None:
        #     s = type(s)(t_idx=min(s.t_idx + 1, env.cfg.horizon_steps - 1),
        #                 uavs=s.uavs, served=s.served, vessel_traj=s.vessel_traj)
        #     continue

        # print('decision_cnt', decision_cnt, 't_idx', s.t_idx, 'action', a)
        # print('decision_cnt', decision_cnt, 's.t_idx', s.t_idx, 's.uavs', s.uavs)
        decision_cnt += 1
        s, r = env.step_real(s, a)
        # print('decision_cnt+1', decision_cnt, 't_idx', s.t_idx, 'action', a)
        # print('decision_cnt+1', decision_cnt, 's.t_idx', s.t_idx, 's.uavs', s.uavs)
        total += float(r)
        print('total', total, 'r', r)

    return EvalResult(
        total_reward=total,
        served_cnt=int(np.sum(s.served)),
        steps=int(s.t_idx),
        decision_cnt=decision_cnt
    )