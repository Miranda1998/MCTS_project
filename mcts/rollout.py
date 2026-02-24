# mcts/rollout.py
from __future__ import annotations
from typing import List, Optional, Callable, Dict, Tuple
import random
import math
import numpy as np
from tqdm import tqdm
from .types import WorldState, Action
from .env import OnlineEnv


# -----------------------------
# helpers
# -----------------------------
def _is_base(env: OnlineEnv, node_id: int) -> bool:
    # 用集合判断最稳（base_set 不一定是 0..B-1）
    return node_id in set(env.cfg.base_set)

def _to_vessel_index(env: OnlineEnv, node_id: int) -> Optional[int]:
    """
    将 node_id 映射到真实船舶索引 v (0..N-1)。
    - 若是基站：返回 None
    - 若是船舶节点：返回 node_id - B
    """
    if _is_base(env, node_id):
        return None
    B = len(env.cfg.base_set)
    return node_id - B

def _finish_time(env: OnlineEnv, a: Action) -> int:
    # 任务完成时刻：观测时刻 + dwell
    return int(a.ob_time_idx + env.cfg.uav_obs_dwell_steps)

def _immediate_reward_proxy(env: OnlineEnv, s: WorldState, a: Action) -> float:
    """
    rollout 内的“即时收益代理”：
    - 去基站：0
    - 去未服务船：reward[v]
    注意：真实收益在 env.step_real 中等 UAV 完成后才结算；
         这里用于“近似DP打分”，不改变状态。
    """
    v = _to_vessel_index(env, a.vessel_id)
    if v is None:
        return 0.0
    if s.served[v]:
        return 0.0
    return float(env.reward[v])


# 为了避免在 ADP 中反复扫描 feasible_pair，做一个轻量 cache：
# key: (id(env), from_node, start_t) -> list of reachable (to_node, earliest_obt)
_REACH_CACHE: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = {}

def _reachable_nodes_earliest(env: OnlineEnv, from_node: int, start_t: int) -> List[Tuple[int, int]]:
    """
    从 (from_node, start_t) 出发，在 feasible_pair 中找所有可达 to_node 的最早到达时刻 obt。
    返回 [(to_node, obt), ...]。

    注意：
    - 这是“轻量索引”，最坏情况仍然会慢；
    - 但它会被 cache，且 rollout_depth 通常不大，所以实际可接受。
    """
    key = (id(env), from_node, start_t)
    if key in _REACH_CACHE:
        return _REACH_CACHE[key]

    # feasible_pair 存的是 (i, k, j, l)
    # 我们要固定 i=from_node,k=start_t，找所有 (j,l)
    out: Dict[int, int] = {}
    fp = env.cfg.feasible_pair

    # 朴素遍历 set 会很慢；但你目前未提供更强索引结构，只能这样。
    # 建议后续把 feasible_pair 预处理成 dict[(i,k)] -> list[(j,l)]，速度会大幅提升。
    for (i, k, j, l) in fp:
        if i == from_node and k == start_t:
            prev = out.get(j, None)
            if prev is None or l < prev:
                out[j] = l

    res = [(j, l) for j, l in out.items()]
    _REACH_CACHE[key] = res
    return res


# -----------------------------
# baseline rollouts
# -----------------------------
def rollout_policy_random(env: OnlineEnv, s: WorldState, rng: random.Random) -> Optional[Action]:
    acts = env.available_actions(s)
    return rng.choice(acts) if acts else None


def rollout_policy_eps_greedy(
    env: OnlineEnv,
    s: WorldState,
    rng: random.Random,
    epsilon: float = 0.10,
) -> Optional[Action]:
    """
    ε-greedy baseline:
    - prob ε: random action
    - prob 1-ε: greedy by reward proxy (ignore travel time, keep it very fast)
    """
    acts = env.available_actions(s)
    if not acts:
        return None

    if rng.random() < epsilon:
        return rng.choice(acts)

    best = None
    best_val = -1e18
    for a in acts:
        v = _to_vessel_index(env, a.vessel_id)
        if v is None:
            val = 0.0
        else:
            if s.served[v]:
                val = 0.0
            else:
                val = float(env.reward[v])
        if val > best_val:
            best_val = val
            best = a

    return best if best is not None else rng.choice(acts)


# -----------------------------
# proposed rollout: ADP
# ----------------------------

def rollout_policy_my_adp(env: OnlineEnv, s: WorldState, rng: random.Random) -> Optional[Action]:

    acts = env.available_actions(s)
    if not acts:
        return None

    next_actions = []

    idle_uavs = set(a.uav_id for a in acts)

    for uid in idle_uavs:
        pos = s.uavs[uid].pos
        t = s.t_idx
        next_pos, next_t, final_route_u = adp(env, s, pos, t, top_k=10)
        if final_route_u is not None:
            next_actions.append(Action(uav_id=uid, vessel_id=next_pos, ob_time_idx=next_t))

    if len(next_actions) != 0:
        a_best = random.choice(next_actions)
        # print("len(next_actions)", len(next_actions))
    else:
        a_best = random.choice(acts)
        # print("random choice!")
    return a_best


def adp_old(env, s, pos, t):
    B = len(env.cfg.base_set)
    all_node = env.cfg.vessels_num + B
    gamma = env.reward

    T = env.cfg.horizon_steps
    V_ik = [[0 for _ in range(T)] for _ in range(all_node)]
    sub_tour_end_at_ik = [[[] for _ in range(T)] for _ in range(all_node)]
    sub_tour_visited = [[[] for _ in range(T)] for _ in range(all_node)]

    # ✅ 只放未服务的船舶节点（node_id = B + vessel_idx）
    rest_task = [B + i for i in range(env.cfg.vessels_num) if s.served[i] == False]

    final_route_u = None

    # 反向遍历
    for k in range(int(T - 1), t, -1):
        for i in rest_task:

            # ---------- (1) 如果从 i,k 可以回到某个基地 T-1，则初始化一个可收尾的子路径 ----------
            # ✅ i 一定是船舶节点（>=B），所以 vessel_id 安全
            for base_id in env.cfg.base_set:
                if (i, k, base_id, T - 1) in env.cfg.feasible_pair:
                    vessel_id = i - B
                    V_ik[i][k] += gamma[vessel_id]
                    sub_tour_end_at_ik[i][k] = [(i, k)]
                    sub_tour_visited[i][k].append(i)
                    break

            V_ik_best = 0
            final_j = None
            final_l = None

            # ---------- (2) 拼接后续最优 suffix ----------
            for j in rest_task:
                for l in range(k + 1, T):

                    # ✅ visited 去重只对船舶节点进行（这里 i 肯定>=B，但我按你要求写清楚）
                    if i >= B and i in sub_tour_visited[j][l]:
                        continue

                    if (i, k, j, l) in env.cfg.feasible_pair:
                        profit_sum_jl = 0.0
                        for (j1, l1) in sub_tour_end_at_ik[j][l]:
                            # ✅ reward 只对船舶节点计算
                            if j1 >= B:
                                vessel_id = j1 - B
                                profit_sum_jl += gamma[vessel_id]
                        V = V_ik[i][k] + profit_sum_jl
                    else:
                        V = -np.inf

                    if V >= V_ik_best:
                        V_ik_best = V
                        final_j = j
                        final_l = l

            # ---------- (3) 如果找到可拼接的 suffix，就更新 i,k 的最优子路径 ----------
            if final_j is not None and len(sub_tour_end_at_ik[final_j][final_l]) != 0:
                sub_tour_end_at_ik[i][k] = [(i, k)]
                sub_tour_visited[i][k] = [i]

                # ✅ i 是船舶节点，reward 安全
                vessel_id = i - B
                V_ik[i][k] = gamma[vessel_id]

                for (j1, l1) in sub_tour_end_at_ik[final_j][final_l]:
                    sub_tour_end_at_ik[i][k].append((j1, l1))
                    sub_tour_visited[i][k].append(j1)
                    if j1 >= B:
                        V_ik[i][k] += gamma[j1 - B]

    # ---------- (4) 从当前 (pos,t) 连接到某个 (j,l) ----------
    V00 = 0.0
    for j in rest_task:
        for l in range(1, T):
            i = pos
            k = t

            if (i, k, j, l) in env.cfg.feasible_pair:
                V = 0.0
                for (j1, l1) in sub_tour_end_at_ik[j][l]:
                    if j1 >= B:
                        V += gamma[j1 - B]
            else:
                V = -np.inf

            if V > V00:
                final_route_u = [(i, k)]
                final_route_u.extend(sub_tour_end_at_ik[j][l])
                V00 = V

    if final_route_u and len(final_route_u) >= 2:
        next_pos, next_t = final_route_u[1]
    else:
        next_pos, next_t = None, None

    return next_pos, next_t, final_route_u


def adp(env, s, pos, t, top_k: int = 10):
    B = len(env.cfg.base_set)
    all_node = env.cfg.vessels_num + B
    gamma = env.reward  # length = vessels_num

    T = env.cfg.horizon_steps
    V_ik = [[0 for _ in range(T)] for _ in range(all_node)]
    sub_tour_end_at_ik = [[[] for _ in range(T)] for _ in range(all_node)]
    sub_tour_visited = [[[] for _ in range(T)] for _ in range(all_node)]

    # ✅ 只选 “未服务 + reward Top-K” 的船舶节点（node_id = B + vessel_idx）
    unserved = [v for v in range(env.cfg.vessels_num) if not s.served[v]]
    if not unserved:
        return None, None, None

    # 排序取 Top-K（reward 越大越优先）
    unserved.sort(key=lambda v: float(gamma[v]), reverse=True)
    if top_k is None or top_k <= 0:
        top_k = len(unserved)
    selected = unserved[: min(top_k, len(unserved))]

    rest_task = [B + v for v in selected]  # 只包含船舶节点，不包含 base
    final_route_u = None

    # 反向遍历
    for k in range(int(T - 1), t, -1):
        for i in rest_task:

            # (1) 若 (i,k) 可回到某基地 (base, T-1)，初始化可收尾路径
            for base_id in env.cfg.base_set:
                if (i, k, base_id, T - 1) in env.cfg.feasible_pair:
                    vessel_id = i - B  # safe because i in rest_task => i>=B
                    V_ik[i][k] = float(gamma[vessel_id])
                    sub_tour_end_at_ik[i][k] = [(i, k)]
                    sub_tour_visited[i][k] = [i]
                    break

            V_ik_best = V_ik[i][k] if sub_tour_end_at_ik[i][k] else -np.inf
            final_j = None
            final_l = None

            # (2) 拼接后续 suffix（仍然按你原逻辑）
            for j in rest_task:
                for l in range(k + 1, T):

                    if sub_tour_end_at_ik[j][l] == []:
                        continue
                    if i in sub_tour_visited[j][l]:
                        continue

                    if (i, k, j, l) in env.cfg.feasible_pair:
                        profit_sum_jl = 0.0
                        for (j1, l1) in sub_tour_end_at_ik[j][l]:
                            if j1 >= B:
                                profit_sum_jl += float(gamma[j1 - B])

                        V = float(gamma[i - B]) + profit_sum_jl
                    else:
                        V = -np.inf

                    if V >= V_ik_best:
                        V_ik_best = V
                        final_j = j
                        final_l = l

            # (3) 更新 i,k 的最优子路径
            if final_j is not None and len(sub_tour_end_at_ik[final_j][final_l]) != 0:
                sub_tour_end_at_ik[i][k] = [(i, k)]
                sub_tour_visited[i][k] = [i]
                V_ik[i][k] = float(gamma[i - B])

                for (j1, l1) in sub_tour_end_at_ik[final_j][final_l]:
                    sub_tour_end_at_ik[i][k].append((j1, l1))
                    sub_tour_visited[i][k].append(j1)
                    if j1 >= B:
                        V_ik[i][k] += float(gamma[j1 - B])

    # (4) 从当前 (pos,t) 连接到某个 (j,l)
    V00 = 0.0
    for j in rest_task:
        for l in range(1, T):
            i = pos
            k = t

            if sub_tour_end_at_ik[j][l] == []:
                continue

            if (i, k, j, l) in env.cfg.feasible_pair:
                V = 0.0
                for (j1, l1) in sub_tour_end_at_ik[j][l]:
                    if j1 >= B:
                        V += float(gamma[j1 - B])
            else:
                V = -np.inf

            if V > V00:
                final_route_u = [(i, k)]
                final_route_u.extend(sub_tour_end_at_ik[j][l])
                V00 = V

    if final_route_u and len(final_route_u) >= 2:
        next_pos, next_t = final_route_u[1]
    else:
        next_pos, next_t = None, None

    return next_pos, next_t, final_route_u


def get_rollout_fn(name: str):
    """
    rollout_fn signature must be: (env, s, rng) -> Optional[Action]
    """
    if name == "random":
        return rollout_policy_random
    if name == "eps_greedy":
        return rollout_policy_eps_greedy
    if name == "adp":
        return rollout_policy_my_adp

    # keep your existing ones if you still want them
    if name == "greedy":
        # reuse eps_greedy with epsilon=0.0 behavior (but keep exactly your old greedy if needed)
        def _greedy(env: OnlineEnv, s: WorldState, rng: random.Random) -> Optional[Action]:
            return rollout_policy_eps_greedy(env, s, rng, epsilon=0.0)
        return _greedy

    raise ValueError(f"Unknown rollout policy: {name}")