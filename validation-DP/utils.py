import copy
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable, DefaultDict
import math
from copy import deepcopy
from dataclasses import dataclass, field
from collections import defaultdict


Arc = Tuple[int, int, int, int]  # (i, k, j, l)

@dataclass
class CSHEOSPInstance:
    # horizon
    T: int
    slot_gap: int

    # resources
    drone_battery: int
    drone_base_num: int

    # tasks
    vessels_num: int
    reward: List[float]

    # feasible arcs
    uav_arcs: List[Arc]


@dataclass
class PathResult:
    value: float
    path: List[Tuple[int, int]]              # list of states (node_id,time)
    selected_vessels: Set[int]


def relative_gap(ub, lb) :
    """Relative gap for maximization: max(0, (UB-LB)/max(eps, |UB|))."""
    if ub is None or not math.isfinite(ub):
        return float("inf")
    denom = max(1e-9, abs(ub))
    return max(0.0, (ub - lb) / denom)


def compute_platform_gammas(inst, lamb, rho, z_aeos, z_air, z_uav):
    """
    Builds modified rewards (修正收益) for each platform/vessel.

    Since the exact equation numbers (5-31/5-34/5-36) are in your doc, but the docx excerpt
    does not expose the full math, we implement a standard "discourage duplicates" shaping:

        gamma_{p,v} = r_v - lamb_v - rho * (others_count_{p,v})

    where others_count is how many *other* platforms currently select vessel v.

    This matches the ADMM narrative of suppressing cross-platform repeated monitoring.
    """
    V = inst.vessels_num
    r = inst.reward

    # precompute total counts
    total = [0]*V
    for S in z_aeos:
        for v in S: total[v]+=1
    for S in z_air:
        for v in S: total[v]+=1
    for S in z_uav:
        for v in S: total[v]+=1

    gamma_aeos=[]
    for s, S in enumerate(z_aeos):
        g=[0.0]*V
        for v in range(V):
            others = total[v] - (1 if v in S else 0)
            g[v] = r[v] - lamb[v] - rho*others + 0.5*rho
        gamma_aeos.append(g)

    gamma_air=[]
    for a, S in enumerate(z_air):
        g=[0.0]*V
        for v in range(V):
            others = total[v] - (1 if v in S else 0)
            g[v] = r[v] - lamb[v] - rho*others + 0.5*rho
        gamma_air.append(g)

    gamma_uav=[]
    for d, S in enumerate(z_uav):
        g=[0.0]*V
        for v in range(V):
            others = total[v] - (1 if v in S else 0)
            g[v] = r[v] - lamb[v] - rho*others + 0.5*rho
        gamma_uav.append(g)

    return gamma_aeos, gamma_air, gamma_uav




def objective_from_repaired_sol(inst, repaired_picks, params):
    obj = 0.0
    # rewards (duplicates counted multiple times)
    for typ, lists in repaired_picks.items():
        for S in lists:
            for v in S:
                obj += inst.reward[v]

    # fixed costs per used platform
    used_aeos = sum(1 for S in repaired_picks.get("AEOS", []) if S)
    used_air  = sum(1 for S in repaired_picks.get("AIR",  []) if S)
    used_uav  = sum(1 for S in repaired_picks.get("UAV",  []) if S)

    obj -= params.fixed_cost_aeos * used_aeos
    obj -= params.fixed_cost_airship * used_air
    obj -= params.fixed_cost_uav * used_uav
    return obj

def lagrangian_value(inst, picks, lamb, params):
    V = inst.vessels_num
    total = [0]*V
    for typ in ("AEOS","AIR","UAV"):
        for S in picks.get(typ, []):
            for v in S:
                if 0 <= v < V:
                    total[v] += 1

    fx = relaxed_objective(inst, picks, params)
    # L = f - sum λ (total-1)
    L = fx - sum(lamb[v]*(total[v]-1) for v in range(V))
    return L


def get_bat_enough(T, k, sub_tour_end_at_jl, drone_battery, drone_base_num):
    if_bat_enough = False
    last_base_t = T - 1
    for (j, l) in sub_tour_end_at_jl:
        if j < drone_base_num:
            last_base_t = min(last_base_t, l)
            break

    if last_base_t - k < drone_battery:
        if_bat_enough = True

    return if_bat_enough

def relaxed_objective(inst, picks, params):
    """Relaxed objective: allow duplicate vessel coverage across platforms.
    Rewards are counted per platform pick; fixed costs apply per used platform (non-empty pick set).
    """
    obj = 0.0
    # rewards (duplicates counted multiple times)
    for typ, lists in picks.items():
        for S in lists:
            for v in S:
                obj += inst.reward[v]

    # fixed costs per used platform
    used_aeos = sum(1 for S in picks.get("AEOS", []) if S)
    used_air  = sum(1 for S in picks.get("AIR",  []) if S)
    used_uav  = sum(1 for S in picks.get("UAV",  []) if S)

    obj -= params.fixed_cost_aeos * used_aeos
    obj -= params.fixed_cost_airship * used_air
    obj -= params.fixed_cost_uav * used_uav
    return obj


def get_fea_grid_time_slot(vessel_coverage_set, grid_set, slot_gap, T):

    fea_grid_time_slot = [[] for t in range(T)]
    for (grid_id, t) in grid_set:
        fea_grid_time_slot[t].append(grid_id)
    return fea_grid_time_slot

def invert_fea_grid_time_slot(fea_grid_time_slot, T, all_grid):

    fea_time_slot_grid = {grid: [] for grid in all_grid}
    for t in range(T):
        for grid_id in fea_grid_time_slot[t]:
            fea_time_slot_grid[(grid_id)].append(t)

    return fea_time_slot_grid


def build_airship_fea_task_set(inst, fea_pair, topk_reward=30):
    B = inst.airship_base_num
    V = inst.vessels_num
    T = inst.T
    task_nodes = set(range(B, B + V))

    reachable_from_base = set()  # tasks that some base can reach (at any time)
    can_return_to_base = set()   # tasks that can return to a base at terminal

    for (u, tu, v, tv) in fea_pair:
        # base -> task
        if u < B and v in task_nodes:
            reachable_from_base.add(v)
        # task -> base at terminal
        if u in task_nodes and v < B and tv == T - 1:
            can_return_to_base.add(u)

    # 1) feasibility filter (出得去 + 回得来)
    fea_task_nodes = reachable_from_base & can_return_to_base

    # 2) reward filter inside feasible set: keep top-k by inst.reward
    # NOTE: task node id -> vessel id = node - B
    if topk_reward is not None and topk_reward > 0 and len(fea_task_nodes) > topk_reward:
        # sort by reward descending
        fea_task_nodes = sorted(
            fea_task_nodes,
            key=lambda n: inst.reward[n - B],
            reverse=True
        )[:topk_reward]
        return sorted(fea_task_nodes)  # keep output stable (ascending node id)

    return sorted(fea_task_nodes)


def build_airship_tasks_at_time(inst, fea_pair, fea_task_nodes):
    B = inst.airship_base_num
    T = inst.T
    fea_task_set = set(fea_task_nodes)

    tasks_at_time = [set() for _ in range(T)]
    for (u, tu, v, tv) in fea_pair:
        if u in fea_task_set:
            tasks_at_time[tu].add(u)
        if v in fea_task_set:
            tasks_at_time[tv].add(v)
    # 转成 list 便于迭代
    tasks_at_time = [list(s) for s in tasks_at_time]
    return tasks_at_time


def compute_primal_residual(total):
    # r = ||max(0, total-1)||_2
    return math.sqrt(sum((max(0, c - 1)) ** 2 for c in total))

def compute_dual_residual(total, total_prev, rho):
    # s = rho * || total - total_prev ||_2
    if total_prev is None:
        return 0.0
    return rho * math.sqrt(sum((total[i] - total_prev[i]) ** 2 for i in range(len(total))))

def update_rho_residual_balancing(rho, r, s, mu=10.0, tau=2.0, rho_min=1e-4, rho_max=1e4):
    if r > mu * s:
        rho = min(rho_max, rho * tau)
    elif s > mu * r:
        rho = max(rho_min, rho / tau)
    return rho


