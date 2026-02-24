from typing import Set
from utils import *
import numpy as np
from tqdm import tqdm
import random
from copy import deepcopy
from copy import deepcopy
NEG = -1e30

def solve_uav_dp(inst, fea_pair, gamma, drone_battery, params):
    """
    UAV DP on grid/time nodes.
    """
    all_node = inst.vessels_num + inst.drone_base_num

    V_ik = [[0 for i in range(inst.T)] for j in range(all_node)]
    sub_tour_end_at_ik = [[[] for i in range(inst.T)] for j in range(all_node)]
    sub_tour_visited = [[[] for i in range(inst.T)] for j in range(all_node)]

    # rest_tasks = [i + inst.drone_base_num for i in range(inst.vessels_num)]
    rest_tasks = [i for i in range(8, inst.vessels_num + inst.drone_base_num)]

    final_route_u = [(0, 0), (0, 0)]

    for k in tqdm(range(int(inst.T - 1), 0, -1)):  # 从 m-1 到 1 反向遍历
        for i in rest_tasks:
            # 如果从船舶到目标节点的路径在可行路径集合中
            for base_id in range(inst.drone_base_num):
                if (i, k, base_id, inst.T - 1) in fea_pair:
                    vessel_id = i - inst.drone_base_num
                    V_ik[i][k] += gamma[vessel_id]  # 设置目标节点的价值为任务的利润
                    sub_tour_end_at_ik[i][k] = [(i, k)]
                    sub_tour_visited[i][k].append(i)
                    break

            V_ik_best = 0
            final_j = None
            # 对每个船舶和时间槽，检查其他船舶和时间槽组合
            for j in rest_tasks:
                for l in range(k + 1, inst.T):
                    if i > inst.drone_base_num and i in sub_tour_visited[j][l]:
                        continue
                    if_bat_enough = get_bat_enough(inst.T, k, sub_tour_end_at_ik[j][l], drone_battery, inst.drone_base_num)

                    if if_bat_enough == False:
                        continue
                    if (i, k, j, l) in fea_pair:
                        profit_sum_jl = 0
                        for (j1, l1) in sub_tour_end_at_ik[j][l]:
                            if j1 > inst.drone_base_num:
                                vessel_id = j1 - inst.drone_base_num
                                profit_sum_jl += gamma[vessel_id]
                        V = V_ik[i][k] + profit_sum_jl
                    else:
                        V = -np.inf

                    # a = random.random()
                    # if V > V_ik_best and a < 0.8:
                    if V >= V_ik_best:
                        V_ik_best = V
                        final_j = j
                        final_l = l

            if final_j and len(sub_tour_end_at_ik[final_j][final_l]) != 0:
                sub_tour_end_at_ik[i][k] = [(i, k)]
                sub_tour_visited[i][k] = [i]
                vessel_id = i - inst.drone_base_num
                V_ik[i][k] = gamma[i - vessel_id]
                for (j1, l1) in sub_tour_end_at_ik[final_j][final_l]:
                    sub_tour_end_at_ik[i][k].append((j1, l1))
                    sub_tour_visited[i][k].append(j1)
                    V_ik[i][k] += gamma[j1 - inst.drone_base_num]
                # V_ik[i][k] = V_ik_best

    V00 = 0
    for j in rest_tasks:
        for l in range(1, inst.T):
            for i in range(inst.drone_base_num):
                for k in range(0, l):
                    if_bat_enough = get_bat_enough(inst.T, k, sub_tour_end_at_ik[j][l], drone_battery,
                                                   inst.drone_base_num)
                    if if_bat_enough == False:
                        continue

                    if (i, k, j, l) in fea_pair:
                        V = V_ik[j][l]
                    else:
                        V = -np.inf
                    if V > V00:
                        final_route_u = [(i, k)]
                        final_route_u.extend(sub_tour_end_at_ik[j][l])
                        V00 = V
                        last_node = sub_tour_end_at_ik[j][l][-1]
                        for base_id in range(inst.drone_base_num):
                            if (last_node[0], last_node[1], base_id, inst.T - 1) in fea_pair:
                                final_route_u.extend([(base_id, inst.T - 1)])

    selected_vessels = {i - inst.drone_base_num for (i, t) in final_route_u if i >= inst.drone_base_num}

    if V00 < params.fixed_cost_uav:
        return PathResult(value=0.0, path=[(0, 0), (0, 0)], selected_vessels=set())
        # return PathResult(value=72.7551587, path=[(8, 0), (21, 5), (26, 7), (8, 12), (32, 21),
        #                                           (30, 27), (27, 29), (8, 35)], selected_vessels={8, 13, 19, 17, 14})


    return PathResult(value=float(V00), path=final_route_u, selected_vessels=selected_vessels)


def repair_feasible_assignment(inst, sol, picks, params=None,rng=None):
    """
    New logic (as requested):
    1) Identify duplicated tasks globally.
    2) For each asset, compute value of remaining tasks AFTER de-duplication
       (interpreted as keeping only NON-duplicated tasks for that asset).
       If remaining_value < fixed_cost => clear ALL tasks of that asset.
    3) If NO asset is cleared in step (2), then perform classic de-duplication:
       keep duplicated task on lower-cost platform, remove from higher-cost.
    4) For feasibility, after step (2) if any asset cleared, we still perform a
       de-duplication pass on the remaining picks to ensure each task appears once.
    """

    # ----- fixed costs -----
    if params is not None:
        sp_cost = {
            "AEOS": float(getattr(params, "fixed_cost_aeos", 10.0)),
            "AIR":  float(getattr(params, "fixed_cost_airship", 20.0)),
            "UAV":  float(getattr(params, "fixed_cost_uav", 30.0)),
        }
    else:
        sp_cost = {"AEOS": 10.0, "AIR": 20.0, "UAV": 30.0}

    platforms = ["AEOS", "AIR", "UAV"]  # must match sol order

    new_sol = deepcopy(sol)
    new_picks = deepcopy(picks)

    # ========== Step 1: find duplicates ==========
    task2owners = {}
    for p in platforms:
        for a_idx, s in enumerate(new_picks.get(p, [])):
            for v in s:
                task2owners.setdefault(v, []).append((p, a_idx))

    duplicated_tasks = {v for v, owners in task2owners.items() if len(owners) > 1}

    # ========== Step 2: asset-level clearing test ==========
    assets_to_clear = []  # list[(p, a_idx)]
    for p in platforms:
        c = sp_cost.get(p, 0.0)
        for a_idx, s in enumerate(new_picks.get(p, [])):
            if not s:
                continue
            # "去重后剩余任务"：按你的表述，这里保留非重复任务（重复任务视为会在去重中丢失）
            remaining = [v for v in s if v not in duplicated_tasks]
            remaining_val = sum(inst.reward[v] for v in remaining)
            if remaining_val < c:
                assets_to_clear.append((p, a_idx))

    any_cleared = len(assets_to_clear) > 0
    if any_cleared:
        for p, a_idx in assets_to_clear:
            new_picks[p][a_idx].clear()

    # ========== Step 3: only if no clearing, do classic cost-priority de-dup ==========
    # ========== Step 4: if clearing happened, still need a de-dup pass for feasibility ==========
    # if rng is None:
    #     # 保证可复现：默认用 params.seed
    #     rng = random.Random(getattr(params, "seed", 0) + 99991)

    task2owners = {}
    for p in platforms:
        for a_idx, s in enumerate(new_picks.get(p, [])):
            for v in s:
                task2owners.setdefault(v, []).append((p, a_idx))

    for v, owners in task2owners.items():
        if len(owners) <= 1:
            continue

        # 1) 先按平台固定成本找最低成本
        min_cost = min(sp_cost.get(p, 10 ** 9) for p, _ in owners)
        best_cost_owners = [oa for oa in owners if sp_cost.get(oa[0], 10 ** 9) == min_cost]

        # 2) 若最低成本对应多个平台类型（理论上你这里不会发生，但写健壮点）
        #    仍然按平台优先级 AEOS < AIR < UAV 选一个平台类型
        best_cost_owners.sort(key=lambda oa: platforms.index(oa[0]))
        best_platform = best_cost_owners[0][0]
        same_platform_owners = [oa for oa in best_cost_owners if oa[0] == best_platform]

        # 3) 同一平台类型内随机保留一个资产（解决你说的 idx 偏置）
        keep_p, keep_a = random.choice(same_platform_owners)

        # 4) 删除其他 owner 上的这个重复任务
        for rm_p, rm_a in owners:
            if (rm_p, rm_a) != (keep_p, keep_a):
                new_picks[rm_p][rm_a].discard(v)

    # ========== prune paths in sol to match new_picks (same style as your original) ==========
    # AEOS union for pruning
    aeos_pick_all = set()
    for s in new_picks.get("AEOS", []):
        aeos_pick_all |= set(s)

    # UAV
    B_uav = inst.drone_base_num
    for a_idx, seq in enumerate(new_sol[2]):
        filtered = []
        for node, t in seq:
            if node < B_uav:  # base
                filtered.append((node, t))
                continue
            v = node - B_uav
            if v in new_picks["UAV"][a_idx]:
                filtered.append((node, t))
        new_sol[2][a_idx] = filtered


    return new_sol, new_picks




