import copy
import random
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from utils import *
from DP import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

@dataclass
class ADMMParams:
    lambda_: float = 0
    rho: float = 0
    max_iter: int = 50
    tol: float = 1e-3
    seed: int = 0

    # stopping criterion on relative UB-LB gap
    gap_tol: float = 0.01

    # fixed costs approximating your MILP objective (-10/-20/-30)
    fixed_cost_aeos: float = 0
    fixed_cost_airship: float = 0
    fixed_cost_uav: float = 0

    time_limit: int = 3600

    # --- LNS ---
    lns_enable: bool = True
    lns_trigger_no_improve: int = 10  # no_improve >= 该阈值时触发
    lns_m: int = 1  # destroy: 最少移除资产数
    lns_n: int = 1  # destroy: 最多移除资产数
    lns_delta: float = 8.0  # gamma 额外奖励
    lns_trials: int = 100  # 触发一次时，尝试几次不同随机destroy
    delta_min: int = 7
    delta_max: int = 9
    lns_removed_platforms: set = field(default_factory=set)  # 用于记录已经去除的平台，防止重复去除

    lns_max_destroy_unique: int = 4


    def __post_init__(self):
        # 确保 lns_removed_platforms 是一个空的 set
        if self.lns_removed_platforms is None:
            self.lns_removed_platforms = set()

@dataclass
class ADMMResult:
    best_obj: float
    best_sol: List
    history: List[Dict[str, float]]


def solve_admm_dp_serial_gs(inst, aeos_num, airships_num, drones_num, params, start_t):
    random.seed(params.seed)
    V = inst.vessels_num
    lamb = [params.lambda_] * V

    z_aeos = [set() for _ in range(aeos_num)]
    z_air  = [set() for _ in range(airships_num)]
    z_uav  = [set() for _ in range(drones_num)]

    UB = float("inf")
    LB = float("-inf")
    history = []
    best_sol = []
    best_picks = None
    best_obj = float("-inf")
    no_improve = 0
    last_LB = float("-inf")

    total_prev = None

    for it in range(1, params.max_iter + 1):
        # ---------- (1~2) Gauss–Seidel: solve one asset, then refresh gammas ----------
        path_aeos = [None] * aeos_num
        path_air  = [None] * airships_num
        path_uav  = [None] * drones_num

        # UAV assets (serial)
        for d in range(drones_num):
            gamma_aeos, gamma_air, gamma_uav = compute_platform_gammas(inst, lamb, params.rho, z_aeos, z_air, z_uav)
            res = solve_uav_dp(inst, inst.uav_arcs, gamma_uav[d], inst.drone_battery, params)
            z_uav[d] = set(res.selected_vessels)
            path_uav[d] = res.path

        # ---------- (3) compute totals ----------
        total = [0] * V
        for S in z_aeos:
            for v in S:
                if 0 <= v < V: total[v] += 1
        for S in z_air:
            for v in S:
                if 0 <= v < V: total[v] += 1
        for S in z_uav:
            for v in S:
                if 0 <= v < V: total[v] += 1

        picks = {"AEOS": z_aeos, "AIR": z_air, "UAV": z_uav}


        # ---------- (5) lambda update (projected subgradient) ----------
        max_dlambda = 0.0
        for v in range(V):
            subgrad = total[v] - 1
            new_l = lamb[v] + params.rho * subgrad
            if new_l < 0.0:
                new_l = 0.0
            dl = abs(new_l - lamb[v])
            if dl > max_dlambda:
                max_dlambda = dl
            lamb[v] = new_l

        # rho update based on residual balancing
        r = compute_primal_residual(total)
        s = compute_dual_residual(total, total_prev, params.rho)
        params.rho = update_rho_residual_balancing(params.rho, r, s, mu=10.0, tau=2.0)
        total_prev = total[:]  # copy

        # ---------- (6) neighborhood search ----------
        sol = [path_aeos, path_air, path_uav]

        # repair（先用你当前最稳定的 repair）
        new_sol, new_picks = repair_feasible_assignment(inst, sol, picks, params)
        print('难道admm中就出错了', new_picks)

        # 重新更新 LB
        obj = objective_from_repaired_sol(inst, new_picks, params)
        print('ADMM-DP求解得到的obj', obj)
        if obj > LB:
            LB = obj
            best_sol = copy.deepcopy(new_sol)
            best_picks = copy.deepcopy(new_picks)
            best_obj = float(LB)
            print('best_obj after repair', best_obj)
            print('best_sol after repair', best_sol)

            # After repair, we need to update z_aeos, z_air, z_uav based on the new picks
            z_aeos, z_air, z_uav = [], [], []

            for a_idx, s in enumerate(new_picks["AEOS"]):
                z_aeos.append(s)
            for a_idx, s in enumerate(new_picks["AIR"]):
                z_air.append(s)
            for a_idx, s in enumerate(new_picks["UAV"]):
                z_uav.append(s)
        # -----------------进入LNS试图改进best_sol-------------------------


        # ----------------------结束LNS---------------------------

        # ---------- (7) history ----------
        gap = relative_gap(UB, LB)
        history.append({
            "iter": float(it),
            "UB": float(UB) if math.isfinite(UB) else float("inf"),
            "LB": float(LB),
            "gap": float(gap),
            "max_dlambda": float(max_dlambda),
            "repaired_obj": float(obj),
            "best_obj": best_obj,
        })
        print(f"[SERIAL-GS] Iter {it}: LB={LB:.4f} gap={gap:.6f} max_dlambda={max_dlambda:.4e}")

        # ---------- stop ----------
        # ---------- stop / trigger LNS ----------
        if LB > last_LB:
            no_improve = 0
            last_LB = LB
        else:
            no_improve += 1


        end_t = time.perf_counter()
        if end_t - start_t > params.time_limit:
            print(f"[SERIAL-GS] Time limit reached. Stopping ADMM.")
            break

    return ADMMResult(best_obj=float(LB), best_sol=best_sol, history=history)
