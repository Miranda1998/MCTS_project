# main.py
from datetime import datetime
import os
import time
import sys
from openpyxl import Workbook
from args import parse_args
from data_loader import Load_Data, Load_Trajectories

# ===== 你的 ADMM 相关仍可保留 =====
# from ADMM import *
# from ADMM_LNS import *

from mcts.env import OnlineEnv, OnlineEnvConfig
from mcts.mcts_core import MCTSPlanner
from mcts.scenario_updater import ScenarioUpdater
from mcts.evaluator import run_episode

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

def _ensure_header_mcts(ws):
    if ws.max_row == 1 and ws["A1"].value is None:
        ws.append([
            "exp_name",
            "rollout_policy",
            "N(vessels)",
            "K(steps)",
            "M(scenarios)",
            "uav_num",
            "tau(min)",
            "budget(ms)",
            "repeat_id",
            "total_reward",
            "served_cnt",
            "decision_cnt",
            "walltime(s)"
        ])

def run_mcts_group(args, wb, rou0, date_today):
    ws_name = "MCTS_result"
    ws = wb[ws_name] if ws_name in wb.sheetnames else wb.create_sheet(ws_name)
    _ensure_header_mcts(ws)

    # load trajectories
    traj_true, traj_mean, scenarios = Load_Trajectories(args, n_vessels=None)
    N, K, _ = traj_true.shape
    M = scenarios.shape[0]

    # reward：若你已有 Reward.xlsx，可在此读取；这里先给占位：全 1
    reward = (1.0 + 0.0 * (os.urandom(1)[0])) * (1.0)  # avoid lint
    reward_vec = (1.0 + 0.0) * (1.0)
    reward_arr = (1.0 + 0.0) * (1.0)
    import numpy as np
    reward = np.ones(N, dtype=float)

    horizon_steps = int(args.horizon_hours * 60.0 / args.slot_gap_min)
    horizon_steps = min(horizon_steps, K)

    info_delay_steps = int(round(args.info_delay_min / args.slot_gap_min))
    uav_obs_dwell_steps = int(round(args.uav_obs_dwell_min / args.slot_gap_min))

    cfg = OnlineEnvConfig(
        slot_gap_min=float(args.slot_gap_min),
        horizon_steps=int(horizon_steps),
        info_delay_steps=int(info_delay_steps),
        uav_speed_kmph=float(args.uav_speed_kmph),
        uav_obs_dwell_steps=int(uav_obs_dwell_steps),
        hit_radius_km_aeos=float(args.hit_radius_km_aeos),
        hit_radius_km_airship=float(args.hit_radius_km_airship),
    )

    env = OnlineEnv(
        traj_true=traj_true,
        traj_mean=traj_mean,
        scenarios=scenarios,
        reward=reward,
        cfg=cfg,
        normal_plan=None  # TODO: 把 ADMM-LNS 输出的常态化监测计划塞进来
    )
    env.set_uav_num(args.uav_num)

    # 观测一致性门限：简单取 AEOS 半径或其比例
    gate = float(args.hit_radius_km_aeos)
    updater = ScenarioUpdater(scenarios=scenarios, gate_km=gate)

    print(f"[MCTS] N={N}, K={horizon_steps}, M={M}, uav={args.uav_num}, tau={args.info_delay_min}min")

    for rep in range(args.repeat):
        t0 = time.perf_counter()
        planner = MCTSPlanner(
            env=env,
            ucb_c=float(args.ucb_c),
            rollout_depth=int(args.rollout_depth),
            rollout_policy=str(args.rollout_policy),
            seed=int(args.seed) + rep
        )
        res = run_episode(
            env=env,
            planner=planner,
            updater=updater,
            budget_ms=int(args.mcts_budget_ms),
            iters=int(args.mcts_iters)
        )
        t1 = time.perf_counter()
        ws.append([
            str(args.exp_name),
            str(args.rollout_policy),
            int(N),
            int(horizon_steps),
            int(M),
            int(args.uav_num),
            float(args.info_delay_min),
            int(args.mcts_budget_ms if args.mcts_iters == 0 else args.mcts_iters),
            int(rep),
            float(res.total_reward),
            int(res.served_cnt),
            int(res.decision_cnt),
            float(t1 - t0)
        ])
        print(f"[rep={rep}] reward={res.total_reward:.4f}, served={res.served_cnt}, decisions={res.decision_cnt}, time={t1-t0:.3f}s")

def main():
    args = parse_args()
    rou0 = os.getcwd()
    wb = Workbook()
    if "Sheet" in wb.sheetnames and len(wb.sheetnames) == 1:
        wb.remove(wb["Sheet"])

    date_now = datetime.now().strftime("%Y_%m%d_%H%M") + "_" + str(args.exp_name)
    result_dir = os.path.join(rou0, 'result', date_now)
    os.makedirs(result_dir, exist_ok=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_path = os.path.join(result_dir, f"exp_{stamp}_{args.mode}.log")
    xlsx_path = os.path.join(result_dir, f"exp_{stamp}_{args.mode}.xlsx")

    old_stdout = sys.stdout
    with open(txt_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(old_stdout, f)
        try:
            print(f"[RUN] mode={args.mode}, exp={args.exp_name}")
            if args.mode == "mcts":
                run_mcts_group(args, wb, rou0, date_today=date_now)
            else:
                # 你的原 ADMM_DP 主流程仍可放这里（保持兼容）
                print("[WARN] admm_dp mode not wired in this template.")
            wb.save(xlsx_path)
            print(f"[OK] wrote excel: {xlsx_path}")
            print(f"[OK] wrote log:   {txt_path}")
        finally:
            sys.stdout = old_stdout
            wb.save(xlsx_path)

if __name__ == '__main__':
    main()