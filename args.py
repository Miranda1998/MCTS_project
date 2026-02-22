# args.py
import argparse
import os

rou0 = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description="HyperHeuristic Experiment")

    # ===== 原有参数（保留）=====
    parser.add_argument("--Timelimit", type=int, default=3600, help="Set the time limit for the solver")
    parser.add_argument("--rouReward", type=str, default=os.path.join(rou0, 'data', 'Reward.xlsx'))
    parser.add_argument("--rouSPs", type=str, default=os.path.join(rou0, 'data', 'SPs_info.json'))
    parser.add_argument("--rouGrid", type=str, default=os.path.join(rou0, 'data_AEOS'))
    parser.add_argument("--rouFeasiblePairs", type=str, default=os.path.join(rou0, 'feasible_pair'))
    parser.add_argument("--MIPgap", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument("--lambda_", type=float, default=5.0)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="admm_dp_result.json")

    # ===== 第四个工作：MCTS 在线层新增参数 =====
    parser.add_argument("--mode", type=str, default="mcts",
                        choices=["mcts", "admm_dp"], help="Run mode")

    # 轨迹输入（你说：真实轨迹、期望轨迹、m个情景集）
    parser.add_argument("--rouTrajTrue", type=str, default=os.path.join(rou0, "data_traj", "traj_true.npy"),
                        help="shape: [N, K, 2] or dict-like npy")
    parser.add_argument("--rouTrajMean", type=str, default=os.path.join(rou0, "data_traj", "traj_mean.npy"),
                        help="shape: [N, K, 2]")
    parser.add_argument("--rouScenarios", type=str, default=os.path.join(rou0, "data_traj", "scenarios.npy"),
                        help="shape: [M, N, K, 2] or a folder of npy files")

    # 时间参数
    parser.add_argument("--slot_gap_min", type=float, default=1.0, help="minutes per step")
    parser.add_argument("--horizon_hours", type=float, default=6.0, help="online horizon (hours)")

    # 信息更新机制
    parser.add_argument("--info_delay_min", type=float, default=10.0, help="tau: minutes")
    parser.add_argument("--hit_radius_km_aeos", type=float, default=25.0)
    parser.add_argument("--hit_radius_km_airship", type=float, default=30.0)

    # UAV在线层
    parser.add_argument("--uav_num", type=int, default=4)
    parser.add_argument("--uav_speed_kmph", type=float, default=80.0)
    parser.add_argument("--uav_obs_dwell_min", type=float, default=1.0, help="time to perform a monitoring")

    # MCTS 参数
    parser.add_argument("--mcts_budget_ms", type=int, default=200, help="per decision time budget")
    parser.add_argument("--mcts_iters", type=int, default=0, help="if >0, use fixed iterations instead of time")
    parser.add_argument("--ucb_c", type=float, default=1.4)
    parser.add_argument("--rollout_depth", type=int, default=25)
    parser.add_argument("--rollout_policy", type=str, default="greedy",
                        choices=["random", "greedy", "lookahead2"])

    # 实验输出
    parser.add_argument("--exp_name", type=str, default="mcts_online_eval")
    parser.add_argument("--repeat", type=int, default=10)

    return parser.parse_args()