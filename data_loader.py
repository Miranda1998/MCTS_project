# data_loader.py
from openpyxl import load_workbook
import numpy as np
import os
import json
import pickle

# ===== 你原来的 Load_Data：不要删，保持兼容 =====
def Load_Data(args, vessels_num, AEOS_num, airships_num, drones_num):
    # （原文件内容保持不变）
    with open(args.rouSPs, 'r') as f:
        config = json.load(f)

    ScheduleHorizon = config['ScheduleHorizon']
    slot_gap = config['SlotGap']
    T = int(ScheduleHorizon * (60 / slot_gap))

    drone_battery = config['drone_battery']
    drone_base_num = config['drone_base_num']
    airship_base_num = config['airship_base_num']

    grid_vesselset_path = os.path.join(args.rouGrid, f"vessels_{vessels_num}", f"vessel_set.pkl")
    with open(grid_vesselset_path, "rb") as f:
        vessel_set = pickle.load(f)

    base_dir = os.path.join(args.rouGrid, f"vessels_{vessels_num}", f"grid_set.npy")
    grid_set = np.load(base_dir)

    drone_feasible_path = os.path.join(args.rouFeasiblePairs,
                                       f"new_feasible_pair4drones_20260109_TASK={vessels_num}_T={T}.npy")
    drone_feasible_pair = np.load(drone_feasible_path, allow_pickle=True)

    airship_feasible_path = os.path.join(args.rouFeasiblePairs,
                                         f"added_new_feasible_pair4airships_20260109_TASK={vessels_num}_T={T}.npy")
    airship_feasible_pair = np.load(airship_feasible_path, allow_pickle=True)

    AEOS_feasible_pair_set = []
    for AEOS in range(AEOS_num):
        satellite_feasible_path = os.path.join(
            args.rouFeasiblePairs,
            f"added_ob_du=1_new_feasible_pair20260126_AEOS={AEOS}_TASK={vessels_num}_T={T}.npy"
        )
        satellite_feasible_pair = np.load(satellite_feasible_path, allow_pickle=True)
        AEOS_feasible_pair_set.append(satellite_feasible_pair)

    tpb = load_workbook(filename=args.rouReward)
    tps = tpb.worksheets[0]
    reward = []
    for row_A in range(1, vessels_num + 1):
        a1 = tps.cell(row=row_A, column=1).value
        if a1:
            reward.append(a1)

    AEOS_feasible_pair_set = [set(map(tuple, x.tolist())) for x in AEOS_feasible_pair_set]
    airship_feasible_pair = set(map(tuple, airship_feasible_pair.tolist()))
    drone_feasible_pair = set(map(tuple, drone_feasible_pair.tolist()))

    sp_cost = {"AEOS": 10, "airship": 20, "drone": 30}

    return (T, slot_gap, drone_battery, drone_base_num, airship_base_num,
            vessel_set, grid_set,
            drone_feasible_pair, airship_feasible_pair, AEOS_feasible_pair_set,
            reward, sp_cost)

# ===== 新增：第四个工作 MCTS 在线评估所需轨迹加载 =====
def Load_Trajectories(args, n_vessels=None):
    """
    Returns:
      traj_true:  [N, K, 2]  (km or same coordinate system)
      traj_mean:  [N, K, 2]
      scenarios:  [M, N, K, 2]
    """
    traj_true = np.load(args.rouTrajTrue, allow_pickle=True)
    traj_mean = np.load(args.rouTrajMean, allow_pickle=True)

    # scenarios 可以是一个 npy（[M,N,K,2]）或文件夹（多个 npy）
    if os.path.isdir(args.rouScenarios):
        files = sorted([f for f in os.listdir(args.rouScenarios) if f.endswith(".npy")])
        scen_list = [np.load(os.path.join(args.rouScenarios, f), allow_pickle=True) for f in files]
        scenarios = np.stack(scen_list, axis=0)  # [M, N, K, 2]
    else:
        scenarios = np.load(args.rouScenarios, allow_pickle=True)

    # 若提供 n_vessels，则截取前 n_vessels
    if n_vessels is not None:
        traj_true = traj_true[:n_vessels]
        traj_mean = traj_mean[:n_vessels]
        scenarios = scenarios[:, :n_vessels]

    # 基本形状检查
    if traj_true.ndim != 3 or traj_true.shape[-1] != 2:
        raise ValueError(f"traj_true shape expected [N,K,2], got {traj_true.shape}")
    if traj_mean.shape != traj_true.shape:
        raise ValueError(f"traj_mean must match traj_true shape, got {traj_mean.shape} vs {traj_true.shape}")
    if scenarios.ndim != 4 or scenarios.shape[1:] != traj_true.shape:
        raise ValueError(f"scenarios shape expected [M,N,K,2] = [M,{traj_true.shape[0]},{traj_true.shape[1]},2], got {scenarios.shape}")

    return traj_true.astype(float), traj_mean.astype(float), scenarios.astype(float)