import os
from openpyxl import load_workbook
import pandas as pd
import json
import numpy as np

# 读取返回的各个变量，并根据变量设置集合等参数
def Load_Data(args):

    # 和问题有关的参数
    with open(args.rouSPs, 'r') as f:
        config = json.load(f)

    # Define variables based on the JSON content
    drone_base_pos = config['drone_base_pos']
    ScheduleHorizon = config['ScheduleHorizon']
    slot_gap = config['SlotGap']
    drone_speed = config['drone_speed']
    drone_battery = config['drone_battery']
    drone_base_num = config['drone_base_num']
    vessel_total_num = config['vessel_total_num']

    T = int(ScheduleHorizon * (60/slot_gap))

    # 读取船舶数据（npy 版）：traj_mean.npy / traj_true.npy 都可以
    traj = np.load(args.rouVessels, allow_pickle=True)  # 期望形状 [N, K, 2]

    # 与原来 excel 版保持一致的数据结构：List[List[List[lat,lon]]]
    vessel_pos = traj.tolist()

    # 任务位置+base位置
    task_pos = [[[0, 0] for t in range(T)] for _ in range(drone_base_num+vessel_total_num)]

    for _ in range(drone_base_num + vessel_total_num):
        for t in range(T):
            if _ < drone_base_num:
                task_pos[_][t] = drone_base_pos[_]
            else:
                task_pos[_][t] = vessel_pos[_-drone_base_num][t]

    return (drone_base_num, vessel_total_num, task_pos, drone_speed, drone_battery, T, slot_gap)


