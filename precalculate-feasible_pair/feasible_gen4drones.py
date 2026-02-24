from geopy.distance import distance
from args import *
from data_load4drones import *
from datetime import datetime
import time
from tqdm import tqdm
import os

args = parse_args()

# 读取返回的各个变量，并根据变量设置集合等参数
(base_num, vessel_total_num, task_pos, UAV_speed, UAV_battery, T, slot_gap) = Load_Data(args)

def get_travel_time(i, k, j, l):
    dis_i_k_j_l = distance(task_pos[i][k], task_pos[j][l]).km
    travel_time = dis_i_k_j_l/UAV_speed * 60 / slot_gap  # 转换为时间片数量
    return travel_time

for TASK in range(10, 60, 20):

    start = time.perf_counter()  # 开始计时
    print('当前任务数量为:', TASK)
    print('当前周期为', T)

    # 创建一个npy文件存储feasible_route
    feasible_pair = []

    for i in tqdm(range(base_num+TASK)):
        for k in range(T):
            for j in range(base_num + TASK):
                for l in range(k, T):
                    travel_time = get_travel_time(i, k, j, l)
                    if i == j:
                        continue
                    if l - k > UAV_battery:
                        continue
                    else:
                        if l - k - 1 <= travel_time + 1 <= l - k:
                        # if k + travel_time + 1 <= l:
                            feasible_pair.append([i, k, j, l])

    # # 获取当前日期
    current_date = datetime.now().strftime('%Y%m%d')  # 格式化日期为YYYYMMDD
    result_dir = os.path.join(rou0, 'result')

    # 如果文件夹路径不存在，则创建文件夹
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 文件路径
    file_path = os.path.join(result_dir,
                             f'new_feasible_pair{current_date}_TASK={TASK}_T={T}.npy')

    np.save(file_path, np.array(feasible_pair, dtype=object))
