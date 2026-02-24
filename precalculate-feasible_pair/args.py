import argparse
import os

rou0 = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description="HyperHeuristic Experiment")

    # 算法时间限制
    parser.add_argument("--Timelimit", type=int, default=3600, help="Set the time limit for the solver")

    parser.add_argument("--rouSPs", type=str, default=os.path.join(rou0, 'data', 'SPs_info.json'),
                        help="Set the route of data of Drone.json")


    parser.add_argument("--rouVessels", type=str, default=os.path.join(rou0, 'data', 'traj_mean.npy'),
                        help="Set the routes of Vessels")

    return parser.parse_args()



