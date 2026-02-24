import argparse
import os

rou0 = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description="HyperHeuristic Experiment")

    # 算法时间限制
    parser.add_argument("--Timelimit", type=int, default=3600, help="Set the time limit for the solver")

    parser.add_argument("--rouReward", type=str, default=os.path.join(rou0, 'data', 'Reward.xlsx'),
                        help="Set the route of Reward")

    parser.add_argument("--rouSPs", type=str, default=os.path.join(rou0, 'data', 'SPs_info.json'),
                        help="Set the route of data of Drone.json")

    parser.add_argument("--rouGrid", type=str, default=os.path.join(rou0, 'data_AEOS'),
                        help="Set the routes of Grid")

    parser.add_argument("--rouFeasiblePairs", type=str, default=os.path.join(rou0, 'feasible_pair'),
                        help="Set the routes of feasible_pairs")

    parser.add_argument("--MIPgap", type=float, default=0.01, help="Set the MIP gap for the solver")

    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument("--lambda_", type=float, default=5.0)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="admm_dp_result.json")

    return parser.parse_args()



