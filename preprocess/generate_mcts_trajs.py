import argparse
import json
import os
import numpy as np
import torch
from model import CVAE

def inverse_points_minmax(points_norm: np.ndarray, mm: dict) -> np.ndarray:
    """
    points_norm: [...,2] in [0,1]
    return: [...,2] in degrees, [lat, lon]
    """
    lat = points_norm[..., 0] * (mm["lat_max"] - mm["lat_min"]) + mm["lat_min"]
    lon = points_norm[..., 1] * (mm["lon_max"] - mm["lon_min"]) + mm["lon_min"]
    return np.stack([lat, lon], axis=-1)

@torch.no_grad()
def gen_mean_and_9scen_latlon(
    x_hist_np: np.ndarray,
    ckpt: str,
    minmax_path: str,
    mean_K: int,
    horizon: int,
    downsample: int,
    seed: int,
    device: str,
):
    # minmax
    with open(minmax_path, "r") as f:
        mm = json.load(f)

    # model
    model = CVAE(x_dim=5, y_dim=2, z_dim=32, hid_dim=128, horizon=horizon).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    x_hist = torch.tensor(x_hist_np[:, :horizon, :], dtype=torch.float32, device=device)
    stat = None

    # ---- traj_mean：用 mus 的均值近似条件期望（更稳定）
    trajs, _, mus, _ = model.sample(x_hist, stat=stat, K=mean_K, seed=seed)
    mu_bar = mus.mean(dim=0).detach().cpu().numpy()  # [N,T,2] normalized

    # ---- 9 条 CVAE 情景
    scen_list = []
    for m in range(9):
        trajs_m, _, _, _ = model.sample(x_hist, stat=stat, K=1, seed=seed + 10_000 + m)
        scen_list.append(trajs_m[0].detach().cpu().numpy())  # [N,T,2] normalized
    scen_9 = np.stack(scen_list, axis=0)  # [9,N,T,2] normalized

    # ---- downsample（可选）
    if downsample > 1:
        mu_bar = mu_bar[:, ::downsample, :].copy()
        scen_9 = scen_9[:, :, ::downsample, :].copy()

    # ---- inverse min-max -> latlon degrees
    traj_mean_latlon = inverse_points_minmax(mu_bar, mm).astype(np.float32)      # [N,K,2]
    scen_9_latlon = inverse_points_minmax(scen_9, mm).astype(np.float32)         # [9,N,K,2]

    return traj_mean_latlon, scen_9_latlon, mm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_hist", type=str, default="x_hist_50.npy",
                    help="x_hist_50.npy, [N,T,5]")
    ap.add_argument("--y_true", type=str, default="y_hist_50.npy",
                    help="y_hist_50.npy, [N,T,2] (normalized)")
    ap.add_argument("--ckpt", type=str, default="best_model.pt",
                    help="trained cvae.pth")
    ap.add_argument("--minmax", type=str, default="minmax_norm.json",
                    help="minmax_norm.json")
    ap.add_argument("--out_dir", type=str, default="mcts_traj_out_latlon")
    ap.add_argument("--horizon", type=int, default=72)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--mean_K", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    x_hist = np.load(args.x_hist, allow_pickle=True)  # [N,T,5]
    y_true = np.load(args.y_true, allow_pickle=True)  # [N,T,2] normalized

    if x_hist.ndim != 3 or x_hist.shape[-1] != 5:
        raise ValueError(f"x_hist expected [N,T,5], got {x_hist.shape}")
    if y_true.ndim != 3 or y_true.shape[-1] != 2:
        raise ValueError(f"y_true expected [N,T,2], got {y_true.shape}")
    if x_hist.shape[0] != y_true.shape[0]:
        raise ValueError("N mismatch between x_hist and y_true")
    if x_hist.shape[1] < args.horizon or y_true.shape[1] < args.horizon:
        raise ValueError("T shorter than horizon")

    # minmax
    with open(args.minmax, "r") as f:
        mm = json.load(f)

    # ---- traj_true：把未来真值 y_true 反归一化成经纬度
    traj_true_norm = y_true[:, :args.horizon, :].copy()
    if args.downsample > 1:
        traj_true_norm = traj_true_norm[:, ::args.downsample, :].copy()
    traj_true_latlon = inverse_points_minmax(traj_true_norm, mm).astype(np.float32)  # [N,K,2]

    # ---- traj_mean + 前9条情景：来自 CVAE
    traj_mean_latlon, scen_9_latlon, _ = gen_mean_and_9scen_latlon(
        x_hist_np=x_hist,
        ckpt=args.ckpt,
        minmax_path=args.minmax,
        mean_K=args.mean_K,
        horizon=args.horizon,
        downsample=args.downsample,
        seed=args.seed,
        device=device
    )

    # ---- scenarios：前9条 + 第10条真值
    scenarios_latlon = np.concatenate([scen_9_latlon, traj_true_latlon[None, ...]], axis=0)  # [10,N,K,2]

    # ---- shape checks
    N, K, _ = traj_true_latlon.shape
    assert traj_mean_latlon.shape == (N, K, 2), traj_mean_latlon.shape
    assert scenarios_latlon.shape == (10, N, K, 2), scenarios_latlon.shape
    assert np.allclose(scenarios_latlon[-1], traj_true_latlon), "last scenario must equal traj_true"

    np.save(os.path.join(args.out_dir, "traj_true.npy"), traj_true_latlon)
    np.save(os.path.join(args.out_dir, "traj_mean.npy"), traj_mean_latlon)
    np.save(os.path.join(args.out_dir, "scenarios.npy"), scenarios_latlon)

    print("[OK] saved (lat, lon in degrees):")
    print(" traj_true :", traj_true_latlon.shape, "->", os.path.join(args.out_dir, "traj_true.npy"))
    print(" traj_mean :", traj_mean_latlon.shape, "->", os.path.join(args.out_dir, "traj_mean.npy"))
    print(" scenarios :", scenarios_latlon.shape, "->", os.path.join(args.out_dir, "scenarios.npy"))
    print(" scenarios[-1] equals traj_true =", bool(np.allclose(scenarios_latlon[-1], traj_true_latlon)))

if __name__ == "__main__":
    main()