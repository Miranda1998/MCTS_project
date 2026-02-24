import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence

def _inverse_minmax_latlon(latlon_norm: np.ndarray, mm: dict) -> np.ndarray:
    """
    latlon_norm: [...,2] in [0,1] with [:,0]=lat_norm, [:,1]=lon_norm
    return: [...,2] in degrees [lat, lon]
    """
    lat = latlon_norm[..., 0] * (mm["lat_max"] - mm["lat_min"]) + mm["lat_min"]
    lon = latlon_norm[..., 1] * (mm["lon_max"] - mm["lon_min"]) + mm["lon_min"]
    return np.stack([lat, lon], axis=-1)

def plot_mcts_latlon_trajs(
    out_dir: str = "mcts_traj_out_latlon",
    x_hist_path: str = "x_hist_50.npy",
    minmax_path: str = "minmax_norm.json",
    n: int = 5,
    vessel_ids: Optional[Sequence[int]] = None,
    scen_ids: Optional[Sequence[int]] = None,
    k_scen: int = 5,
    show_points: bool = False,
    save_dir: Optional[str] = None,
):
    """
    out_dir: 包含 traj_true.npy / traj_mean.npy / scenarios.npy 的目录（经纬度度）
    x_hist_path: 归一化的历史输入 [N,T,5]，前两列为 (lat_norm, lon_norm)
    minmax_path: 用于把历史轨迹反归一化为经纬度

    画图内容（每条船一张图）：
      - Hist 6h (lat/lon, from x_hist, inverse minmax)
      - True next 6h (traj_true)
      - Mean next 6h (traj_mean)
      - Scenarios next 6h (scenarios, 默认画前 k_scen 条；如给 scen_ids 则按指定画)
    """
    # ---- load minmax
    with open(minmax_path, "r") as f:
        mm = json.load(f)

    # ---- load data
    x_hist = np.load(x_hist_path, allow_pickle=True)  # [N,T,5] normalized
    traj_true = np.load(os.path.join(out_dir, "traj_true.npy"), allow_pickle=True)   # [N,K,2] lat/lon deg
    traj_mean = np.load(os.path.join(out_dir, "traj_mean.npy"), allow_pickle=True)   # [N,K,2] lat/lon deg
    scenarios = np.load(os.path.join(out_dir, "scenarios.npy"), allow_pickle=True)   # [M,N,K,2] lat/lon deg

    N, K, _ = traj_true.shape
    if x_hist.shape[0] != N:
        raise ValueError(f"N mismatch: x_hist has {x_hist.shape[0]}, but traj_true has {N}")

    M = scenarios.shape[0]

    # ---- choose vessels
    if vessel_ids is None:
        vessel_ids = list(range(min(n, N)))
    else:
        vessel_ids = [int(v) for v in vessel_ids if 0 <= int(v) < N]

    # ---- choose scenarios
    if scen_ids is None:
        scen_ids = list(range(min(k_scen, M)))
    else:
        scen_ids = [int(s) for s in scen_ids if 0 <= int(s) < M]
        if len(scen_ids) == 0:
            raise ValueError("scen_ids is empty after filtering by [0, M).")

    # ---- inverse minmax for history (first two cols: lat_norm, lon_norm)
    hist_norm_latlon = x_hist[:, :, :2]  # [N,T,2]
    hist_latlon = _inverse_minmax_latlon(hist_norm_latlon, mm)  # [N,T,2] deg

    # ---- save dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # ---- plot per vessel
    for vid in vessel_ids:
        plt.figure()

        # history
        h = hist_latlon[vid]         # [T,2]
        # future
        tt = traj_true[vid]          # [K,2]
        tm = traj_mean[vid]          # [K,2]

        if show_points:
            plt.scatter(h[:, 1], h[:, 0], s=10, label="Hist (6h)", marker="o")
            plt.scatter(tt[:, 1], tt[:, 0], s=14, label="True (next 6h)", marker="x")
            plt.scatter(tm[:, 1], tm[:, 0], s=14, label="Mean (CVAE)", marker="^")
        else:
            plt.plot(h[:, 1], h[:, 0], label="Hist (6h)")
            plt.plot(tt[:, 1], tt[:, 0], label="True (next 6h)")
            plt.plot(tm[:, 1], tm[:, 0], label="Mean (CVAE)")

        # scenarios
        for j, sid in enumerate(scen_ids):
            sc = scenarios[sid, vid]  # [K,2]
            # 避免图例爆炸：只给第一条情景加 legend
            lbl = f"Scenario {sid}" if j == 0 else None
            if show_points:
                plt.scatter(sc[:, 1], sc[:, 0], s=8, alpha=0.55, label=lbl)
            else:
                plt.plot(sc[:, 1], sc[:, 0], alpha=0.55, label=lbl)

        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")
        plt.title(f"Vessel {vid} | K={K} | M={M} (last scenario often = true)")
        plt.grid(True)
        plt.legend()

        if save_dir is not None:
            out_path = os.path.join(save_dir, f"vessel_{vid}.png")
            plt.savefig(out_path, dpi=220, bbox_inches="tight")

        plt.show()



if __name__ == "__main__":
    plot_mcts_latlon_trajs(
        out_dir="mcts_traj_out_latlon",
        x_hist_path="x_hist_50.npy",
        minmax_path="minmax_norm.json",
        n=1,
        k_scen=1,  # 画前6条情景
        show_points=False,
        scen_ids=[3, 4, 5]
    )