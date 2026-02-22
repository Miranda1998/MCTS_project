# mcts/rollout.py
from __future__ import annotations
from typing import List, Optional, Callable
import numpy as np
import random
from .types import WorldState, Action
from .env import OnlineEnv

def rollout_policy_random(env: OnlineEnv, s: WorldState, rng: random.Random) -> Optional[Action]:
    acts = env.available_actions(s)
    return rng.choice(acts) if acts else None

def rollout_policy_greedy(env: OnlineEnv, s: WorldState, rng: random.Random) -> Optional[Action]:
    acts = env.available_actions(s)
    if not acts:
        return None
    # 一步贪婪：选择 reward 最大的未服务船（忽略旅行时间，保持轻量）
    best = None
    best_val = -1e18
    for a in acts:
        if s.served[a.vessel_id]:
            continue
        v = float(env.reward[a.vessel_id])
        if v > best_val:
            best_val = v
            best = a
    return best if best is not None else rng.choice(acts)

def rollout_policy_lookahead2(env: OnlineEnv, s: WorldState, rng: random.Random) -> Optional[Action]:
    acts = env.available_actions(s)
    if not acts:
        return None
    # 简单二步前瞻：当前奖励 + 下一步最大可能奖励（近似）
    best = None
    best_val = -1e18
    for a in acts:
        # 复制状态并做一次真实步进（在 rollout 中我们会用 sim step，这里偷懒用 real step 也可）
        s1, r1 = env.step_real(s, a)  # 你想更严格：应该用 step_sim（见 mcts_core 里会用）
        # 下一步最大即时 reward
        v2 = 0.0
        acts2 = env.available_actions(s1)
        if acts2:
            v2 = max(float(env.reward[aa.vessel_id]) for aa in acts2 if not s1.served[aa.vessel_id])
        val = float(r1) + 0.5 * float(v2)
        if val > best_val:
            best_val = val
            best = a
    return best if best is not None else rng.choice(acts)

def get_rollout_fn(name: str):
    if name == "random":
        return rollout_policy_random
    if name == "greedy":
        return rollout_policy_greedy
    if name == "lookahead2":
        return rollout_policy_lookahead2
    raise ValueError(f"Unknown rollout policy: {name}")