# mcts/mcts_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import time
import math
import random
import numpy as np
from .types import WorldState, Action
from .env import OnlineEnv
from .rollout import get_rollout_fn

@dataclass
class Node:
    state: WorldState
    parent: Optional["Node"] = None
    parent_action: Optional[Action] = None
    children: Dict[Action, "Node"] = field(default_factory=dict)
    N: int = 0
    W: float = 0.0
    untried: Optional[List[Action]] = None

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

class MCTSPlanner:
    def __init__(self, env: OnlineEnv, ucb_c: float, rollout_depth: int,
                 rollout_policy: str, seed: int = 0):
        self.env = env
        self.ucb_c = ucb_c
        self.rollout_depth = rollout_depth
        self.rollout_fn = get_rollout_fn(rollout_policy)
        self.rng = random.Random(seed)

    def plan(self, root_state: WorldState, budget_ms: int, iters: int = 0) -> Optional[Action]:
        root = Node(state=root_state)
        root.untried = self.env.available_actions(root_state)

        if not root.untried:
            return None

        if iters > 0:
            for _ in range(iters):
                self._iterate(root)
        else:
            t_end = time.perf_counter() + budget_ms / 1000.0
            while time.perf_counter() < t_end:
                self._iterate(root)

        # 选访问次数最多的动作
        best_a = None
        best_n = -1
        for a, ch in root.children.items():
            if ch.N > best_n:
                best_n = ch.N
                best_a = a
        return best_a

    def _iterate(self, root: Node) -> None:
        node = root

        # Selection
        while node.untried is not None and len(node.untried) == 0 and len(node.children) > 0:
            node = self._select_ucb(node)

        # Expansion
        if node.untried is None:
            node.untried = self.env.available_actions(node.state)
        if node.untried:
            a = node.untried.pop(self.rng.randrange(len(node.untried)))
            s2, r = self.env.step_real(node.state, a)  # 你可替换为 step_sim（见下方说明）
            child = Node(state=s2, parent=node, parent_action=a)
            child.untried = self.env.available_actions(s2)
            node.children[a] = child
            node = child
            value = r + self._rollout(node.state)
        else:
            value = self._rollout(node.state)

        # Backpropagation
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent

    def _select_ucb(self, node: Node) -> Node:
        assert node.children
        logN = math.log(max(1, node.N))
        best, best_score = None, -1e18
        for a, ch in node.children.items():
            if ch.N == 0:
                score = float("inf")
            else:
                score = ch.Q() + self.ucb_c * math.sqrt(logN / ch.N)
            if score > best_score:
                best_score = score
                best = ch
        return best  # type: ignore

    def _rollout(self, s: WorldState) -> float:
        total = 0.0
        cur = s
        for _ in range(self.rollout_depth):
            a = self.rollout_fn(self.env, cur, self.rng)
            if a is None:
                break
            cur, r = self.env.step_real(cur, a)  # 更严格：rollout 用 step_sim（下面说明）
            total += float(r)
            if cur.t_idx >= self.env.cfg.horizon_steps - 1:
                break
        return total