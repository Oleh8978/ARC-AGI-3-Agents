"""Stagnation-aware, goal-directed empirical transition graph.

Same empirical (position, action) -> next_position graph as
``goal_directed_agent.TransitionGraph`` (BFS over known transitions),
plus three fixes diagnosed from live runs:

1. ``bfs_to_frontier_goal_directed`` -- the plain ``bfs_to_frontier``
   picks the CLOSEST unexplored node by hop count, which can be a
   goal-irrelevant dead end (confirmed live: 35 graph nodes / 260
   actions, never connecting to the goal, because a short dead-end
   branch kept winning over a longer goal-relevant one). This variant
   scores frontier nodes by hops + Manhattan-distance-to-goal instead.

2. ``is_stagnant`` / ``steps_since_new_node`` -- lets a caller detect
   "no new graph node discovered in N steps" and force frontier-directed
   exploration instead of drifting in already-known territory.

3. ``goal_biased_exploration(..., hazard_positions=...)`` -- optionally
   penalises candidate actions that would land adjacent to a known
   hazardous cell, so goal-directedness doesn't walk the agent straight
   into something that resets/kills it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

Pos = tuple[int, int]


@dataclass
class StagnationAwareTransitionGraph:
    action_direction: dict  # {action_name: (dx, dy)}
    edges: dict = field(default_factory=lambda: defaultdict(dict))
    visit_count: dict = field(default_factory=lambda: defaultdict(int))
    STAGNATION_THRESHOLD: int = 15

    def __post_init__(self) -> None:
        self._step = 0
        self._last_new_node_step = 0
        self._known_nodes: set = set()

    # ------------------------------------------------------------------
    def record(self, pos_before: Pos, action_name: str, pos_after: Pos) -> None:
        moved = pos_before != pos_after
        self.edges[pos_before][action_name] = pos_after if moved else None
        self.visit_count[pos_before] += 1
        self._step += 1
        for p in (pos_before, pos_after):
            if p not in self._known_nodes:
                self._known_nodes.add(p)
                self._last_new_node_step = self._step

    @property
    def steps_since_new_node(self) -> int:
        return self._step - self._last_new_node_step

    def is_stagnant(self) -> bool:
        return self.steps_since_new_node >= self.STAGNATION_THRESHOLD

    # ------------------------------------------------------------------
    def bfs_path(self, start: Pos, goal: Pos, goal_radius: int = 0) -> Optional[list]:
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) <= goal_radius:
            return []
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            for action_name, next_pos in self.edges.get(pos, {}).items():
                if next_pos is None or next_pos in visited:
                    continue
                new_path = path + [action_name]
                if abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1]) <= goal_radius:
                    return new_path
                visited.add(next_pos)
                queue.append((next_pos, new_path))
        return None

    def frontier_nodes(self) -> list:
        all_dirs = set(self.action_direction.keys())
        return [pos for pos, m in self.edges.items() if set(m.keys()) < all_dirs]

    def bfs_to_frontier(self, start: Pos) -> Optional[list]:
        """Nearest frontier node by hop count only. Kept for contrast
        with ``bfs_to_frontier_goal_directed`` -- see module docstring."""
        frontiers = set(self.frontier_nodes())
        if not frontiers or start in frontiers:
            return None
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            for action_name, next_pos in self.edges.get(pos, {}).items():
                if next_pos is None or next_pos in visited:
                    continue
                new_path = path + [action_name]
                if next_pos in frontiers:
                    return new_path
                visited.add(next_pos)
                queue.append((next_pos, new_path))
        return None

    def bfs_to_frontier_goal_directed(self, start: Pos, goal: Pos) -> Optional[list]:
        """Among ALL frontier nodes reachable from start, picks the one
        minimising (hops_from_start + manhattan_to_goal), not hops
        alone -- prefers a longer goal-relevant branch over a shorter
        goal-irrelevant one."""
        frontiers = set(self.frontier_nodes())
        if not frontiers or start in frontiers:
            return None
        best_path: Optional[list] = None
        best_score: Optional[int] = None
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            for action_name, next_pos in self.edges.get(pos, {}).items():
                if next_pos is None or next_pos in visited:
                    continue
                new_path = path + [action_name]
                if next_pos in frontiers:
                    score = len(new_path) + abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                    if best_score is None or score < best_score:
                        best_score = score
                        best_path = new_path
                visited.add(next_pos)
                queue.append((next_pos, new_path))
        return best_path

    def goal_biased_exploration(
        self,
        pos: Pos,
        candidates: list,
        goal: Pos,
        hazard_positions: Optional[set] = None,
    ):
        hazard_positions = hazard_positions or set()
        scored = []
        for action in candidates:
            edge_dict = self.edges.get(pos, {})
            if action.name not in edge_dict:
                d = self.action_direction.get(action.name, (0, 0))
                predicted = (pos[0] + d[0], pos[1] + d[1])
                visit_penalty = 0
                manhattan_to_goal = abs(predicted[0] - goal[0]) + abs(predicted[1] - goal[1])
                landing = predicted
            else:
                next_pos = edge_dict[action.name]
                if next_pos is None:
                    visit_penalty = 999
                    manhattan_to_goal = 0
                    landing = pos
                else:
                    visit_penalty = self.visit_count.get(next_pos, 0)
                    manhattan_to_goal = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                    landing = next_pos

            hazard_penalty = 0.0
            for hz in hazard_positions:
                if abs(landing[0] - hz[0]) + abs(landing[1] - hz[1]) <= 1:
                    hazard_penalty = 50.0
                    break

            combined = visit_penalty + 3.0 * manhattan_to_goal + hazard_penalty
            scored.append((action, combined))

        scored.sort(key=lambda t: t[1])
        return scored[0][0] if scored else None
