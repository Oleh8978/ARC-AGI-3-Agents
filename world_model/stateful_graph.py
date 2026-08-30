"""State-augmented transition graph: nodes are (position, held) pairs,
not position alone.

Why this exists (the actual key/door bug): a door cell's passability
depends on whether the key has been collected. A PLAIN position-keyed
graph records that same door cell as "sometimes passable, sometimes
not" with no way to plan around it -- BFS either refuses to route
through it at all, or routes through it at the wrong time and records a
false dead end it never retries. ``tests/test_key_door_mechanic.py``
demonstrates this directly: a naive position-only graph cannot
distinguish "door, key not yet held" from "door, key held", while this
graph -- keyed by (position, held) -- represents them as two different
nodes, so BFS naturally refuses a route through the locked state and
finds one through the unlocked state, discovered from data, with no
"shapes must match" rule hardcoded anywhere in this file.

The ``held`` component is deliberately generic: it can be a plain
boolean-ish indicator color (as in the rotator/door test), or richer
per-game state your own perception layer derives -- e.g. a
``(has_key: bool, key_orientation: int)`` tuple for a game where the key
must additionally be ROTATED to match the door's orientation (per the
mechanics writeup's rotator/orientation-matching level: the key only
opens the door once its facing matches the door's, and it rotates one
step clockwise each time you touch the plus-shaped tile again). As long
as whatever you pack into ``held`` is hashable and stable frame to
frame, this graph and its BFS work unchanged -- the ordering/rotation
LOGIC itself doesn't need to be hardcoded here; it falls out of the
learned (state, action) -> state transitions, exactly like the
confirmed rotator/door case.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

Pos = tuple[int, int]
State = tuple  # (Pos, held) -- held may be any hashable value, incl. None


@dataclass
class StatefulTransitionGraph:
    action_direction: dict  # {action_name: (dx, dy)}
    edges: dict = field(default_factory=lambda: defaultdict(dict))
    visit_count: dict = field(default_factory=lambda: defaultdict(int))
    STAGNATION_THRESHOLD: int = 15

    def __post_init__(self) -> None:
        self._step = 0
        self._last_new_node_step = 0
        self._known_nodes: set = set()

    # ------------------------------------------------------------------
    def record(self, state_before: State, action_name: str, state_after: State) -> None:
        moved = state_before != state_after
        self.edges[state_before][action_name] = state_after if moved else None
        self.visit_count[state_before] += 1
        self._step += 1
        for s in (state_before, state_after):
            if s not in self._known_nodes:
                self._known_nodes.add(s)
                self._last_new_node_step = self._step

    @property
    def steps_since_new_node(self) -> int:
        return self._step - self._last_new_node_step

    def is_stagnant(self) -> bool:
        return self.steps_since_new_node >= self.STAGNATION_THRESHOLD

    @staticmethod
    def _pos_of(state: State) -> Pos:
        return state[0]

    # ------------------------------------------------------------------
    def bfs_path(self, start: State, goal_pos: Pos, goal_radius: int = 0) -> Optional[list]:
        """BFS over (position, held) states. Reaches ANY state whose
        POSITION component is within ``goal_radius`` of ``goal_pos`` --
        held/key state at arrival doesn't matter for "did we get there",
        only for which edges were legally traversable along the way."""
        sx, sy = self._pos_of(start)
        if abs(sx - goal_pos[0]) + abs(sy - goal_pos[1]) <= goal_radius:
            return []
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            state, path = queue.popleft()
            for action_name, next_state in self.edges.get(state, {}).items():
                if next_state is None or next_state in visited:
                    continue
                new_path = path + [action_name]
                nx, ny = self._pos_of(next_state)
                if abs(nx - goal_pos[0]) + abs(ny - goal_pos[1]) <= goal_radius:
                    return new_path
                visited.add(next_state)
                queue.append((next_state, new_path))
        return None

    def frontier_nodes(self) -> list:
        all_dirs = set(self.action_direction.keys())
        return [state for state, m in self.edges.items() if set(m.keys()) < all_dirs]

    def bfs_to_frontier_goal_directed(self, start: State, goal_pos: Pos) -> Optional[list]:
        frontiers = set(self.frontier_nodes())
        if not frontiers or start in frontiers:
            return None
        best_path: Optional[list] = None
        best_score: Optional[int] = None
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            state, path = queue.popleft()
            for action_name, next_state in self.edges.get(state, {}).items():
                if next_state is None or next_state in visited:
                    continue
                new_path = path + [action_name]
                if next_state in frontiers:
                    nx, ny = self._pos_of(next_state)
                    score = len(new_path) + abs(nx - goal_pos[0]) + abs(ny - goal_pos[1])
                    if best_score is None or score < best_score:
                        best_score = score
                        best_path = new_path
                visited.add(next_state)
                queue.append((next_state, new_path))
        return best_path

    def goal_biased_exploration(
        self,
        state: State,
        candidates: list,
        goal_pos: Pos,
        hazard_positions: Optional[set] = None,
    ):
        hazard_positions = hazard_positions or set()
        pos = self._pos_of(state)
        scored = []
        for action in candidates:
            edge_dict = self.edges.get(state, {})
            if action.name not in edge_dict:
                d = self.action_direction.get(action.name, (0, 0))
                predicted = (pos[0] + d[0], pos[1] + d[1])
                visit_penalty = 0
                manhattan_to_goal = abs(predicted[0] - goal_pos[0]) + abs(predicted[1] - goal_pos[1])
                landing = predicted
            else:
                next_state = edge_dict[action.name]
                if next_state is None:
                    visit_penalty = 999
                    manhattan_to_goal = 0
                    landing = pos
                else:
                    visit_penalty = self.visit_count.get(next_state, 0)
                    npos = self._pos_of(next_state)
                    manhattan_to_goal = abs(npos[0] - goal_pos[0]) + abs(npos[1] - goal_pos[1])
                    landing = npos

            hazard_penalty = 0.0
            for hz in hazard_positions:
                if abs(landing[0] - hz[0]) + abs(landing[1] - hz[1]) <= 1:
                    hazard_penalty = 50.0
                    break

            combined = visit_penalty + 3.0 * manhattan_to_goal + hazard_penalty
            scored.append((action, combined))

        scored.sort(key=lambda t: t[1])
        return scored[0][0] if scored else None
