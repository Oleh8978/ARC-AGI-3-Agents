"""
Fix for the ACTUAL root cause of ls20 being unsolvable by every previous
iteration: confirmed via external research (not guessing) that ls20 is a
key/door puzzle, not a plain maze:

  "The avatar moves around a grid, holding a key. The level is won by
   reaching a locked door, which opens only when the key's shape matches
   the door's. Certain tiles act as 'rotators' that cycle the held key
   through different shapes when the avatar walks over them. Every move
   costs one unit of health." (Pwin under naive play: 1/355.)

Every previous fix in this codebase (goal-directed frontier BFS,
complete-the-frontier, hazard avoidance, network retry) operated on a
graph keyed by POSITION ALONE. That graph structurally cannot represent
"the door is open only when I'm holding key-shape X" -- from the graph's
point of view the door cell just looks randomly sometimes-passable,
which is indistinguishable from "not explored yet". No amount of better
exploration fixes a wrong state representation.

This module widens the graph key to (position, held_indicator_state).
Critically, it does NOT hardcode "shapes must match" as a rule -- it
just gives the SAME empirical recording process (record a transition,
BFS over recorded transitions) a big enough state space to discover the
rule from data: if the door only transitions successfully when
held_state == 3, that shows up naturally as "from (door_approach, 3),
action X succeeds; from (door_approach, 1), action X is blocked" once
both have been observed. This generalizes to other ARC-AGI-3 games with
similar "carry a state, use it conditionally" mechanics without
per-game special-casing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

Pos = tuple[int, int]
HeldState = Optional[int]  # color of the object identified as "held/carried", or None
State = tuple[Pos, HeldState]


@dataclass
class StatefulTransitionGraph:
    """Same BFS/frontier-search shape as StagnationAwareTransitionGraph,
    but every node is (position, held_state) instead of position alone.
    """
    GOAL_BIAS_WEIGHT = 1.0
    STAGNATION_STEPS = 15
    HAZARD_PENALTY = 8

    def __init__(self, action_direction: dict[str, tuple[int, int]]) -> None:
        self._action_direction = action_direction
        self.edges: dict[State, dict[str, Optional[State]]] = {}
        self.visit_count: dict[State, int] = {}
        self._known_states: set[State] = set()
        self.steps_since_new_node: int = 0

    # ------------------------------------------------------------------
    def record(self, before: State, action_name: str, after: State) -> None:
        was_known_before = before in self._known_states
        was_known_after = after in self._known_states
        moved = before != after
        self.edges.setdefault(before, {})[action_name] = after if moved else None
        self.visit_count[before] = self.visit_count.get(before, 0) + 1
        self._known_states.add(before)
        self._known_states.add(after)
        if was_known_before and was_known_after:
            self.steps_since_new_node += 1
        else:
            self.steps_since_new_node = 0

    def is_stagnant(self) -> bool:
        return self.steps_since_new_node >= self.STAGNATION_STEPS

    # ------------------------------------------------------------------
    def frontier_nodes(self) -> list[State]:
        all_dirs = set(self._action_direction.keys())
        return [s for s, known in self.edges.items() if set(known.keys()) < all_dirs]

    def bfs_path(self, start: State, goal_pos: Pos, goal_radius: int = 2) -> Optional[list[str]]:
        """BFS to ANY state whose POSITION component is within goal_radius
        of goal_pos, regardless of held_state -- we don't hardcode which
        held_state is required; we just search the empirically-recorded
        graph for whatever sequence of actions has actually been observed
        to work."""
        def close_enough(s: State) -> bool:
            (px, py), _ = s
            return abs(px - goal_pos[0]) + abs(py - goal_pos[1]) <= goal_radius

        if close_enough(start):
            return []
        queue: deque[tuple[State, list[str]]] = deque([(start, [])])
        visited: set[State] = {start}
        while queue:
            state, path = queue.popleft()
            for action_name, next_state in self.edges.get(state, {}).items():
                if next_state is None or next_state in visited:
                    continue
                new_path = path + [action_name]
                if close_enough(next_state):
                    return new_path
                visited.add(next_state)
                queue.append((next_state, new_path))
        return None

    def bfs_to_frontier_goal_directed(self, start: State, goal_pos: Pos, max_candidates: int = 40) -> Optional[list[str]]:
        frontiers = set(self.frontier_nodes())
        if not frontiers or start in frontiers:
            return None
        queue: deque[tuple[State, list[str]]] = deque([(start, [])])
        visited: set[State] = {start}
        candidates: list[tuple[int, list[str], State]] = []
        while queue and len(candidates) < max_candidates:
            state, path = queue.popleft()
            for action_name, next_state in self.edges.get(state, {}).items():
                if next_state is None or next_state in visited:
                    continue
                new_path = path + [action_name]
                if next_state in frontiers:
                    candidates.append((len(new_path), new_path, next_state))
                    continue
                visited.add(next_state)
                queue.append((next_state, new_path))
        if not candidates:
            return None

        def score(c):
            steps, _, (pos, _) = c
            manhattan = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
            return steps + manhattan

        candidates.sort(key=score)
        return candidates[0][1]

    # ------------------------------------------------------------------
    def goal_biased_exploration(self, state: State, candidates, goal_pos: Pos, hazard_positions: Optional[set] = None):
        pos, held = state
        scored = []
        edge_dict = self.edges.get(state, {})
        for action in candidates:
            if action.name not in edge_dict:
                d = self._action_direction.get(action.name, (0, 0))
                predicted_pos = (pos[0] + d[0], pos[1] + d[1])
                visit_penalty = 0
                manhattan_to_goal = abs(predicted_pos[0] - goal_pos[0]) + abs(predicted_pos[1] - goal_pos[1])
                result_pos = predicted_pos
            else:
                next_state = edge_dict[action.name]
                if next_state is None:
                    visit_penalty = 999
                    manhattan_to_goal = 0
                    result_pos = pos
                else:
                    next_pos, _ = next_state
                    visit_penalty = self.visit_count.get(next_state, 0)
                    manhattan_to_goal = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])
                    result_pos = next_pos

            hazard_penalty = 0
            if hazard_positions:
                for hp in hazard_positions:
                    if abs(result_pos[0] - hp[0]) <= 1 and abs(result_pos[1] - hp[1]) <= 1:
                        hazard_penalty = self.HAZARD_PENALTY
                        break

            combined = visit_penalty + self.GOAL_BIAS_WEIGHT * manhattan_to_goal + hazard_penalty
            scored.append((action, combined))
        scored.sort(key=lambda t: t[1])
        return scored[0][0] if scored else None
