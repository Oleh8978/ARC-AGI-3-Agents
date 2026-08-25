"""
Fix for a real bug found via live testing on ls20 (see conversation log /
README_INTEGRATION.md): HypothesisWorldAgent got stuck for 130+ consecutive
steps in a concave maze pocket, never reaching the visible goal room.

Root cause (confirmed via visualize_recording.py screenshots + graph_nodes
staying flat at 20 for 60 straight steps): TransitionGraph.goal_biased_
exploration's `visit_penalty + 3.0 * manhattan_to_goal` scoring lets the
3x Manhattan weight overpower novelty-seeking whenever unexplored territory
is geometrically "farther" from the goal in straight-line terms than
already-explored, dead-end space nearby -- exactly what a concave/notched
maze wall produces. bfs_to_frontier (the intended escape hatch) never
fires because it's gated behind "all 4 directions known from the exact
current cell", which the greedy heuristic can perpetually avoid completing
if one direction always scores worst.

Two independent, minimal fixes, both subclassed here rather than edited
into goal_directed_agent.py -- that file stays untouched as the ablation
baseline:

1. GOAL_BIAS_WEIGHT lowered from 3.0 to 1.0 -- novelty and goal-direction
   are weighted closer to equally, so a genuinely unexplored cell isn't
   automatically outscored by a nearby, already-dead-ended, goal-adjacent
   one.
2. Explicit stagnation detection: if no new graph node has been discovered
   in STAGNATION_STEPS actions, force a frontier-BFS attempt on the NEXT
   action regardless of whether the current cell's 4 directions are fully
   charted. This is the actual escape hatch -- (1) alone reduces how often
   you get stuck, it doesn't guarantee you never do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Pos = tuple[int, int]


@dataclass
class StagnationAwareTransitionGraph:
    """Same interface as goal_directed_agent.TransitionGraph (bfs_path,
    frontier_nodes, bfs_to_frontier are inherited via composition below,
    not duplicated) plus stagnation tracking and a rebalanced
    goal_biased_exploration.

    Deliberately does NOT subclass TransitionGraph directly (dataclass
    field-ordering with an already-@dataclass base is fragile across
    Python versions); instead it WRAPS one and forwards the read-only
    methods, overriding only the two that needed fixing plus record().
    """
    GOAL_BIAS_WEIGHT = 1.0  # was 3.0 in goal_directed_agent.py
    STAGNATION_STEPS = 15

    def __init__(self, action_direction: dict[str, tuple[int, int]]) -> None:
        # local import to avoid a hard dependency at module load time
        from agents.templates.goal_directed_agent import TransitionGraph
        self._inner = TransitionGraph()
        self._action_direction = action_direction
        self._known_positions: set[Pos] = set()
        self.steps_since_new_node: int = 0

    # ------------------------------------------------------------------
    @property
    def edges(self):
        return self._inner.edges

    @property
    def visit_count(self):
        return self._inner.visit_count

    def bfs_path(self, start: Pos, goal: Pos, goal_radius: int = 2) -> Optional[list[str]]:
        return self._inner.bfs_path(start, goal, goal_radius=goal_radius)

    def frontier_nodes(self) -> list[Pos]:
        return self._inner.frontier_nodes()

    def bfs_to_frontier(self, start: Pos) -> Optional[list[str]]:
        return self._inner.bfs_to_frontier(start)

    def bfs_to_frontier_goal_directed(self, start: Pos, goal: Pos, max_candidates: int = 40) -> Optional[list[str]]:
        """Fix for the real remaining bottleneck found via live testing:
        the original bfs_to_frontier returns the FIRST frontier node BFS
        encounters (nearest by hop count from `start`), completely ignoring
        where the goal is. After 260 live actions and 35 graph nodes, the
        agent still never connected to the goal -- most stagnation-escapes
        were very likely spent exploring territory that happened to be
        nearest in hop-count but geometrically irrelevant to the goal.

        This does a full BFS from `start` collecting EVERY frontier node
        reachable within a bounded search (max_candidates), each tagged
        with (path_length, position). It then picks the frontier node that
        minimises path_length + manhattan_distance_to_goal -- i.e. an A*-
        style admissible-ish heuristic for "which unexplored direction is
        actually worth investigating first", instead of "whichever is
        fewest hops away regardless of direction".
        """
        frontiers = set(self._inner.frontier_nodes())
        if not frontiers or start in frontiers:
            return None

        from collections import deque as _deque
        queue = _deque([(start, [])])
        visited: set[Pos] = {start}
        candidates: list[tuple[int, list[str], Pos]] = []

        while queue and len(candidates) < max_candidates:
            pos, path = queue.popleft()
            for action_name, next_pos in self._inner.edges.get(pos, {}).items():
                if next_pos is None or next_pos in visited:
                    continue
                new_path = path + [action_name]
                if next_pos in frontiers:
                    candidates.append((len(new_path), new_path, next_pos))
                    continue  # don't expand past a frontier node itself
                visited.add(next_pos)
                queue.append((next_pos, new_path))

        if not candidates:
            return None

        def score(c: tuple[int, list[str], Pos]) -> int:
            steps, _, pos = c
            manhattan = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            return steps + manhattan

        candidates.sort(key=score)
        return candidates[0][1]

    # ------------------------------------------------------------------
    def record(self, pos_before: Pos, action_name: str, pos_after: Pos) -> None:
        was_known_before = pos_before in self._known_positions
        was_known_after = pos_after in self._known_positions
        self._inner.record(pos_before, action_name, pos_after)
        self._known_positions.add(pos_before)
        self._known_positions.add(pos_after)
        if was_known_before and was_known_after:
            self.steps_since_new_node += 1
        else:
            self.steps_since_new_node = 0

    def is_stagnant(self) -> bool:
        return self.steps_since_new_node >= self.STAGNATION_STEPS

    # ------------------------------------------------------------------
    HAZARD_PENALTY = 8  # soft, not a hard wall -- see hazard_positions docstring

    def goal_biased_exploration(self, pos, candidates, goal, hazard_positions: Optional[set] = None):
        """Same structure as the original, GOAL_BIAS_WEIGHT lowered, plus an
        optional soft penalty for moving adjacent to a known hazard.

        hazard_positions: current (x, y) positions of objects whose color
        matched HypothesisEngine.likely_hazard_colors() as of THIS frame
        (mobile objects, so this must be recomputed every call -- a
        position that was hazardous 5 steps ago may not be now). Found
        via live testing on ls20: a mobile object caused a measurable
        resource-bar drain when the player moved adjacent to it (confirmed
        via pixel-level bar measurement + a screenshot showing contact
        immediately before the drain). This is a soft penalty (HAZARD_
        PENALTY=8, comparable to a few Manhattan-units at GOAL_BIAS_WEIGHT
        =1.0), not a hard wall: if the only route runs past the hazard,
        the agent should still take it rather than refuse to progress.
        """
        scored = []
        edge_dict = self._inner.edges.get(pos, {})
        for action in candidates:
            if action.name not in edge_dict:
                d = self._action_direction.get(action.name, (0, 0))
                predicted = (pos[0] + d[0], pos[1] + d[1])
                visit_penalty = 0
                manhattan_to_goal = abs(predicted[0] - goal[0]) + abs(predicted[1] - goal[1])
                result_pos = predicted
            else:
                next_pos = edge_dict[action.name]
                if next_pos is None:
                    visit_penalty = 999
                    manhattan_to_goal = 0
                    result_pos = pos
                else:
                    visit_penalty = self._inner.visit_count.get(next_pos, 0)
                    manhattan_to_goal = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
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