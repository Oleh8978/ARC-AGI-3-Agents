"""Move-budget (the ~21-segment yellow bar) and lives tracking, plus the
route-comparison logic the writeup describes in its "yellow bonus"
section (variants A/B/C): try the direct KEY->DOOR route first; only
detour through a yellow bonus if the direct route can't survive on the
current budget; when several bonuses exist, compare every detour option
and keep the shortest one that survives.

Deliberately kept SEPARATE from the (position, held) state graph in
``stateful_graph.py``: folding an exact move-count into every graph node
would explode the state space (21 possible values x every position x
every held-value). Geometry/mechanics (can I reach X from Y, is the door
open) are learned/graph problems; "can I afford this route on my current
budget" is a simple arithmetic check layered on top of whatever shortest
paths the graph (or ``footprint.bfs_shortest_path``) already gives you.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

Pos = tuple[int, int]
GraphBFS = Callable[[Pos, Pos], Optional[list]]


@dataclass(frozen=True)
class ResourceState:
    moves_remaining: int
    max_moves: int
    lives: int = 3

    def after_moves(self, n: int) -> "ResourceState":
        return replace(self, moves_remaining=max(0, self.moves_remaining - n))

    def after_bonus(self) -> "ResourceState":
        """Yellow bonus is a full RESET, not a fixed top-up (per the
        writeup: touching it sets moves_remaining back to the maximum,
        it doesn't just add a few moves)."""
        return replace(self, moves_remaining=self.max_moves)

    def survives(self, n_moves: int) -> bool:
        return n_moves <= self.moves_remaining


def plan_key_door_route(
    graph_bfs: GraphBFS,
    start: Pos,
    key_pos: Pos,
    door_pos: Pos,
    bonus_positions: list,
    resource: ResourceState,
) -> Optional[tuple]:
    """Returns (total_moves, action_path, label) for the cheapest route
    that is actually survivable on the current move budget, checking:

      - direct:            START -> KEY -> DOOR
      - via_bonus_before:  START -> BONUS -> KEY -> DOOR   (per bonus)
      - via_bonus_after:   START -> KEY -> BONUS -> DOOR   (per bonus)

    matching the writeup's variants A (no detour needed), B (detour
    required), and C (several bonuses, compare and pick the best). A
    detour is never taken just because a bonus exists -- only when it's
    part of the cheapest SURVIVABLE candidate. Returns None if nothing
    found is survivable (e.g. no path exists yet, or every option
    exhausts the budget) -- in that case, keep exploring rather than
    attempting a route you know you can't finish.
    """
    candidates: list = []

    to_key = graph_bfs(start, key_pos)
    key_to_door = graph_bfs(key_pos, door_pos)
    if to_key is not None and key_to_door is not None:
        if resource.survives(len(to_key)):
            after_key = resource.after_moves(len(to_key))
            if after_key.survives(len(key_to_door)):
                total = len(to_key) + len(key_to_door)
                candidates.append((total, to_key + key_to_door, "direct"))

    for i, bonus in enumerate(bonus_positions):
        # START -> BONUS -> KEY -> DOOR
        s1 = graph_bfs(start, bonus)
        s2 = graph_bfs(bonus, key_pos)
        s3 = graph_bfs(key_pos, door_pos)
        if s1 is not None and s2 is not None and s3 is not None:
            if resource.survives(len(s1)):
                r2 = resource.after_bonus()
                if r2.survives(len(s2)):
                    r3 = r2.after_moves(len(s2))
                    if r3.survives(len(s3)):
                        total = len(s1) + len(s2) + len(s3)
                        candidates.append((total, s1 + s2 + s3, f"via_bonus_{i}_before_key"))

        # START -> KEY -> BONUS -> DOOR
        s1b = graph_bfs(start, key_pos)
        s2b = graph_bfs(key_pos, bonus)
        s3b = graph_bfs(bonus, door_pos)
        if s1b is not None and s2b is not None and s3b is not None:
            if resource.survives(len(s1b)):
                r2 = resource.after_moves(len(s1b))
                if r2.survives(len(s2b)):
                    r3 = r2.after_bonus()
                    if r3.survives(len(s3b)):
                        total = len(s1b) + len(s2b) + len(s3b)
                        candidates.append((total, s1b + s2b + s3b, f"via_bonus_{i}_after_key"))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0]
