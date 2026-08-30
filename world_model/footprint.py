"""Footprint-aware geometry: is a WIDTH x HEIGHT actor (the game's 5x5
player) actually able to occupy a given position, and the position graph
that follows from that.

The key difference from every earlier graph in this codebase
(``TransitionGraph`` / ``StagnationAwareTransitionGraph``): those are
built purely EMPIRICALLY, one recorded (pos, action)->pos edge at a
time, because the RULES of movement (does an action move you, does a
wall block you) are not known in advance and must be learned by
interacting. Footprint validity is different: for a fixed-size actor on
a grid you can already see, whether a 5x5 square fits in a corridor is a
plain geometric fact computable from a SINGLE frame, with no
interaction needed at all. So this module computes the full position
graph directly from the grid, instead of waiting to empirically bump
into every wall first.

This does not replace the empirical action-delta learning in
``hypotheses.py`` (a game can still have extra rules on top of plain
wall-blocking, e.g. the "white lines push the player forward" mechanic
from the writeup) -- it just gives you the CORRECT static geometry layer
to plan over, which the current agents don't have at all (they plan
over a single point).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

Pos = tuple[int, int]


def build_passable_grid(
    grid: np.ndarray,
    wall_colors: set,
) -> np.ndarray:
    """Boolean grid, True = a 1x1 cell an actor's footprint may occupy.
    Every color NOT in ``wall_colors`` is treated as passable -- this
    includes the key, door, and yellow-bonus cells themselves (the
    player must be able to walk onto them), so pass only the colors that
    are actually structural obstacles.

    ``wall_colors`` must be supplied explicitly: it is not safely
    inferable from a single frame (a wall color and a decorative,
    walkable color can look identical). Good ways to obtain it in
    practice: (a) visually inspect one real captured frame and read off
    the wall color, or (b) derive it from ``HypothesisEngine`` by
    watching which color's cells NEVER get overlapped by the player's
    footprint across many recorded frames, across many attempted moves.
    """
    return ~np.isin(grid, list(wall_colors))


def valid_top_lefts(passable: np.ndarray, width: int, height: int) -> set:
    """All (x, y) top-left positions where a WIDTH x HEIGHT footprint
    fits entirely within passable cells."""
    h, w = passable.shape
    valid = set()
    for y in range(h - height + 1):
        for x in range(w - width + 1):
            if passable[y : y + height, x : x + width].all():
                valid.add((x, y))
    return valid


def build_footprint_graph(
    passable: np.ndarray,
    width: int,
    height: int,
    action_direction: dict,
) -> dict:
    """Full {pos: {action_name: next_pos_or_None}} position graph for a
    WIDTH x HEIGHT footprint, computed directly from geometry -- no
    exploration needed. Feed this straight into ``bfs_shortest_path``,
    or seed a ``StagnationAwareTransitionGraph.edges`` with it if you
    also want visit-count / stagnation bookkeeping on top.
    """
    valid = valid_top_lefts(passable, width, height)
    graph: dict = {}
    for pos in valid:
        x, y = pos
        moves: dict = {}
        for name, (dx, dy) in action_direction.items():
            nxt = (x + dx, y + dy)
            moves[name] = nxt if nxt in valid else None
        graph[pos] = moves
    return graph


def bfs_shortest_path(graph: dict, start: Pos, goal: Pos, goal_radius: int = 0) -> Optional[list]:
    """Plain shortest-path BFS over a precomputed {pos: {action: next_pos}}
    graph (e.g. from ``build_footprint_graph``), reaching any cell within
    ``goal_radius`` Manhattan distance of ``goal``."""
    if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) <= goal_radius:
        return []
    if start not in graph:
        return None
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        pos, path = queue.popleft()
        for action_name, next_pos in graph.get(pos, {}).items():
            if next_pos is None or next_pos in visited:
                continue
            new_path = path + [action_name]
            if abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1]) <= goal_radius:
                return new_path
            visited.add(next_pos)
            queue.append((next_pos, new_path))
    return None
