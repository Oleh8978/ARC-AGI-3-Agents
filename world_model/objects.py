"""Object-centric perception for ARC-AGI-3 frames.

The earlier perception layer (``ColorRegionTracker`` / ``color_centroids``
in ``goal_directed_agent.py``) treats every color as ONE blob: it averages
the centroid of every cell of that color anywhere on the grid. That is
exactly wrong for a game whose rules (per the mechanics writeup) depend on
*shape*, not just color:

  - the player is a specific 5x5 bicolor pattern (2 rows orange over 3
    rows dark blue) -- a "blue" cell far away that belongs to a wall is
    NOT part of the player and must not pull its centroid around;
  - the key is a specific plus-shaped 5-cell pattern in gray/dark-gray;
  - the door is a specific 4-cell dark-blue pattern (not just "some dark
    blue pixels");
  - a yellow bonus and the yellow move-budget bar are both "yellow" but
    are different objects at different places on screen.

``extract_objects`` fixes this by doing real connected-component
labelling per color, so each on-screen shape becomes its own object with
its own cell set, bounding box, and (translation-invariant) shape
signature -- which is what you actually need to recognise "the 5x5
player-shaped thing" or "the plus-shaped key" reliably frame to frame,
instead of hoping a single global color centroid happens to track the
right pixels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

Cell = tuple[int, int]  # (x, y)


@dataclass(frozen=True)
class GameObject:
    color: int
    cells: frozenset  # frozenset[Cell]

    @property
    def centroid(self) -> tuple[float, float]:
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(min_x, min_y, max_x, max_y), inclusive."""
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def width(self) -> int:
        x0, _, x1, _ = self.bbox
        return x1 - x0 + 1

    @property
    def height(self) -> int:
        _, y0, _, y1 = self.bbox
        return y1 - y0 + 1

    def shape_signature(self) -> tuple:
        """Cell offsets relative to the bounding-box top-left corner.
        Translation-invariant fingerprint of a shape's outline -- lets a
        caller recognise "this is a 5x5-block-shaped thing" or "this is
        the plus-shaped key" independent of where it currently sits on
        the grid. Two objects with the same color AND the same
        shape_signature are almost certainly the same logical object
        seen in two different frames.
        """
        x0, y0, _, _ = self.bbox
        return tuple(sorted((cx - x0, cy - y0) for cx, cy in self.cells))


def extract_objects(grid: np.ndarray, background: Optional[int] = None) -> list[GameObject]:
    """4-connected connected-component labelling, one object per
    same-colored connected region (NOT one object per color overall).

    ``background``, if given, is the color skipped as "empty space". If
    omitted, the most frequent color in the grid is treated as
    background, which is the standard ARC convention and matches what
    the rest of this codebase already assumes.
    """
    h, w = grid.shape
    if background is None:
        vals, counts = np.unique(grid, return_counts=True)
        background = int(vals[np.argmax(counts)])

    visited = np.zeros_like(grid, dtype=bool)
    objects: list[GameObject] = []

    for y in range(h):
        for x in range(w):
            if visited[y, x]:
                continue
            color = int(grid[y, x])
            if color == background:
                visited[y, x] = True
                continue
            queue: deque[Cell] = deque([(x, y)])
            visited[y, x] = True
            cells: list[Cell] = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < w
                        and 0 <= ny < h
                        and not visited[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            objects.append(GameObject(color=color, cells=frozenset(cells)))
    return objects


def merge_by_color(objects: list[GameObject]) -> dict:
    """color -> (centroid_x, centroid_y, total_cell_count), merging every
    connected component of the same color into one weighted-average
    position. This is the old ColorRegionTracker-style aggregate,
    reproduced here (on top of real per-component objects) for callers
    that only need "where is this color, roughly" -- e.g. bulk delta
    tracking in ``hypotheses.py``, where the synthetic/offline tests
    exercise exactly one component per color.
    """
    from collections import defaultdict

    agg: dict = defaultdict(list)
    for obj in objects:
        cx, cy = obj.centroid
        agg[obj.color].append((cx, cy, obj.size))
    out = {}
    for color, parts in agg.items():
        total = sum(p[2] for p in parts)
        cx = sum(p[0] * p[2] for p in parts) / total
        cy = sum(p[1] * p[2] for p in parts) / total
        out[color] = (cx, cy, total)
    return out


def find_by_shape(objects: list[GameObject], signature: tuple, color: Optional[int] = None):
    """Returns the first object matching an exact shape_signature (and
    color, if given). Use this to recognise a specific known pattern --
    e.g. the game's 5x5 player block or the plus-shaped key -- once
    you've captured its signature once, instead of trusting color alone.
    """
    for obj in objects:
        if color is not None and obj.color != color:
            continue
        if obj.shape_signature() == signature:
            return obj
    return None


def footprint_cells(top_left: Cell, width: int, height: int) -> frozenset:
    """All (x, y) cells a WIDTH x HEIGHT footprint occupies with its
    top-left corner at ``top_left``. Use this (NOT a single point) to
    check whether a large actor -- e.g. the game's 5x5 player -- can
    legally occupy a candidate position: every cell in the returned set
    must be free of walls/out-of-bounds, not just the center point.
    """
    x0, y0 = top_left
    return frozenset((x0 + dx, y0 + dy) for dx in range(width) for dy in range(height))


def footprint_is_valid(top_left: Cell, width: int, height: int, blocked: frozenset) -> bool:
    """True iff every cell of a WIDTH x HEIGHT footprint at ``top_left``
    is free of ``blocked`` cells (walls / out-of-bounds / hazards).
    """
    return footprint_cells(top_left, width, height).isdisjoint(blocked)


def detect_held_indicator(
    objects: list[GameObject],
    player_center: tuple[float, float],
    exclude: set,
    radius: float,
) -> Optional[int]:
    """Best-effort 'what state indicator is the player currently
    carrying' signal: the color of the nearest small object within
    ``radius`` of the player that isn't the player or excluded goal
    color(s). Used as the ``held`` component of a (position, held) state
    for state-augmented planning (see ``stateful_graph.py``) -- e.g. "do
    I currently hold the key" or "which way is the key rotated".

    This is a generic proxy, not a hardcoded parser of any one game's
    HUD layout. If a game encodes carried-key state via a small HUD icon
    at a fixed screen position rather than "near the player", detect it
    with a dedicated region-based reader instead and feed its color in
    as ``held`` directly -- the state-augmented graph itself doesn't
    care where the signal comes from, only that it's stable and
    observable every frame.
    """
    px, py = player_center
    best_color = None
    best_dist = None
    for obj in objects:
        if obj.color in exclude:
            continue
        if obj.size > 50:
            continue
        cx, cy = obj.centroid
        d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if d <= radius and (best_dist is None or d < best_dist):
            best_dist = d
            best_color = obj.color
    return best_color
