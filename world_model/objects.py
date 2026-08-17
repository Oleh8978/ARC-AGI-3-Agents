"""
Object-centric perception layer.

Replaces the old "one centroid per color" view (goal_directed_agent.py's
color_centroids) with actual connected-component objects. This is the
prerequisite for anything beyond single-player-in-a-maze games: multiple
same-colored objects, moving obstacles, collectibles, etc. all need to be
distinguishable as separate entities, not blurred into one centroid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GameObject:
    """A single connected-component region of one color in one frame."""
    color: int
    cells: frozenset[tuple[int, int]]  # (x, y) grid coordinates

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def centroid(self) -> tuple[float, float]:
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(min_x, min_y, max_x, max_y)"""
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return (min(xs), min(ys), max(xs), max(ys))


def extract_objects(grid: np.ndarray, background_colors: Optional[set[int]] = None,
                     max_object_cells: int = 400) -> list[GameObject]:
    """Connected-component labeling per color (4-connectivity).

    background_colors: colors to skip entirely (e.g. the most common color,
    which is almost always the game's background/floor). If None, the single
    most frequent color in the grid is treated as background automatically.

    max_object_cells: objects bigger than this are dropped. A legitimate
    game object is small; a huge blob is background that wasn't caught by
    the frequency heuristic (e.g. a big empty room).
    """
    if background_colors is None:
        colors, counts = np.unique(grid, return_counts=True)
        background_colors = {int(colors[np.argmax(counts)])}

    objects: list[GameObject] = []
    for color in np.unique(grid):
        color = int(color)
        if color in background_colors:
            continue
        mask = grid == color
        labeled, n = ndimage.label(mask, structure=np.array([[0, 1, 0],
                                                               [1, 1, 1],
                                                               [0, 1, 0]]))
        for comp_id in range(1, n + 1):
            ys, xs = np.where(labeled == comp_id)
            if len(xs) == 0 or len(xs) > max_object_cells:
                continue
            cells = frozenset(zip(xs.tolist(), ys.tolist()))
            objects.append(GameObject(color=color, cells=cells))
    return objects


def match_objects(
    before: list[GameObject], after: list[GameObject], max_centroid_jump: float = 6.0
) -> list[tuple[Optional[GameObject], Optional[GameObject]]]:
    """Greedy nearest-neighbour matching of objects between two frames.

    Matching is same-color + closest centroid + similar size, capped by
    max_centroid_jump (objects that moved further than this in one action
    are treated as a different object / not matched, rather than an absurd
    teleport hypothesis).

    Returns list of (before_obj_or_None, after_obj_or_None) pairs.
    None on one side means "appeared" or "disappeared".
    """
    pairs: list[tuple[Optional[GameObject], Optional[GameObject]]] = []
    used_after: set[int] = set()

    # sort by size descending so large stable objects match first
    for b in sorted(before, key=lambda o: -o.size):
        best_j, best_dist = None, None
        bx, by = b.centroid
        for j, a in enumerate(after):
            if j in used_after or a.color != b.color:
                continue
            ax, ay = a.centroid
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            size_ratio = min(a.size, b.size) / max(a.size, b.size)
            if dist > max_centroid_jump or size_ratio < 0.5:
                continue
            if best_dist is None or dist < best_dist:
                best_j, best_dist = j, dist
        if best_j is not None:
            used_after.add(best_j)
            pairs.append((b, after[best_j]))
        else:
            pairs.append((b, None))  # disappeared

    for j, a in enumerate(after):
        if j not in used_after:
            pairs.append((None, a))  # appeared

    return pairs
