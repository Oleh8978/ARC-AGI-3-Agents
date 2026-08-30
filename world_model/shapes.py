"""Known shape signatures from the mechanics writeup, used to identify
player / key / door BY SHAPE, not by "smallest static color blob" (the
old ``GoalDetector`` heuristic, which has no way to tell a key apart
from a door apart from a decorative tile of similar size).

Per the writeup:
  - Player: 5 wide x 5 tall. Top 2 rows one color ("orange"), bottom 3
    rows a second color ("dark blue"). 25 cells total (10 + 15).
  - Key: plus-shape, 5 cells, gray/dark-gray (color varies per level --
    some levels have color-changing tiles the key's color reacts to, so
    KEY IDENTIFICATION HERE IS BY SHAPE ONLY, never by color).
  - Door: 4 dark-blue cells in a specific asymmetric pattern (so it
    isn't confused with any other 4-cell dark-blue decoration):

        [ ][B][B]
        [ ][ ][B]
        [B][ ][ ]

None of this is guessed -- these are the literal patterns given. If a
level's actual pixel data differs even slightly (off-by-one, mirrored),
``shape_signature()`` will NOT match and these detectors will return
None -- in that case, verify the exact cell coordinates against one real
captured frame and adjust the *_SIG tuples below; don't loosen the match
tolerance, or you'll re-introduce the "confused with decoration" bug
this file exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .objects import GameObject

# ---------------------------------------------------------------------------
# Canonical shapes (cell offsets relative to bounding-box top-left)
# ---------------------------------------------------------------------------

def _rect(width: int, height: int) -> tuple:
    return tuple(sorted((x, y) for y in range(height) for x in range(width)))


PLAYER_TOP_SIG = _rect(5, 2)      # 2 rows x 5 wide = 10 cells (orange)
PLAYER_BOTTOM_SIG = _rect(5, 3)   # 3 rows x 5 wide = 15 cells (dark blue)
PLAYER_WIDTH = 5
PLAYER_HEIGHT = 5

KEY_SIG = tuple(sorted([(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]))  # plus, 5 cells

DOOR_SIG = tuple(sorted([(1, 0), (2, 0), (2, 1), (0, 2)]))  # 4 dark-blue cells
DOOR_BBOX_W = 3
DOOR_BBOX_H = 3

# IMPORTANT: unlike the key (a connected plus-shape), the door pattern as
# given is NOT 4-connected -- cell (0, 2) is diagonally/further isolated
# from the connected 3-cell group {(1,0),(2,0),(2,1)}. A plain
# connected-component object detector (``objects.extract_objects``) will
# therefore always see the door as TWO separate objects and never match
# DOOR_SIG against either one. ``detect_door`` below scans the raw grid
# directly with a sliding 3x3 window instead of relying on connectivity,
# specifically to handle this.


def _find_pattern_by_color_mask(
    grid: np.ndarray, pattern_offsets: tuple, bbox_w: int, bbox_h: int
) -> Optional[tuple]:
    """Slides a bbox_w x bbox_h window over the grid; at each position,
    checks every color present in that window to see if the set of
    cells equal to that color (ignoring connectivity) exactly matches
    ``pattern_offsets``. Returns (top_left, color) of the first match,
    or None. Use this for known compound/disconnected patterns like the
    door; use ``objects.extract_objects`` + ``shape_signature()`` for
    ordinary connected shapes like the player or the key.
    """
    h, w = grid.shape
    pattern_set = frozenset(pattern_offsets)
    for y in range(h - bbox_h + 1):
        for x in range(w - bbox_w + 1):
            window = grid[y : y + bbox_h, x : x + bbox_w]
            for color in np.unique(window):
                color = int(color)
                cells = frozenset(
                    (dx, dy)
                    for dy in range(bbox_h)
                    for dx in range(bbox_w)
                    if int(window[dy, dx]) == color
                )
                if cells == pattern_set:
                    return (x, y), color
    return None


@dataclass(frozen=True)
class PlayerSprite:
    """The FULL 5x5 player footprint (both color bands merged), for
    footprint/collision purposes -- not just whichever single color the
    calibration tracker happened to lock onto."""

    top_left: tuple  # (x, y) of the 5x5 bounding box
    top_color: int
    bottom_color: int

    @property
    def width(self) -> int:
        return PLAYER_WIDTH

    @property
    def height(self) -> int:
        return PLAYER_HEIGHT

    @property
    def centroid(self) -> tuple:
        x0, y0 = self.top_left
        return (x0 + (PLAYER_WIDTH - 1) / 2, y0 + (PLAYER_HEIGHT - 1) / 2)


def detect_player(objects: list) -> Optional[PlayerSprite]:
    """Finds a PLAYER_TOP_SIG-shaped object sitting directly above a
    PLAYER_BOTTOM_SIG-shaped object of a DIFFERENT color, same x-range,
    adjacent rows (top's bottom row + 1 == bottom's top row). Returns
    the merged 5x5 sprite, or None if no such pair exists in this frame.
    """
    tops = [o for o in objects if o.shape_signature() == PLAYER_TOP_SIG]
    bottoms = [o for o in objects if o.shape_signature() == PLAYER_BOTTOM_SIG]
    for top in tops:
        tx0, ty0, tx1, ty1 = top.bbox
        for bottom in bottoms:
            if bottom.color == top.color:
                continue
            bx0, by0, bx1, by1 = bottom.bbox
            if bx0 == tx0 and bx1 == tx1 and by0 == ty1 + 1:
                return PlayerSprite(top_left=(tx0, ty0), top_color=top.color, bottom_color=bottom.color)
    return None


def detect_key(objects: list) -> Optional[GameObject]:
    """Finds the plus-shaped key object BY SHAPE, regardless of its
    current color (some levels recolor it via puzzle tiles)."""
    for obj in objects:
        if obj.shape_signature() == KEY_SIG:
            return obj
    return None


def detect_door(grid: np.ndarray) -> Optional[GameObject]:
    """Finds the door by its specific 4-cell pattern via a direct grid
    scan (NOT connected-component matching -- see the note above
    DOOR_SIG: the pattern is not 4-connected, so ``extract_objects``
    would never produce one matching object for it). Takes the raw grid,
    not an object list. Returns a ``GameObject`` built from the matched
    cells so callers get the same ``.bbox`` / ``.centroid`` API as
    ``detect_key``.
    """
    match = _find_pattern_by_color_mask(grid, DOOR_SIG, DOOR_BBOX_W, DOOR_BBOX_H)
    if match is None:
        return None
    (x0, y0), color = match
    cells = frozenset((x0 + dx, y0 + dy) for dx, dy in DOOR_SIG)
    return GameObject(color=color, cells=cells)


def classify_frame(grid: np.ndarray, objects: list) -> dict:
    """Best-effort role classification for one frame:
    {'player': PlayerSprite|None, 'key': GameObject|None, 'door': GameObject|None}.
    Takes BOTH the raw grid (door needs a direct scan) and the
    pre-extracted object list (player/key use connected components).
    Does NOT attempt to classify yellow bonuses here -- their shape isn't
    given precisely in the writeup, so identify them separately (e.g. via
    HypothesisEngine.active_vanish_rules() where a color vanishes near the
    player and the move-budget resets, which IS a reliable, learnable
    signature -- see resources.py).
    """
    return {
        "player": detect_player(objects),
        "key": detect_key(objects),
        "door": detect_door(grid),
    }
