"""Calibration helper: dumps every distinct object (color, size, bbox,
shape_signature) found in one real captured frame, so the exact
PLAYER_TOP_SIG / PLAYER_BOTTOM_SIG / KEY_SIG / DOOR_SIG constants in
shapes.py can be corrected against REAL data instead of the mechanics
writeup's prose description (confirmed live: they don't match yet --
`classify_frame` found nothing for an entire ls20 run).

Usage
-----
Against a recording produced by this repo's Recorder (JSON Lines, one
frame per line, each with a "frame" field holding the raw grid):

    python3 -m world_model.calibrate path/to/some.recording.jsonl

Or, from inside a running agent (e.g. temporarily in choose_action, for
one frame only):

    from world_model.calibrate import dump_grid
    dump_grid(grid)  # grid = grid_to_array(latest_frame.frame)

What to look for in the output:
  - The PLAYER should be the two objects that move between frames: one
    small rectangle atop another, different colors, in the same
    x-range, adjacent rows. Compare their `width`/`height` against
    PLAYER_TOP_SIG (5 wide x 2 tall) / PLAYER_BOTTOM_SIG (5 wide x 3
    tall) in shapes.py -- if the real player is a different size
    (e.g. 3x3, or a single color), update PLAYER_WIDTH/PLAYER_HEIGHT and
    the *_SIG tuples to match.
  - The KEY should be a small (~5 cell) object that DISAPPEARS after
    the player walks onto it once. Compare its cell layout to KEY_SIG.
  - The DOOR should be a small, PERSISTENT object (never disappears,
    unlike the key) near where the level actually ends. Compare its
    cell layout to DOOR_SIG -- remember DOOR_SIG is intentionally
    disconnected in the writeup's own diagram, so `detect_door` scans
    the raw grid directly rather than using connected components; if
    the real door pattern IS connected, a plain shape_signature() match
    via extract_objects will work just as well and is simpler to debug.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .objects import extract_objects


def dump_grid(grid: np.ndarray, label: str = "") -> None:
    objects = extract_objects(grid)
    print(f"--- frame {label}: grid shape {grid.shape}, {len(objects)} objects ---")
    for obj in sorted(objects, key=lambda o: o.size):
        x0, y0, x1, y1 = obj.bbox
        print(
            f"  color={obj.color:>3}  size={obj.size:>4}  "
            f"bbox=({x0},{y0})-({x1},{y1})  {obj.width}x{obj.height}  "
            f"shape_signature={obj.shape_signature()}"
        )


def dump_recording(path: str, max_frames: int = 5) -> None:
    with open(path) as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Real recordings from this repo's Recorder wrap each frame as
            # {"timestamp": ..., "data": {"frame": [[[...64x64 grid...]]], ...}}
            # -- one JSON line per step, "frame" is a list of layers (usually
            # just one), each layer a 64x64 grid of color indices. Confirmed
            # directly against a real ls20 recording, 2026-08-30.
            data = record.get("data", record)
            frame = data.get("frame")
            if frame is None:
                continue
            grid = np.array(frame[0])
            dump_grid(grid, label=f"{count} (step-in-file, state={data.get('state')})")
            count += 1
            if count >= max_frames:
                break
    if count == 0:
        print("No frames with a 'data.frame' field found -- print one raw line "
              "(json.loads(line)) to see its actual keys and adjust dump_recording() above.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m world_model.calibrate path/to/some.recording.jsonl [max_frames]")
        sys.exit(1)
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    dump_recording(sys.argv[1], max_frames=max_frames)
