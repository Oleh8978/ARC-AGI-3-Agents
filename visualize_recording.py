"""
Visualize a local ARC-AGI-3 recording.jsonl file as a sequence of PNG
frames (and an optional GIF), so we can SEE what the game actually looks
like without relying on the (currently inaccessible) browser replay UI.

Usage:
    python visualize_recording.py recordings/<file>.recording.jsonl
    python visualize_recording.py recordings/<file>.recording.jsonl --steps 10-20

Outputs:
    recording_frames/frame_0000.png, frame_0001.png, ...
    recording_frames/animation.gif  (if Pillow is available)

Also prints a per-step summary table (action taken, state, levels_completed)
so we can correlate the death at step ~128 with what's visually happening.
"""

import json
import sys
import argparse
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("NOTE: Pillow not installed (pip install Pillow) — will still dump "
          "a text summary and raw grids, just no PNG/GIF images.")

# ARC-AGI-3 uses a fixed 16-color palette (same as ARC-AGI-1/2)
ARC_PALETTE = [
    (0, 0, 0),        # 0 black
    (0, 116, 217),    # 1 blue
    (255, 65, 54),    # 2 red
    (46, 204, 64),    # 3 green
    (255, 220, 0),    # 4 yellow
    (170, 170, 170),  # 5 grey
    (240, 18, 190),   # 6 magenta/pink
    (255, 133, 27),   # 7 orange
    (127, 219, 255),  # 8 light blue / cyan
    (135, 12, 37),    # 9 dark red/maroon
]


def color_for(v: int):
    return ARC_PALETTE[v % len(ARC_PALETTE)]


def frame_to_image(frame, cell_size: int = 12):
    """frame is list[list[list[int]]] — a stack of 2D grids (layers).
    We render the FIRST layer (most ARC-AGI-3 games are single-layer;
    if multi-layer, this can be extended to composite them)."""
    grid = frame[0] if frame else [[0]]
    h, w = len(grid), len(grid[0]) if grid else 0
    img = Image.new("RGB", (w * cell_size, h * cell_size))
    pixels = img.load()
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            c = color_for(v)
            for dy in range(cell_size):
                for dx in range(cell_size):
                    pixels[x * cell_size + dx, y * cell_size + dy] = c
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize ARC-AGI-3 recording.jsonl frames")
    parser.add_argument("path", help="Path to recording.jsonl")
    parser.add_argument(
        "--steps", type=str, default=None,
        help="Explicit step range to render, e.g. '10-20'. "
             "Overrides auto death-detection."
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    out_dir = Path("recording_frames")
    out_dir.mkdir(exist_ok=True)

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    print(f"Loaded {len(entries)} recorded steps from {path.name}\n")

    # Print compact summary table
    print(f"{'step':>5} {'action':<10} {'state':<14} {'score':>6}")
    print("-" * 40)
    death_step = None
    explicit_window = None
    if args.steps:
        try:
            lo, hi = args.steps.split("-")
            explicit_window = range(int(lo), min(len(entries), int(hi) + 1))
        except ValueError:
            print(f"Invalid --steps format: {args.steps!r}, expected e.g. '10-20'")
            sys.exit(1)

    for i, e in enumerate(entries):
        d = e.get("data", e)  # tolerate both wrapped/unwrapped formats
        action = d.get("action_input", {}).get("id", "?")
        state = d.get("state", "?")
        score = d.get("score", "?")
        if state == "GAME_OVER" and death_step is None:
            death_step = i
        marker = "  <-- DEATH" if state == "GAME_OVER" else ""
        show_row = (
            i < 5 or i > len(entries) - 5
            or (death_step is not None and abs(i - death_step) <= 3)
            or (explicit_window is not None and i in explicit_window)
        )
        if show_row:
            print(f"{i:>5} {str(action):<10} {state:<14} {str(score):>6}{marker}")

    if explicit_window is not None:
        print(f"\n>>> Using explicit step range: {args.steps}")
        window = explicit_window
    elif death_step is not None:
        print(f"\n>>> Death detected at recorded step {death_step}")
        window = range(max(0, death_step - 5), min(len(entries), death_step + 3))
    else:
        print("\n>>> No GAME_OVER found in this recording")
        window = range(0, min(len(entries), 10))

    if HAS_PIL:
        print(f"\nSaving PNG frames for steps {list(window)} to {out_dir}/")
        for i in window:
            d = entries[i].get("data", entries[i])
            frame = d.get("frame")
            if not frame:
                continue
            img = frame_to_image(frame)
            img.save(out_dir / f"frame_{i:04d}.png")
        print("Done. Open the PNGs to see exactly what happened around the death.")

        # Also save a full animation for the whole run if not too long
        if len(entries) <= 250:
            print("Building full animation.gif (all steps)...")
            frames_imgs = []
            for e in entries:
                d = e.get("data", e)
                frame = d.get("frame")
                if frame:
                    frames_imgs.append(frame_to_image(frame))
            if frames_imgs:
                frames_imgs[0].save(
                    out_dir / "animation.gif",
                    save_all=True,
                    append_images=frames_imgs[1:],
                    duration=150,
                    loop=0,
                )
                print(f"Saved {out_dir}/animation.gif ({len(frames_imgs)} frames)")
    else:
        print("\nInstall Pillow to get PNG/GIF output: pip install Pillow")


if __name__ == "__main__":
    main()