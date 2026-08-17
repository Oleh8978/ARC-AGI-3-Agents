"""
Offline validation against REAL recorded frames.

This is the "не впасти обличчям в грязь" step: before this code ever
touches a live game, it has to prove itself against actual recorded
gameplay from recordings/*.jsonl (already produced by your own agents,
per ARC-AGI-3-Agents/agents/recorder.py). No network, no API key needed.

Run:
    python tests/test_offline_replay.py /path/to/ARC-AGI-3-Agents/recordings

What "passing" means here, concretely (read this before trusting the
output):
  1. It must not crash on any real recorded frame sequence.
  2. best_player_color() must converge to a stable, non-None value within
     the recording and NOT flip-flop late in the run (a sign the rule
     induction is unstable, not that the game changed).
  3. active_constant_delta count must be > 0 (it actually learned
     something) but not equal to number of colors * actions (that would
     mean nothing was ever falsified, i.e. falsification logic is dead
     code).
  4. Report is printed per recording AND aggregated -- read the printed
     numbers yourself, don't just trust the exit code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from world_model.objects import extract_objects
from world_model.hypotheses import HypothesisEngine

ACTION_ID_TO_NAME = {
    1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4",
    5: "ACTION5", 6: "ACTION6", 7: "ACTION7",
}


def load_recording(path: Path) -> list[dict]:
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            steps.append(json.loads(line))
    return steps


def replay_one(path: Path, verbose: bool = True) -> dict:
    steps = load_recording(path)
    engine = HypothesisEngine()

    prev_grid = None
    prev_objects = None
    player_color_history: list[int | None] = []

    n_frames_processed = 0
    for i, step in enumerate(steps):
        d = step["data"]
        frame = d.get("frame")
        if not frame:
            continue
        grid = np.array(frame[0], dtype=np.int16)
        action_id = d.get("action_input", {}).get("id")
        action_name = ACTION_ID_TO_NAME.get(action_id)

        objects = extract_objects(grid)

        if prev_grid is not None and action_name is not None:
            # This is the transition that PRODUCED this frame: the action
            # recorded on THIS step is the one that was taken to get from
            # prev_grid to grid (matches the recorder's convention where
            # action_input is the action that led to this frame).
            engine.observe(prev_objects, action_name, objects)
            n_frames_processed += 1

        prev_grid, prev_objects = grid, objects
        player_color_history.append(engine.best_player_color())

    summary = engine.summary()
    summary["recording"] = path.name
    summary["frames_processed"] = n_frames_processed
    summary["total_steps_in_file"] = len(steps)

    # stability check: did the guessed player color change in the last
    # third of the run? if so, flag it -- that's a real problem, not noise.
    tail = player_color_history[len(player_color_history) * 2 // 3:]
    tail_nonnull = [c for c in tail if c is not None]
    summary["player_color_stable_in_tail"] = (
        len(set(tail_nonnull)) <= 1 if tail_nonnull else False
    )
    summary["player_color_never_identified"] = summary["best_player_color"] is None

    if verbose:
        print(f"\n--- {path.name} ---")
        for k, v in summary.items():
            if k != "recording":
                print(f"  {k}: {v}")

    return summary


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python test_offline_replay.py <recordings_dir> [max_files]")
        sys.exit(1)

    rec_dir = Path(sys.argv[1])
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    files = sorted(rec_dir.glob("*.recording.jsonl"))[:max_files]

    if not files:
        print(f"No .recording.jsonl files found in {rec_dir}")
        sys.exit(1)

    print(f"Replaying {len(files)} recording(s) from {rec_dir} ...")
    results = [replay_one(f) for f in files]

    print("\n=== AGGREGATE ===")
    n = len(results)
    n_identified = sum(1 for r in results if not r["player_color_never_identified"])
    n_stable = sum(1 for r in results if r["player_color_stable_in_tail"])
    avg_active = sum(r["active_constant_delta"] for r in results) / n
    avg_falsified = sum(r["falsified_constant_delta"] for r in results) / n

    print(f"recordings replayed:                 {n}")
    print(f"player color identified:             {n_identified}/{n}")
    print(f"player color stable in final third:  {n_stable}/{n}")
    print(f"avg active (non-falsified) rules:    {avg_active:.1f}")
    print(f"avg falsified rules:                 {avg_falsified:.1f}")

    if n_identified < n:
        print("\n[WARN] player color not identified on all recordings --"
              " look at those files specifically before trusting the agent on live games.")
    if avg_falsified == 0:
        print("\n[WARN] nothing was ever falsified across ANY recording --"
              " falsification logic is probably not being exercised. Investigate"
              " before believing the CEGIS claim in the paper.")
    if n_stable < n:
        print("\n[WARN] player color guess flipped late in some run(s) --"
              " that's a real instability, not noise. Do not paper over it.")


if __name__ == "__main__":
    main()
