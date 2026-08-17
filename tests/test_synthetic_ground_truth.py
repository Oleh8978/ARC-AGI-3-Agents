"""
Synthetic ground-truth validation.

Why this exists: the real recordings/*.jsonl in this repo have a data
quality problem -- action_input.id is 0 (RESET) on every single recorded
frame, in all 25 files, even when state == NOT_FINISHED. That means we
cannot currently trust them to prove the hypothesis engine learns
action -> movement correctly (see terminal output / README note this
script prints). Until that's fixed and re-verified on a machine with a
live connection to the ARC-AGI-3 server, THIS script is the source of
truth for "does the code actually work".

Method: build a tiny synthetic grid world with KNOWN rules (we wrote them,
we know ground truth), feed it through the exact same ObjectExtractor +
HypothesisEngine used in production, and assert the engine recovers the
known rules and rejects the wrong ones. If this fails, the bug is in
world_model/, not in recording data -- fix here first.

Run:
    python tests/test_synthetic_ground_truth.py
Exit code 0 = all assertions passed. Non-zero = something is broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from world_model.objects import extract_objects
from world_model.hypotheses import HypothesisEngine

BG = 0
PLAYER = 2
WALL_COLOR = 1
GOAL = 3
COIN = 4

ACTION_DELTA = {
    "ACTION1": (0, -1),   # up
    "ACTION2": (0, 1),    # down
    "ACTION3": (-1, 0),   # left
    "ACTION4": (1, 0),    # right
}


def make_grid(player_xy, walls, goal_xy=None, coin_xy=None, size=10) -> np.ndarray:
    g = np.full((size, size), BG, dtype=np.int16)
    for (x, y) in walls:
        g[y, x] = WALL_COLOR
    if goal_xy:
        g[goal_xy[1], goal_xy[0]] = GOAL
    if coin_xy:
        g[coin_xy[1], coin_xy[0]] = COIN
    px, py = player_xy
    g[py, px] = PLAYER
    return g


def step_ground_truth(player_xy, action, walls, size=10):
    """The actual known rule: player moves by ACTION_DELTA, blocked by
    walls and the grid boundary. This is the ground truth the engine
    must (approximately) recover."""
    dx, dy = ACTION_DELTA[action]
    nx, ny = player_xy[0] + dx, player_xy[1] + dy
    if not (0 <= nx < size and 0 <= ny < size):
        return player_xy
    if (nx, ny) in walls:
        return player_xy
    return (nx, ny)


def run_case(name, assertion_fn):
    try:
        assertion_fn()
        print(f"[PASS] {name}")
        return True
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        return False


def test_learns_constant_delta_with_no_walls():
    """Open room, no walls: every ACTION1..4 should converge to a stable,
    non-falsified delta matching ACTION_DELTA exactly."""
    engine = HypothesisEngine()
    player = (5, 5)
    walls: set = set()

    actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"] * 4
    for action in actions:
        before_grid = make_grid(player, walls)
        before_objs = extract_objects(before_grid)
        new_player = step_ground_truth(player, action, walls)
        after_grid = make_grid(new_player, walls)
        after_objs = extract_objects(after_grid)
        engine.observe(before_objs, action, after_objs)
        player = new_player

    for action, expected_delta in ACTION_DELTA.items():
        got = engine.predict_delta(PLAYER, action)
        assert got == expected_delta, (
            f"{action}: expected {expected_delta}, engine predicted {got} "
            f"(hypothesis object: {engine.constant_delta.get((PLAYER, action))})"
        )

    assert engine.best_player_color() == PLAYER, (
        f"expected best_player_color()=={PLAYER}, got {engine.best_player_color()}"
    )


def test_falsifies_delta_when_wall_sometimes_blocks():
    """Player tries to move right repeatedly; a wall blocks it exactly
    once mid-sequence. The engine MUST falsify the ACTION4 constant-delta
    hypothesis for PLAYER (movement isn't a fixed vector -- sometimes 0,
    sometimes (1,0)) rather than silently averaging or ignoring it."""
    engine = HypothesisEngine()
    player = (2, 5)
    walls = {(4, 5)}  # a wall two steps to the right

    for _ in range(5):
        before_grid = make_grid(player, walls)
        before_objs = extract_objects(before_grid)
        new_player = step_ground_truth(player, "ACTION4", walls)
        after_grid = make_grid(new_player, walls)
        after_objs = extract_objects(after_grid)
        engine.observe(before_objs, "ACTION4", after_objs)
        player = new_player

    h = engine.constant_delta.get((PLAYER, "ACTION4"))
    assert h is not None, "expected a hypothesis object to exist for (PLAYER, ACTION4)"
    assert h.falsified, (
        f"expected ACTION4/PLAYER hypothesis to be FALSIFIED (blocked once, moved "
        f"other times) but it wasn't. deltas_seen={h.deltas_seen}"
    )
    assert engine.predict_delta(PLAYER, "ACTION4") is None, (
        "predict_delta must refuse to predict for a falsified hypothesis"
    )


def test_identifies_immobile_background_object():
    """A wall/goal marker that never moves should end up in
    active_immobile_colors(), and must NOT be mistaken for the player
    even though it's a distinct, trackable object."""
    engine = HypothesisEngine()
    player = (1, 1)
    walls: set = set()
    goal_xy = (8, 8)

    actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"] * 3
    for action in actions:
        before_grid = make_grid(player, walls, goal_xy=goal_xy)
        before_objs = extract_objects(before_grid)
        new_player = step_ground_truth(player, action, walls)
        after_grid = make_grid(new_player, walls, goal_xy=goal_xy)
        after_objs = extract_objects(after_grid)
        engine.observe(before_objs, action, after_objs)
        player = new_player

    assert GOAL in engine.active_immobile_colors(), (
        f"expected GOAL color {GOAL} in active_immobile_colors(), "
        f"got {engine.active_immobile_colors()}"
    )
    assert engine.best_player_color() == PLAYER, (
        f"expected best_player_color()=={PLAYER}, got {engine.best_player_color()} "
        f"-- engine may be confusing the immobile goal marker for the player"
    )


def test_vanish_near_rule_detects_coin_pickup():
    """A coin that disappears exactly when the player becomes adjacent to
    it should trigger an active vanish_near rule; a coin that disappears
    for no adjacency reason (e.g. a timer) should NOT."""
    engine = HypothesisEngine()
    player = (0, 5)
    coin_xy = (2, 5)
    walls: set = set()

    # step 1: move right, not yet adjacent
    before_grid = make_grid(player, walls, coin_xy=coin_xy)
    before_objs = extract_objects(before_grid)
    player = step_ground_truth(player, "ACTION4", walls)
    after_grid = make_grid(player, walls, coin_xy=coin_xy)
    after_objs = extract_objects(after_grid)
    engine.observe(before_objs, "ACTION4", after_objs)

    # step 2: move right again -> now adjacent to coin -> coin vanishes
    before_grid = make_grid(player, walls, coin_xy=coin_xy)
    before_objs = extract_objects(before_grid)
    player = step_ground_truth(player, "ACTION4", walls)
    after_grid = make_grid(player, walls, coin_xy=None)  # coin picked up
    after_objs = extract_objects(after_grid)
    engine.observe(before_objs, "ACTION4", after_objs)

    rules = engine.active_vanish_rules()
    assert any(r.color_a == COIN and r.color_b == PLAYER for r in rules), (
        f"expected an active vanish_near(COIN, PLAYER) rule, got {rules}"
    )


def main() -> None:
    print("=== Synthetic ground-truth tests (world_model engine) ===\n")
    results = [
        run_case("learns constant delta, no walls", test_learns_constant_delta_with_no_walls),
        run_case("falsifies delta when wall sometimes blocks", test_falsifies_delta_when_wall_sometimes_blocks),
        run_case("identifies immobile background object, not player", test_identifies_immobile_background_object),
        run_case("vanish-near rule detects coin pickup", test_vanish_near_rule_detects_coin_pickup),
    ]
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} passed")

    print("\n--- KNOWN DATA ISSUE (separate from this test) ---")
    print("recordings/*.recording.jsonl in this repo all show action_input.id == 0")
    print("(RESET) on every frame, including frames where state == NOT_FINISHED.")
    print("That contradicts choose_action() in goal_directed_agent.py, which only")
    print("returns RESET when state is NOT_PLAYED or GAME_OVER. Do NOT trust those")
    print("recordings for action-conditioned validation until you've confirmed on a")
    print("machine with live server access that fresh recordings populate")
    print("action_input.id correctly. This synthetic test is deliberately")
    print("independent of that data path.")

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
