"""
Isolated proof that bfs_to_frontier_goal_directed beats the original
nearest-by-hops bfs_to_frontier on a branching maze -- the exact failure
mode diagnosed from the live run (35 graph nodes / 260 actions, never
connecting to goal): the old frontier selection can pick a CLOSER (by hop
count) but goal-IRRELEVANT branch over a slightly farther one that
actually leads toward the goal.

Maze (a Y-junction): from J, a SHORT branch (2 cells) goes to a dead end
far from goal; a LONGER branch (4 cells) goes toward the goal.

    J--o--o                    (short branch, 2 hops, away from goal)
    |
    o--o--o--o....G            (long branch, toward goal, more hops)

Old logic (fewest hops wins): picks the short branch first -- wastes
exploration on a dead end.
New logic (hops + manhattan-to-goal wins): should prefer the branch that
is actually goal-directed, even though it's more hops away.
"""

from __future__ import annotations

import sys
import types
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _install_stubs_if_needed() -> None:
    try:
        import arcengine  # noqa: F401
        return
    except Exception:
        pass

    class GameState(str, Enum):
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    class GameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

    class FrameData:
        pass

    class FrameDataRaw(FrameData):
        pass

    m = types.ModuleType("arcengine")
    m.FrameData = FrameData
    m.FrameDataRaw = FrameDataRaw
    m.GameAction = GameAction
    m.GameState = GameState
    sys.modules["arcengine"] = m

    class EnvironmentWrapper:
        pass

    arc_agi_mod = types.ModuleType("arc_agi")
    arc_agi_mod.EnvironmentWrapper = EnvironmentWrapper
    sys.modules["arc_agi"] = arc_agi_mod

    class EnvironmentScorecard:
        pass

    arc_agi_scorecard_mod = types.ModuleType("arc_agi.scorecard")
    arc_agi_scorecard_mod.EnvironmentScorecard = EnvironmentScorecard
    sys.modules["arc_agi.scorecard"] = arc_agi_scorecard_mod

    try:
        import pydantic  # noqa: F401
    except Exception:
        pydantic_mod = types.ModuleType("pydantic")

        class ValidationError(Exception):
            pass

        pydantic_mod.ValidationError = ValidationError
        sys.modules["pydantic"] = pydantic_mod


_install_stubs_if_needed()

from world_model.stagnation_graph import StagnationAwareTransitionGraph  # noqa: E402

ACTION_DIRECTION = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}


def build_known_graph_with_two_branches():
    """Manually construct a graph (bypassing real gameplay) representing
    the Y-junction described above, with junction J at (0,0), short dead
    end branch going UP (away from goal), long branch going RIGHT
    (toward goal, far off to the right)."""
    g = StagnationAwareTransitionGraph(ACTION_DIRECTION)

    # fully explore the junction itself (0,0) first -- in real gameplay,
    # STAGNATION only fires after the agent has been moving around, so the
    # current position is rarely a frontier itself. ACTION2/ACTION3 are
    # walls here (no movement) so (0,0) becomes fully charted.
    g.record((0, 0), "ACTION2", (0, 0))
    g.record((0, 0), "ACTION3", (0, 0))

    # short branch: J(0,0) -up-> (0,-1) -up-> (0,-2) [dead end, frontier]
    g.record((0, 0), "ACTION1", (0, -1))
    g.record((0, -1), "ACTION1", (0, -2))
    # try one direction FROM (0,-2) so it appears as a key in edges with
    # <4 known directions (a wall bump — no movement) -- this is what
    # actually makes it recognisable as a frontier node; a position that
    # was only ever arrived AT, never acted FROM, isn't a graph key yet.
    g.record((0, -2), "ACTION3", (0, -2))

    # long branch: J(0,0) -right-> (1,0) -right-> (2,0) -right-> (3,0) [frontier]
    g.record((0, 0), "ACTION4", (1, 0))
    g.record((1, 0), "ACTION4", (2, 0))
    g.record((2, 0), "ACTION4", (3, 0))
    g.record((3, 0), "ACTION3", (3, 0))
    # (3,0) also a frontier node (only 2 of 4 directions explored)

    goal = (10, 0)  # far to the right -- long branch direction is correct
    return g, goal


def main() -> None:
    g, goal = build_known_graph_with_two_branches()
    start = (0, 0)

    old_path = g.bfs_to_frontier(start)
    new_path = g.bfs_to_frontier_goal_directed(start, goal)

    print("=== Frontier selection comparison ===")
    print(f"start={start} goal={goal}")
    print(f"OLD bfs_to_frontier (nearest by hops):        {old_path}")
    print(f"NEW bfs_to_frontier_goal_directed:              {new_path}")

    ok = True
    if old_path == ["ACTION1"]:
        print("\n[CONFIRMED] OLD logic picks the SHORT branch (ACTION1, away "
              "from goal) -- exactly the wasted-exploration bug from the live run.")
    else:
        print(f"\n[NOTE] OLD logic didn't pick the short branch as expected "
              f"(got {old_path}) -- maze construction may need adjustment.")

    if new_path == ["ACTION4"]:
        print("[PASS] NEW logic picks the LONG branch (ACTION4, toward goal) "
              "-- correctly prioritises goal-relevant exploration.")
    else:
        print(f"[FAIL] NEW logic did not prefer the goal-directed branch "
              f"(got {new_path})")
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
