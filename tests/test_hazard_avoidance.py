"""
Isolated proof that goal_biased_exploration's hazard penalty actually
changes the decision when a hazard sits near one candidate but not
another, WITHOUT breaking normal scoring when there's no hazard nearby
(covered separately by the regression in test_agent_integration_synthetic.py,
which passes unchanged with hazard_positions=None).

Scenario: player at (0,0), goal far to the east. Two candidate actions:
ACTION4 (east, toward goal, uncharted) lands adjacent to a hazard.
ACTION1 (north, away from goal, uncharted) does not. Without the hazard
penalty, ACTION4 wins easily (lower manhattan). With a hazard at (2,0)
(adjacent to ACTION4's landing cell (1,0)), the penalty should be enough
to flip the choice toward ACTION1 in this deliberately close scenario.
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

    class FrameData:
        pass

    m = types.ModuleType("arcengine")
    m.FrameData = FrameData
    m.FrameDataRaw = FrameData
    m.GameAction = GameAction
    m.GameState = GameState
    sys.modules["arcengine"] = m

    arc_agi_mod = types.ModuleType("arc_agi")
    arc_agi_mod.EnvironmentWrapper = type("EnvironmentWrapper", (), {})
    sys.modules["arc_agi"] = arc_agi_mod
    arc_agi_scorecard_mod = types.ModuleType("arc_agi.scorecard")
    arc_agi_scorecard_mod.EnvironmentScorecard = type("EnvironmentScorecard", (), {})
    sys.modules["arc_agi.scorecard"] = arc_agi_scorecard_mod

    try:
        import pydantic  # noqa: F401
    except Exception:
        pydantic_mod = types.ModuleType("pydantic")
        pydantic_mod.ValidationError = type("ValidationError", (Exception,), {})
        sys.modules["pydantic"] = pydantic_mod


_install_stubs_if_needed()

from world_model.stagnation_graph import StagnationAwareTransitionGraph  # noqa: E402

ACTION_DIRECTION = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}


class FakeAction:
    def __init__(self, name):
        self.name = name


CANDIDATES = [FakeAction("ACTION1"), FakeAction("ACTION4")]  # only the two relevant ones


def main() -> None:
    pos = (0, 0)
    goal = (10, 0)  # far east -- ACTION4 (east) is normally clearly better

    g = StagnationAwareTransitionGraph(ACTION_DIRECTION)

    # --- without hazard: ACTION4 (toward goal) should win easily ---
    choice_no_hazard = g.goal_biased_exploration(pos, CANDIDATES, goal, hazard_positions=None)
    print(f"No hazard nearby -> chosen action: {choice_no_hazard.name}")

    # --- with hazard adjacent to ACTION4's landing cell (1,0) ---
    hazard_positions = {(2, 0)}  # adjacent (radius 1) to (1,0), NOT to (0,-1)
    choice_with_hazard = g.goal_biased_exploration(pos, CANDIDATES, goal, hazard_positions=hazard_positions)
    print(f"Hazard at {hazard_positions} -> chosen action: {choice_with_hazard.name}")

    ok = True
    if choice_no_hazard.name != "ACTION4":
        print(f"[FAIL] expected ACTION4 without hazard, got {choice_no_hazard.name}")
        ok = False
    else:
        print("[PASS] without hazard info, prefers the goal-directed action (ACTION4)")

    if choice_with_hazard.name != "ACTION1":
        print(f"[FAIL] expected hazard penalty to flip choice to ACTION1, got {choice_with_hazard.name}")
        ok = False
    else:
        print("[PASS] hazard penalty flips the choice to the safer action (ACTION1)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
