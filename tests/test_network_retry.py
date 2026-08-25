"""
Isolated test for call_with_network_retry (agents/templates/
hypothesis_world_agent.py) -- proves two things without any real network:

1. A function that raises requests.exceptions.RequestException (the
   parent class of ReadTimeout, ConnectionError -- the exact failure mode
   seen live against three.arcprize.org) a few times, then succeeds, IS
   retried and its eventual success IS returned.
2. A function that raises a non-network exception (e.g. a real bug) is
   NEVER retried -- it propagates immediately on the first failure. This
   matters: silently retrying real bugs would hide them.
"""

from __future__ import annotations

import sys
import types
from enum import Enum
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _install_stubs_if_needed() -> None:
    try:
        import arcengine  # noqa: F401
        import arc_agi  # noqa: F401
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

        def is_complex(self) -> bool:
            return self is GameAction.ACTION6

        def is_simple(self) -> bool:
            return not self.is_complex()

        @classmethod
        def from_id(cls, action_id: int) -> "GameAction":
            for a in cls:
                if a.value == action_id:
                    return a
            raise ValueError(f"unknown action id {action_id}")

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

try:
    import requests  # noqa: F401
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

from agents.templates.hypothesis_world_agent import call_with_network_retry  # noqa: E402


def test_retries_then_succeeds_on_network_error() -> bool:
    if not HAVE_REQUESTS:
        print("[SKIP] requests not installed here -- run on your machine "
              "with the real venv for this test to mean anything")
        return True

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ReadTimeout("simulated read timeout")
        return "success"

    with patch("time.sleep"):  # don't actually wait during the test
        result = call_with_network_retry(flaky, attempts=3, delay_seconds=2.0, log_prefix="[test]")

    ok = result == "success" and calls["n"] == 3
    print(f"[{'PASS' if ok else 'FAIL'}] retried {calls['n']} times, got result={result!r}")
    return ok


def test_does_not_retry_non_network_error() -> bool:
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("a real bug, not a network issue")

    raised = False
    try:
        with patch("time.sleep"):
            call_with_network_retry(broken, attempts=3, delay_seconds=2.0, log_prefix="[test]")
    except ValueError:
        raised = True

    ok = raised and calls["n"] == 1
    print(f"[{'PASS' if ok else 'FAIL'}] non-network error propagated immediately "
          f"after {calls['n']} call(s) (expected 1, no retries)")
    return ok


def test_retries_on_swallowed_network_error_valueerror() -> bool:
    """Reproduces the EXACT live crash: arc_agi/remote_wrapper.py catches
    the real requests exception internally and returns None, which the
    base Agent then turns into ValueError('Received None frame data from
    environment') -- a different exception type than the original network
    error. This must ALSO be retried, or the retry mechanism is useless
    for the actual failure mode seen in practice (confirmed live: the
    first version of call_with_network_retry did NOT catch this and the
    run crashed after only 4 actions)."""
    calls = {"n": 0}

    def flaky_via_sdk_swallow():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("Received None frame data from environment")
        return "success"

    with patch("time.sleep"):
        result = call_with_network_retry(flaky_via_sdk_swallow, attempts=3, delay_seconds=2.0, log_prefix="[test]")

    ok = result == "success" and calls["n"] == 3
    print(f"[{'PASS' if ok else 'FAIL'}] retried {calls['n']} times on the swallowed-network-error "
          f"ValueError signature, got result={result!r}")
    return ok


def main() -> None:
    print("=== call_with_network_retry isolated tests ===")
    results = [
        test_retries_then_succeeds_on_network_error(),
        test_does_not_retry_non_network_error(),
        test_retries_on_swallowed_network_error_valueerror(),
    ]
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} passed")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
