"""World-model package: object-centric perception, rule/hypothesis
induction, and empirical transition graphs (plain and state-augmented)
for the ARC-AGI-3 agents in this repo.

This package was referenced (imported) by
``agents/templates/hypothesis_world_agent.py`` and by every test under
``tests/`` (``test_synthetic_ground_truth.py``, ``test_offline_replay.py``,
``test_key_door_mechanic.py``, ``test_frontier_goal_direction.py``,
``test_hazard_avoidance.py``) but the folder itself was never actually
present in the delivered archive -- every one of those imports was a
guaranteed ``ModuleNotFoundError``. That is the actual root cause of
"the game mechanic doesn't work": the mechanic-aware code path could
never run at all, so every live run silently fell back to whatever else
was importable (the plain, key/door-unaware ``goal_directed_agent.py``).
"""

from .objects import GameObject, extract_objects, detect_held_indicator
from .hypotheses import HypothesisEngine, DeltaHypothesis, VanishNearRule
from .stagnation_graph import StagnationAwareTransitionGraph
from .stateful_graph import StatefulTransitionGraph, State
from .shapes import (
    PlayerSprite,
    detect_player,
    detect_key,
    detect_door,
    classify_frame,
    PLAYER_TOP_SIG,
    PLAYER_BOTTOM_SIG,
    KEY_SIG,
    DOOR_SIG,
)
from .footprint import (
    build_passable_grid,
    valid_top_lefts,
    build_footprint_graph,
    bfs_shortest_path,
)
from .resources import ResourceState, plan_key_door_route
from .sequential_planner import KeyDoorPlanner, PlanResult
from .journal import ObjectJournal, ObjectRecord, GoalCandidate

__all__ = [
    "GameObject",
    "extract_objects",
    "detect_held_indicator",
    "HypothesisEngine",
    "DeltaHypothesis",
    "VanishNearRule",
    "StagnationAwareTransitionGraph",
    "StatefulTransitionGraph",
    "State",
    "PlayerSprite",
    "detect_player",
    "detect_key",
    "detect_door",
    "classify_frame",
    "PLAYER_TOP_SIG",
    "PLAYER_BOTTOM_SIG",
    "KEY_SIG",
    "DOOR_SIG",
    "build_passable_grid",
    "valid_top_lefts",
    "build_footprint_graph",
    "bfs_shortest_path",
    "ResourceState",
    "plan_key_door_route",
    "KeyDoorPlanner",
    "PlanResult",
    "ObjectJournal",
    "ObjectRecord",
    "GoalCandidate",
]
