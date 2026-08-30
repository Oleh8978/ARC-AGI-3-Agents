"""Top-level orchestrator: given one raw frame + the current move-budget
state + whether the key has been collected, returns the next action to
take. This is the piece an ``Agent.choose_action`` implementation should
actually call every frame; it ties together every other module in this
package:

  1. ``objects.extract_objects`` + ``shapes.classify_frame`` -- find the
     player / key / door BY SHAPE (never by "smallest static color").
  2. ``footprint.build_passable_grid`` + ``build_footprint_graph`` --
     compute the full 5x5-footprint-aware position graph directly from
     this frame's geometry (no exploration needed for this part).
  3. ``resources.plan_key_door_route`` -- compare the direct KEY->DOOR
     route against every single-bonus-detour route, survivability
     checked against the current move budget, and pick the cheapest
     survivable one (the writeup's variants A/B/C).
  4. Returns just the FIRST action of the winning plan -- re-plan every
     frame rather than blindly executing a stored plan, so a wrong
     wall-color guess or an unexpected push-tile is corrected on the
     very next frame instead of compounding.

What this deliberately does NOT do:
  - It does not identify yellow bonuses (shape not given precisely in
    the writeup -- wire in your own detector, e.g. via
    ``HypothesisEngine.active_vanish_rules()``, and pass their positions
    in as ``bonus_positions``).
  - It does not implement the key-rotation/orientation-matching
    mechanic. That mechanic is naturally representable by extending
    ``held`` in ``stateful_graph.StatefulTransitionGraph`` to something
    like ``(has_key, key_orientation)`` and letting BFS over that
    augmented state discover the rule empirically -- deliberately not
    hardcoded here (see ``stateful_graph.py``'s module docstring).
  - It does not know the wall color -- you must supply it (see
    ``footprint.build_passable_grid``'s docstring for how to obtain it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .footprint import build_footprint_graph, build_passable_grid, bfs_shortest_path
from .objects import extract_objects
from .resources import ResourceState, plan_key_door_route
from .shapes import classify_frame

Pos = tuple[int, int]


@dataclass
class PlanResult:
    next_action: Optional[str]
    full_path: Optional[list]
    label: Optional[str]  # "direct" / "via_bonus_0_before_key" / "to_door" / None


class KeyDoorPlanner:
    def __init__(self, action_direction: dict, wall_colors: set, footprint_size: int = 5):
        self.action_direction = action_direction
        self.wall_colors = wall_colors
        self.footprint_size = footprint_size

    def plan(
        self,
        grid: np.ndarray,
        resource: ResourceState,
        has_key: bool,
        bonus_positions: Optional[list] = None,
    ) -> PlanResult:
        bonus_positions = bonus_positions or []

        objects = extract_objects(grid)
        classified = classify_frame(grid, objects)
        player = classified["player"]
        key_obj = classified["key"]
        door_obj = classified["door"]

        if player is None or door_obj is None:
            # Can't even see the essentials this frame -- caller should
            # fall back to exploration (e.g. StagnationAwareTransitionGraph)
            # rather than trust a plan built on missing information.
            return PlanResult(None, None, None)

        passable = build_passable_grid(grid, wall_colors=self.wall_colors)
        graph = build_footprint_graph(
            passable, self.footprint_size, self.footprint_size, self.action_direction
        )

        def graph_bfs(a: Pos, b: Pos):
            return bfs_shortest_path(graph, a, b)

        start = player.top_left
        door_pos = (door_obj.bbox[0], door_obj.bbox[1])

        if not has_key:
            if key_obj is None:
                return PlanResult(None, None, None)
            key_pos = (key_obj.bbox[0], key_obj.bbox[1])
            result = plan_key_door_route(
                graph_bfs, start, key_pos, door_pos, bonus_positions, resource
            )
            if result is None:
                return PlanResult(None, None, None)
            total, path, label = result
            return PlanResult(path[0] if path else None, path, label)

        # Key already held: just go straight to the door.
        path = graph_bfs(start, door_pos)
        if not path:
            return PlanResult(None, None, None)
        return PlanResult(path[0], path, "to_door")
