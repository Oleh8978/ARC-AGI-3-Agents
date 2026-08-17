"""
Hypothesis-World Agent: object-centric perception + CEGIS-style hypothesis
falsification, feeding the SAME proven TransitionGraph/GoalDetector/BFS
planner from goal_directed_agent.py.

Deliberately does NOT rewrite the planner. goal_directed_agent.py's BFS
planning, frontier exploration, and goal-biased fallback exploration are
reused unchanged (imported below) -- they were validated across 6+ live
runs per the docstring in that file, and per our own critique, that logic
was never the weak part. The weak part was:
  1. player identification via a single hardcoded ACTION_DIRECTION table
     matched against raw color-centroid drift (breaks on multi-instance
     colors, can't tell player from a scenery object that happens to
     drift with camera-like motion), and
  2. zero model of any object other than "player" and "goal" -- no way to
     reason about hazards, collectibles, or moving obstacles.

This file replaces (1) with HypothesisEngine (per-color, per-action,
falsifiable delta hypotheses over real connected-component objects) and
adds the scaffolding for (2) via active_immobile_colors()/
active_vanish_rules(), without yet wiring vanish-rules into planning --
that's flagged as a TODO, not silently pretended to be done.

Drop into: agents/templates/hypothesis_world_agent.py
Run:       uv run main.py --agent=hypothesisworldagent --game=ls20

BEFORE trusting this on a live game: run
    python tests/test_agent_integration_synthetic.py
which drives this exact class (choose_action/append_frame) through a
synthetic multi-step maze with known ground truth, with NO network and
NO arcengine session required. See that file for what "passing" proves
and does not prove.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .goal_directed_agent import (
    ACTION_DIRECTION,
    GoalDetector,
    Pos,
    TransitionGraph,
    color_centroids,
    grid_to_array,
)
from world_model.objects import extract_objects
from world_model.hypotheses import HypothesisEngine

logger = logging.getLogger()


class HypothesisWorldAgent(Agent):
    """Same planning loop as GoalDirectedAgent; player/goal identification
    driven by HypothesisEngine + object-centric perception instead of
    ColorRegionTracker + raw centroids."""

    MAX_ACTIONS = 500
    CALIBRATION_CYCLE = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    # HypothesisEngine needs at least 2 supporting observations per
    # (color, action) pair before best_player_color() will consider it
    # (see HypothesisEngine.best_player_color). With 4 actions that's a
    # firm floor; we cycle a bit longer to get repeated evidence per action.
    MIN_CALIBRATION_STEPS = 10

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.engine = HypothesisEngine()
        self.goal_detector = GoalDetector()
        self.world_model = TransitionGraph()
        self.player_color: Optional[int] = None
        self.goal_color: Optional[int] = None
        self._pending_action: Optional[GameAction] = None
        self._pending_objects_before = None
        self._pending_pos_before: Optional[Pos] = None
        self._action_count = 0
        self._planned_path: list[str] = []
        self._last_levels_completed: int = 0
        random.seed(random.randint(0, 10**9))

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # ------------------------------------------------------------------
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET

        grid = grid_to_array(latest_frame.frame)
        candidates_raw = [GameAction.from_id(a) for a in latest_frame.available_actions]
        candidates = [a for a in candidates_raw if a is not GameAction.RESET]
        if not candidates:
            candidates = [a for a in GameAction
                          if a is not GameAction.RESET and a.is_simple()]

        self.goal_detector.observe(grid, self.player_color)

        if self.player_color is None and self._action_count >= self.MIN_CALIBRATION_STEPS:
            self.player_color = self.engine.best_player_color()
            if self.player_color is not None:
                logger.info(f"[hyp-agent] player color identified: {self.player_color} "
                            f"(engine summary: {self.engine.summary()})")

        if self.player_color is not None and self.goal_color is None:
            self.goal_color = self.goal_detector.best_goal_color(exclude={self.player_color})
            if self.goal_color is not None:
                logger.info(f"[hyp-agent] goal color identified: {self.goal_color}")

        action = self._select_action(grid, candidates)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})

        # stash pre-action state for append_frame
        self._pending_objects_before = extract_objects(grid)
        centroids = color_centroids(grid)
        if self.player_color and self.player_color in centroids:
            px, py, _ = centroids[self.player_color]
            self._pending_pos_before = (round(px), round(py))
        else:
            self._pending_pos_before = None
        self._pending_action = action
        return action

    def _select_action(self, grid: np.ndarray, candidates: list[GameAction]) -> GameAction:
        # ── Phase 1: calibration (cycle through actions to get hypothesis
        #    evidence; unlike the old tracker, we don't gate on a
        #    convergence score here -- best_player_color() itself refuses
        #    to answer until evidence across >=2 actions is consistent) ──
        if self._action_count < self.MIN_CALIBRATION_STEPS:
            name = self.CALIBRATION_CYCLE[self._action_count % len(self.CALIBRATION_CYCLE)]
            matching = [a for a in candidates if a.name == name]
            return matching[0] if matching else random.choice(candidates)

        if self.player_color is None or self.goal_color is None:
            return random.choice(candidates)

        centroids = color_centroids(grid)
        if self.player_color not in centroids:
            return random.choice(candidates)

        px, py, _ = centroids[self.player_color]
        current_pos: Pos = (round(px), round(py))
        goal_pos = self.goal_detector.goal_position(self.goal_color)
        if goal_pos is None:
            return random.choice(candidates)

        # ── unchanged BFS planning, reused from GoalDirectedAgent ────────
        if self._planned_path:
            next_action_name = self._planned_path[0]
            expected_next = self.world_model.edges.get(current_pos, {}).get(next_action_name)
            if expected_next is None:
                self._planned_path = []

        if not self._planned_path:
            path = self.world_model.bfs_path(current_pos, goal_pos)
            if path:
                self._planned_path = path
                logger.info(f"[hyp-agent] BFS path found: {len(path)} steps "
                            f"from {current_pos} to {goal_pos}")

        if self._planned_path:
            next_name = self._planned_path[0]
            matching = [a for a in candidates if a.name == next_name]
            if matching:
                return matching[0]
            self._planned_path = []

        all_dir_names = set(ACTION_DIRECTION.keys())
        current_known = set(self.world_model.edges.get(current_pos, {}).keys())
        if all_dir_names <= current_known:
            frontier_path = self.world_model.bfs_to_frontier(current_pos)
            if frontier_path:
                self._planned_path = frontier_path

        if self._planned_path:
            next_name = self._planned_path[0]
            matching = [a for a in candidates if a.name == next_name]
            if matching:
                return matching[0]
            self._planned_path = []

        best = self.world_model.goal_biased_exploration(current_pos, candidates, goal_pos)
        if best is not None and random.random() > 0.15:
            return best
        return random.choice(candidates)

    # ------------------------------------------------------------------
    def append_frame(self, frame: FrameData) -> None:
        super().append_frame(frame)
        if self._pending_action is None or self._pending_objects_before is None:
            return

        grid_after = grid_to_array(frame.frame)
        objects_after = extract_objects(grid_after)

        # feed the hypothesis engine (replaces tracker.observe)
        self.engine.observe(self._pending_objects_before, self._pending_action.name, objects_after)

        # record position transition in the BFS world model (unchanged)
        if self._pending_pos_before is not None and self.player_color is not None:
            centroids_after = color_centroids(grid_after)
            if self.player_color in centroids_after:
                ax, ay, _ = centroids_after[self.player_color]
                pos_after: Pos = (round(ax), round(ay))
                self.world_model.record(self._pending_pos_before, self._pending_action.name, pos_after)
                if (self._planned_path
                        and pos_after != self.world_model.edges.get(
                            self._pending_pos_before, {}
                        ).get(self._pending_action.name)):
                    self._planned_path = []

        self._action_count += 1

        if frame.levels_completed > self._last_levels_completed:
            logger.info(f"[hyp-agent] LEVEL COMPLETE! {self._last_levels_completed} -> "
                        f"{frame.levels_completed}. Keeping graph+engine, resetting goal.")
            self._last_levels_completed = frame.levels_completed
            self.goal_detector = GoalDetector()
            self.goal_color = None
            self._planned_path = []

        if self._action_count % 20 == 0:
            logger.info(f"[diag] step={self._action_count} player={self.player_color} "
                        f"goal={self.goal_color} graph_nodes={len(self.world_model.edges)} "
                        f"levels={frame.levels_completed} engine={self.engine.summary()}")
