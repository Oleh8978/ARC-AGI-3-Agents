"""
Hypothesis-World Agent: object-centric perception + CEGIS-style hypothesis
falsification + STATE-AUGMENTED planning (position + has_key).

──────────────────────────────────────────────────────────────────────────
Two concrete bugs fixed here, diagnosed directly from a live run log
(ls20, 2026-08-29 10:23-10:27, 307 actions, 0 levels completed):

BUG 1 -- RESET boundary corrupts the hypothesis engine.
  choose_action() returns GameAction.RESET immediately when
  latest_frame.state is NOT_PLAYED/GAME_OVER, WITHOUT clearing
  self._pending_action / self._pending_objects_before. The next
  append_frame() call then still has the STALE pending values from
  several frames earlier, and pairs them with the frame AFTER reset
  (i.e. the fresh level start) as if the old action had caused that
  transition. That is a completely bogus (color, action) -> huge-jump
  observation fed straight into HypothesisEngine.observe() every single
  time the game auto-resets (e.g. on running out of the move budget).
  This is directly visible in the log: 'falsified_constant_delta'
  climbs from 0 to 16 over the run while 'active_constant_delta' shrinks,
  and 'best_player_color' eventually flips to None -- exactly what
  repeated bogus giant-delta observations would do to a falsification
  engine.
  Fix: clear all three pending_* fields (and the stale planned path)
  whenever RESET is issued, so append_frame's existing early-return
  guard (`if self._pending_action is None: return`) skips recording
  anything across that boundary.

BUG 2 -- goal identification never distinguished KEY from DOOR.
  The previous version reused `goal_directed_agent.GoalDetector`, which
  picks exactly ONE static small color as "the goal" and BFS's straight
  to it for the entire game. Per the mechanics writeup, ls20-style games
  need KEY reached before DOOR, and the OLD code's docstring even
  admits level 1 was once completed by luck when a rotator happened to
  get crossed mid-exploration -- not by design.
  Fix: use world_model.shapes.classify_frame() to identify player / key
  / door BY SHAPE each frame (grounded, doesn't depend on the
  potentially-falsified HypothesisEngine color at all), track has_key
  by watching the key object disappear while the player is standing on
  its last known position, and plan to the key first, then the door --
  exactly the KEY->DOOR sequencing the writeup describes. The state fed
  into StatefulTransitionGraph is now (player_pos, has_key: bool)
  instead of (player_pos, some-nearby-color-guess).

Falls back to the OLD single-goal GoalDetector/color-centroid path ONLY
if shape-based player/door detection can't find anything in a given
frame (e.g. this specific rendering doesn't match the exact pixel
pattern from the writeup) -- so the agent stays functional rather than
going fully idle while the shapes are re-calibrated against a real
captured frame.

BUG 3 -- confirmed live: shapes.py's exact pixel patterns don't match
this game's real rendering. A second live run (2026-08-29 11:20, ls20)
shows `door=None key=None` for the entire run -- classify_frame() never
found a match, so the agent silently fell back to the OLD single-color
goal ("(fallback) goal color identified: 1") and still couldn't
sequence key-before-door, still 0 levels completed. The exact
PLAYER_TOP_SIG / PLAYER_BOTTOM_SIG / KEY_SIG / DOOR_SIG constants in
world_model/shapes.py were built from the mechanics WRITEUP's prose
description, not from a real captured frame's actual color-index grid
-- they need calibrating against one (see world_model/calibrate.py,
also delivered this round) before the shape-based path can ever engage.

Fix (this round): make the FALLBACK path itself key/door-aware instead
of single-goal, using data the engine already collects with no shape
assumptions at all: among GoalDetector's static small-color candidates,
a color that DISAPPEARS when the player touches it
(HypothesisEngine.active_vanish_rules(), color_a=candidate,
color_b=player_color) behaves like a one-time pickup -- i.e. a KEY --
while a persistent candidate that never vanishes behaves like the DOOR.
This reuses _door_pos/_last_key_pos/has_key unchanged, so every
downstream planning method (BFS, stagnation, hazard-avoidance) works
identically regardless of which path (shape or fallback) populated them.

Drop into: agents/templates/hypothesis_world_agent.py
Run:       uv run main.py --agent=hypothesisworldagent --game=ls20
"""

from __future__ import annotations

import logging
import random
import time as _time
from typing import Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .goal_directed_agent import (
    ACTION_DIRECTION,
    GoalDetector,
    Pos,
    color_centroids,
    grid_to_array,
)
from world_model.objects import extract_objects
from world_model.hypotheses import HypothesisEngine
from world_model.stateful_graph import StatefulTransitionGraph, State
from world_model.shapes import classify_frame
from world_model.journal import ObjectJournal

logger = logging.getLogger()


def _looks_like_network_failure(e: Exception, requests_module) -> bool:
    """Two known signatures of the same transient server-side hiccup
    against three.arcprize.org, confirmed via live testing:
    1. requests.exceptions.RequestException surfaces directly in some
       cases (ReadTimeout, ConnectionError).
    2. In OTHER cases, arc_agi/remote_wrapper.py catches the network
       exception itself internally (it logs "Failed to perform action...
       Read timed out" and returns None instead of re-raising), and the
       base Agent.do_action_request()/_convert_raw_frame_data() then
       raises ValueError("Received None frame data from environment")
       -- a different exception type carrying the same underlying cause.
    Only these two specific, evidence-backed signatures are treated as
    retryable; anything else (a real bug, a validation error) is not.
    """
    if requests_module is not None and isinstance(e, requests_module.exceptions.RequestException):
        return True
    if isinstance(e, ValueError) and "Received None frame data from environment" in str(e):
        return True
    return False


def call_with_network_retry(fn, attempts: int, delay_seconds: float, log_prefix: str = ""):
    """Calls fn() and retries up to `attempts` times ONLY on a known
    transient network-failure signature (see _looks_like_network_failure).
    Uses exponential backoff (delay_seconds * 2**attempt): live testing
    showed failures more severe than a slow server response (DNS
    resolution failures, SSL handshake timeouts).
    """
    try:
        import requests as _requests
    except ImportError:
        _requests = None

    last_exc: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if not _looks_like_network_failure(e, _requests):
                raise
            last_exc = e
            if attempt < attempts - 1:
                wait = delay_seconds * (2 ** attempt)
                logger.warning(f"{log_prefix} attempt {attempt + 1}/{attempts} failed: "
                                f"{e!r} -- retrying in {wait}s")
                _time.sleep(wait)
    assert last_exc is not None
    raise last_exc


class HypothesisWorldAgent(Agent):
    """State is now (player_pos, has_key), and the goal is either the
    KEY (before collection) or the DOOR (after) -- not one generic
    static goal for the whole game."""

    MAX_ACTIONS = 500
    CALIBRATION_CYCLE = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    MIN_CALIBRATION_STEPS = 10
    ACTION_RETRY_ATTEMPTS = 5
    ACTION_RETRY_DELAY_SECONDS = 5.0
    # How close the player must be to the key's last known position, at
    # the moment the key disappears from the frame, to count it as a
    # real pickup (vs. a one-frame detection glitch elsewhere on screen).
    KEY_PICKUP_RADIUS = 4.0

    def do_action_request(self, action: GameAction) -> FrameData:
        return call_with_network_retry(
            lambda: super(HypothesisWorldAgent, self).do_action_request(action),
            attempts=self.ACTION_RETRY_ATTEMPTS,
            delay_seconds=self.ACTION_RETRY_DELAY_SECONDS,
            log_prefix="[hyp-agent] action request",
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.engine = HypothesisEngine()
        self.goal_detector = GoalDetector()  # fallback path only
        self.world_model = StatefulTransitionGraph(ACTION_DIRECTION)
        self.journal = ObjectJournal()
        self._journal_goal_pos: Optional[Pos] = None
        self.player_color: Optional[int] = None  # fallback path only
        self.has_key: bool = False
        self._last_key_pos: Optional[Pos] = None
        self._door_pos: Optional[Pos] = None
        # Fallback (non-shape) key/door role assignment -- see BUG 3 fix.
        self._fallback_roles_assigned: bool = False
        self._fallback_key_color: Optional[int] = None
        self._fallback_door_color: Optional[int] = None
        self._pending_action: Optional[GameAction] = None
        self._pending_objects_before = None
        self._pending_state_before: Optional[State] = None
        self._action_count = 0
        self._planned_path: list[str] = []
        self._last_levels_completed: int = 0
        random.seed(random.randint(0, 10**9))

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # ------------------------------------------------------------------
    def _player_pos(self, grid: np.ndarray, objects: list) -> Optional[Pos]:
        """Shape-based player position first (grounded: doesn't depend
        on the possibly-falsified HypothesisEngine color at all).
        Falls back to the old color-centroid approach only if the exact
        5x5 bicolor pattern from the writeup isn't found in this frame
        (e.g. this rendering differs slightly -- verify shapes.py's
        PLAYER_TOP_SIG/PLAYER_BOTTOM_SIG against a real captured frame
        if this fallback is triggered often)."""
        classified = classify_frame(grid, objects)
        player = classified["player"]
        if player is not None:
            cx, cy = player.centroid
            return (round(cx), round(cy))

        if self.player_color is None and self._action_count >= self.MIN_CALIBRATION_STEPS:
            self.player_color = self.engine.best_player_color()
            if self.player_color is not None:
                logger.info(f"[hyp-agent] (fallback) player color identified: {self.player_color} "
                            f"(engine summary: {self.engine.summary()})")
        if self.player_color is None:
            return None
        centroids = color_centroids(grid)
        if self.player_color not in centroids:
            return None
        px, py, _ = centroids[self.player_color]
        return (round(px), round(py))

    def _ranked_static_candidates(self, exclude: set) -> list:
        """Same filtering GoalDetector.best_goal_color() uses (static,
        small, low-position-variance), but returns ALL qualifying colors
        ranked by score instead of just the top one -- needed to
        consider a KEY and a DOOR as two separate candidates instead of
        collapsing to a single goal."""
        detector = self.goal_detector
        scored: dict = {}
        for color, positions in detector.static_positions.items():
            if color in exclude or len(positions) < 3:
                continue
            avg_size = float(np.mean(detector.color_sizes[color]))
            if avg_size > 200 or avg_size < 1:
                continue
            xs, ys = zip(*positions)
            if float(np.var(xs) + np.var(ys)) > 1.0:
                continue
            scored[color] = 1.0 / (avg_size + 1.0)
        return sorted(scored, key=lambda c: scored[c], reverse=True)

    def _try_fallback_key_door(self, grid: np.ndarray, player_pos: Optional[Pos]) -> None:
        """No-shape-assumptions key/door role assignment (BUG 3 fix):
        among the static goal-like candidates, a color that VANISHES
        when the player touches it acts like a one-time KEY pickup; a
        persistent candidate that never vanishes acts like the DOOR.
        Assigned once (like player_color), then tracked the same way
        the shape-based path tracks has_key -- populates the SAME
        _door_pos / _last_key_pos / has_key fields, so every planning
        method downstream is unaffected by which path filled them in.
        """
        if self.player_color is None:
            return
        if not self._fallback_roles_assigned:
            ranked = self._ranked_static_candidates(exclude={self.player_color})
            if not ranked:
                return
            vanish_rules = self.engine.active_vanish_rules()
            key_like = [
                c for c in ranked
                if any(r.color_a == c and r.color_b == self.player_color for r in vanish_rules)
            ]
            persistent = [c for c in ranked if c not in key_like]
            if key_like and persistent:
                self._fallback_key_color = key_like[0]
                self._fallback_door_color = persistent[0]
                self._fallback_roles_assigned = True
                logger.info(f"[hyp-agent] (fallback) KEY candidate={self._fallback_key_color} "
                            f"(vanishes on touch), DOOR candidate={self._fallback_door_color} (persistent)")
            elif len(ranked) >= 1 and self._action_count > 60:
                # No vanish-rule evidence yet after a reasonable amount of
                # exploration -- accept the single best candidate as the
                # door with no separate key (matches old single-goal
                # behaviour rather than blocking forever).
                self._fallback_door_color = ranked[0]
                self._fallback_roles_assigned = True
                logger.info(f"[hyp-agent] (fallback) no vanish evidence -- "
                            f"treating {ranked[0]} as DOOR with no separate key")
            else:
                return

        if self._fallback_door_color is not None:
            self._door_pos = self.goal_detector.goal_position(self._fallback_door_color)
        if self._fallback_key_color is not None:
            centroids = color_centroids(grid)
            if self._fallback_key_color in centroids:
                cx, cy, _ = centroids[self._fallback_key_color]
                self._last_key_pos = (round(cx), round(cy))
            elif (
                not self.has_key
                and self._last_key_pos is not None
                and player_pos is not None
            ):
                dist = abs(player_pos[0] - self._last_key_pos[0]) + abs(player_pos[1] - self._last_key_pos[1])
                if dist <= self.KEY_PICKUP_RADIUS:
                    self.has_key = True
                    logger.info(f"[hyp-agent] (fallback) KEY COLLECTED near {self._last_key_pos} "
                                f"-- now heading to door {self._door_pos}")

    def _update_key_door(self, grid: np.ndarray, objects: list, player_pos: Optional[Pos]) -> None:
        """Tracks has_key / door position, shape-based first (BUG 2 fix);
        if shape detection finds NEITHER key nor door this frame, tries
        the no-shape-assumptions fallback instead (BUG 3 fix). has_key
        flips to True only when the key DISAPPEARS while the player is
        standing near its last known position -- a real pickup, not a
        one-frame detection glitch elsewhere on screen."""
        classified = classify_frame(grid, objects)
        key_obj = classified["key"]
        door_obj = classified["door"]
        shape_found_anything = key_obj is not None or door_obj is not None or self._door_pos is not None

        if door_obj is not None:
            self._door_pos = (door_obj.bbox[0], door_obj.bbox[1])

        if key_obj is not None:
            self._last_key_pos = (key_obj.bbox[0], key_obj.bbox[1])
        elif (
            not self.has_key
            and self._last_key_pos is not None
            and player_pos is not None
        ):
            dist = abs(player_pos[0] - self._last_key_pos[0]) + abs(player_pos[1] - self._last_key_pos[1])
            if dist <= self.KEY_PICKUP_RADIUS:
                self.has_key = True
                logger.info(f"[hyp-agent] KEY COLLECTED near {self._last_key_pos} "
                            f"-- now heading to door {self._door_pos}")

        if not shape_found_anything:
            self._try_fallback_key_door(grid, player_pos)

        if self._door_pos is None:
            self._try_journal_goal(objects, player_pos)

    def _try_journal_goal(self, objects: list, player_pos: Optional[Pos]) -> None:
        """Third-tier fallback (BUG 4 fix): when NEITHER shape detection
        NOR the vanish-rule dual-role heuristic found a door, fall back
        to the empirical object journal -- ranks every small, static,
        in-play-area object by how "unexplained" it still is (never
        vanished, never definitively ruled out by contact) and treats
        the top-ranked one as the door. Confirmed useful directly against
        a real ls20 recording: this correctly surfaced the one genuine
        anomaly object inside the maze that neither shapes.py nor the
        color-based fallback had any way to name."""
        candidates = self.journal.goal_candidates()
        if candidates:
            self._door_pos = candidates[0].position

    def _current_goal_pos(self) -> Optional[Pos]:
        if self._door_pos is not None:
            return self._last_key_pos if (not self.has_key and self._last_key_pos is not None) else self._door_pos
        return None

    # ------------------------------------------------------------------
    def _current_state(self, grid: np.ndarray, objects: list) -> Optional[State]:
        pos = self._player_pos(grid, objects)
        if pos is None:
            return None
        return (pos, self.has_key)

    # ------------------------------------------------------------------
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # BUG 1 fix: clear pending_* so append_frame's early-return
            # guard skips recording a transition across this boundary --
            # otherwise the next append_frame() pairs a STALE pending
            # action/objects with the fresh post-reset frame and feeds
            # HypothesisEngine a bogus giant-delta observation.
            self._pending_action = None
            self._pending_objects_before = None
            self._pending_state_before = None
            self._planned_path = []
            return GameAction.RESET

        grid = grid_to_array(latest_frame.frame)
        candidates_raw = [GameAction.from_id(a) for a in latest_frame.available_actions]
        candidates = [a for a in candidates_raw if a is not GameAction.RESET]
        if not candidates:
            candidates = [a for a in GameAction
                          if a is not GameAction.RESET and a.is_simple()]

        objects = extract_objects(grid)
        player_pos = self._player_pos(grid, objects)

        self.journal.ingest(self._action_count, objects, player_pos)

        # Needed by _try_fallback_key_door()'s candidate ranking even
        # when shape-based detection succeeds this frame (cheap; keeps
        # fallback data warm in case shapes stop matching later).
        self.goal_detector.observe(grid, self.player_color)
        self._update_key_door(grid, objects, player_pos)

        action = self._select_action(grid, candidates, objects)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})

        # stash pre-action state for append_frame
        self._pending_objects_before = objects
        self._pending_state_before = self._current_state(grid, objects)
        self._pending_action = action
        return action

    def _select_action(self, grid: np.ndarray, candidates: list[GameAction], objects: list) -> GameAction:
        if self._action_count < self.MIN_CALIBRATION_STEPS:
            name = self.CALIBRATION_CYCLE[self._action_count % len(self.CALIBRATION_CYCLE)]
            matching = [a for a in candidates if a.name == name]
            return matching[0] if matching else random.choice(candidates)

        current_state = self._current_state(grid, objects)
        if current_state is None:
            return random.choice(candidates)
        current_pos, _current_held = current_state

        goal_pos = self._current_goal_pos()
        if goal_pos is None:
            return random.choice(candidates)

        # ── BFS planning over (position, has_key) state space ───────
        if self._planned_path:
            next_action_name = self._planned_path[0]
            expected_next = self.world_model.edges.get(current_state, {}).get(next_action_name)
            if expected_next is None:
                self._planned_path = []

        if not self._planned_path:
            path = self.world_model.bfs_path(current_state, goal_pos)
            if path:
                self._planned_path = path
                logger.info(f"[hyp-agent] BFS path found: {len(path)} steps "
                            f"from {current_state} to goal near {goal_pos}")

        if self._planned_path:
            next_name = self._planned_path[0]
            matching = [a for a in candidates if a.name == next_name]
            if matching:
                return matching[0]
            self._planned_path = []

        # COMPLETE-THE-FRONTIER: only when stagnant AND standing on a
        # frontier state ourselves -- see stagnation_graph.py history for
        # why this must be scoped this narrowly (an unconditional version
        # regressed a simple maze from 55 steps to a full FAIL).
        if self.world_model.is_stagnant():
            current_known_map = self.world_model.edges.get(current_state, {})
            untried = [name for name in ACTION_DIRECTION if name not in current_known_map]
            if untried:
                matching = [a for a in candidates if a.name in untried]
                if matching:
                    logger.info("[hyp-agent] STAGNATION + standing on a frontier state -- "
                                "trying untried direction directly")
                    return matching[0]

        if not self._planned_path and self.world_model.is_stagnant():
            frontier_path = self.world_model.bfs_to_frontier_goal_directed(current_state, goal_pos)
            if frontier_path:
                self._planned_path = frontier_path
                logger.info(f"[hyp-agent] STAGNATION ({self.world_model.steps_since_new_node} "
                            f"steps no new state) -- forcing frontier BFS: {len(frontier_path)} steps")

        all_dir_names = set(ACTION_DIRECTION.keys())
        current_known = set(self.world_model.edges.get(current_state, {}).keys())
        if not self._planned_path and all_dir_names <= current_known:
            frontier_path = self.world_model.bfs_to_frontier_goal_directed(current_state, goal_pos)
            if frontier_path:
                self._planned_path = frontier_path

        if self._planned_path:
            next_name = self._planned_path[0]
            matching = [a for a in candidates if a.name == next_name]
            if matching:
                return matching[0]
            self._planned_path = []

        hazard_positions = self._current_hazard_positions(objects)
        best = self.world_model.goal_biased_exploration(
            current_state, candidates, goal_pos, hazard_positions=hazard_positions
        )
        if best is not None and random.random() > 0.15:
            return best
        return random.choice(candidates)

    def _current_hazard_positions(self, objects: list) -> set:
        if self.player_color is None:
            return set()
        hazard_colors = self.engine.likely_hazard_colors(self.player_color)
        if not hazard_colors:
            return set()
        positions = set()
        for obj in objects:
            if obj.color in hazard_colors:
                cx, cy = obj.centroid
                positions.add((round(cx), round(cy)))
        return positions

    # ------------------------------------------------------------------
    def append_frame(self, frame: FrameData) -> None:
        super().append_frame(frame)
        if self._pending_action is None or self._pending_objects_before is None:
            # BUG 1 fix: this now correctly triggers right after a RESET
            # (see choose_action), skipping the bogus cross-reset
            # observation that used to corrupt the hypothesis engine.
            return

        grid_after = grid_to_array(frame.frame)
        objects_after = extract_objects(grid_after)

        self.engine.observe(self._pending_objects_before, self._pending_action.name, objects_after)

        if self._pending_state_before is not None:
            state_after = self._current_state(grid_after, objects_after)
            if state_after is not None:
                self.world_model.record(self._pending_state_before, self._pending_action.name, state_after)
                if (self._planned_path
                        and state_after != self.world_model.edges.get(
                            self._pending_state_before, {}
                        ).get(self._pending_action.name)):
                    self._planned_path = []

        self._action_count += 1

        if frame.levels_completed > self._last_levels_completed:
            logger.info(f"[hyp-agent] LEVEL COMPLETE! {self._last_levels_completed} -> "
                        f"{frame.levels_completed}. Keeping graph+engine, resetting goal/key state.")
            self._last_levels_completed = frame.levels_completed
            self.goal_detector = GoalDetector()
            self.journal = ObjectJournal()
            self.has_key = False
            self._last_key_pos = None
            self._door_pos = None
            self._fallback_roles_assigned = False
            self._fallback_key_color = None
            self._fallback_door_color = None
            self._planned_path = []

        if self._action_count % 20 == 0:
            steps_to_goal = None
            if self._pending_state_before is not None and self._door_pos is not None:
                goal_pos = self._current_goal_pos()
                if goal_pos is not None:
                    path = self.world_model.bfs_path(self._pending_state_before, goal_pos)
                    steps_to_goal = len(path) if path is not None else "unknown (no path learned yet)"
            logger.info(f"[diag] step={self._action_count} player_color={self.player_color} "
                        f"has_key={self.has_key} door={self._door_pos} key={self._last_key_pos} "
                        f"fallback_key_color={self._fallback_key_color} fallback_door_color={self._fallback_door_color} "
                        f"steps_to_goal={steps_to_goal} "
                        f"graph_nodes={len(self.world_model.edges)} "
                        f"levels={frame.levels_completed} engine={self.engine.summary()}")

        if self._action_count % 50 == 0:
            logger.info("\n" + self.journal.describe())
