"""
Phase 3: World-Model-Induction Agent via Empirical Transition Graph
====================================================================

This agent builds an explicit graph of (position, action) -> next_position
transitions through active interaction with the game environment. BFS over
this empirically-learned graph replaces the Manhattan-greedy heuristic from
Phase 2, which failed fundamentally because maze walls make straight-line
distance a misleading proxy for actual reachability.

Key findings from Phase 2 debugging (6+ live runs on ls20):
  - Player (color 9) and goal (color 1) correctly identified via
    ColorRegionTracker/GoalDetector after ~6 calibration steps.
  - Best-ever position (20,38), dist=6.5 reached consistently, but
    Manhattan-greedy could never break through: ACTION1 looks like it
    should reduce dist by 1, but maze walls block the straight path.
  - 40+ consecutive directional cycles at (20,38) never made progress —
    confirmed this is a genuine wall, not a fine-alignment issue.
  - ACTION5 confirmed not available for ls20 game type.

Fix: every observed (pos_before, action) -> pos_after transition is stored
in a TransitionGraph. BFS over this graph finds the true shortest path
through known-reachable cells. Unknown territory falls back to novelty-
seeking exploration that actively prioritises under-explored regions,
ensuring the graph grows to eventually connect the start and goal.

This is "world model induction" in the ARC-AGI-3 paper sense:
  - The graph IS the learned world model
  - Each action is simultaneously exploratory AND informative about
    the game's transition function
  - BFS over the graph gives the theoretically-optimal policy for
    the learned world model (not just a heuristic approximation)

Drop into: agents/templates/goal_directed_agent.py
Run:       uv run main.py --agent=goaldirectedagent --game=ls20
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()


# ─────────────────────────────────────────────────────────────────────────────
# Frame utilities (unchanged from Phase 2, validated empirically)
# ─────────────────────────────────────────────────────────────────────────────

def grid_to_array(frame: list[list[list[int]]]) -> np.ndarray:
    return np.array(frame[0] if frame else [[0]], dtype=np.int16)


def color_centroids(grid: np.ndarray) -> dict[int, tuple[float, float, int]]:
    """Returns {color: (centroid_x, centroid_y, cell_count)}."""
    out: dict[int, tuple[float, float, int]] = {}
    for color in np.unique(grid):
        ys, xs = np.where(grid == color)
        out[int(color)] = (float(xs.mean()), float(ys.mean()), int(len(xs)))
    return out


ACTION_DIRECTION = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}

Pos = tuple[int, int]  # (round_x, round_y)


# ─────────────────────────────────────────────────────────────────────────────
# ColorRegionTracker (Phase 2, proven to work)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColorRegionTracker:
    movement_evidence: dict[int, list[tuple[float, float, float, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    color_sizes: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    calibration_steps: int = 0
    MIN_CALIBRATION_STEPS: int = 6

    def observe(self, action: GameAction, before: np.ndarray, after: np.ndarray) -> None:
        direction = ACTION_DIRECTION.get(action.name)
        if direction is None:
            return
        self.calibration_steps += 1
        cb = color_centroids(before)
        ca = color_centroids(after)
        for color, (bx, by, bsize) in cb.items():
            if color not in ca:
                continue
            ax, ay, _ = ca[color]
            self.color_sizes[color].append(bsize)
            self.movement_evidence[color].append((direction[0], direction[1], ax - bx, ay - by))

    def is_calibrated(self) -> bool:
        return self.calibration_steps >= self.MIN_CALIBRATION_STEPS

    def best_player_color(self) -> Optional[int]:
        scores: dict[int, float] = {}
        for color, evidence in self.movement_evidence.items():
            if len(evidence) < 2:
                continue
            avg_size = float(np.mean(self.color_sizes[color]))
            if avg_size > 200:
                continue
            agreements = []
            for edx, edy, adx, ady in evidence:
                em = (edx**2 + edy**2)**0.5
                am = (adx**2 + ady**2)**0.5
                if em == 0 or am < 0.3:
                    agreements.append(0.0)
                    continue
                agreements.append(max(0.0, (edx * adx + edy * ady) / (em * am)))
            scores[color] = float(np.mean(agreements))
        if not scores:
            return None
        best = max(scores, key=lambda c: scores[c])
        return best if scores[best] >= 0.5 else None


# ─────────────────────────────────────────────────────────────────────────────
# GoalDetector (Phase 2, proven to work)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoalDetector:
    static_positions: dict[int, list[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    color_sizes: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))

    def observe(self, grid: np.ndarray, player_color: Optional[int]) -> None:
        for color, (cx, cy, size) in color_centroids(grid).items():
            if color == player_color:
                continue
            self.static_positions[color].append((cx, cy))
            self.color_sizes[color].append(size)

    def best_goal_color(self, exclude: set[int]) -> Optional[int]:
        candidates: dict[int, float] = {}
        for color, positions in self.static_positions.items():
            if color in exclude or len(positions) < 3:
                continue
            avg_size = float(np.mean(self.color_sizes[color]))
            if avg_size > 200 or avg_size < 1:
                continue
            xs, ys = zip(*positions)
            if float(np.var(xs) + np.var(ys)) > 1.0:
                continue
            candidates[color] = 1.0 / (avg_size + 1.0)
        if not candidates:
            return None
        return max(candidates, key=lambda c: candidates[c])

    def goal_position(self, color: int) -> Optional[Pos]:
        positions = self.static_positions.get(color)
        if not positions:
            return None
        xs, ys = zip(*positions)
        return (round(float(np.mean(xs))), round(float(np.mean(ys))))


# ─────────────────────────────────────────────────────────────────────────────
# TransitionGraph — the "learned world model"
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitionGraph:
    """Empirical graph of known (position, action) -> next_position transitions.

    Each node is a discretised player position (round(cx), round(cy)).
    Each edge is labelled with the action that caused the transition.
    Edges with zero actual movement (wall collision) are recorded but
    flagged as non-traversable — BFS ignores them.

    This is the "world model" referenced in the paper title. It is built
    purely through interaction (not pixel-parsing), so it naturally represents
    only the portions of the maze the agent has physically visited.
    """
    # edges[pos][action_name] = pos_after (None if action is wall-blocked here)
    edges: dict[Pos, dict[str, Optional[Pos]]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # visit_count[pos] = how many times we've been at this position
    visit_count: dict[Pos, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, pos_before: Pos, action_name: str, pos_after: Pos) -> None:
        moved = pos_before != pos_after
        self.edges[pos_before][action_name] = pos_after if moved else None
        self.visit_count[pos_before] += 1

    def bfs_path(self, start: Pos, goal: Pos) -> Optional[list[str]]:
        """BFS over known transitions. Returns list of action names, or None
        if goal is not yet reachable in the known graph."""
        if start == goal:
            return []
        queue: deque[tuple[Pos, list[str]]] = deque([(start, [])])
        visited: set[Pos] = {start}
        while queue:
            pos, path = queue.popleft()
            for action_name, next_pos in self.edges.get(pos, {}).items():
                if next_pos is None or next_pos in visited:
                    continue
                new_path = path + [action_name]
                if next_pos == goal:
                    return new_path
                visited.add(next_pos)
                queue.append((next_pos, new_path))
        return None  # goal not yet connected in known graph

    def goal_biased_exploration(
        self,
        pos: Pos,
        candidates: list[GameAction],
        goal: Pos,
    ) -> Optional[GameAction]:
        """Choose an action that balances NOVELTY (expanding the graph into
        unknown territory) and GOAL-DIRECTION (preferring nodes geometrically
        closer to the goal, measured by Manhattan distance as a heuristic).

        Pure novelty-seeking (v1) was the critical failure mode:
        undirected exploration preferentially moved AWAY from the goal because
        the graph already covered nearby-goal territory somewhat, so 'least
        visited' always pointed south/east. The agent spent the bulk of steps
        in the lower half of the maze, never discovering the true path north.

        This is essentially greedy best-first search biased toward the goal
        region over unknown portions of the graph. Once a path IS discovered,
        BFS (called before this) takes over optimally.
        """
        known = self.edges.get(pos, {})
        scored = []
        for action in candidates:
            edge_dict = self.edges.get(pos, {})
            if action.name not in edge_dict:
                # Uncharted: predict position from action direction
                d = ACTION_DIRECTION.get(action.name, (0, 0))
                predicted = (pos[0] + d[0], pos[1] + d[1])
                visit_penalty = 0  # unexplored = maximum novelty
                manhattan_to_goal = abs(predicted[0] - goal[0]) + abs(predicted[1] - goal[1])
            else:
                next_pos = edge_dict[action.name]
                if next_pos is None:
                    # Confirmed wall — heavily penalise to avoid retrying
                    visit_penalty = 999
                    manhattan_to_goal = 0  # doesn't matter, penalty dominates
                else:
                    visit_penalty = self.visit_count.get(next_pos, 0)
                    manhattan_to_goal = abs(next_pos[0]-goal[0]) + abs(next_pos[1]-goal[1])

            # Lower combined score = better candidate
            # manhattan_to_goal weighted 3x to bias toward goal region
            combined = visit_penalty + 3.0 * manhattan_to_goal
            scored.append((action, combined))

        scored.sort(key=lambda t: t[1])
        return scored[0][0] if scored else None


# ─────────────────────────────────────────────────────────────────────────────
# The Agent
# ─────────────────────────────────────────────────────────────────────────────

class GoalDirectedAgent(Agent):
    """Phase 3 agent: builds an empirical transition graph through active
    exploration, then uses BFS over the graph to find the true shortest path
    to the goal, bypassing the Manhattan-distance heuristic that failed in
    Phase 2 due to maze walls."""

    MAX_ACTIONS = 200
    CALIBRATION_CYCLE = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tracker = ColorRegionTracker()
        self.goal_detector = GoalDetector()
        self.world_model = TransitionGraph()
        self.player_color: Optional[int] = None
        self.goal_color: Optional[int] = None
        self._pending_action: Optional[GameAction] = None
        self._pending_grid_before: Optional[np.ndarray] = None
        self._pending_pos_before: Optional[Pos] = None
        self._action_count = 0
        self._planned_path: list[str] = []  # BFS-computed action sequence
        random.seed(random.randint(0, 10**9))

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

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

        if self.player_color is None and self.tracker.is_calibrated():
            self.player_color = self.tracker.best_player_color()
            if self.player_color is not None:
                logger.info(f"[goal-agent] player color identified: {self.player_color}")

        if self.player_color is not None and self.goal_color is None:
            self.goal_color = self.goal_detector.best_goal_color(exclude={self.player_color})
            if self.goal_color is not None:
                logger.info(f"[goal-agent] goal color identified: {self.goal_color}")

        action = self._select_action(grid, candidates)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})

        # Store pre-action state for graph recording in append_frame
        centroids = color_centroids(grid)
        if self.player_color and self.player_color in centroids:
            px, py, _ = centroids[self.player_color]
            self._pending_pos_before = (round(px), round(py))
        self._pending_action = action
        self._pending_grid_before = grid
        return action

    def _select_action(self, grid: np.ndarray, candidates: list[GameAction]) -> GameAction:
        # ── Phase 1: calibration ─────────────────────────────────────────────
        if not self.tracker.is_calibrated():
            name = self.CALIBRATION_CYCLE[self._action_count % len(self.CALIBRATION_CYCLE)]
            matching = [a for a in candidates if a.name == name]
            return matching[0] if matching else random.choice(candidates)

        # ── Fallback: player/goal not yet identified ─────────────────────────
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

        # ── Phase 3: BFS over world model ────────────────────────────────────
        # Re-plan if path is empty or stale (first step of path must still
        # match our current position's known edge)
        if self._planned_path:
            next_action_name = self._planned_path[0]
            expected_next = self.world_model.edges.get(current_pos, {}).get(next_action_name)
            if expected_next is None:
                self._planned_path = []  # edge not yet known, replan

        if not self._planned_path:
            path = self.world_model.bfs_path(current_pos, goal_pos)
            if path:
                self._planned_path = path
                logger.info(
                    f"[goal-agent] BFS path found: {len(path)} steps "
                    f"from {current_pos} to {goal_pos}: {path[:5]}..."
                )

        if self._planned_path:
            next_name = self._planned_path[0]
            matching = [a for a in candidates if a.name == next_name]
            if matching:
                return matching[0]
            # action not in candidates this step — skip it
            self._planned_path = []

        # ── Exploration: build the graph toward the goal ─────────────────────
        # No BFS path yet — the goal isn't reachable in the known graph.
        # Use goal-biased exploration: prefer actions leading to under-visited
        # positions that are ALSO geometrically closer to the goal. Pure
        # novelty-seeking (v1 "least_visited_action") was the critical failure:
        # it drove the agent toward the southern part of the maze (less visited)
        # but further from the goal — the graph grew but never connected to
        # goal position (20,32), leaving bfs_steps=? for the entire run.
        best = self.world_model.goal_biased_exploration(current_pos, candidates, goal_pos)
        if best is not None and random.random() > 0.15:
            return best
        return random.choice(candidates)

    def append_frame(self, frame: FrameData) -> None:
        super().append_frame(frame)
        if self._pending_action is None or self._pending_grid_before is None:
            return

        grid_after = grid_to_array(frame.frame)

        # Always feed calibration tracker
        self.tracker.observe(self._pending_action, self._pending_grid_before, grid_after)

        # Record position transition in the world model
        if self._pending_pos_before is not None and self.player_color is not None:
            centroids_after = color_centroids(grid_after)
            if self.player_color in centroids_after:
                ax, ay, _ = centroids_after[self.player_color]
                pos_after: Pos = (round(ax), round(ay))
                self.world_model.record(
                    self._pending_pos_before,
                    self._pending_action.name,
                    pos_after,
                )
                # Invalidate plan if we ended up somewhere unexpected
                if (self._planned_path
                        and pos_after != self.world_model.edges.get(
                            self._pending_pos_before, {}
                        ).get(self._pending_action.name)):
                    self._planned_path = []

        self._action_count += 1
        if self._action_count % 20 == 0:
            n_nodes = len(self.world_model.edges)
            n_edges = sum(len(v) for v in self.world_model.edges.values())
            goal_pos = (self.goal_detector.goal_position(self.goal_color)
                        if self.goal_color else None)
            centroids = color_centroids(grid_after)
            pos_info = ""
            if self.player_color and self.player_color in centroids:
                px, py, _ = centroids[self.player_color]
                cur = (round(px), round(py))
                if goal_pos:
                    dist = abs(cur[0] - goal_pos[0]) + abs(cur[1] - goal_pos[1])
                    path = self.world_model.bfs_path(cur, goal_pos)
                    bfs_len = len(path) if path is not None else "?"
                    pos_info = (f" pos={cur} goal={goal_pos} "
                                f"manhattan={dist:.1f} bfs_steps={bfs_len}")
            logger.info(
                f"[diag] step={self._action_count} "
                f"player={self.player_color} goal={self.goal_color} "
                f"graph_nodes={n_nodes} graph_edges={n_edges} "
                f"levels={frame.levels_completed}{pos_info}"
            )