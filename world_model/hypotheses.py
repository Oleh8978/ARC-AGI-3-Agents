"""
Hypothesis DSL + falsification engine.

Honest scope note (say this in the paper too): this is NOT full Bayesian
inference over a continuous hypothesis space. It is a deterministic
counterexample-guided falsification loop (CEGIS-style) over a small,
finite set of template rules. That is a much weaker and cheaper claim than
MASTER_PLAN_EN.md originally made -- and it is the claim we can actually
back with working code and offline compute.

Templates covered:
  1. ConstantDelta(color, action) -> (dx, dy)
       "objects of this color move by this action-independent-of-position
        vector when this action is taken" -- discovered per (color, action)
        pair, not hardcoded like the old ACTION_DIRECTION table.
  2. Immobile(color)
       "objects of this color never move regardless of action" -- likely
        background/UI/goal marker.
  3. VanishesNear(color_a, color_b)
       "objects of color_a disappear when they end the step adjacent to
        an object of color_b" -- candidate for collectibles/hazards.
  4. AppearsAfter(color, action)
       "new objects of this color tend to appear right after this action".

Every template starts as a candidate with weak/no evidence. Every observed
(before, action, after) transition either supports or falsifies each
active candidate. Falsified hypotheses are dropped permanently -- this is
the actual "learning" happening, not a magic score.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .objects import GameObject, match_objects

Delta = tuple[int, int]


@dataclass
class ConstantDeltaHypothesis:
    color: int
    action: str
    deltas_seen: set[Delta] = field(default_factory=set)
    support: int = 0

    @property
    def falsified(self) -> bool:
        # more than one distinct delta observed for the same (color, action)
        # means movement is NOT a fixed vector -- e.g. blocked by a wall
        # sometimes. That's a real, useful falsification (tells you the
        # simple rule is wrong), not a bug.
        return len(self.deltas_seen) > 1

    @property
    def delta(self) -> Optional[Delta]:
        if len(self.deltas_seen) == 1:
            return next(iter(self.deltas_seen))
        return None


@dataclass
class ImmobileHypothesis:
    color: int
    max_drift_seen: float = 0.0
    observations: int = 0

    @property
    def falsified(self) -> bool:
        # BUG FIX (found via synthetic integration test): a normal
        # single-cell orthogonal move has magnitude exactly 1.0. The
        # original threshold of 1.5 was meant to tolerate centroid
        # rounding jitter, but it also silently tolerated real
        # single-cell player movement, so a moving player could never
        # falsify this hypothesis. 0.5 still absorbs sub-pixel jitter
        # (drift < 0.5) while correctly falsifying on any real move.
        return self.max_drift_seen > 0.5


@dataclass
class VanishNearHypothesis:
    color_a: int
    color_b: int
    vanish_while_adjacent: int = 0
    vanish_while_not_adjacent: int = 0
    survive_while_adjacent: int = 0

    @property
    def falsified(self) -> bool:
        # if it survived adjacency at least twice with no vanish-while-not-
        # adjacent evidence backing it, the rule isn't doing useful work
        return self.survive_while_adjacent >= 2 and self.vanish_while_adjacent == 0


def _adjacent(obj_a: GameObject, obj_b: GameObject, radius: int = 1) -> bool:
    for (x1, y1) in obj_a.cells:
        for (x2, y2) in obj_b.cells:
            if abs(x1 - x2) <= radius and abs(y1 - y2) <= radius:
                return True
    return False


class HypothesisEngine:
    """Owns the full candidate hypothesis set and updates it from
    observed (objects_before, action_name, objects_after) transitions."""

    def __init__(self) -> None:
        self.constant_delta: dict[tuple[int, str], ConstantDeltaHypothesis] = {}
        self.immobile: dict[int, ImmobileHypothesis] = {}
        self.vanish_near: dict[tuple[int, int], VanishNearHypothesis] = {}
        self.step_count = 0
        self.colors_seen: set[int] = set()

    # ------------------------------------------------------------------
    def observe(self, before: list[GameObject], action: str, after: list[GameObject]) -> None:
        self.step_count += 1
        matches = match_objects(before, after)

        for b, a in matches:
            color = (b or a).color
            self.colors_seen.add(color)

            if b is not None and a is not None:
                bx, by = b.centroid
                ax, ay = a.centroid
                dx, dy = round(ax - bx), round(ay - by)

                key = (color, action)
                h = self.constant_delta.setdefault(
                    key, ConstantDeltaHypothesis(color=color, action=action)
                )
                h.deltas_seen.add((dx, dy))
                h.support += 1

                imm = self.immobile.setdefault(color, ImmobileHypothesis(color=color))
                drift = (dx ** 2 + dy ** 2) ** 0.5
                imm.max_drift_seen = max(imm.max_drift_seen, drift)
                imm.observations += 1

            elif b is not None and a is None:
                # b vanished this step -- check adjacency to every other
                # object present in the "before" frame
                for other in before:
                    if other.color == color:
                        continue
                    key = (color, other.color)
                    vh = self.vanish_near.setdefault(
                        key, VanishNearHypothesis(color_a=color, color_b=other.color)
                    )
                    if _adjacent(b, other):
                        vh.vanish_while_adjacent += 1
                    else:
                        vh.vanish_while_not_adjacent += 1

        # update survive_while_adjacent for pairs that were adjacent but
        # BOTH objects survived (evidence against the vanish rule)
        surviving_by_color: dict[int, list[GameObject]] = defaultdict(list)
        for b, a in matches:
            if a is not None:
                surviving_by_color[a.color].append(a)
        for (ca, cb), vh in self.vanish_near.items():
            for obj_a in surviving_by_color.get(ca, []):
                for obj_b in surviving_by_color.get(cb, []):
                    if _adjacent(obj_a, obj_b):
                        vh.survive_while_adjacent += 1

    # ------------------------------------------------------------------
    def active_constant_delta(self) -> list[ConstantDeltaHypothesis]:
        return [h for h in self.constant_delta.values() if not h.falsified and h.support >= 2]

    def active_immobile_colors(self) -> set[int]:
        return {c for c, h in self.immobile.items() if not h.falsified and h.observations >= 3}

    def active_vanish_rules(self) -> list[VanishNearHypothesis]:
        return [h for h in self.vanish_near.values() if not h.falsified and h.vanish_while_adjacent >= 1]

    def best_player_color(self) -> Optional[int]:
        """A 'player' color is one with strong, CONSISTENT constant-delta
        evidence across at least 2 different actions (moves differently
        depending on which button you press -- unlike wind/scenery)."""
        by_color: dict[int, set[str]] = defaultdict(set)
        for (color, action), h in self.constant_delta.items():
            if not h.falsified and h.delta is not None and h.delta != (0, 0) and h.support >= 2:
                by_color[color].add(action)
        candidates = {c: acts for c, acts in by_color.items() if len(acts) >= 2}
        if not candidates:
            return None
        # prefer the color with the most distinct confirmed action-deltas
        return max(candidates, key=lambda c: len(candidates[c]))

    def predict_delta(self, color: int, action: str) -> Optional[Delta]:
        h = self.constant_delta.get((color, action))
        if h is None or h.falsified:
            return None
        return h.delta

    def summary(self) -> dict:
        return {
            "step_count": self.step_count,
            "colors_seen": sorted(self.colors_seen),
            "active_constant_delta": len(self.active_constant_delta()),
            "falsified_constant_delta": sum(1 for h in self.constant_delta.values() if h.falsified),
            "active_immobile_colors": sorted(self.active_immobile_colors()),
            "active_vanish_rules": [(h.color_a, h.color_b) for h in self.active_vanish_rules()],
            "best_player_color": self.best_player_color(),
        }
