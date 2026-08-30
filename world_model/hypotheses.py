"""Active hypothesis induction over object transitions.

For every (color, action) pair, maintains a hypothesis "this color moves
by a constant (dx, dy) delta when this action is taken" and falsifies it
the moment two DIFFERENT non-trivial outcomes are observed for the same
(color, action) pair (e.g. moves freely most of the time, but is blocked
by a wall once) -- this is the CEGIS ("falsify, don't average") behaviour
the paper's Phase 2/3 writeups describe and that ``test_synthetic_ground_truth.py``
checks directly.

Also tracks:
  - active_immobile_colors(): colors that are present across many frames
    but never move -- static scenery / goal markers, and specifically
    NOT the player, however visually similar.
  - active_vanish_rules(): "color A disappears exactly when adjacent to
    color B" rules (key/coin pickup, hazard contact, etc.), each scored
    by confirmations vs. disconfirmations so a coincidental disappearance
    doesn't get promoted to a rule.
  - best_player_color(): the color most consistent with being player-
    controlled (moves under player action, isn't the giant static goal
    region).
  - likely_hazard_colors(): colors whose *adjacency* correlates with the
    player itself vanishing (a proxy for "touching this resets/kills
    you" -- distinct from a benign pickup, which is a vanish_near rule
    where the PLAYER survives).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .objects import GameObject, merge_by_color

Delta = tuple[int, int]


@dataclass
class DeltaHypothesis:
    """'color X moves by a constant delta under action A' -- falsified
    the moment more than one distinct delta value has been observed."""

    deltas_seen: list = field(default_factory=list)
    falsified: bool = False

    def observe(self, delta: Delta) -> None:
        self.deltas_seen.append(delta)
        if len(set(self.deltas_seen)) > 1:
            self.falsified = True

    def value(self) -> Optional[Delta]:
        if self.falsified or not self.deltas_seen:
            return None
        return self.deltas_seen[0]


@dataclass
class VanishNearRule:
    """'color_a disappears when adjacent to color_b' -- e.g. a key/coin
    (color_a) picked up by the player (color_b), or the player (color_a)
    dying/resetting next to a hazard (color_b)."""

    color_a: int
    color_b: int
    confirmations: int = 0
    disconfirmations: int = 0

    @property
    def active(self) -> bool:
        return self.confirmations > 0 and self.disconfirmations == 0


class HypothesisEngine:
    # Centroid-distance treated as "adjacent" for vanish-near detection.
    # Generous enough to cover a one-cell step onto/next-to an object
    # while still not being so wide it treats far-apart objects as
    # touching.
    ADJACENCY_RADIUS = 1.6
    # An object bigger than this many cells is treated as background
    # scenery, not a candidate hazard/pickup, when scoring hazards.
    MAX_HAZARD_SIZE = 40

    def __init__(self) -> None:
        self.constant_delta: dict[tuple[int, str], DeltaHypothesis] = {}
        self.color_sizes: dict[int, list[int]] = defaultdict(list)
        self.color_seen_frames: dict[int, int] = defaultdict(int)
        self.color_moved_ever: dict[int, bool] = defaultdict(bool)
        self._vanish_rules: dict[tuple[int, int], VanishNearRule] = {}

    def observe(
        self,
        before_objects: list[GameObject],
        action_name: str,
        after_objects: list[GameObject],
    ) -> None:
        before = merge_by_color(before_objects)
        after = merge_by_color(after_objects)

        for color, (bx, by, bsize) in before.items():
            self.color_sizes[color].append(bsize)
            self.color_seen_frames[color] += 1
            if color in after:
                ax, ay, _ = after[color]
                delta = (round(ax - bx), round(ay - by))
                if delta != (0, 0):
                    self.color_moved_ever[color] = True
                key = (color, action_name)
                h = self.constant_delta.setdefault(key, DeltaHypothesis())
                h.observe(delta)

        vanished = set(before.keys()) - set(after.keys())
        for va in vanished:
            vax, vay, _ = before[va]
            for vb, (bx, by, _) in before.items():
                if vb == va:
                    continue
                dist = ((vax - bx) ** 2 + (vay - by) ** 2) ** 0.5
                key = (va, vb)
                rule = self._vanish_rules.setdefault(key, VanishNearRule(va, vb))
                if dist <= self.ADJACENCY_RADIUS:
                    rule.confirmations += 1
                else:
                    rule.disconfirmations += 1

    def predict_delta(self, color: int, action_name: str) -> Optional[Delta]:
        h = self.constant_delta.get((color, action_name))
        if h is None:
            return None
        return h.value()

    def active_immobile_colors(self) -> set:
        out = set()
        for color, seen in self.color_seen_frames.items():
            if seen >= 3 and not self.color_moved_ever[color]:
                out.add(color)
        return out

    def active_vanish_rules(self) -> list:
        return [r for r in self._vanish_rules.values() if r.active]

    def best_player_color(self) -> Optional[int]:
        scores: dict[int, float] = defaultdict(float)
        for (color, _action), h in self.constant_delta.items():
            if h.falsified or not h.deltas_seen:
                continue
            dx, dy = h.value()
            if dx == 0 and dy == 0:
                continue
            scores[color] += 1.0
        if not scores:
            return None

        def avg_size(c: int) -> float:
            sizes = self.color_sizes.get(c, [1])
            return sum(sizes) / len(sizes)

        # Most action-consistent, non-trivial mover wins; ties broken by
        # smallest average footprint (the controllable sprite is usually
        # compact, not a sprawling background region).
        return max(scores, key=lambda c: (scores[c], -avg_size(c)))

    def likely_hazard_colors(self, player_color: int) -> set:
        """Colors whose adjacency correlates with the PLAYER color
        itself vanishing -- i.e. touching them makes the player
        disappear (death/reset), as opposed to an ordinary pickup where
        the OTHER color vanishes and the player survives."""
        return {
            r.color_b
            for r in self.active_vanish_rules()
            if r.color_a == player_color
        }

    def summary(self) -> dict:
        active = sum(1 for h in self.constant_delta.values() if not h.falsified)
        falsified = sum(1 for h in self.constant_delta.values() if h.falsified)
        return {
            "best_player_color": self.best_player_color(),
            "active_constant_delta": active,
            "falsified_constant_delta": falsified,
            "active_immobile_colors": sorted(self.active_immobile_colors()),
            "active_vanish_rules": [
                (r.color_a, r.color_b) for r in self.active_vanish_rules()
            ],
        }
