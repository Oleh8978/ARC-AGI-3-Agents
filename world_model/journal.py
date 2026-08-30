"""Object journal: a per-episode catalog of every distinct object type
actually seen during play, built empirically from observed frames --
NOT assumed from a text description.

Why this exists (confirmed directly against a real ls20 recording,
2026-08-30): shapes.py's hardcoded PLAYER/KEY/DOOR pixel patterns,
written from the mechanics writeup's prose, did not fully match this
game's real rendering. Concretely, in that recording:
  - The player WAS found correctly (5x2 top + 5x3 bottom, exact match).
  - The "key"-shaped L-icon turned out to be a fixed HUD element that
    never moves and never disappears across all 500 frames -- not a
    walkable pickup at all.
  - There is a real 42-unit move-budget bar (two same-row objects whose
    widths sum to a constant, one shrinking ~1/step) and a 3-life cycle
    before a true GAME_OVER -- exactly the resource system the writeup
    described, just not the exact color assumed.
  - The one genuinely unexplained static object INSIDE the main play
    area (not in any HUD margin) never vanished even after the player's
    footprint fully covered its cell once -- its role is still open.

A hardcoded shape table breaks the moment any one of these assumptions
is wrong. A journal that logs what's ACTUALLY observed, and ranks
candidates by behavior (moves like the player? vanishes on contact?
sits untouched in the main play area?), degrades gracefully instead --
and gives you and the agent something to literally read and reference
mid-game, which is the point.

Usage
-----
    journal = ObjectJournal()
    ...
    journal.ingest(step, objects, player_pos)
    ...
    print(journal.describe())              # human-readable notes
    candidates = journal.goal_candidates()  # ranked list of GoalCandidate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .objects import GameObject

Pos = tuple[int, int]
Signature = tuple  # (color, width, height, size)


@dataclass
class ObjectRecord:
    signature: Signature
    positions: set = field(default_factory=set)  # capped, see MAX_POSITIONS
    first_step: int = 0
    last_step: int = 0
    times_seen: int = 0
    vanish_events: list = field(default_factory=list)  # steps where it disappeared
    touched_without_vanishing: int = 0  # player footprint overlapped it, nothing happened
    first_touched_step: Optional[int] = None
    last_seen_position: Optional[Pos] = None

    MAX_POSITIONS = 30

    def note_seen(self, step: int, position: Pos) -> None:
        self.times_seen += 1
        self.last_step = step
        self.last_seen_position = position
        if len(self.positions) < self.MAX_POSITIONS:
            self.positions.add(position)

    @property
    def is_static(self) -> bool:
        """1-2 distinct positions across the whole episode -- doesn't
        move like the player does, but isn't necessarily immobile scenery
        either (could just be a small goal marker)."""
        return len(self.positions) <= 2

    @property
    def moves_like_player(self) -> bool:
        return len(self.positions) > 5

    @property
    def blinks(self) -> bool:
        """True if this instance's presence gaps come in rapid, tight
        succession (average gap-to-gap spacing <= 3 steps) rather than
        occasional/sparse -- a deliberate on/off blink animation, not a
        one-time pickup or a rare rendering glitch. Confirmed live: one
        real static object started blinking in EXACTLY this pattern the
        moment the player's footprint first covered it -- strong evidence
        of a "you are on/near the target" visual cue."""
        gaps = self.vanish_events
        if len(gaps) < 4:
            return False
        spacings = [b - a for a, b in zip(gaps, gaps[1:])]
        return (sum(spacings) / len(spacings)) <= 3.0

    @property
    def started_blinking_after_touch(self) -> Optional[int]:
        """Step at which blinking began, if it began AFTER (not before)
        the player's footprint first overlapped this object -- i.e. the
        blink looks triggered BY player contact rather than being present
        from the start. Returns None if not blinking or blinking predates
        any recorded touch."""
        if not self.blinks or not self.vanish_events:
            return None
        return self.vanish_events[0]


class ObjectJournal:
    """Ingests extract_objects() output frame by frame and builds the
    catalog described above. Identity key is (color, width, height,
    size) -- coarser than exact cell layout, so small rendering jitter
    (anti-aliasing-like single-pixel differences) doesn't fragment one
    real object into many records; if that ever proves too coarse for a
    specific game, tighten to include shape_signature() as well.
    """

    def __init__(self, player_signature_hint: Optional[Signature] = None) -> None:
        self.records: dict = {}
        self.player_signature: Optional[Signature] = player_signature_hint
        self._player_positions_seen: set = set()
        self._step = 0

    @staticmethod
    def _sig(obj: GameObject) -> Signature:
        return (obj.color, obj.width, obj.height, obj.size)

    def ingest(self, step: int, objects: list, player_pos: Optional[Pos]) -> None:
        self._step = step
        if player_pos is not None:
            self._player_positions_seen.add(player_pos)

        seen_this_frame: set = set()
        for obj in objects:
            base_sig = self._sig(obj)
            pos = (obj.bbox[0], obj.bbox[1])
            # Identity includes POSITION: two on-screen objects that happen
            # to share (color, w, h, size) but sit in different places (e.g.
            # two 1x1 dots of the same color, or two identical resource-bar
            # segments at different widths) are physically different things
            # and must not be merged into one record, or a "vanish" of one
            # gets falsely masked by the continued presence of the other.
            identity = base_sig + pos
            seen_this_frame.add(identity)
            record = self.records.get(identity)
            if record is None:
                record = ObjectRecord(signature=base_sig, first_step=step)
                self.records[identity] = record
            record.note_seen(step, pos)

            if (
                player_pos is not None
                and pos != player_pos
                and self._footprint_overlaps(player_pos, pos, obj.width, obj.height)
            ):
                record.touched_without_vanishing += 1
                if record.first_touched_step is None:
                    record.first_touched_step = step

        # anything present in a PREVIOUS frame but missing THIS EXACT frame:
        # a real presence gap for that specific instance (could be a pickup,
        # or could be a blink/animation cycle -- both are meaningful signals,
        # neither is assumed here).
        for identity, record in self.records.items():
            if identity not in seen_this_frame and record.last_step == step - 1:
                record.vanish_events.append(step)

    @staticmethod
    def _footprint_overlaps(player_pos: Pos, obj_pos: Pos, obj_w: int, obj_h: int,
                             player_w: int = 5, player_h: int = 5) -> bool:
        px, py = player_pos
        ox, oy = obj_pos
        return not (
            px + player_w <= ox or ox + obj_w <= px
            or py + player_h <= oy or oy + obj_h <= py
        )

    # ------------------------------------------------------------------
    def _playable_bbox(self) -> Optional[tuple]:
        """Bounding box of everywhere the player has actually been --
        used to exclude fixed HUD panels/corner icons (which sit outside
        this box for the whole episode) from goal candidates, without
        hardcoding any pixel margin for a specific game."""
        if not self._player_positions_seen:
            return None
        xs = [p[0] for p in self._player_positions_seen]
        ys = [p[1] for p in self._player_positions_seen]
        pad = 6
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    def goal_candidates(self, exclude_signatures: Optional[set] = None) -> list:
        """Ranks static, small, in-play-area object records as possible
        goal markers -- highest-ranked first. Doesn't assume a "vanishes
        on touch" pickup model; a persistent, untouched marker (a door
        that simply hasn't been reached yet) ranks just as legitimately
        as one with vanish evidence, since both are equally consistent
        with "this is the target, we just haven't solved it yet"."""
        exclude_signatures = exclude_signatures or set()
        bbox = self._playable_bbox()
        scored = []
        for identity, record in self.records.items():
            sig = record.signature
            if sig in exclude_signatures or self.player_signature == sig:
                continue
            if not record.is_static or record.last_seen_position is None:
                continue
            color, w, h, size = sig
            if size > 30:  # too big to be a marker (floor/wall/background)
                continue
            if bbox is not None:
                x, y = record.last_seen_position
                bx0, by0, bx1, by1 = bbox
                if not (bx0 <= x <= bx1 and by0 <= y <= by1):
                    continue  # outside the area the player has ever reached -- likely a HUD element
            score = 0.0
            score += 5.0 if record.vanish_events else 0.0  # a presence gap is notable either way
            score += 4.0 if record.blinks else 0.0  # a deliberate blink is a strong "interactive marker" signal
            score += 3.0 if (record.blinks and record.first_touched_step is not None
                              and record.vanish_events
                              and record.vanish_events[0] >= record.first_touched_step) else 0.0
            score += 2.0 if record.touched_without_vanishing == 0 else 0.0  # not yet ruled out
            score -= 1.0 * min(size, 10) / 10.0  # smaller is more marker-like
            scored.append(GoalCandidate(signature=sig, position=record.last_seen_position,
                                         score=score, record=record))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored

    def describe(self) -> str:
        """Human-readable dump of everything currently known -- meant to
        be logged periodically so a person can literally read what the
        agent has learned about the objects in this game so far."""
        lines = [f"=== Object journal @ step {self._step} ({len(self.records)} tracked instances) ==="]
        for identity, r in sorted(self.records.items(), key=lambda kv: -kv[1].times_seen):
            color, w, h, size = r.signature
            kind = "PLAYER-LIKE (moves a lot)" if r.moves_like_player else (
                "static" if r.is_static else "semi-mobile")
            gap_note = f", presence gaps at steps {r.vanish_events[:6]}{'...' if len(r.vanish_events) > 6 else ''}" \
                if r.vanish_events else ""
            blink_note = ""
            if r.blinks:
                blink_note = " [BLINKING"
                if r.first_touched_step is not None:
                    blink_note += f" -- started right after player touch at step {r.vanish_events[0]}" \
                        if r.vanish_events and r.vanish_events[0] >= r.first_touched_step else ""
                blink_note += "]"
            touch_note = f", touched {r.touched_without_vanishing}x with no effect" \
                if r.touched_without_vanishing else ""
            lines.append(
                f"  color={color:>3} {w}x{h} size={size:>4}  [{kind}]  "
                f"seen {r.times_seen}x (step {r.first_step}-{r.last_step})  "
                f"last_pos={r.last_seen_position}{gap_note}{blink_note}{touch_note}"
            )
        return "\n".join(lines)


@dataclass
class GoalCandidate:
    signature: Signature
    position: Pos
    score: float
    record: ObjectRecord
