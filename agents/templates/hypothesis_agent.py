"""
Active Hypothesis-Driven World Model Induction Agent for ARC-AGI-3
=====================================================================

Core idea:
Instead of passive/random/graph-frontier exploration, this agent maintains
an explicit set of candidate hypotheses about game mechanics, and selects
actions that maximize EXPECTED INFORMATION GAIN over that hypothesis space
(Bayesian experimental design applied to interactive RL).

This is Phase 1 / 2 skeleton — perception reuse + hypothesis generation +
naive information-gain action selection. NOT yet integrated with the full
graph-explorer baseline (that comes next, as a fallback/backup strategy).

Drop this file into: agents/templates/hypothesis_agent.py
Then it auto-registers as agent name "hypothesisagent" (see agents/__init__.py)

Run:
    uv run main.py --agent=hypothesisagent --game=ls20
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()


# ──────────────────────────────────────────────────────────────────
# Layer 1: Perception — minimal reusable frame processing
# (Reimplementation of the segmentation idea from the graph-explorer
#  baseline; swap with their actual Frame Processor once integrated)
# ──────────────────────────────────────────────────────────────────

def grid_to_array(frame: list[list[list[int]]]) -> np.ndarray:
    """ARC-AGI-3 frames are lists of 2D color grids (one per 'layer')."""
    return np.array(frame, dtype=np.int16)


def state_hash(frame: list[list[list[int]]]) -> str:
    """Cheap content hash for state deduplication."""
    arr = grid_to_array(frame)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def segment_colors(arr: np.ndarray) -> dict[int, int]:
    """Count of cells per color, per layer summed. A crude but fast
    'what objects/colors are present' signal for hypothesis matching."""
    colors, counts = np.unique(arr, return_counts=True)
    return dict(zip(colors.tolist(), counts.tolist()))


# ──────────────────────────────────────────────────────────────────
# Layer 2: Hypothesis representation
# ──────────────────────────────────────────────────────────────────

@dataclass
class Hypothesis:
    """A candidate rule about game mechanics.

    Kept deliberately simple for v1: a hypothesis predicts whether a
    given action, from a state with given color-signature, will change
    levels_completed or significantly alter the frame. More refined
    predicate templates (object permanence, symmetry, etc.) get added
    in Phase 2 once this skeleton round-trips correctly.
    """

    id: str
    description: str
    # predicate: (action, prev_color_sig) -> predicted P(progress)
    predict_fn: Any
    # running Bayesian confidence (not yet falsified)
    alive: bool = True
    support: int = 0       # times observation matched prediction
    contradict: int = 0    # times observation contradicted prediction

    @property
    def posterior_weight(self) -> float:
        """Simple Beta-Bernoulli style confidence; more support => more weight,
        more contradiction => exponentially less weight (falsification-biased)."""
        if not self.alive:
            return 0.0
        # Laplace-smoothed accuracy, penalized for contradictions
        total = self.support + self.contradict
        if total == 0:
            return 1.0  # untested hypotheses start neutral
        acc = (self.support + 1) / (total + 2)
        return acc


@dataclass
class HypothesisSpace:
    """Maintains the live set of hypotheses and computes information gain
    for candidate actions; updates posterior after each observed transition."""

    hypotheses: list[Hypothesis] = field(default_factory=list)
    # per-action history of (color_sig_before -> outcome) for empirical
    # information-gain estimation when we don't have an explicit predicate
    action_outcomes: dict[str, list[bool]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # (state_hash, action_name) pairs that previously led to GAME_OVER.
    # This was a critical gap in v1: the agent silently auto-reset on death
    # without ever recording WHICH action killed it, so it kept repeating
    # fatal moves indefinitely. Death is the single strongest signal in the
    # game and must dominate the info-gain heuristic, not be ignored by it.
    known_fatal_moves: set[tuple[str, str]] = field(default_factory=set)

    def entropy(self) -> float:
        """Entropy over which hypotheses remain plausible (alive, weighted)."""
        weights = np.array([h.posterior_weight for h in self.hypotheses if h.alive])
        if len(weights) == 0:
            return 0.0
        p = weights / (weights.sum() + 1e-9)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def is_known_fatal(self, state_hash_: str, action: GameAction) -> bool:
        return (state_hash_, action.name) in self.known_fatal_moves

    def expected_info_gain(self, action: GameAction) -> float:
        """Estimate how much entropy would drop if we took `action` next.

        IMPORTANT CAVEAT (discovered empirically, ls20):
        In navigation-style games, almost every action changes the frame
        (player moves -> new pixel position -> new hash), so frame-change
        saturates near 100% for ALL actions. That makes binary frame-change
        entropy useless as a discriminating signal — every action looks
        "maximally informative" forever, which is really "uninformative
        because uniform". This was visible in diagnostics: hypothesis_entropy
        stayed at 0.000 and unique_states_seen tracked action count almost
        1:1 — i.e. we were doing undirected novelty-seeking, not learning.

        v1.1 fix: weight by RECENCY of state revisits instead of raw
        frame-change. An action that leads to states we've already deeply
        explored (low marginal novelty) scores lower than one that leads
        to genuinely under-explored regions. This is closer to a proper
        visit-count-based exploration bonus (UCB-style) than naive
        binary-outcome entropy.
        """
        key = action.name
        history = self.action_outcomes[key]
        if len(history) < 2:
            return 2.0  # unexplored actions are maximally informative by default

        # history now stores visit-count deltas (see update below) rather
        # than booleans: lower recent novelty -> lower priority
        recent = history[-10:]
        avg_novelty = sum(recent) / len(recent)
        # avg_novelty in [0,1]: fraction of recent uses of this action that
        # led to a state we hadn't seen before. High = still discovering
        # new territory via this action. Low = exhausted, revisiting old ground.
        if avg_novelty <= 0.02:
            return 0.05
        return avg_novelty * 2.0  # rescale to keep comparable magnitude to before

    def update(self, action: GameAction, made_progress: bool) -> None:
        self.action_outcomes[action.name].append(made_progress)
        for h in self.hypotheses:
            if not h.alive:
                continue
            predicted = h.predict_fn(action)
            if predicted is None:
                continue  # hypothesis doesn't apply to this action
            if predicted == made_progress:
                h.support += 1
            else:
                h.contradict += 1
                # falsify hypotheses that are consistently wrong
                if h.contradict >= 3 and h.contradict > 2 * h.support:
                    h.alive = False
                    logger.info(f"[hypothesis] falsified: {h.description}")


def default_hypotheses() -> list[Hypothesis]:
    """Seed hypotheses. Phase 2 will generate these dynamically from
    observed transitions instead of hand-seeding; this is the v1 skeleton."""
    return [
        Hypothesis(
            id="movement_helps",
            description="Directional actions (up/down/left/right) tend to make progress",
            predict_fn=lambda a: True
            if a.name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
            else None,
        ),
        Hypothesis(
            id="interact_helps",
            description="Interact-style actions (5,6,7) tend to make progress",
            predict_fn=lambda a: True if a.name in ("ACTION5", "ACTION6", "ACTION7") else None,
        ),
    ]


# ──────────────────────────────────────────────────────────────────
# Layer 3: The Agent
# ──────────────────────────────────────────────────────────────────

class HypothesisAgent(Agent):
    """Active hypothesis-driven agent: at each step, picks the action
    with the highest expected information gain about game mechanics,
    falling back to weighted-random exploration when signal is flat."""

    MAX_ACTIONS = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hspace = HypothesisSpace(hypotheses=default_hypotheses())
        self._last_levels_completed = 0
        self._last_color_sig: Optional[dict[int, int]] = None
        self._visited_states: set[str] = set()
        self._pending_state_hash: Optional[str] = None
        seed = random.randint(0, 10**9)
        random.seed(seed)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET

        # Track state novelty (cheap proxy for "have I seen this before")
        h = state_hash(latest_frame.frame)
        is_new_state = h not in self._visited_states
        self._visited_states.add(h)

        # available_actions comes back as list[int] (raw codes), not GameAction
        # objects. GameAction is a multi-value Enum (id, action_type pair),
        # so GameAction(1) does NOT do value-lookup — must use .from_id().
        # This was the source of the "'int' object has no attribute 'name'" crash.
        raw_candidates = [GameAction.from_id(a) for a in latest_frame.available_actions]
        candidates = [a for a in raw_candidates if a is not GameAction.RESET]
        if not candidates:
            candidates = [a for a in GameAction if a is not GameAction.RESET]

        # Filter out actions we KNOW are fatal from this exact state — death
        # is the strongest possible signal and must override info-gain
        # scoring entirely, not just nudge it. Previously the agent had no
        # memory of which action killed it and could repeat the same fatal
        # move after every auto-reset.
        safe_candidates = [
            a for a in candidates if not self.hspace.is_known_fatal(h, a)
        ]
        if safe_candidates:
            candidates = safe_candidates
        # if ALL candidates from this state are known-fatal, we have no
        # choice but to try one anyway (shouldn't normally happen)

        # Score each candidate action by expected information gain.
        # Shuffle BEFORE sorting so ties (e.g. all actions equally untried,
        # or all equally "predictable") don't always resolve to the same
        # action — this was the actual cause of the ACTION1 lock-in: once
        # several actions hit identical low scores, Python's stable sort
        # kept returning the same first-in-list action every single step.
        random.shuffle(candidates)
        scored = [(a, self.hspace.expected_info_gain(a)) for a in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)

        # Softmax-ish selection: mostly greedy on info gain, with exploration noise
        # to avoid getting stuck if early estimates are misleading
        if random.random() < 0.25:
            action = random.choice(candidates)
        else:
            action = scored[0][0]

        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})

        action.reasoning = {
            "strategy": "active_hypothesis_driven",
            "expected_info_gain": dict((a.name, round(g, 3)) for a, g in scored[:3]),
            "live_hypotheses": [h.description for h in self.hspace.hypotheses if h.alive],
            "hypothesis_entropy": round(self.hspace.entropy(), 3),
            "new_state": is_new_state,
        }

        # Stash the chosen action so we can update hypotheses once we see the result
        self._pending_action = action
        self._pending_levels_completed = latest_frame.levels_completed
        self._pending_state_hash = h  # state hash computed at top of this method
        return action

    def append_frame(self, frame: FrameData) -> None:
        """Hook into frame append to update hypothesis posteriors with the
        outcome of the action we just took.

        v1.1: uses STATE NOVELTY (is this a never-before-seen state) instead
        of raw frame-change. Empirically, in ls20, nearly every action
        changes the frame (player moves -> new position), so frame-change
        saturated at ~100% for all actions and carried zero discriminating
        signal — confirmed by diagnostics showing hypothesis_entropy=0.000
        and unique_states tracking action count almost 1:1. True state
        novelty (have we visited this exact frame before) is sparser and
        more useful: it naturally decays as an action's local neighborhood
        gets exhausted, which is what we want for directed exploration.
        levels_completed is still tracked separately to falsify the
        hand-seeded progress hypotheses (a coarser, rarer signal).
        """
        super().append_frame(frame)
        if hasattr(self, "_pending_action"):
            new_hash = state_hash(frame.frame)
            is_novel_state = new_hash not in self._visited_states
            made_progress = frame.levels_completed > self._pending_levels_completed

            # CRITICAL: record fatal (state, action) pairs. Previously the
            # agent had zero memory of what killed it and would repeat the
            # same fatal move after every auto-reset (confirmed in logs:
            # GAME_OVER + reset happened mid-run, and the agent kept the
            # exact same action-selection behavior afterward with no
            # adaptation). Death must be remembered and actively avoided,
            # not just absorbed into the generic novelty signal.
            if frame.state is GameState.GAME_OVER:
                self.hspace.known_fatal_moves.add(
                    (self._pending_state_hash, self._pending_action.name)
                )
                logger.info(
                    f"[diag] FATAL MOVE recorded: state={self._pending_state_hash} "
                    f"action={self._pending_action.name} "
                    f"(total known fatal moves: {len(self.hspace.known_fatal_moves)})"
                )

            # Novelty signal drives action selection (replaces frame-change)
            self.hspace.action_outcomes[self._pending_action.name].append(is_novel_state)

            # Lightweight diagnostic: every 20 actions, dump a summary so we
            # can see what the agent has learned without needing to parse
            # the recording.jsonl file separately.
            self._action_count = getattr(self, "_action_count", 0) + 1
            if self._action_count % 20 == 0:
                stats = {
                    name: f"{sum(h)}/{len(h)}"
                    for name, h in self.hspace.action_outcomes.items()
                }
                logger.info(
                    f"[diag] step={self._action_count} "
                    f"levels_completed={frame.levels_completed} "
                    f"state_novelty_rate(per action): {stats} "
                    f"hypothesis_entropy={self.hspace.entropy():.3f} "
                    f"unique_states_seen={len(self._visited_states)}"
                )
            # Sparse signal drives hypothesis falsification (level completion)
            for h in self.hspace.hypotheses:
                if not h.alive:
                    continue
                predicted = h.predict_fn(self._pending_action)
                if predicted is None:
                    continue
                if predicted == made_progress:
                    h.support += 1
                else:
                    h.contradict += 1
                    if h.contradict >= 3 and h.contradict > 2 * h.support:
                        h.alive = False
                        logger.info(f"[hypothesis] falsified: {h.description}")
