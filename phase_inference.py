"""Derived ACTION/WAIT/CUTSCENE/TERMINAL classifier -- no phase RAM byte needed."""

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum, auto


class Phase(Enum):
    ACTION   = auto()
    WAIT     = auto()
    CUTSCENE = auto()
    TERMINAL = auto()


@dataclass
class TickSignals:
    shots_fired_delta: int
    shots_hit_delta: int
    life_delta: int          # negative means damage taken
    timer_delta: int         # signed change across the tick
    cleared_guess: bool
    dead_guess: bool
    can_fire_probe: bool
    cutscene_ui_detected: bool = False


class PhaseInferer:
    """Majority vote over a short window to stop phase flickering at transitions."""

    def __init__(self, vote_window: int = 3):
        self.hist = deque(maxlen=vote_window)
        self.last = Phase.WAIT

    def infer_raw(self, s: TickSignals) -> Phase:
        if s.cleared_guess or s.dead_guess:
            return Phase.TERMINAL

        interactive = (
            s.can_fire_probe
            or s.shots_fired_delta > 0
            or s.shots_hit_delta > 0
            or s.life_delta < 0
        )
        if interactive:
            return Phase.ACTION

        if s.cutscene_ui_detected:
            return Phase.CUTSCENE

        return Phase.WAIT

    def infer(self, s: TickSignals) -> Phase:
        raw = self.infer_raw(s)

        # Terminal is absorbing -- never vote our way back out of it.
        if raw is Phase.TERMINAL:
            self.last = Phase.TERMINAL
            return self.last

        self.hist.append(raw)
        self.last = Counter(self.hist).most_common(1)[0][0]
        return self.last

    def reset(self):
        self.hist.clear()
        self.last = Phase.WAIT
