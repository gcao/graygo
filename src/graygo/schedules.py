"""Fresh-record schedules and episode choices from the implemented design.

A2 and 9x9 have explicit different epsilon clocks. Stopping/resuming or changing
an execution stage never rescales a clock. Label eligibility uses the logical
mode, not whichever fallback policy acted during the hand-off.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LearnerSchedule:
    epsilon_decay_records: int
    reservoir_open: int=5_000_000
    handoff_end: int=7_000_000
    best_response_probability: float=.1

    def __post_init__(self):
        if type(self.epsilon_decay_records) is not int or self.epsilon_decay_records<1:
            raise ValueError("explicit positive epsilon clock required")
        if type(self.reservoir_open) is not int or type(self.handoff_end) is not int or not 0<=self.reservoir_open<self.handoff_end:
            raise ValueError("invalid hand-off endpoints")
        if not 0<=self.best_response_probability<=1:
            raise ValueError("invalid logical mode probability")

    @classmethod
    def board(cls):return cls(40_000_000)
    @classmethod
    def a2(cls):return cls(7_500_000)

    def epsilon(self,records):
        _records(records)
        return .01+.09*max(0.,1-records/self.epsilon_decay_records)

    def ramp(self,records):
        _records(records)
        return min(1.,max(0.,(records-self.reservoir_open)/(self.handoff_end-self.reservoir_open)))

    def choose_episode(self,records,rng,*,average_ready,uniform_comparison=False):
        _records(records)
        mode_draw,ramp_draw=map(float,rng.random(2))
        best_response=mode_draw<self.best_response_probability
        ramp=self.ramp(records)
        use_average=not best_response and average_ready and ramp_draw<ramp
        acting="average" if use_average else ("uniform" if uniform_comparison and not best_response else "q")
        return EpisodeChoice(records,best_response,acting,self.epsilon(records),ramp,mode_draw,ramp_draw)

    def label_eligible(self,choice,records,*,supplemental=False,opening_enabled=True):
        _records(records)
        return bool(opening_enabled and records>=self.reservoir_open and choice.best_response_mode and not supplemental)

    def opening_allowed(self,records,*,scope,pilot_ready=False):
        _records(records)
        if scope not in ("development","confirmation"):
            raise ValueError("scope must be fixed development or confirmation")
        return records>=self.reservoir_open and (scope=="confirmation" or pilot_ready)


@dataclass(frozen=True)
class EpisodeChoice:
    start_record: int
    best_response_mode: bool
    acting_policy: str
    epsilon: float
    ramp_probability: float
    mode_draw: float
    ramp_draw: float


def _records(records):
    if isinstance(records,(bool,np.bool_)) or not isinstance(records,(int,np.integer)) or records<0:
        raise ValueError("nonnegative cumulative fresh-record count required")
