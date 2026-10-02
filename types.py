from dataclasses import dataclass
import numpy as np

def sealed(a, dtype=float):
    out = np.array(a, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out

@dataclass(frozen=True)
class RobotProfile:
    id: int
    archetype: int
    nominal: np.ndarray

    def __post_init__(self):
        object.__setattr__(self,"nominal",sealed(self.nominal))

@dataclass(frozen=True)
class RobotInternal:
    profile: RobotProfile
    sensitivity: np.ndarray
    reliability: float

    def __post_init__(self):
        object.__setattr__(self,"sensitivity",sealed(self.sensitivity))

@dataclass(frozen=True)
class TaskBatch:
    ids: np.ndarray
    types: np.ndarray
    requirements: np.ndarray
    difficulty: np.ndarray
    reward: np.ndarray
    penalty: np.ndarray
    positions: np.ndarray

    def __post_init__(self):
        for name in ("ids","types","requirements","difficulty","reward","penalty","positions"):
            object.__setattr__(self,name,sealed(getattr(self,name),int if name in ("ids","types") else float))

@dataclass(frozen=True)
class DecisionInput:
    round_index: int
    profiles: tuple[RobotProfile, ...]
    telemetry: np.ndarray
    tasks: TaskBatch
    distances: np.ndarray
    costs: np.ndarray
    raw: np.ndarray

    def __post_init__(self):
        object.__setattr__(self,"profiles",tuple(self.profiles))
        for name in ("telemetry","distances","costs","raw"):
            object.__setattr__(self,name,sealed(getattr(self,name)))

@dataclass(frozen=True)
class SimulationTruth:
    battery: np.ndarray
    degradation: np.ndarray
    effective: np.ndarray
    probabilities: np.ndarray
    uniforms: np.ndarray

    def __post_init__(self):
        for name in ("battery","degradation","effective","probabilities","uniforms"):
            object.__setattr__(self,name,sealed(getattr(self,name)))

@dataclass(frozen=True)
class Feedback:
    round_index: int
    robot_id: int
    task_id: int
    raw_probability: float
    outcome: int

@dataclass(frozen=True)
class Assignment:
    tasks: tuple[int, ...]
