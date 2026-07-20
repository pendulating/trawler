# Vendored from CI-RL (github.com/EricGLan/CI-RL) - Apache 2.0 License
from typing import NamedTuple


class SimulatedObservation(NamedTuple):
    observation: str
    thought_summary: str
    log: str
    def __str__(self):
        return self.observation
    def __repr__(self):
        return self.observation
