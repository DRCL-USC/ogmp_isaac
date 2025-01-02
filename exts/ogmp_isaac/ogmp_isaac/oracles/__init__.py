from .base_oracle import BaseOracle
from .box.push import PushBoxOracle
from .dive.dive import DiveOracle
from .locomotion.velocity import VelocityLocomotionOracle
from .recover.reach import ReachOracle
from .soccer.kick import KickSoccerOracle
from .soccer.stop import StopSoccerOracle

__all__ = [
    "BaseOracle",
    "KickSoccerOracle",
    "StopSoccerOracle",
    "DiveOracle",
    "VelocityLocomotionOracle",
    "ReachOracle",
    "PushBoxOracle",
]
