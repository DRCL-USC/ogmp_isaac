# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Iterable
from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.actuators import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass

from . import actuator_pd
from .actuator_pd import IdentifiedActuator


@configclass
class HectorV1p5ImplicitPDActuatorCfg(ImplicitActuatorCfg):
    """Configuration for the implicit PD actuator model in Hector v1.5."""

    class_type: type = actuator_pd.HectorV1p5ImplicitPDActuator

    knee_gear_ratio: float = 2.0


@configclass
class HectorV1p5IdealPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for the ideal PD actuator model in Hector v1.5."""

    class_type: type = actuator_pd.HectorV1p5IdealPDActuator

    knee_gear_ratio: float = 2.0


@configclass
class HectorV1p5DCActuatorCfg(DCMotorCfg):
    """Configuration for direct control (DC) motor actuator model in Hector v1.5."""

    class_type: type = actuator_pd.HectorV1p5DCActuator

    knee_gear_ratio: float = 2.0

    friction_static: float = MISSING
    """ (in N-m)."""
    activation_vel: float = MISSING
    """ (in Rad/s)."""
    friction_dynamic: float = MISSING
    """ (in N-m-s/Rad)."""


@configclass
class IdentifiedActuatorCfg(DCMotorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type = IdentifiedActuator

    friction_static: float = MISSING
    """ (in N-m)."""
    activation_vel: float = MISSING
    """ (in Rad/s)."""
    friction_dynamic: float = MISSING
    """ (in N-m-s/Rad)."""
