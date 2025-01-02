# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.core.utils.types import ArticulationActions

from omni.isaac.lab.actuators import DCMotor, IdealPDActuator, ImplicitActuator
from omni.isaac.lab.utils import DelayBuffer, LinearInterpolation

if TYPE_CHECKING:
    from .actuator_cfg import (
        HectorV1p5DCActuatorCfg,
        HectorV1p5IdealPDActuatorCfg,
        HectorV1p5ImplicitPDActuatorCfg,
        IdentifiedActuatorCfg,
    )


class HectorV1p5ImplicitPDActuator(ImplicitActuator):
    """Implicit actuator model for Hector V1.5 robot.

    This class implements an implicit actuator model for the Hector V1.5 robot. The model is based on the
    joint stiffness and damping parameters provided in the configuration instance passed to the class.

    """

    cfg: HectorV1p5ImplicitPDActuatorCfg
    """The configuration for the actuator model."""

    def reset(self, env_ids: Sequence[int]):
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:

        # knee command with gear ratio

        # assume the command is in motor space (policy pov) , now transform it to joint space (sim pov)
        control_action.joint_positions[:, 6] = control_action.joint_positions[:, 6] / self.cfg.knee_gear_ratio
        control_action.joint_positions[:, 7] = control_action.joint_positions[:, 7] / self.cfg.knee_gear_ratio

        control_action.joint_velocities[:, 6] = control_action.joint_velocities[:, 6] / self.cfg.knee_gear_ratio
        control_action.joint_velocities[:, 7] = control_action.joint_velocities[:, 7] / self.cfg.knee_gear_ratio

        # knee-ankle coupling

        # get the knee joint positions
        q_j_knee_left = joint_pos[:, 6]
        q_j_knee_right = joint_pos[:, 7]
        qdot_j_knee_left = joint_vel[:, 6]
        qdot_j_knee_right = joint_vel[:, 7]

        # clip the ankle commands to the joint limits, q_m_ankle = nominal: 25.0 deg, min: -8.5 deg, max: 81.0 deg
        control_action.joint_positions[:, 8] = torch.clip(
            control_action.joint_positions[:, 8], min=-0.5846853, max=0.977384
        )
        control_action.joint_positions[:, 9] = torch.clip(
            control_action.joint_positions[:, 9], min=-0.5846853, max=0.977384
        )

        # assume the ankle command is in the motor space (coming from policy pov) , now transform it to joint space
        q_j_ankle_left = control_action.joint_positions[:, 8] - q_j_knee_left
        q_j_ankle_right = control_action.joint_positions[:, 9] - q_j_knee_right

        # qdot_j_ankle_left = control_action.joint_velocities[:,8]  - qdot_j_knee_left
        # qdot_j_ankle_right = control_action.joint_velocities[:,9]  - qdot_j_knee_right

        # overwrite the commands to the joint space (simulation pov)
        control_action.joint_positions[:, 8] = q_j_ankle_left
        control_action.joint_positions[:, 9] = q_j_ankle_right

        # dummy: store approximate torques for reward computation
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts

        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        return control_action


class HectorV1p5IdealPDActuator(IdealPDActuator):
    cfg: HectorV1p5IdealPDActuatorCfg
    """The configuration for the actuator model."""

    def reset(self, env_ids: Sequence[int]):
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:

        # NOTE: IdealPD but assum PD in joint space like implicitPD

        # assume the command is in motor space (policy pov) , now transform it to joint space (sim pov)
        control_action.joint_positions[:, 6] = control_action.joint_positions[:, 6] / self.cfg.knee_gear_ratio
        control_action.joint_positions[:, 7] = control_action.joint_positions[:, 7] / self.cfg.knee_gear_ratio

        control_action.joint_velocities[:, 6] = control_action.joint_velocities[:, 6] / self.cfg.knee_gear_ratio
        control_action.joint_velocities[:, 7] = control_action.joint_velocities[:, 7] / self.cfg.knee_gear_ratio

        # knee-ankle coupling

        # get the knee joint positions
        q_j_knee_left = joint_pos[:, 6]
        q_j_knee_right = joint_pos[:, 7]
        qdot_j_knee_left = joint_vel[:, 6]
        qdot_j_knee_right = joint_vel[:, 7]

        # clip the ankle commands to the joint limits, q_m_ankle = nominal: 25.0 deg, min: -8.5 deg, max: 81.0 deg
        control_action.joint_positions[:, 8] = torch.clip(
            control_action.joint_positions[:, 8], min=-0.5846853, max=0.977384
        )
        control_action.joint_positions[:, 9] = torch.clip(
            control_action.joint_positions[:, 9], min=-0.5846853, max=0.977384
        )

        # assume the command is in the motor space (coming from policy pov) , now transform it to joint space
        q_j_ankle_left = control_action.joint_positions[:, 8] - q_j_knee_left
        q_j_ankle_right = control_action.joint_positions[:, 9] - q_j_knee_right

        # overwrite the commands to the joint space (simulation pov), for ankle q_m = A * q_j + B so no scaling needed
        control_action.joint_positions[:, 8] = q_j_ankle_left
        control_action.joint_positions[:, 9] = q_j_ankle_right

        # store approximate torques for reward computation
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts

        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action


class HectorV1p5DCActuator(DCMotor):
    cfg: HectorV1p5DCActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: HectorV1p5DCActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(self.cfg.friction_static, 0.0)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(self.cfg.friction_dynamic, 0.0)

    def reset(self, env_ids: Sequence[int]):
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:

        # NOTE: IdealPD but assum PD in joint space like implicitPD
        self._joint_vel[:] = joint_vel

        # assume the command is in motor space (policy pov) , now transform it to joint space (sim pov)
        control_action.joint_positions[:, 6] = control_action.joint_positions[:, 6] / self.cfg.knee_gear_ratio
        control_action.joint_positions[:, 7] = control_action.joint_positions[:, 7] / self.cfg.knee_gear_ratio

        control_action.joint_velocities[:, 6] = control_action.joint_velocities[:, 6] / self.cfg.knee_gear_ratio
        control_action.joint_velocities[:, 7] = control_action.joint_velocities[:, 7] / self.cfg.knee_gear_ratio

        # knee-ankle coupling

        # get the knee joint positions
        q_j_knee_left = joint_pos[:, 6]
        q_j_knee_right = joint_pos[:, 7]
        qdot_j_knee_left = joint_vel[:, 6]
        qdot_j_knee_right = joint_vel[:, 7]

        # clip the ankle commands to the joint limits, q_m_ankle = nominal: 25.0 deg, min: -8.5 deg, max: 81.0 deg
        control_action.joint_positions[:, 8] = torch.clip(
            control_action.joint_positions[:, 8], min=-0.5846853, max=0.977384
        )
        control_action.joint_positions[:, 9] = torch.clip(
            control_action.joint_positions[:, 9], min=-0.5846853, max=0.977384
        )

        # assume the command is in the motor space (coming from policy pov) , now transform it to joint space
        q_j_ankle_left = control_action.joint_positions[:, 8] - q_j_knee_left
        q_j_ankle_right = control_action.joint_positions[:, 9] - q_j_knee_right

        # qdot_j_ankle_left = control_action.joint_velocities[:,8]  - qdot_j_knee_left
        # qdot_j_ankle_right = control_action.joint_velocities[:,9]  - qdot_j_knee_right

        # overwrite the commands to the joint space (simulation pov), for ankle q_m = A * q_j + B so no scaling needed
        control_action.joint_positions[:, 8] = q_j_ankle_left
        control_action.joint_positions[:, 9] = q_j_ankle_right
        # control_action.joint_velocities[:,8] = qdot_j_ankle_left
        # control_action.joint_velocities[:,9] = qdot_j_ankle_right

        # store approximate torques for reward computation
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts

        # clip the torques based on the motor limits
        control_action.joint_efforts = self._clip_effort(self.computed_effort)

        # apply friction model on the torque
        control_action.joint_efforts = control_action.joint_efforts - (
            self.friction_static * torch.tanh(joint_vel / self.activation_vel) + self.friction_dynamic * joint_vel
        )
        # control_action.joint_efforts = control_action.joint_efforts - (self.friction_dynamic * joint_vel)

        self.applied_effort = control_action.joint_efforts
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action


class IdentifiedActuator(DCMotor):
    cfg: IdentifiedActuatorCfg

    def __init__(self, cfg: IdentifiedActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(self.cfg.friction_static, 0.0)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(self.cfg.friction_dynamic, 0.0)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # call the base method
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # apply friction model on the torque
        control_action.joint_efforts = control_action.joint_efforts - (
            self.friction_static * torch.tanh(joint_vel / self.activation_vel) + self.friction_dynamic * joint_vel
        )

        self.applied_effort = control_action.joint_efforts
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action
