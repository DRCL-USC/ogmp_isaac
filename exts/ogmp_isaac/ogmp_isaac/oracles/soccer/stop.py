import torch
from dataclasses import dataclass, field

from ..base_oracle import BaseOracle, BaseReference
from ..oracle_utils import euler_to_quaternion_horizon, interpolate


@dataclass
class StopReference(BaseReference):
    base_pos: torch.Tensor = field(init=False)
    base_ori: torch.Tensor = field(init=False)
    base_lin_vel: torch.Tensor = field(init=False)
    base_ang_vel: torch.Tensor = field(init=False)
    ball_pos: torch.Tensor = field(init=False)
    ball_vel: torch.Tensor = field(init=False)

    def __post_init__(self):
        # Initialize tensors after class is fully constructed
        self.base_pos = torch.zeros((self.num_envs, self.prediction_horizon, 3), device=self.device)
        self.base_ori = torch.zeros((self.num_envs, self.prediction_horizon, 4), device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, self.prediction_horizon, 3), device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, self.prediction_horizon, 3), device=self.device)
        self.ball_pos = torch.zeros((self.num_envs, self.prediction_horizon, 3), device=self.device)
        self.ball_vel = torch.zeros((self.num_envs, self.prediction_horizon, 3), device=self.device)


class StopSoccerOracle(BaseOracle):
    def __init__(self, reach_thresh=0.3, detach_thresh=0.3, **kwargs):
        super().__init__(**kwargs)
        self.reference = StopReference(
            num_envs=self.num_envs,
            prediction_horizon=self.prediction_horizon,
            device=self.device,
        )
        self.reach_thresh = reach_thresh
        self.detach_thresh = detach_thresh
        self.mode_reference_functions = [self._reach_reference, self._manipulate_reference, self._detach_reference]

    def reset(self, env_ids):
        super().reset(env_ids)
        self.reference.base_pos[env_ids] = 0
        self.reference.base_ori[env_ids] = 0
        self.reference.base_lin_vel[env_ids] = 0
        self.reference.base_ang_vel[env_ids] = 0
        self.reference.ball_pos[env_ids] = 0
        self.reference.ball_vel[env_ids] = 0

    def _set_robot_reference(self, env_ids, start_pos, end_pos, yaw):
        self.reference.base_pos[env_ids, :, :2] = interpolate(start_pos, end_pos, self.prediction_horizon)
        self.reference.base_pos[env_ids, :, 2] = self.nominal_height
        euler_angles = torch.cat(
            (torch.zeros_like(yaw).unsqueeze(1), torch.zeros_like(yaw).unsqueeze(1), yaw.unsqueeze(1)), dim=-1
        )
        interpolated_euler = interpolate(euler_angles, euler_angles, self.prediction_horizon)
        self.reference.base_ori[env_ids] = euler_to_quaternion_horizon(interpolated_euler)
        self.reference.base_lin_vel[env_ids, :-1] = torch.diff(self.reference.base_pos[env_ids], dim=1) / self.env_dt
        self.reference.base_lin_vel[env_ids, -1] = self.reference.base_lin_vel[env_ids, -2]

    def _reach_reference(self, feedback):
        env_ids = feedback["env_ids"]
        mask = self.modes[env_ids] == 0
        env_ids = env_ids[mask]
        if env_ids.numel() == 0:
            return
        robot_pos = feedback["robot_pos"][env_ids, :2]
        ball_pos = feedback["ball_pos"][env_ids, :2]

        yaw = self._compute_heading(robot_pos, ball_pos)
        robot_end_pos = self._get_end_pos(robot_pos, yaw)

        self._set_robot_reference(env_ids, robot_pos, robot_end_pos, yaw)

    def _manipulate_reference(self, feedback):
        env_ids = feedback["env_ids"]
        mask = self.modes[env_ids] == 1
        env_ids = env_ids[mask]
        if env_ids.numel() == 0:
            return
        robot_pos = feedback["robot_pos"][env_ids, :2]
        detach_pos = feedback["target_pos"][env_ids]

        yaw = self._compute_heading(robot_pos, detach_pos)
        robot_end_pos = self._get_end_pos(robot_pos, yaw)

        self._set_robot_reference(env_ids, robot_pos, robot_end_pos, yaw)

    def _detach_reference(self, feedback):
        env_ids = feedback["env_ids"]
        mask = self.modes[env_ids] == 2
        env_ids = env_ids[mask]
        if env_ids.numel() == 0:
            return
        ball_pos = feedback["ball_pos"][env_ids, :2]
        detach_pos = feedback["target_pos"][env_ids]

        self._set_robot_reference(env_ids, detach_pos, detach_pos, torch.zeros((env_ids.numel(),), device=self.device))

        self.reference.ball_pos[env_ids, :, :2] = ball_pos.unsqueeze(1).repeat(1, self.prediction_horizon, 1)
        self.reference.ball_pos[env_ids, :, 2] = 0.07

    def check_transitions(self, feedback):
        robot_pos = feedback["robot_pos"][:, :2]
        ball_pos = feedback["ball_pos"][:, :2]
        detach_pos = feedback["target_pos"]

        reach_dist = torch.linalg.norm(ball_pos - robot_pos, dim=-1)
        detach_dist = torch.linalg.norm(robot_pos - detach_pos, dim=-1)

        reach_modes = self.modes == 0
        manipulate_modes = self.modes == 1
        detach_modes = self.modes == 2

        reach_mode_mask = (reach_dist > self.reach_thresh) & (detach_dist > self.detach_thresh)
        manipulate_mode_mask = (reach_dist <= self.reach_thresh) & (detach_dist > self.detach_thresh)
        detach_mode_mask = (reach_dist <= self.reach_thresh) & (detach_dist <= self.detach_thresh)

        self.modes[reach_modes & manipulate_mode_mask] = 1
        self.modes[reach_modes & detach_mode_mask] = 2
        self.modes[manipulate_modes & reach_mode_mask] = 0
        self.modes[manipulate_modes & detach_mode_mask] = 2
        self.modes[detach_modes & manipulate_mode_mask] = 1

        return (
            (reach_modes & (manipulate_mode_mask | detach_mode_mask))
            | (manipulate_modes & (reach_mode_mask | detach_mode_mask))
            | (detach_modes & manipulate_mode_mask)
            | ((self.phase == 0).squeeze(-1))
        )
