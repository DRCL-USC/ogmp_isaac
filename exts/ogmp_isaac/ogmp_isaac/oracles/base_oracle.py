import torch
from dataclasses import dataclass


@dataclass
class BaseReference:
    num_envs: int
    prediction_horizon: int
    device: torch.device


class BaseOracle:
    def __init__(
        self,
        rollout_time=1.0,
        env_dt=0.033,
        speed=0.5,
        num_envs=1,
        nominal_height=0.55,
    ):
        self.nominal_height = nominal_height
        self.rollout_time = rollout_time
        self.env_dt = env_dt
        self.speed = speed
        self.prediction_horizon = int(rollout_time / env_dt)
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.phase = torch.zeros((num_envs, 1), dtype=torch.int64, device=self.device)
        self.reference = BaseReference(num_envs, self.prediction_horizon, self.device)
        self.modes = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.mode_reference_functions = []

    def reset(self, env_ids):
        self.modes[env_ids] = 0
        self.phase[env_ids] = 0

    def generate_reference(self, feedback):
        for func in self.mode_reference_functions:
            func(feedback)
        return

    def check_transitions(self, feedback):
        raise NotImplementedError

    def _compute_heading(self, start_pos, end_pos):
        heading = end_pos - start_pos
        yaw = torch.atan2(heading[:, 1], heading[:, 0])
        return yaw

    def _get_end_pos(self, robot_pos, yaw, speed=None):
        if speed is None:
            speed = self.speed
        return torch.stack(
            (
                robot_pos[:, 0] + speed * self.rollout_time * torch.cos(yaw),
                robot_pos[:, 1] + speed * self.rollout_time * torch.sin(yaw),
            ),
            dim=-1,
        )
