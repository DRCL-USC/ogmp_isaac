import torch

from ..oracle_utils import interpolate
from ..soccer.stop import StopReference, StopSoccerOracle


class KickSoccerOracle(StopSoccerOracle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reference = StopReference(
            num_envs=self.num_envs,
            prediction_horizon=self.prediction_horizon,
            device=self.device,
        )

    def _detach_reference(self, feedback):
        env_ids = feedback["env_ids"]
        mask = self.modes[env_ids] == 2
        env_ids = env_ids[mask]
        if env_ids.numel() == 0:
            return
        ball_pos = feedback["ball_pos"][env_ids, :2]
        target_pos = feedback["target_pos"][env_ids]

        self._set_robot_reference(env_ids, target_pos, target_pos, torch.zeros((env_ids.numel(),), device=self.device))

        yaw = feedback["heading"][env_ids]
        ball_end_pos = self._get_end_pos(ball_pos, yaw, speed=1.5)

        self.reference.ball_pos[env_ids, :, :2] = interpolate(ball_pos, ball_end_pos, self.prediction_horizon)
        self.reference.ball_pos[env_ids, :, 2] = 0.07
        self.reference.ball_vel[env_ids, :-1] = torch.diff(self.reference.ball_pos[env_ids], dim=1) / self.env_dt
        self.reference.ball_vel[env_ids, -1] = self.reference.ball_vel[env_ids, -2]
