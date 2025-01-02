import math
import torch
from dataclasses import dataclass, field

from ..base_oracle import BaseOracle, BaseReference
from ..oracle_utils import euler_to_quaternion_horizon, interpolate, quaternion_to_euler


@dataclass
class DiveReference(BaseReference):
    base_pos: torch.Tensor = field(init=False)
    base_ori: torch.Tensor = field(init=False)
    base_lin_vel: torch.Tensor = field(init=False)
    base_ang_vel: torch.Tensor = field(init=False)

    def __post_init__(self):
        # Initialize tensors after class is fully constructed
        self.base_pos = torch.zeros((self.num_envs, self.prediction_horizon, 3), device=self.device)
        self.base_ori = torch.zeros((self.num_envs, self.prediction_horizon, 4), device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, self.prediction_horizon, 3), device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, self.prediction_horizon, 3), device=self.device)


class DiveOracle(BaseOracle):
    def __init__(self, time_take_off=0.5, platform_height=1.5, nominal_height=0.55, **kwargs):
        self.time_take_off = time_take_off
        self.time_of_flight = math.sqrt(2 * (platform_height + nominal_height) / 9.81)
        rollout_time = self.time_take_off + self.time_of_flight
        kwargs["rollout_time"] = rollout_time
        kwargs["nominal_height"] = nominal_height
        super().__init__(**kwargs)
        self.reference = DiveReference(
            num_envs=self.num_envs,
            prediction_horizon=self.prediction_horizon,
            device=self.device,
        )
        self.mode_reference_functions = [
            self._dive_reference,
            self._land_reference,
        ]

        self.land_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.switch2land = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device).bool()

    def reset(self, env_ids):
        super().reset(env_ids)
        self.land_position[env_ids] = 0
        self.switch2land[env_ids] = False
        self.reference.base_pos[env_ids] = 0
        self.reference.base_ori[env_ids] = 0
        self.reference.base_lin_vel[env_ids] = 0
        self.reference.base_ang_vel[env_ids] = 0

    def check_transitions(self, feedback):
        condition = (self.phase == 0).squeeze(-1)
        condition |= self._check_mode_transitions(feedback)

        return condition

    def _check_mode_transitions(self, feedback):
        land_mask = (feedback["robot_pos"][:, 2] - self.nominal_height < 0.1) | (
            self.switch2land == True & (self.phase == 0).squeeze(-1)
        )
        self.modes[land_mask] = 1
        return land_mask

    def _dive_reference(self, feedback):
        env_ids = feedback["env_ids"]
        mask = self.modes[env_ids] == 0  # & ~self.switch2land[env_ids]
        env_ids = env_ids[mask]
        if len(env_ids) == 0:
            return

        # total time for mode
        total_time = self.time_take_off + self.time_of_flight
        angle_disp_in_take_off = 2 * torch.pi * self.time_take_off / total_time
        angle_disp_in_flight = 2 * torch.pi * self.time_of_flight / total_time

        robot_pos = feedback["robot_pos"][env_ids, :3]
        robot_ori = feedback["robot_ori"][env_ids, :4]

        flip_dir = feedback["flip_dir"][env_ids]

        robot_ori = quaternion_to_euler(robot_ori)
        self.rollout_time = self.time_take_off
        flat_end_pos = torch.zeros((len(env_ids), 3), device=self.device)
        flat_end_pos[:, :2] = torch.stack(
            (
                robot_pos[:, 0] + flip_dir[:, 0] * self.speed * self.rollout_time,
                robot_pos[:, 1] + flip_dir[:, 1] * self.speed * self.rollout_time,
            ),
            dim=-1,
        )
        flat_end_pos[:, 2] = robot_pos[:, 2]

        robot_end_ori_tf = torch.ones((len(env_ids), 3), device=self.device) * angle_disp_in_take_off
        robot_end_ori_tf[:, 0] *= -flip_dir[:, 1]
        robot_end_ori_tf[:, 1] *= flip_dir[:, 0]
        robot_end_ori_tf[:, 2] = 0
        n_flat_end_pos = self.time_take_off / self.env_dt
        self._set_robot_reference(
            env_ids, robot_pos, flat_end_pos, robot_ori, robot_end_ori_tf, end_index=int(n_flat_end_pos)
        )

        self.rollout_time = self.time_of_flight
        dive_end_pos = torch.zeros((len(env_ids), 3), device=self.device)
        dive_end_pos[:, :2] = torch.stack(
            (
                flat_end_pos[:, 0] + flip_dir[:, 0] * self.speed * self.rollout_time,
                flat_end_pos[:, 1] + flip_dir[:, 1] * self.speed * self.rollout_time,
            ),
            dim=-1,
        )
        dive_end_pos[:, 2] = self.nominal_height
        n_dive_end_pos = self.time_of_flight / self.env_dt

        robot_end_ori_fl = torch.ones((len(env_ids), 3), device=self.device) * 2 * torch.pi
        robot_end_ori_fl[:, 0] *= -flip_dir[:, 1]
        robot_end_ori_fl[:, 1] *= flip_dir[:, 0]
        robot_end_ori_fl[:, 2] = 0
        self._set_robot_reference(
            env_ids,
            flat_end_pos,
            dive_end_pos,
            robot_end_ori_tf,
            robot_end_ori_fl,
            start_index=int(n_flat_end_pos),
            end_index=int(n_flat_end_pos + n_dive_end_pos),
        )

        self.land_position[env_ids] = dive_end_pos.clone()
        self.switch2land[env_ids] = True

    def _land_reference(self, feedback):
        env_ids = feedback["env_ids"]
        mask = self.modes[env_ids] == 1
        env_ids = env_ids[mask]
        if len(env_ids) == 0:
            return
        # robot_pos = feedback['robot_pos'][env_ids, :3]
        # robot_ori = feedback['robot_ori'][env_ids, :4]
        # robot_ori = quaternion_to_euler(robot_ori)

        robot_end_ori = torch.zeros((len(env_ids), 3), device=self.device)
        # robot_end_pos = robot_pos.clone()
        # robot_end_pos[:, 2] = self.nominal_height

        self._set_robot_reference(
            env_ids, self.land_position[env_ids], self.land_position[env_ids], robot_end_ori, robot_end_ori
        )

    def _set_robot_reference(self, env_ids, start_pos, end_pos, start_ori, end_ori, start_index=0, end_index=None):
        if end_index is None:
            end_index = self.prediction_horizon
        self.reference.base_pos[env_ids, start_index:end_index] = interpolate(
            start_pos, end_pos, end_index - start_index
        )
        interpolated_euler = interpolate(start_ori, end_ori, end_index - start_index)
        self.reference.base_ori[env_ids, start_index:end_index] = euler_to_quaternion_horizon(interpolated_euler)
        self.reference.base_lin_vel[env_ids, start_index : end_index - 1] = (
            torch.diff(self.reference.base_pos[env_ids, start_index:end_index], dim=1) / self.env_dt
        )
        self.reference.base_lin_vel[env_ids, end_index - 1] = self.reference.base_lin_vel[env_ids, end_index - 2]
        self.reference.base_ang_vel[env_ids, start_index : end_index - 1] = (
            torch.diff(interpolated_euler, dim=1) / self.env_dt
        )
        self.reference.base_ang_vel[env_ids, end_index - 1] = self.reference.base_ang_vel[env_ids, end_index - 2]


if __name__ == "__main__":
    oracle = DiveOracle(
        rollout_time=3.0,
        env_dt=0.033,
        speed=0.5,
        num_envs=3,
        reference_dim=13,
        nominal_height=0.5,
        time_take_off=1.0,
        platform_height=1.5,
    )
    oracle.modes[2] = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feedback = {
        "env_ids": torch.tensor([0, 1, 2], device=device),
        "robot_pos": torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.5]], device=device),
        "robot_ori": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            device=device,
        ),
        "flip_dir": torch.tensor([[0, 1], [1, 0], [0, 1]], device=device),
    }
    euler_angles = (torch.rand((3000, 3), device=device) * 2 * math.pi).unsqueeze(1)
    assert torch.allclose(quaternion_to_euler(euler_to_quaternion_horizon(euler_angles)), euler_angles, atol=1e-3)
    print("All tests passed!")
    oracle.generate_reference(feedback)
    # print(oracle.reference)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        oracle.reference[0, :, 0].cpu().numpy(),
        oracle.reference[0, :, 1].cpu().numpy(),
        oracle.reference[0, :, 2].cpu().numpy(),
    )
    ax.plot(
        oracle.reference[1, :, 0].cpu().numpy(),
        oracle.reference[1, :, 1].cpu().numpy(),
        oracle.reference[1, :, 2].cpu().numpy(),
    )
    ax.plot(
        oracle.reference[2, :, 0].cpu().numpy(),
        oracle.reference[2, :, 1].cpu().numpy(),
        oracle.reference[2, :, 2].cpu().numpy(),
    )
    plt.show()
