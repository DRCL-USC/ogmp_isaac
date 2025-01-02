import os
import torch

from ogmplm.tasks.base_env import BaseEnv, BaseEnvCfg

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects")


@configclass
class FlatBoxEnvCfg(BaseEnvCfg):
    marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
            ),
            "base": sim_utils.CuboidCfg(
                size=(0.125, 0.19, 0.248),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)),
            ),
        },
    )

    rewards = {
        "base_pos": {"weight": 0.4, "exp_coeff": 3.0},
        "base_ori": {"weight": 0.23, "exp_coeff": 5.0},
        "base_lin_vel": {"weight": 0.3, "exp_coeff": 2.0},
        "box_closeness": {"weight": 0.5, "threshold": 0.5},
        "preference": {
            "weight": -1.0,
        },
        "torque_exp": {"weight": 0.15, "exp_coeff": 0.05},
        "action": {"weight": 0.15, "exp_coeff": 1.0},
    }
    observations = [
        "base_z",
        "base_ori",
        "joint_pos",
        "base_lin_vel",
        "base_ang_vel",
        "joint_vel",
        "box_dist",
        "target_dist",
        "sinusoid_phase",
    ]
    terminations = [
        "ogmp_pos_x",
        "ogmp_pos_y",
        "ogmp_pos_z",
        "box_pos_x",
        "box_pos_y",
    ]
    ogmp_error_terminations = {
        "base_pos_x": 0.4,
        "base_pos_y": 0.4,
        "base_pos_z": 0.1,
        "box_pos_x": 0.2,
        "box_pos_y": 0.2,
    }
    oracle = {
        "name": "PushBoxOracle",
        "params": {
            "speed": 0.8,
            "reach_thresh": 0.3,
            "detach_thresh": 0.3,
        },
    }
    box_start = 1.0
    box_height = 0.5
    height_to_file_name = {
        "0.5": "box_0p5m.usd",
        "1.0": "box_1m.usd",
        "1.5": "box_1p5m.usd",
    }
    omni_direction_lim = [0.0, 360.0]
    target = 3.0


class FlatBoxEnv(BaseEnv):
    cfg: FlatBoxEnvCfg

    def __init__(self, cfg: FlatBoxEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.target_pos = torch.zeros((self.num_envs, 2), device=self.sim.device)
        self.heading_angles = torch.zeros((self.num_envs,), device=self.sim.device)
        self.start_angle = torch.deg2rad(torch.tensor(self.cfg.omni_direction_lim[0], device=self.sim.device))
        self.end_angle = torch.deg2rad(torch.tensor(self.cfg.omni_direction_lim[1], device=self.sim.device))
        if self.cfg.visualize_markers:
            self.marker = VisualizationMarkers(self.cfg.marker_cfg)
        self.oracle.box_com = self.box.data.default_root_state[0, 2]

    def _setup_scene(self):
        box_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/box",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(ASSETS_DIR, self.cfg.height_to_file_name[str(self.cfg.box_height)]),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(2.0, 0.0, self.cfg.box_height / 2),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
        )
        self.box = RigidObject(box_cfg)
        self.scene.rigid_objects["box"] = self.box
        super()._setup_scene()

    def _get_rewards(self) -> torch.Tensor:
        # Current yaw is okay, no penalty
        yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)[2]
        self.oracle.reference.base_ori[self.env_indices, self.oracle.phase.squeeze()] = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        return super()._get_rewards()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or env_ids.numel() == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.box.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Randomize heading
        self.heading_angles[env_ids] = (
            torch.rand((env_ids.numel(),), device=self.sim.device) * (self.end_angle - self.start_angle)
            + self.start_angle
        )
        cos_heading = torch.cos(self.heading_angles[env_ids])
        sin_heading = torch.sin(self.heading_angles[env_ids])

        # Randomize box position
        box_start_state = self.box.data.default_root_state[env_ids]
        box_start_state[:, 0] = cos_heading * self.cfg.box_start
        box_start_state[:, 1] = sin_heading * self.cfg.box_start
        box_start_state[:, :3] += self.scene.env_origins[env_ids]
        self.box.write_root_pose_to_sim(box_start_state[:, :7], env_ids)
        self.box.write_root_velocity_to_sim(box_start_state[:, 7:], env_ids)

        # Randomize target position
        self.target_pos[env_ids, 0] = cos_heading * self.cfg.target
        self.target_pos[env_ids, 1] = sin_heading * self.cfg.target
        self.target_pos[env_ids, :2] += self.scene.env_origins[env_ids, :2]

    def _get_feedback(self):
        feedback = {
            "robot_pos": self.robot.data.root_pos_w[:, :3],
            "box_pos": self.box.data.root_pos_w[:, :3],
            "target_pos": self.target_pos,
            "heading": self.heading_angles,
        }
        return feedback

    def render_marker_visualization(self):

        # robot base pose traj
        ref_base_pos_traj = self.oracle.reference.base_pos.view(-1, 3).clone()
        ref_base_ori_traj = self.oracle.reference.base_ori.view(-1, 4).clone()

        # box pose
        ref_box_pos_traj = self.oracle.reference.box_pos.view(-1, 3).clone()
        ref_box_ori_traj = torch.zeros((self.num_envs * self.oracle.prediction_horizon, 4), device=self.sim.device)
        ref_box_ori_traj[:, 3] = 1.0

        # get current robot base pose ref
        ref_base_pos = torch.gather(
            self.oracle.reference.base_pos, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(1)
        ref_base_ori = torch.gather(
            self.oracle.reference.base_ori, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 4)
        ).squeeze(1)
        # get current box pose ref
        ref_box_pos = torch.gather(
            self.oracle.reference.box_pos, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(1)
        ref_box_ori = torch.zeros((self.num_envs, 4), device=self.sim.device)
        ref_box_ori[:, 3] = 1.0

        # stack the current base pose ref
        marker_pos = torch.cat((ref_base_pos_traj, ref_base_pos, ref_box_pos_traj, ref_box_pos), dim=0)
        marker_ori = torch.cat((ref_base_ori_traj, ref_base_ori, ref_box_ori_traj, ref_box_ori), dim=0)

        marker_indices = torch.arange(2 * (self.oracle.prediction_horizon + 1), device=self.sim.device).repeat(
            self.num_envs
        )
        marker_indices *= 0  # frames
        marker_indices[
            self.oracle.prediction_horizon * self.num_envs : self.oracle.prediction_horizon * self.num_envs
            + self.num_envs
        ] = 1  # base
        marker_indices[2 * self.oracle.prediction_horizon * self.num_envs + self.num_envs :] = 2  # box

        self.marker.visualize(marker_pos, marker_ori, marker_indices=marker_indices)
