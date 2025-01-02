import os
import torch

from ogmp_isaac.assets import PLATFORM_CFG
from ogmp_isaac.tasks.base_env import BaseEnv, BaseEnvCfg
from ogmp_isaac.utils.terminations import *

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects")


@configclass
class DiveEnvCfg(BaseEnvCfg):
    platform: ArticulationCfg = PLATFORM_CFG.replace(prim_path="/World/envs/env_.*/platform")
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
        "sinusoid_phase",
    ]
    terminations = [
        "ogmp_pos_z",
    ]
    ogmp_error_terminations = {
        "base_pos_z": 0.2,
    }


class DiveEnv(BaseEnv):
    cfg: DiveEnvCfg

    def __init__(self, cfg: DiveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.flip_dir = torch.zeros((self.num_envs, 2), dtype=torch.int64, device=self.sim.device)
        self.flip_options = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.int64, device=self.sim.device)
        if self.cfg.visualize_markers:
            self.marker = VisualizationMarkers(self.cfg.marker_cfg)

    def _setup_scene(self):
        self.platform = Articulation(self.cfg.platform)
        super()._setup_scene()
        self.scene.articulations["platform"] = self.platform

    def _apply_action(self):
        super()._apply_action()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = term_time_out(self)
        self.target_pos = torch.gather(
            self.oracle.reference.base_pos, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(1)
        self.target_ori = torch.gather(
            self.oracle.reference.base_ori, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 4)
        ).squeeze(1)
        self.target_state = torch.cat((self.target_pos, self.target_ori), dim=-1)
        died = torch.zeros((self.num_envs,), device=self.sim.device).bool()
        for func in self.termination_funcs:
            died |= func(self)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        self.platform.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 0] += 1.0
        default_root_state[:, 2] += 1.5

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.platform.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        # default_root_state[:, 0] -= 1.0
        default_root_state[:, 2] += 0.75

        self.platform.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.platform.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.flip_dir[env_ids] = self.flip_options[torch.randint(0, 4, (len(env_ids),), device=self.sim.device)]
        # self.flip_dir[env_ids, 0] = -1
        # self.flip_dir[env_ids, 1] = 0

        self.max_modes[env_ids] = 0

    def _get_feedback(self):
        feedback = {
            "robot_pos": self.robot.data.root_pos_w,
            "robot_ori": self.robot.data.root_quat_w,
            "flip_dir": self.flip_dir,
        }
        return feedback

    def render_marker_visualization(self):

        # only robot base pose traj
        ref_base_pos_traj = self.oracle.reference.base_pos.view(-1, 3).clone()
        ref_base_ori_traj = self.oracle.reference.base_ori.view(-1, 4).clone()

        # get current robot base pose ref
        ref_base_pos = torch.gather(
            self.oracle.reference.base_pos, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(1)
        ref_base_ori = torch.gather(
            self.oracle.reference.base_ori, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 4)
        ).squeeze(1)

        # stack the current base pose ref
        marker_pos = torch.cat((ref_base_pos_traj, ref_base_pos), dim=0)
        marker_ori = torch.cat((ref_base_ori_traj, ref_base_ori), dim=0)

        marker_indices = torch.arange(self.oracle.prediction_horizon + 1, device=self.sim.device).repeat(self.num_envs)
        marker_indices *= 0  # frames
        marker_indices[self.oracle.prediction_horizon * self.num_envs :] = 1  # base

        self.marker.visualize(marker_pos, marker_ori, marker_indices=marker_indices)
