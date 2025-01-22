import os
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_rotate_inverse

from ogmp_isaac.assets import *
from ogmp_isaac.assets import GOAL_DEPTH, GOALPOSTS_CFG
from ogmp_isaac.tasks.base_env import BaseEnv, BaseEnvCfg

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects")


@configclass
class SoccerEnvCfg(BaseEnvCfg):
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "soccer_ball.usd"),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.07),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    goalposts: ArticulationCfg = GOALPOSTS_CFG.replace(prim_path="/World/envs/env_.*/Goalpost")

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
            "ball": sim_utils.SphereCfg(
                radius=0.07,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)),
            ),
        },
    )

    rewards = {
        "base_pos": {"weight": 0.4, "exp_coeff": 3.0},
        "base_ori": {"weight": 0.23, "exp_coeff": 5.0},
        "base_lin_vel": {"weight": 0.3, "exp_coeff": 2.0},
        "ball_closeness": {"weight": 0.5, "threshold": 0.5},
        "preference": {"weight": -1.0,},
        "ball_pos": {"weight": 1.0, "exp_scale": 2.0},
        "ball_vel": {"weight": 1.0, "exp_scale": 2.0},
        "torque_exp": {"weight": 0.15, "exp_coeff": 0.05},
        "action": {"weight": 0.15, "exp_coeff": 1.0},
        "goal": {"weight": 0.0},
    }
    observations = [
        "base_z",
        "base_ori",
        "joint_pos",
        "base_lin_vel",
        "base_ang_vel",
        "joint_vel",
        "ball_dist",
        "target_dist",
        "sinusoid_phase",
    ]
    terminations = {
        "base_pos_x": 0.4,
        "base_pos_y": 0.4,
        "base_pos_z": 0.2,
        "ball_pos_x": 0.4,
    }
    oracle = {
        "name": "KickSoccerOracle",
        "params": {
            "speed": 0.8,
            "reach_thresh": 0.4,
            "detach_thresh": 0.4,
        },
    }
    ball_start = 1.0
    ball_start_vel = 0.0
    omni_direction_lim = [0.0, 360.0]
    target = 3.0
    goal_pos = 5.0
    drag_coeff = 1.5

    visualize_goalpost = False


class SoccerEnv(BaseEnv):
    cfg: SoccerEnvCfg

    def __init__(self, cfg: SoccerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.target_pos = torch.zeros((self.num_envs, 2), device=self.sim.device)
        self.heading_angles = torch.zeros((self.num_envs,), device=self.sim.device)
        self.start_angle = torch.deg2rad(torch.tensor(self.cfg.omni_direction_lim[0], device=self.sim.device))
        self.end_angle = torch.deg2rad(torch.tensor(self.cfg.omni_direction_lim[1], device=self.sim.device))
        self.cfg.goal_pos = self.cfg.target + 2.0

        if self.cfg.visualize_markers:
            self.marker = VisualizationMarkers(self.cfg.marker_cfg)

    def _setup_scene(self):
        self.ball = RigidObject(self.cfg.ball)
        self.scene.rigid_objects["ball"] = self.ball
        if self.cfg.visualize_goalpost:
            self.goalposts = Articulation(self.cfg.goalposts)
            self.scene.articulations["goalposts"] = self.goalposts

        super()._setup_scene()
    
    def _apply_action(self):
        super()._apply_action()
        self.apply_drag_force()

    def apply_drag_force(self):
        # F = -c * v^2
        drag_force_w = torch.zeros((self.num_envs, 3), device=self.sim.device)
        drag_force_w[:, :2] = (
            -self.cfg.drag_coeff
            * torch.square(self.ball.data.root_lin_vel_w[:, :2])
            * torch.sign(self.ball.data.root_lin_vel_w[:, :2])
        )
        drag_force_b = quat_rotate_inverse(self.ball.data.root_quat_w, drag_force_w).unsqueeze(1)
        self.ball.set_external_force_and_torque(drag_force_b, torch.zeros_like(drag_force_b))
        self.ball.write_data_to_sim()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or env_ids.numel() == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.ball.reset(env_ids)
        if self.cfg.visualize_goalpost:
            self.goalposts.reset(env_ids)
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

        # Randomize ball position
        ball_start_state = self.ball.data.default_root_state[env_ids]
        ball_start_state[:, 0] = cos_heading * self.cfg.ball_start
        ball_start_state[:, 1] = sin_heading * self.cfg.ball_start
        ball_start_state[:, :3] += self.scene.env_origins[env_ids]
        self.ball.write_root_pose_to_sim(ball_start_state[:, :7], env_ids)

        # Randomize ball velocity
        vel_heading = torch.rand((env_ids.numel(),), device=self.sim.device) * (torch.pi)
        ball_start_state[:, 7] = torch.cos(vel_heading) * self.cfg.ball_start_vel
        ball_start_state[:, 8] = torch.sin(vel_heading) * self.cfg.ball_start_vel
        self.ball.write_root_velocity_to_sim(ball_start_state[:, 7:], env_ids)

        # Randomize target position
        self.target_pos[env_ids, 0] = cos_heading * self.cfg.target
        self.target_pos[env_ids, 1] = sin_heading * self.cfg.target
        self.target_pos[env_ids, :2] += self.scene.env_origins[env_ids, :2]

        if self.cfg.visualize_goalpost:
            goalposts_start_state = self.goalposts.data.default_root_state[env_ids]
            goalposts_start_state[:, 0] = cos_heading * (self.cfg.goal_pos + GOAL_DEPTH)
            goalposts_start_state[:, 1] = sin_heading * (self.cfg.goal_pos + GOAL_DEPTH)
            goalposts_start_state[:, :3] += self.scene.env_origins[env_ids]
            _, _, init_yaw = euler_xyz_from_quat(goalposts_start_state[:, 3:7])
            target_yaw = init_yaw + self.heading_angles[env_ids]
            goalposts_start_state[:, 3:7] = quat_from_euler_xyz(
                torch.zeros_like(target_yaw), torch.zeros_like(target_yaw), target_yaw
            )
            self.goalposts.write_root_pose_to_sim(goalposts_start_state[:, :7], env_ids)
            self.goalposts.write_root_velocity_to_sim(goalposts_start_state[:, 7:], env_ids)

        # self.is_goal[env_ids] = False

    def _get_feedback(self):
        feedback = {
            "robot_pos": self.robot.data.root_pos_w[:, :3],
            "ball_pos": self.ball.data.root_pos_w[:, :3],
            "target_pos": self.target_pos,
            "heading": self.heading_angles,
        }
        return feedback

    def render_marker_visualization(self):

        # robot base pose traj
        ref_base_pos_traj = self.oracle.reference.base_pos.view(-1, 3).clone()
        ref_base_ori_traj = self.oracle.reference.base_ori.view(-1, 4).clone()

        # ball pose
        ref_ball_pos_traj = self.oracle.reference.ball_pos.view(-1, 3).clone()
        ref_ball_ori_traj = torch.zeros((self.num_envs * self.oracle.prediction_horizon, 4), device=self.sim.device)
        ref_ball_ori_traj[:, 3] = 1.0

        # get current robot base pose ref
        ref_base_pos = torch.gather(
            self.oracle.reference.base_pos, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(1)
        ref_base_ori = torch.gather(
            self.oracle.reference.base_ori, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 4)
        ).squeeze(1)
        # get current ball pose ref
        ref_ball_pos = torch.gather(
            self.oracle.reference.ball_pos, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(1)
        ref_ball_ori = torch.zeros((self.num_envs, 4), device=self.sim.device)
        ref_ball_ori[:, 3] = 1.0

        # stack the current base pose ref
        marker_pos = torch.cat((ref_base_pos_traj, ref_base_pos, ref_ball_pos_traj, ref_ball_pos), dim=0)
        marker_ori = torch.cat((ref_base_ori_traj, ref_base_ori, ref_ball_ori_traj, ref_ball_ori), dim=0)

        marker_indices = torch.arange(2 * (self.oracle.prediction_horizon + 1), device=self.sim.device).repeat(
            self.num_envs
        )
        marker_indices *= 0  # frames
        marker_indices[
            self.oracle.prediction_horizon * self.num_envs : self.oracle.prediction_horizon * self.num_envs
            + self.num_envs
        ] = 1  # base
        marker_indices[2 * self.oracle.prediction_horizon * self.num_envs + self.num_envs :] = 2  # ball

        self.marker.visualize(marker_pos, marker_ori, marker_indices=marker_indices)
