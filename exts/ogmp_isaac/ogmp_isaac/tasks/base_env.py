from __future__ import annotations

import torch

import omni.isaac.core.utils.torch as torch_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg

# Sensors
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

import ogmp_isaac.oracles as oracles
import ogmp_isaac.utils.observations as observations
import ogmp_isaac.utils.rewards as rewards
import ogmp_isaac.utils.terminations as terminations
from ogmp_isaac.assets import *

torch.set_printoptions(profile="short", sci_mode=False)


@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    seed: int = 0

    # simulation
    decimation = 4
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # sim: SimulationCfg = SimulationCfg(dt=0.003, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    # robot
    robot_model: str = "HECTOR_V1P5"
    robot: ArticulationCfg = None

    # env
    num_actions = 1
    num_observations = 1
    episode_length_s = 15.0
    env_dt = 1.0 / 30
    rewards = {}
    terminations = []
    observations = []

    # oracle
    oracle = {
        "name": "Oracle",
        "params": {
            "speed": 0.5,
        },
    }
    ogmp_error_terminations = {}

    visualize_markers = False


class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg

    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if "HECTOR_V1P5" in self.cfg.robot_model:
            self.hector_knee_ratio = self.robot.actuators["legs"].cfg.knee_gear_ratio

        self.reward_funcs = []
        self.termination_funcs = []
        self.observation_funcs = []
        self.compose_reward_funcs()
        self.compose_termination_funcs()
        self.compose_observation_funcs()

        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        self.oracle = getattr(oracles, self.cfg.oracle["name"])(num_envs=self.num_envs, **self.cfg.oracle["params"])

        self.previous_actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.sim.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.sim.device)
        self.max_modes = torch.zeros(self.num_envs, dtype=torch.int64, device=self.sim.device)
        self.env_indices = torch.arange(self.num_envs, device=self.sim.device)
        self.previous_torques = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.sim.device)

    def compose_reward_funcs(self):
        for key in self.cfg.rewards.keys():
            self.reward_funcs.append(getattr(rewards, "rew_" + key))

    def compose_termination_funcs(self):
        for term in self.cfg.terminations:
            self.termination_funcs.append(getattr(terminations, "term_" + term))

    def compose_observation_funcs(self):
        for obs in self.cfg.observations:
            self.observation_funcs.append(getattr(observations, "obs_" + obs))

    def _setup_scene(self):
        robot_cfg = ROBOTS[self.cfg.robot_model]
        self.cfg.num_actions = robot_cfg.num_joints
        self.max_joint_action = robot_cfg.max_mpcl_diff * torch.sqrt(
            torch.tensor(self.cfg.num_actions, device=self.sim.device)
        )
        self.max_joint_torque = torch.tensor(robot_cfg.max_joint_torque, device=self.sim.device)
        self.max_total_joint_vel = torch.tensor(robot_cfg.max_total_joint_vel, device=self.sim.device)
        self.max_total_joint_acc = torch.tensor(robot_cfg.max_total_joint_acc, device=self.sim.device)
        self.motor_pos_cmd_limits = torch.tensor(robot_cfg.mpcl, device=self.sim.device)

        self.cfg.robot = robot_cfg.articulation_cfg.replace(prim_path="/World/envs/env_.*/Robot")

        self.robot = Articulation(self.cfg.robot)

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articultion to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # contact sensors
        if robot_cfg.bad_contact_bodies:
            self.bad_absolute_contact_sensors = []
            for i, body_name in enumerate(robot_cfg.bad_contact_bodies):
                bcs_cfg = ContactSensorCfg(
                    prim_path="/World/envs/env_.*/Robot/" + body_name,
                    history_length=3,
                    debug_vis=False,
                    track_air_time=True,
                )
                self.bad_absolute_contact_sensors.append(ContactSensor(bcs_cfg))
                self.scene.sensors[f"bad_contact_sensor_{i}"] = self.bad_absolute_contact_sensors[-1]

        if robot_cfg.feet_bodies:
            self.feet_contact_sensors = []
            for i, body_name in enumerate(robot_cfg.feet_bodies):
                fcs_cfg = ContactSensorCfg(
                    prim_path="/World/envs/env_.*/Robot/" + body_name,
                    history_length=3,
                    debug_vis=False,
                    track_air_time=True,
                )
                self.feet_contact_sensors.append(ContactSensor(fcs_cfg))
                self.scene.sensors[f"feet_contact_sensor_{i}"] = self.feet_contact_sensors[-1]

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_torques = self.robot.data.applied_torque.clone()
        self.previous_actions = self.actions
        self.actions = actions.clone()

        feedback = self._get_feedback()
        feedback["env_ids"] = self.robot._ALL_INDICES
        net_condition = self.oracle.check_transitions(feedback)
        env_ids = torch.where(net_condition)[0]
        if env_ids.numel() > 0:
            feedback["env_ids"] = env_ids
            self.oracle.generate_reference(feedback)
            self.oracle.phase[env_ids] = 0
            self.max_modes[env_ids] = torch.max(self.max_modes[env_ids], self.oracle.modes[env_ids])

        if self.cfg.visualize_markers:
            self.render_marker_visualization()

    def _apply_action(self):
        angles = torch.clamp(self.actions, self.motor_pos_cmd_limits[:, 0], self.motor_pos_cmd_limits[:, 1])
        self.robot.set_joint_position_target(angles, joint_ids=self._joint_dof_idx)

    def _get_observations(self) -> dict:
        obs = []
        for func in self.observation_funcs:
            obs.append(func(self))
        obs = torch.cat(obs, dim=-1)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        rewards = torch.zeros((self.num_envs,), device=self.sim.device)
        for func in self.reward_funcs:
            rewards += func(self)
        self.oracle.phase += 1
        self.oracle.phase = self.oracle.phase % self.oracle.prediction_horizon
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = terminations.term_time_out(self)
        died = torch.zeros((self.num_envs,), device=self.sim.device).bool()
        for func in self.termination_funcs:
            died |= func(self)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or env_ids.numel() == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self.oracle.reset(env_ids)
        self.max_modes[env_ids] = 0

    def _get_feedback(self):
        feedback = {
            "robot_pos": self.robot.data.root_pos_w[:, :3],
        }
        return feedback

    def render_marker_visualization(self):
        # implement this in the derived classes as necessary
        pass

    @staticmethod
    def seed(seed: int = -1) -> int:
        """
        Identical to function from DirectRLEnv, but sets torch_deterministic=True
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed, torch_deterministic=True)
