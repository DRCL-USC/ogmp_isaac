from dataclasses import dataclass

from omni.isaac.lab.assets.articulation import ArticulationCfg


@dataclass
class RobotCfg:
    articulation_cfg: ArticulationCfg
    mpcl: list  # Motor Position Command Limits
    bad_contact_bodies: list
    feet_bodies: list
    num_joints: int
    max_joint_torque: float
    max_total_joint_vel: float
    max_total_joint_acc: float
    max_mpcl_diff: float
