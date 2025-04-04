# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation import ArticulationCfg

from ogmp_isaac.assets.actuators import IdentifiedActuatorCfg
from ogmp_isaac.assets.robots.robot_cfg import RobotCfg

# from berkeley_humanoid.assets import ISAAC_ASSET_DIR

BERKELEY_HUMANOID_HXX_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*HR", ".*HAA"],
    effort_limit=20.0,
    velocity_limit=23,
    saturation_effort=402,
    stiffness={".*": 10.0},
    damping={".*": 1.5},
    armature={".*": 6.9e-5 * 81},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

BERKELEY_HUMANOID_HFE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*HFE"],
    effort_limit=30.0,
    velocity_limit=20,
    saturation_effort=443,
    stiffness={".*": 15.0},
    damping={".*": 1.5},
    armature={".*": 9.4e-5 * 81},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

BERKELEY_HUMANOID_KFE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*KFE"],
    effort_limit=30.0,
    velocity_limit=14,
    saturation_effort=560,
    stiffness={".*": 15.0},
    damping={".*": 1.5},
    armature={".*": 1.5e-4 * 81},
    friction_static=0.8,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

BERKELEY_HUMANOID_FFE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*FFE"],
    effort_limit=20.0,
    velocity_limit=23,
    saturation_effort=402,
    stiffness={".*": 1.0},
    damping={".*": 0.01},  # 0.1
    armature={".*": 6.9e-5 * 81},
    friction_static=1.0,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

BERKELEY_HUMANOID_FAA_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*FAA"],
    effort_limit=5.0,
    velocity_limit=42,
    saturation_effort=112,
    stiffness={".*": 1.0},
    damping={".*": 0.01},  # 0.1
    armature={".*": 6.1e-6 * 81},
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.005,
)

import os

ASSETS_DIR = os.path.dirname(os.path.realpath(__file__))

BERKELEY_HUMANOID_ART_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAAC_ASSET_DIR}/Robots/berkeley_humanoid.usd",
        usd_path=os.path.join(ASSETS_DIR, "berkeley_humanoid.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.8), # when fix_root_link=False
        pos=(0.0, 0.0, 0.515),
        joint_pos={
            "LL_HR": -0.071,
            "LR_HR": 0.071,
            "LL_HAA": 0.103,
            "LR_HAA": -0.103,
            "LL_HFE": -0.463,
            "LR_HFE": -0.463,
            "LL_KFE": 0.983,
            "LR_KFE": 0.983,
            "LL_FFE": -0.350,
            "LR_FFE": -0.350,
            "LL_FAA": 0.126,
            "LR_FAA": -0.126,
        },
    ),
    actuators={
        "hxx": BERKELEY_HUMANOID_HXX_ACTUATOR_CFG,
        "hfe": BERKELEY_HUMANOID_HFE_ACTUATOR_CFG,
        "kfe": BERKELEY_HUMANOID_KFE_ACTUATOR_CFG,
        "ffe": BERKELEY_HUMANOID_FFE_ACTUATOR_CFG,
        "faa": BERKELEY_HUMANOID_FAA_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)

BERKELEY_HUMANOID_MPCL = [[-2.05, 2.05]] * 12
BERKELEY_HUMANOID_MPCL_DIFF = max([max(BERKELEY_HUMANOID_MPCL[i]) - min(BERKELEY_HUMANOID_MPCL[i]) for i in range(12)])

BERKELEY_HUMANOID_CFG = RobotCfg(
    articulation_cfg=BERKELEY_HUMANOID_ART_CFG,
    mpcl=BERKELEY_HUMANOID_MPCL,
    bad_contact_bodies=[],
    feet_bodies=[],
    num_joints=12,
    max_joint_torque=100.0,
    max_total_joint_vel=84.85,
    max_total_joint_acc=177.70,
    max_mpcl_diff=BERKELEY_HUMANOID_MPCL_DIFF,
)
