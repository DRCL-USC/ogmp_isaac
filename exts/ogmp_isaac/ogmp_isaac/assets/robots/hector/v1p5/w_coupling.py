import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils import configclass

from ogmp_isaac.assets.actuators import (
    HectorV1p5DCActuatorCfg,
    HectorV1p5IdealPDActuatorCfg,
    HectorV1p5ImplicitPDActuatorCfg,
)
from ogmp_isaac.assets.robots.robot_cfg import RobotCfg

##
# Configuration
##


ASSETS_DIR = os.path.dirname(os.path.realpath(__file__))

# implicict actuator
HECTOR_V1P5_ART_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "w_coupling.usd"),
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
            # fix_root_link=True, # NOTE: comment this before training
            # solver_position_iteration_count=4,
            # solver_velocity_iteration_count=0,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),  # when fix_root_link=False
        # pos=(0.0, 0.0, 3.00), # when fix_root_link=True
        joint_pos={
            "l_hip_yaw_joint": 0.0,
            "r_hip_yaw_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "r_hip_pitch_joint": 0.0,
            "l_knee_joint": 0.0,
            "r_knee_joint": 0.0,
            "l_ankle_joint": 0.0,
            "r_ankle_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": HectorV1p5ImplicitPDActuatorCfg(
            knee_gear_ratio=2.0,
            joint_names_expr=[
                "l_hip_yaw_joint",
                "r_hip_yaw_joint",
                "l_hip_roll_joint",
                "r_hip_roll_joint",
                "l_hip_pitch_joint",
                "r_hip_pitch_joint",
                "l_knee_joint",
                "r_knee_joint",
                "l_ankle_joint",
                "r_ankle_joint",
            ],
            effort_limit={
                "l_hip_yaw_joint": 33.5,
                "r_hip_yaw_joint": 33.5,
                "l_hip_roll_joint": 33.5,
                "r_hip_roll_joint": 33.5,
                "l_hip_pitch_joint": 33.5,
                "r_hip_pitch_joint": 33.5,
                "l_knee_joint": 67.0,  # motor_tau_max*knee_gear_ratio
                "r_knee_joint": 67.0,  # motor_tau_max*knee_gear_ratio
                "l_ankle_joint": 33.5,
                "r_ankle_joint": 33.5,
            },
            velocity_limit={
                "l_hip_yaw_joint": 21.0,
                "r_hip_yaw_joint": 21.0,
                "l_hip_roll_joint": 21.0,
                "r_hip_roll_joint": 21.0,
                "l_hip_pitch_joint": 21.0,
                "r_hip_pitch_joint": 21.0,
                "l_knee_joint": 10.5,  # motor_speed_max/knee_gear_ratio
                "r_knee_joint": 10.5,  # motor_speed_max/knee_gear_ratio
                "l_ankle_joint": 21.0,
                "r_ankle_joint": 21.0,
            },
            stiffness={
                "l_hip_yaw_joint": 30.0,
                "r_hip_yaw_joint": 30.0,
                "l_hip_roll_joint": 30.0,
                "r_hip_roll_joint": 30.0,
                "l_hip_pitch_joint": 30.0,
                "r_hip_pitch_joint": 30.0,
                "l_knee_joint": 120.0,  # kp*knee_gear_ratio^2
                "r_knee_joint": 120.0,  # kp*knee_gear_ratio^2
                "l_ankle_joint": 30.0,
                "r_ankle_joint": 30.0,
            },
            damping={
                "l_hip_yaw_joint": 0.5,
                "r_hip_yaw_joint": 0.5,
                "l_hip_roll_joint": 0.5,
                "r_hip_roll_joint": 0.5,
                "l_hip_pitch_joint": 0.5,
                "r_hip_pitch_joint": 0.5,
                "l_knee_joint": 2.0,  # kd*knee_gear_ratio^2
                "r_knee_joint": 2.0,  # kd*knee_gear_ratio^2
                "l_ankle_joint": 0.5,
                "r_ankle_joint": 0.5,
            },
        ),
    },
)

# idealPD actuator
HECTOR_V1P5_IPD_ART_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "w_coupling.usd"),
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
            # fix_root_link=True, # NOTE: comment this before training
            # solver_position_iteration_count=4,
            # solver_velocity_iteration_count=0,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),  # when fix_root_link=False
        # pos=(0.0, 0.0, 1.00), # when fix_root_link=True
        joint_pos={
            "l_hip_yaw_joint": 0.0,
            "r_hip_yaw_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "r_hip_pitch_joint": 0.0,
            "l_knee_joint": 0.0,
            "r_knee_joint": 0.0,
            "l_ankle_joint": 0.0,
            "r_ankle_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": HectorV1p5IdealPDActuatorCfg(
            knee_gear_ratio=2.0,
            joint_names_expr=[
                "l_hip_yaw_joint",
                "r_hip_yaw_joint",
                "l_hip_roll_joint",
                "r_hip_roll_joint",
                "l_hip_pitch_joint",
                "r_hip_pitch_joint",
                "l_knee_joint",
                "r_knee_joint",
                "l_ankle_joint",
                "r_ankle_joint",
            ],
            effort_limit={
                "l_hip_yaw_joint": 33.5,
                "r_hip_yaw_joint": 33.5,
                "l_hip_roll_joint": 33.5,
                "r_hip_roll_joint": 33.5,
                "l_hip_pitch_joint": 33.5,
                "r_hip_pitch_joint": 33.5,
                "l_knee_joint": 67.0,  # motor_tau_max*knee_gear_ratio
                "r_knee_joint": 67.0,  # motor_tau_max*knee_gear_ratio
                "l_ankle_joint": 33.5,
                "r_ankle_joint": 33.5,
            },
            velocity_limit={
                "l_hip_yaw_joint": 21.0,
                "r_hip_yaw_joint": 21.0,
                "l_hip_roll_joint": 21.0,
                "r_hip_roll_joint": 21.0,
                "l_hip_pitch_joint": 21.0,
                "r_hip_pitch_joint": 21.0,
                "l_knee_joint": 10.5,  # motor_speed_max/knee_gear_ratio
                "r_knee_joint": 10.5,  # motor_speed_max/knee_gear_ratio
                "l_ankle_joint": 21.0,
                "r_ankle_joint": 21.0,
            },
            stiffness={
                "l_hip_yaw_joint": 15.0,
                "r_hip_yaw_joint": 15.0,
                "l_hip_roll_joint": 15.0,
                "r_hip_roll_joint": 15.0,
                "l_hip_pitch_joint": 20.0,
                "r_hip_pitch_joint": 20.0,
                "l_knee_joint": 20.0,  # kp*knee_gear_ratio^2
                "r_knee_joint": 20.0,  # kp*knee_gear_ratio^2
                "l_ankle_joint": 10.0,
                "r_ankle_joint": 10.0,
            },
            damping={
                "l_hip_yaw_joint": 1.0,
                "r_hip_yaw_joint": 1.0,
                "l_hip_roll_joint": 1.0,
                "r_hip_roll_joint": 1.0,
                "l_hip_pitch_joint": 0.5,
                "r_hip_pitch_joint": 0.5,
                "l_knee_joint": 0.1,  # kd*knee_gear_ratio^2
                "r_knee_joint": 0.1,  # kd*knee_gear_ratio^2
                "l_ankle_joint": 0.05,
                "r_ankle_joint": 0.05,
            },
        ),
    },
)


# DC actuator
HECTOR_V1P5_DC_ART_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "w_coupling.usd"),
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
            # fix_root_link=True, # NOTE: comment this before training
            # solver_position_iteration_count=4,
            # solver_velocity_iteration_count=0,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),  # when fix_root_link=False
        # pos=(0.0, 0.0, 1.00), # when fix_root_link=True
        joint_pos={
            "l_hip_yaw_joint": 0.0,
            "r_hip_yaw_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "r_hip_pitch_joint": 0.0,
            "l_knee_joint": 0.0,
            "r_knee_joint": 0.0,
            "l_ankle_joint": 0.0,
            "r_ankle_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": HectorV1p5DCActuatorCfg(
            knee_gear_ratio=2.0,
            joint_names_expr=[
                "l_hip_yaw_joint",
                "r_hip_yaw_joint",
                "l_hip_roll_joint",
                "r_hip_roll_joint",
                "l_hip_pitch_joint",
                "r_hip_pitch_joint",
                "l_knee_joint",
                "r_knee_joint",
                "l_ankle_joint",
                "r_ankle_joint",
            ],
            effort_limit={
                "l_hip_yaw_joint": 33.5,
                "r_hip_yaw_joint": 33.5,
                "l_hip_roll_joint": 33.5,
                "r_hip_roll_joint": 33.5,
                "l_hip_pitch_joint": 33.5,
                "r_hip_pitch_joint": 33.5,
                "l_knee_joint": 67.0,  # motor_tau_max*knee_gear_ratio
                "r_knee_joint": 67.0,  # motor_tau_max*knee_gear_ratio
                "l_ankle_joint": 33.5,
                "r_ankle_joint": 33.5,
            },
            velocity_limit={
                "l_hip_yaw_joint": 21.0,
                "r_hip_yaw_joint": 21.0,
                "l_hip_roll_joint": 21.0,
                "r_hip_roll_joint": 21.0,
                "l_hip_pitch_joint": 21.0,
                "r_hip_pitch_joint": 21.0,
                "l_knee_joint": 10.5,  # motor_speed_max/knee_gear_ratio
                "r_knee_joint": 10.5,  # motor_speed_max/knee_gear_ratio
                "l_ankle_joint": 21.0,
                "r_ankle_joint": 21.0,
            },
            stiffness={
                "l_hip_yaw_joint": 15.0,
                "r_hip_yaw_joint": 15.0,
                "l_hip_roll_joint": 15.0,
                "r_hip_roll_joint": 15.0,
                "l_hip_pitch_joint": 20.0,
                "r_hip_pitch_joint": 20.0,
                "l_knee_joint": 20.0,  # kp*knee_gear_ratio^2
                "r_knee_joint": 20.0,  # kp*knee_gear_ratio^2
                "l_ankle_joint": 10.0,
                "r_ankle_joint": 10.0,
            },
            damping={
                "l_hip_yaw_joint": 1.0,
                "r_hip_yaw_joint": 1.0,
                "l_hip_roll_joint": 1.0,
                "r_hip_roll_joint": 1.0,
                "l_hip_pitch_joint": 0.5,
                "r_hip_pitch_joint": 0.5,
                "l_knee_joint": 0.1,  # kd*knee_gear_ratio^2
                "r_knee_joint": 0.1,  # kd*knee_gear_ratio^2
                "l_ankle_joint": 0.05,
                "r_ankle_joint": 0.05,
            },
            friction_static={
                ".*": 0.2,
            },
            friction_dynamic={
                ".*": 0.02,
            },
            activation_vel={
                ".*": 0.1,
            },
            saturation_effort=402,
            armature={".*": 6.9e-5 * 81},
        ),
    },
)

HECTOR_V1P5_MPCL = [
    [-0.7900000214576721, 0.7900000214576721],  # l_hip_yaw_joint
    [-0.7900000214576721, 0.7900000214576721],  # r_hip_yaw_joint
    [-0.7900000214576721, 0.7900000214576721],  # l_hip_roll_joint
    [-0.7900000214576721, 0.7900000214576721],  # r_hip_roll_joint
    [-1.0499999523162842, 1.1299999952316284],  # l_hip_pitch_joint
    [-1.0499999523162842, 1.1299999952316284],  # r_hip_pitch_joint
    [-1.74, 3.5],  # l_knee_joint, joint limits * knee_gear_ratio
    [-1.74, 3.5],  # r_knee_joint, joint limits * knee_gear_ratio
    [-1.5700000524520874, 0.7900000214576721],  # l_ankle_joint
    [-1.5700000524520874, 0.7900000214576721],  # r_ankle_joint
]

epsilon = 0.0
HECTOR_V1P5_DC_MPCL = [[bl - epsilon, bu + epsilon] for bl, bu in HECTOR_V1P5_MPCL]
HECTOR_V1P5_MPCL_DIFF = max([max(HECTOR_V1P5_MPCL[i]) - min(HECTOR_V1P5_MPCL[i]) for i in range(10)])
HECTOR_V1P5_DC_MPCL_DIFF = max([max(HECTOR_V1P5_DC_MPCL[i]) - min(HECTOR_V1P5_DC_MPCL[i]) for i in range(10)])

HECTOR_V1P5_DC_BAD_CONTACT_BODIES = [
    "l_thigh",
    "r_thigh",
    "l_thigh1_trans",
    "r_thigh1_trans",
    "l_thigh2_trans",
    "r_thigh2_trans",
    "l_calf",
    "r_calf",
]
HECTOR_V1P5_DC_FEET_BODIES = ["l_toe", "r_toe"]

HECTOR_V1P5_CFG = RobotCfg(
    articulation_cfg=HECTOR_V1P5_ART_CFG,
    mpcl=HECTOR_V1P5_MPCL,
    bad_contact_bodies=[],
    feet_bodies=[],
    num_joints=10,
    max_joint_torque=134.0,
    max_total_joint_vel=61.22,  # sqrt(8*21^2 + 2*10.5^2),
    max_total_joint_acc=122.45,  # sqrt(8*42^2 + 2*21^2),
    max_mpcl_diff=HECTOR_V1P5_MPCL_DIFF,
)

HECTOR_V1P5_IPD_CFG = RobotCfg(
    articulation_cfg=HECTOR_V1P5_IPD_ART_CFG,
    mpcl=HECTOR_V1P5_MPCL,
    bad_contact_bodies=HECTOR_V1P5_DC_BAD_CONTACT_BODIES,
    feet_bodies=HECTOR_V1P5_DC_FEET_BODIES,
    num_joints=10,
    max_joint_torque=134.0,
    max_total_joint_vel=61.22,  # sqrt(8*21^2 + 2*10.5^2),
    max_total_joint_acc=122.45,  # sqrt(8*42^2 + 2*21^2),
    max_mpcl_diff=HECTOR_V1P5_MPCL_DIFF,
)

HECTOR_V1P5_DC_CFG = RobotCfg(
    articulation_cfg=HECTOR_V1P5_DC_ART_CFG,
    mpcl=HECTOR_V1P5_DC_MPCL,
    bad_contact_bodies=HECTOR_V1P5_DC_BAD_CONTACT_BODIES,
    feet_bodies=HECTOR_V1P5_DC_FEET_BODIES,
    num_joints=10,
    max_joint_torque=134.0,
    max_total_joint_vel=61.22,  # sqrt(8*21^2 + 2*10.5^2),
    max_total_joint_acc=122.45,  # sqrt(8*42^2 + 2*21^2),
    max_mpcl_diff=HECTOR_V1P5_DC_MPCL_DIFF,
)
