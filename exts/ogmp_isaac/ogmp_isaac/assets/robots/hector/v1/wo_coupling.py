import os

from ogmp_isaac.assets.actuators import IdentifiedActuatorCfg
from ogmp_isaac.assets.robots.robot_cfg import RobotCfg

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

##
# Configuration
##


ASSETS_DIR = os.path.dirname(os.path.realpath(__file__))

HECTOR_V1_ART_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "wo_coupling.usd"),
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={  # The order of the joints is how the tensor is returned in robot.data.joint_pos
            "L_hip_joint": 0.0,
            "L_hip2_joint": 0.0,
            "R_hip_joint": 0.0,
            "L_thigh_joint": 0.0,
            "R_hip2_joint": 0.0,
            "L_calf_joint": 0.0,
            "R_thigh_joint": 0.0,
            "L_toe_joint": 0.0,
            "R_calf_joint": 0.0,
            "R_toe_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"],
            effort_limit=33.5,
            velocity_limit=21.0,
            stiffness={
                ".*_hip_joint": 30.0,
                ".*_hip2_joint": 30.0,
                ".*_thigh_joint": 30.0,
                ".*_calf_joint": 30.0,
                ".*_toe_joint": 30.0,
            },
            damping={
                ".*_hip_joint": 0.5,
                ".*_hip2_joint": 0.5,
                ".*_thigh_joint": 0.5,
                ".*_calf_joint": 0.5,
                ".*_toe_joint": 0.5,
            },
        )
    },
)

HECTOR_V1_DC_ART_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "wo_coupling.usd"),
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={  # The order of the joints is how the tensor is returned in robot.data.joint_pos
            "L_hip_joint": 0.0,
            "L_hip2_joint": 0.0,
            "R_hip_joint": 0.0,
            "L_thigh_joint": 0.0,
            "R_hip2_joint": 0.0,
            "L_calf_joint": 0.0,
            "R_thigh_joint": 0.0,
            "L_toe_joint": 0.0,
            "R_calf_joint": 0.0,
            "R_toe_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdentifiedActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"],
            effort_limit=33.5,
            velocity_limit=21.0,
            stiffness={
                ".*": 15.0,
            },
            damping={
                ".*": 0.6,
            },
            friction_static={
                ".*": 0.45,
            },
            activation_vel={
                ".*": 0.1,
            },
            friction_dynamic={
                ".*": 0.023,
            },
            saturation_effort=None,
        )
    },
)

# JOINT_ANGLE_LIMITS = [
#     [-0.261, 0.261],  # L_hip_joint
#     [-0.523, 0.261],  # L_hip2_joint
#     [-0.261, 0.261],  # R_hip_joint
#     [-3.14, 3.14],  # L_thigh_joint
#     [-0.261, 0.523],  # R_hip2_joint
#     [-3.14, 3.14],  # L_calf_joint
#     [-3.14, 3.14],  # R_thigh_joint
#     [-1.4, 1.4],  # L_toe_joint
#     [-3.14, 3.14],  # R_calf_joint
#     [-1.4, 1.4],  # R_toe_joint
# ]

HECTOR_V1_MPCL = [[-3.14, 3.14]] * 10
HECTOR_V1_MPCL_DIFF = max([max(HECTOR_V1_MPCL[i]) - min(HECTOR_V1_MPCL[i]) for i in range(10)])

HECTOR_V1_CFG = RobotCfg(
    articulation_cfg=HECTOR_V1_ART_CFG,
    mpcl=HECTOR_V1_MPCL,
    bad_contact_bodies=[],
    feet_bodies=[],
    num_joints=10,
    max_joint_torque=105.94,
    max_total_joint_vel=21.0,
    max_total_joint_acc=42.0,
    max_mpcl_diff=HECTOR_V1_MPCL_DIFF,
)

HECTOR_V1_DC_CFG = RobotCfg(
    articulation_cfg=HECTOR_V1_DC_ART_CFG,
    mpcl=HECTOR_V1_MPCL,
    bad_contact_bodies=[],
    feet_bodies=[],
    num_joints=10,
    max_joint_torque=105.94,
    max_total_joint_vel=21.0,
    max_total_joint_acc=42.0,
    max_mpcl_diff=HECTOR_V1_MPCL_DIFF,
)
