import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab_assets import G1_CFG as G1_ART_CFG
from omni.isaac.lab_assets import G1_MINIMAL_CFG as G1_MINIMAL_ART_CFG

from ogmp_isaac.assets.actuators import IdentifiedActuatorCfg
from ogmp_isaac.assets.robots.robot_cfg import RobotCfg

G1_DC_ART_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=1.0,
            angular_damping=1.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),  # if fix_root_link=False
        # pos=(0.0, 0.0, 1.74), # if fix_root_link=True
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0,
            "right_one_joint": -1.0,
            "left_two_joint": 0.52,
            "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdentifiedActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            # stiffness={
            #             ".*_hip_yaw_joint": 150.0,
            #             ".*_hip_roll_joint": 150.0,
            #             ".*_hip_pitch_joint": 200.0,
            #             ".*_knee_joint": 200.0,
            #             "torso_joint": 200.0,
            #             },
            stiffness=50.0,
            damping=1.0,
            armature=0.01,
            friction_static=0.3,
            activation_vel=0.1,
            friction_dynamic=0.02,
            saturation_effort=3000,
        ),
        "feet": IdentifiedActuatorCfg(
            effort_limit=20,
            velocity_limit=100.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=1.0,
            damping=0.1,
            armature=0.01,
            friction_static=0.3,
            activation_vel=0.1,
            friction_dynamic=0.02,
            saturation_effort=200,
        ),
        "arms": IdentifiedActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=0.1,
            damping=0.01,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
            friction_static=0.3,
            activation_vel=0.1,
            friction_dynamic=0.02,
            saturation_effort=3000,
        ),
    },
)

G1_DC_SPLIT_ART_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=1.0,
            angular_damping=1.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),  # if fix_root_link=False
        # pos=(0.0, 0.0, 1.74), # if fix_root_link=True
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0,
            "right_one_joint": -1.0,
            "left_two_joint": 0.52,
            "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdentifiedActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            # stiffness={
            #             ".*_hip_yaw_joint": 150.0,
            #             ".*_hip_roll_joint": 150.0,
            #             ".*_hip_pitch_joint": 200.0,
            #             ".*_knee_joint": 200.0,
            #             "torso_joint": 200.0,
            #             },
            stiffness=50.0,
            damping=1.0,
            armature=0.01,
            friction_static=0.3,
            activation_vel=0.1,
            friction_dynamic=0.02,
            saturation_effort=3000,
        ),
        "feet": IdentifiedActuatorCfg(
            effort_limit=20,
            velocity_limit=100.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=1.0,
            damping=0.1,
            armature=0.01,
            friction_static=0.3,
            activation_vel=0.1,
            friction_dynamic=0.02,
            saturation_effort=200,
        ),
        "arms": IdentifiedActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=20.0,
            damping=0.2,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
            },
            friction_static=0.3,
            activation_vel=0.1,
            friction_dynamic=0.02,
            saturation_effort=3000,
        ),
        "hands": IdentifiedActuatorCfg(
            joint_names_expr=[
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=0.1,
            damping=0.01,
            armature={
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
            friction_static=0.0,
            activation_vel=0.1,
            friction_dynamic=0.0,
            saturation_effort=3000,
        ),
    },
)

G1_MPCL = [[-3.14, 3.14]] * 37
G1_BAD_CONTACT_BODIES = [
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
]

G1_MPCL_DIFF = max([max(G1_MPCL[i]) - min(G1_MPCL[i]) for i in range(37)])

G1_CFG = RobotCfg(
    articulation_cfg=G1_ART_CFG,
    mpcl=G1_MPCL,
    bad_contact_bodies=G1_BAD_CONTACT_BODIES,
    feet_bodies=[],
    num_joints=37,
    max_joint_torque=1723.84,
    max_total_joint_vel=100.0,
    max_total_joint_acc=200.0,
    max_mpcl_diff=G1_MPCL_DIFF,
)

G1_DC_CFG = RobotCfg(
    articulation_cfg=G1_DC_ART_CFG,
    mpcl=G1_MPCL,
    bad_contact_bodies=G1_BAD_CONTACT_BODIES,
    feet_bodies=[],
    num_joints=37,
    max_joint_torque=1723.84,
    max_total_joint_vel=100.0,
    max_total_joint_acc=200.0,
    max_mpcl_diff=G1_MPCL_DIFF,
)

G1_MINIMAL_CFG = RobotCfg(
    articulation_cfg=G1_MINIMAL_ART_CFG,
    mpcl=G1_MPCL,
    bad_contact_bodies=G1_BAD_CONTACT_BODIES,
    feet_bodies=[],
    num_joints=37,
    max_joint_torque=1723.84,
    max_total_joint_vel=100.0,
    max_total_joint_acc=200.0,
    max_mpcl_diff=G1_MPCL_DIFF,
)

G1_SPLIT_CFG = RobotCfg(
    articulation_cfg=G1_DC_SPLIT_ART_CFG,
    mpcl=G1_MPCL,
    bad_contact_bodies=G1_BAD_CONTACT_BODIES,
    feet_bodies=[],
    num_joints=37,
    max_joint_torque=1723.84,
    max_total_joint_vel=100.0,
    max_total_joint_acc=200.0,
    max_mpcl_diff=G1_MPCL_DIFF,
)
