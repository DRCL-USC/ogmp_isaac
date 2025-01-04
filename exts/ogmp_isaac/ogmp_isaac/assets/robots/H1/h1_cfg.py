import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab_assets import H1_CFG as H1_ART_CFG
from omni.isaac.lab_assets import H1_MINIMAL_CFG as H1_MINIMAL_ART_CFG

from ogmp_isaac.assets.actuators import IdentifiedActuatorCfg
from ogmp_isaac.assets.robots.robot_cfg import RobotCfg

H1_DC_ART_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
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
            # fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 2.05), # if fix_root_link=True
        pos=(0.0, 0.0, 1.05),  # if fix_root_link=False
        joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": -0.28,  # -16 degrees
            ".*_knee": 0.79,  # 45 degrees
            ".*_ankle": -0.52,  # -30 degrees
            "torso": 0.0,
            ".*_shoulder_pitch": 0.28,
            ".*_shoulder_roll": 0.0,
            ".*_shoulder_yaw": 0.0,
            ".*_elbow": 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdentifiedActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
            effort_limit=300,
            velocity_limit=100.0,
            # stiffness={
            #     ".*_hip_yaw": 150.0,
            #     ".*_hip_roll": 150.0,
            #     ".*_hip_pitch": 200.0,
            #     ".*_knee": 200.0,
            #     "torso": 200.0,
            # },
            # damping={
            #     ".*_hip_yaw": 5.0,
            #     ".*_hip_roll": 5.0,
            #     ".*_hip_pitch": 5.0,
            #     ".*_knee": 5.0,
            #     "torso": 5.0,
            # },
            stiffness=80.0,
            damping=1.0,
            armature=0.01,
            friction_static=0.3,
            activation_vel=0.1,
            friction_dynamic=0.02,
            saturation_effort=3000,
        ),
        "feet": IdentifiedActuatorCfg(
            joint_names_expr=[".*_ankle"],
            effort_limit=100,
            velocity_limit=100.0,
            # stiffness={".*_ankle": 20.0},
            # damping={".*_ankle": 4.0},
            stiffness=5.0,
            damping=0.1,
            armature=0.01,
            friction_static=0.3,
            activation_vel=0.1,
            friction_dynamic=0.02,
            saturation_effort=200,
        ),
        "arms": IdentifiedActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            effort_limit=300,
            velocity_limit=100.0,
            # stiffness={
            #     ".*_shoulder_pitch": 40.0,
            #     ".*_shoulder_roll": 40.0,
            #     ".*_shoulder_yaw": 40.0,
            #     ".*_elbow": 40.0,
            # },
            # damping={
            #     ".*_shoulder_pitch": 10.0,
            #     ".*_shoulder_roll": 10.0,
            #     ".*_shoulder_yaw": 10.0,
            #     ".*_elbow": 10.0,
            # },
            stiffness=5.0,
            damping=0.1,
            friction_static=0.3,
            activation_vel=0.1,
            friction_dynamic=0.02,
            saturation_effort=3000,
        ),
    },
)

H1_MPCL = [[-3.14, 3.14]] * 19
H1_BAD_CONTACT_BODIES = [
    "torso_link",
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
H1_FEET_BODIES = ["left_ankle_link", "right_ankle_link"]

H1_MPCL_DIFF = max([max(H1_MPCL[i]) - min(H1_MPCL[i]) for i in range(19)])

H1_CFG = RobotCfg(
    articulation_cfg=H1_ART_CFG,
    mpcl=H1_MPCL,
    bad_contact_bodies=H1_BAD_CONTACT_BODIES,
    feet_bodies=H1_FEET_BODIES,
    num_joints=19,
    max_joint_torque=1244.99,
    max_total_joint_vel=100.0,
    max_total_joint_acc=200.0,
    max_mpcl_diff=H1_MPCL_DIFF,
)

H1_MINIMAL_CFG = RobotCfg(
    articulation_cfg=H1_MINIMAL_ART_CFG,
    mpcl=H1_MPCL,
    bad_contact_bodies=H1_BAD_CONTACT_BODIES,
    feet_bodies=H1_FEET_BODIES,
    num_joints=19,
    max_joint_torque=1244.99,
    max_total_joint_vel=100.0,
    max_total_joint_acc=200.0,
    max_mpcl_diff=H1_MPCL_DIFF,
)

H1_DC_CFG = RobotCfg(
    articulation_cfg=H1_DC_ART_CFG,
    mpcl=H1_MPCL,
    bad_contact_bodies=H1_BAD_CONTACT_BODIES,
    feet_bodies=H1_FEET_BODIES,
    num_joints=19,
    max_joint_torque=1244.99,
    max_total_joint_vel=100.0,
    max_total_joint_acc=200.0,
    max_mpcl_diff=H1_MPCL_DIFF,
)
