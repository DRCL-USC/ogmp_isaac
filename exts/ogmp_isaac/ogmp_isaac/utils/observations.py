import torch


def obs_base_z(env):
    return env.robot.data.root_pos_w[:, 2:3]


def obs_base_ori(env):
    return env.robot.data.root_quat_w


def obs_joint_pos(env):
    return env.robot.data.joint_pos


def obs_joint_pos_hector_coupling(env):
    joint_pos = env.robot.data.joint_pos.clone()

    # offset knee joint from ankle joint
    joint_pos[:, 8] += joint_pos[:, 6]
    joint_pos[:, 9] += joint_pos[:, 7]

    # scale knee to motor sapce
    joint_pos[:, 6] = joint_pos[:, 6] * env.hector_knee_ratio
    joint_pos[:, 7] = joint_pos[:, 7] * env.hector_knee_ratio

    return joint_pos


def obs_projected_gravity_body(env):
    return env.robot.data.projected_gravity_b


def obs_base_lin_vel(env):
    return env.robot.data.root_lin_vel_w


def obs_base_lin_vel_body(env):
    return env.robot.data.root_lin_vel_b


def obs_base_ang_vel(env):
    return env.robot.data.root_ang_vel_w


def obs_base_ang_vel_body(env):
    return env.robot.data.root_ang_vel_b


def obs_joint_vel(env):
    return env.robot.data.joint_vel


def obs_joint_vel_hector_coupling(env):
    joint_vel = env.robot.data.joint_vel.clone()

    # offset knee joint from ankle joint
    joint_vel[:, 8] += joint_vel[:, 6]
    joint_vel[:, 9] += joint_vel[:, 7]

    # scale knee to motor sapce
    joint_vel[:, 6] = joint_vel[:, 6] * env.hector_knee_ratio
    joint_vel[:, 7] = joint_vel[:, 7] * env.hector_knee_ratio

    return joint_vel


def obs_joint_torque(env):
    return env.robot.data.applied_torque


def obs_joint_torque_hector_coupling(env):
    joint_torque = env.robot.data.applied_torque.clone()

    # scale knee to motor sapce
    joint_torque[:, 6] = joint_torque[:, 6] / env.hector_knee_ratio
    joint_torque[:, 7] = joint_torque[:, 7] / env.hector_knee_ratio

    return joint_torque


def obs_sinusoid_phase(env):
    return torch.cat(
        [
            torch.sin(2 * torch.pi * env.oracle.phase.float() / env.oracle.prediction_horizon),
            torch.cos(2 * torch.pi * env.oracle.phase.float() / env.oracle.prediction_horizon),
        ],
        dim=-1,
    )


def obs_ball_dist(env):
    return (env.ball.data.root_pos_w - env.robot.data.root_pos_w)[:, :2]


def obs_box_dist(env):
    return (env.box.data.root_pos_w - env.robot.data.root_pos_w)[:, :2]


def obs_target_dist(env):
    return env.target_pos - env.robot.data.root_pos_w[:, :2]
