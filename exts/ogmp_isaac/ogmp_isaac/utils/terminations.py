import torch


def term_time_out(env):
    return env.episode_length_buf >= env.max_episode_length - 1


def term_toe_y(env):
    r_toe_pos_y = env.robot.data.body_pos_w[:, env.r_toe_id, 1]
    l_toe_pos_y = env.robot.data.body_pos_w[:, env.l_toe_id, 1]
    base_pos_y = env.robot.data.root_pos_w[:, 1]
    return (torch.abs(r_toe_pos_y - base_pos_y) > env.cfg.terminations["toe_y"]["threshold"]) | (
        torch.abs(l_toe_pos_y - base_pos_y) > env.cfg.terminations["toe_y"]["threshold"]
    )


def term_base_pos_x(env):
    target_pos_x = torch.gather(
        env.oracle.reference.base_pos[:, :, :1], 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 1)
    ).squeeze()
    error_x = torch.abs(env.robot.data.root_pos_w[:, 0] - target_pos_x)
    return error_x > env.cfg.terminations["base_pos_x"]["threshold"]


def term_base_pos_y(env):
    target_pos_y = torch.gather(
        env.oracle.reference.base_pos[:, :, 1:2], 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 1)
    ).squeeze()
    error_y = torch.abs(env.robot.data.root_pos_w[:, 1] - target_pos_y)
    return error_y > env.cfg.terminations["base_pos_y"]["threshold"]


def term_base_pos_z(env):
    target_pos_z = torch.gather(
        env.oracle.reference.base_pos[:, :, 2:3], 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 1)
    ).squeeze()
    error_z = torch.abs(env.robot.data.root_pos_w[:, 2] - target_pos_z)
    return error_z > env.cfg.terminations["base_pos_z"]["threshold"]


def term_ball_pos_x(env):
    target_pos_x = torch.gather(
        env.oracle.reference.ball_pos[:, :, 0:1], 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 1)
    ).squeeze()
    error_ball_pos_x = torch.abs(env.ball.data.root_pos_w[:, 0] - target_pos_x)
    return (error_ball_pos_x > env.cfg.terminations["ball_pos_x"]["threshold"]) & (env.oracle.modes >= 2)


def term_ball_pos_y(env):
    target_pos_y = torch.gather(
        env.oracle.reference.ball_pos[:, :, 1:2], 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 1)
    ).squeeze()
    error_ball_pos_y = torch.abs(env.ball.data.root_pos_w[:, 1] - target_pos_y)
    return (error_ball_pos_y > env.cfg.terminations["ball_pos_y"]["threshold"]) & (env.oracle.modes >= 2)


def term_box_pos_x(env):
    target_pos_x = torch.gather(
        env.oracle.reference.box_pos[:, :, 0:1], 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 1)
    ).squeeze()
    error_box_pos_x = torch.abs(env.box.data.root_pos_w[:, 0] - target_pos_x)
    return (error_box_pos_x > env.cfg.terminations["box_pos_x"]["threshold"]) & (env.oracle.modes >= 2)


def term_box_pos_y(env):
    target_pos_y = torch.gather(
        env.oracle.reference.box_pos[:, :, 1:2], 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 1)
    ).squeeze()
    error_box_pos_y = torch.abs(env.box.data.root_pos_w[:, 1] - target_pos_y)
    return (error_box_pos_y > env.cfg.terminations["box_pos_y"]["threshold"]) & (env.oracle.modes >= 2)
