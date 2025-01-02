import torch


@torch.jit.script
def euler_to_quaternion_horizon(euler_angles: torch.Tensor) -> torch.Tensor:
    roll = euler_angles[:, :, 0]
    pitch = euler_angles[:, :, 1]
    yaw = euler_angles[:, :, 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quaternions = torch.stack((w, x, y, z), dim=-1)
    return quaternions


@torch.jit.script
def quaternion_to_euler(quaternions: torch.Tensor) -> torch.Tensor:
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = torch.asin(2 * (w * y - z * x))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    euler_angles = torch.stack((roll, pitch, yaw), dim=-1)
    return euler_angles


@torch.jit.script
def interpolate(start: torch.Tensor, stop: torch.Tensor, num: int):
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    steps = steps.view((1, num, 1))

    # Expand start and stop to match the shape of steps
    start_expanded = start[:, None, :]
    stop_expanded = stop[:, None, :]

    out = start_expanded + steps * (stop_expanded - start_expanded)

    return out
