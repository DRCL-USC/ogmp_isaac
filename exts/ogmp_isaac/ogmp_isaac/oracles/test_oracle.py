import torch

from ogmplm.oracles.reach_oracle import ReachOracle

if __name__ == "__main__":
    oracle = ReachOracle(
        rollout_time=3.0,
        env_dt=0.033,
        speed=0.5,
        num_envs=3,
        nominal_height=0.5,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feedback = {
        "env_ids": torch.tensor([0, 1, 2], device=device),
        "robot_pos": torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.5]], device=device),
        "robot_ori": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            device=device,
        ),
    }
    oracle.generate_reference(feedback)
    print(oracle.reference.base_pos[0, 0, :])
    print(oracle.reference.base_ori[0, 0, :])
    print(oracle.reference.base_lin_vel[0, 0, :])
    print(oracle.reference.base_ang_vel[0, 0, :])
    print(oracle.reference.base_pos[0, -1, :])
    print(oracle.reference.base_ori[0, -1, :])
    print(oracle.reference.base_lin_vel[0, -1, :])
    print(oracle.reference.base_ang_vel[0, -1, :])
