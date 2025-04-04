# NOTE: set fix_root_link=True in the articulation config to fix the root link of the robot to the world frame
"""
This script demonstrates tests model, motor and joint position limits

"""


import argparse
import torch

torch.set_printoptions(precision=2, sci_mode=False)

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a bipedal robot.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

##
# Pre-defined configs
##
# from ogmp_isaac.assets import HECTOR_V1P5_CFG , HECTOR_V1P5_MPCL, GOALPOSTS_CFG
# from ogmp_isaac.assets import BERKELEY_HUMANOID_CFG, BERKELEY_HUMANOID_MPCL
# from ogmp_isaac.assets import HECTOR_V1P5_IPD_CFG, HECTOR_V1P5_MPCL
# from ogmp_isaac.assets import HECTOR_V1_DC_CFG, HECTOR_V1_MPCL
from ogmp_isaac.assets import HECTOR_V1P5_DC_CFG, HECTOR_V1P5_DC_MPCL


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            device="cuda",
            # use_gpu_pipeline=True,
            dt=1 / 120,
            render_interval=4,
            # dt=1/1000,
            # render_interval=10,
            # physx=sim_utils.PhysxCfg(
            #                         use_gpu=True, solver_type=0,
            #                         min_position_iteration_count=255, min_velocity_iteration_count=255,
            #                         max_position_iteration_count=255, max_velocity_iteration_count=255
            #                     )
        )
    )
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    n_hectors = 5
    # Ground-plane
    # cfg = sim_utils.GroundPlaneCfg()
    # cfg.func("/World/defaultGroundPlane", cfg)
    import omni.isaac.lab.terrains as terrain_gen

    terrain_generator_cfg = TerrainGeneratorCfg(
        size=(2.5, 2.5),
        border_width=20.0,
        num_rows=64,
        num_cols=64,
        horizontal_scale=0.1,
        vertical_scale=0.00,
        slope_threshold=0.0,
        use_cache=False,
        sub_terrains={
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.75, 0.75),
                step_width=0.5,
                platform_width=0.5,
                border_width=1.0,
                holes=False,
            ),
        },
    )
    cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    cfg.num_envs = 5
    cfg.env_spacing = 5.0
    cfg.class_type(cfg)
    # origins = torch.tensor([ [ i - 0.5*n_hectors,0.0, 0.0] for i in range(n_hectors) ], device=sim.device)
    # arrange the robots in a grid
    # origins = torch.tensor([ [ i - 0.25*n_hectors, i - 0.25*n_hectors, 0.0] for i in range(n_hectors) ], device=sim.device)
    # origins = torch.tensor([ [i-0.5*n_hectors,j-0.5*n_hectors,0.0] for i in range(int(n_hectors/2)) for j in range(int(n_hectors/2))], device=sim.device)

    # split n_hectors into a rectangle closest to a square
    # n_cols = int(n_hectors**0.5)
    # n_rows = int(n_hectors/n_cols)

    # origins = torch.tensor([ [i-n_cols/2, j-n_rows/2, 1.0] for i in range(n_cols) for j in range(n_rows)], device=sim.device)
    origins = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.25, 1.25, 1.5], [3.0, 0.0, 1.0], [4.0, 0.0, 1.0]], device=sim.device
    )

    # load a goalpost
    # goalposts = Articulation(GOALPOSTS_CFG.replace(prim_path="/World/Goalposts"))

    # load n_hectors and append to robots
    # ROBOT_MPCL = BERKELEY_HUMANOID_MPCL
    # ROBOT_MPCL = HECTOR_V1_MPCL
    # ROBOT_MPCL = HECTOR_V1P5_MPCL
    ROBOT_MPCL = HECTOR_V1P5_DC_MPCL
    robots = []
    for i in range(n_hectors):
        # robot = Articulation(HECTOR_V1P5_CFG.replace(prim_path=f"/World/Hector_v1p5_{i}"))
        # robot = Articulation(BERKELEY_HUMANOID_CFG.replace(prim_path=f"/World/Berkeley_Humanoid_{i}"))
        # robot = Articulation(HECTOR_V1P5_IPD_CFG.replace(prim_path=f"/World/Hector_v1p5_ipd_{i}"))
        robot = Articulation(HECTOR_V1P5_DC_CFG.replace(prim_path=f"/World/Hector_v1p5_dc_{i}"))
        robots.append(robot)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # print robot information of the first instance
    robot = robots[0]
    print(f"Robot 0")

    # print bodies and masses
    print("robot bodies")
    for i, (body, mass) in enumerate(zip(robot.body_names, robot.data.default_mass.flatten())):
        print(f"\tbody {i}: {body} with mass: {mass}")

    # print joints
    joint_ids, joint_names = robot.find_joints(".*")

    print("robot joints")
    for i, (
        joint_id,
        joint_name,
        joint_friction,
        joint_limits,
    ) in enumerate(zip(joint_ids, robot.joint_names, robot.data.joint_friction[0], robot.data.default_joint_limits[0])):
        print(
            "\tjnt",
            joint_id,
            joint_name,
            torch.round(joint_friction, decimals=2).tolist(),
            torch.round(joint_limits, decimals=2).tolist(),
        )

    for actuator in robot.actuators:
        print("actuator group: ", actuator)
        actuator = robot.actuators[actuator]
        i = 0
        for ajn, akp, akd in zip(
            actuator.joint_names,
            actuator.stiffness.flatten(),
            actuator.damping.flatten(),
        ):
            print(
                "\tact",
                i,
                ajn,
                torch.round(akp, decimals=2).tolist(),
                torch.round(akd, decimals=2).tolist(),
                ROBOT_MPCL[i],
            )
            i += 1

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    mode = -1

    count_per_dof = 500

    joint_positions = []
    joint_velocities = []
    motor_commands = []
    joint_pos_target = None
    ROBOT_MPCL = torch.tensor(ROBOT_MPCL, device=sim.device)
    print(ROBOT_MPCL.shape)
    while simulation_app.is_running():
        # reset
        if count % count_per_dof == 0:
            sim_time = 0.0
            count = 0
            for index, robot in enumerate(robots):
                # reset dof state
                joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
                # print robot mass
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_state_to_sim(root_state)
                robot.reset()
                robot.write_data_to_sim()
            # reset command
            print(">>>>>>>> Reset!")
            mode += 1
            if mode == int(0.5 * len(ROBOT_MPCL)):
                break
            print(f"Mode: {mode}", "testing joint", robot.joint_names[2 * mode], robot.joint_names[2 * mode + 1])

        # apply action to the robot
        for index, robot in enumerate(robots):
            joint_pos_target = robot.data.default_joint_pos.clone()

            # sine wave, with time period as 0.5 count_per_dof
            # joint_pos_target[:,2*mode  ] = 6.28 * torch.sin(2.0*torch.pi*torch.tensor(sim_time)/(sim_dt*count_per_dof))
            # joint_pos_target[:,2*mode+1] = 6.28 * torch.sin(2.0*torch.pi*torch.tensor(sim_time)/(sim_dt*count_per_dof))

            # interpolate bw low and high limits with a sine wave, with time period as 0.5 count_per_dof

            l_p_u_b2 = (ROBOT_MPCL[2 * mode, 1] + ROBOT_MPCL[2 * mode, 0]) / 2.0
            l_m_u_b2 = (ROBOT_MPCL[2 * mode, 1] - ROBOT_MPCL[2 * mode, 0]) / 2.0
            joint_pos_target[:, 2 * mode] = l_p_u_b2 + l_m_u_b2 * torch.sin(
                2.0 * torch.pi * torch.tensor(sim_time) / (sim_dt * count_per_dof)
            )

            l_p_u_b2 = (ROBOT_MPCL[2 * mode + 1, 1] + ROBOT_MPCL[2 * mode + 1, 0]) / 2.0
            l_m_u_b2 = (ROBOT_MPCL[2 * mode + 1, 1] - ROBOT_MPCL[2 * mode + 1, 0]) / 2.0
            joint_pos_target[:, 2 * mode + 1] = l_p_u_b2 + l_m_u_b2 * torch.sin(
                2.0 * torch.pi * torch.tensor(sim_time) / (sim_dt * count_per_dof)
            )

            # apply random caommands sampled from the joint limits for all motors
            # joint_pos_target = torch.rand_like(joint_pos_target) * (ROBOT_MPCL[:,1] - ROBOT_MPCL[:,0]) + ROBOT_MPCL[:,0]
            robot.set_joint_position_target(joint_pos_target)
            robot.write_data_to_sim()

        motor_commands.append(joint_pos_target.clone())
        joint_positions.append(robot.data.joint_pos.clone())
        joint_velocities.append(robot.data.joint_vel.clone())
        # step the simulation
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1

        # update buffers
        for index, robot in enumerate(robots):
            robot.update(sim_dt)

    # make buffers into tensors
    joint_positions = torch.stack(joint_positions)
    joint_velocities = torch.stack(joint_velocities)
    motor_commands = torch.stack(motor_commands)

    # reshape the tensors
    joint_positions = joint_positions.permute(0, 2, 1).squeeze()
    joint_velocities = joint_velocities.permute(0, 2, 1).squeeze()
    motor_commands = motor_commands.permute(0, 2, 1).squeeze()

    # move it to cpu and make it numpy
    joint_positions = joint_positions.cpu().numpy()
    joint_velocities = joint_velocities.cpu().numpy()
    motor_commands = motor_commands.cpu().numpy()

    print(joint_velocities.shape, joint_positions.shape, motor_commands.shape)

    # make a plot 2x5

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(
        4,
        # 5,
        int(joint_positions.shape[1] / 2),
        figsize=(40, 10),
    )
    fig.suptitle("jpos, jvel and mcom")

    joint_limits = robot.data.default_joint_limits[0].cpu().numpy()

    for i in range(int(joint_positions.shape[1] / 2)):
        axs[0, i].plot(motor_commands[:, 2 * i], label="motor command")
        axs[0, i].plot(joint_positions[:, 2 * i], label="joint position")
        axs[1, i].plot(motor_commands[:, 2 * i + 1], label="motor command")
        axs[1, i].plot(joint_positions[:, 2 * i + 1], label="joint position")

        axs[0, i].axhline(y=joint_limits[2 * i, 0], color="b", linestyle="--")
        axs[0, i].axhline(y=joint_limits[2 * i, 1], color="b", linestyle="--")
        axs[1, i].axhline(y=joint_limits[2 * i + 1, 0], color="b", linestyle="--")
        axs[1, i].axhline(y=joint_limits[2 * i + 1, 1], color="b", linestyle="--")

        axs[2, i].plot(joint_velocities[:, 2 * i], label="joint velocity")
        axs[3, i].plot(joint_velocities[:, 2 * i + 1], label="joint velocity")

    for ax in axs.flat:
        ax.set(xlabel="time", ylabel="value")
        ax.grid()
        ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
