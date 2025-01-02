import argparse
import os
import subprocess
import time
from datetime import datetime

# Define the arguments for the script
parser = argparse.ArgumentParser(description="Run multiple experiments in parallel")
parser.add_argument(
    "--exp_logpath",
    type=str,
    default="/home/lkrajan/drcl_projects/ogmp_v2/ogmp_isaac/logs/dive/2024Oct12_17-14-38_sanity_rnn_all_dir",
    help="path to the experiment log directory",
)
# parser.add_argument("--num_parallel", type=int, default=1, help="Number of parallel experiments to run (1 or 2)")
parser.add_argument("--two_in_parallel", action="store_true", default=False, help="run two experiments in parallel")
args = parser.parse_args()


# read the sub folders in the experiment log directory
exp_folders = os.listdir(args.exp_logpath)

exp_variants = [exp_folder for exp_folder in exp_folders if os.path.isdir(args.exp_logpath + "/" + exp_folder)]


# Base command and arguments
# base_command = ["python3", "scripts/rsl_rl/train.py", "--headless", "--num_envs", "3000"]
base_command = ["python3", "scripts/rsl_rl/train.py", "--headless"]


print("running experiment:", args.exp_logpath)

# Run the experiments one by one or two in parallel
i = 0
while i < len(exp_variants):
    processes = []
    print("\tvariant", i, ":", exp_variants[i], "| started at:", datetime.now().strftime("%Y%b%d_%H-%M-%S"))
    # log_file_1 = f"outputs/output_{yaml_files[i].replace('.yaml', '')}.log"
    log_file_1 = args.exp_logpath + "/" + exp_variants[i] + "/output.log"
    with open(log_file_1, "w") as log_1:
        # Run the first instance
        yaml_path_1 = os.path.abspath(os.path.join(args.exp_logpath, exp_variants[i], "exp_conf.yaml"))
        command_1 = base_command + ["--yaml_config", yaml_path_1]
        processes.append(subprocess.Popen(command_1, stdout=log_1, stderr=subprocess.STDOUT))

    time.sleep(60)

    # Check if there's a second file available and run the second instance
    if args.two_in_parallel and i + 1 < len(exp_variants):
        i += 1
        print("\tvariant", i, ":", exp_variants[i], "| started at:", datetime.now().strftime("%Y%b%d_%H-%M-%S"))
        # log_file_2 = f"outputs/output_{yaml_files[i].replace('.yaml', '')}.log"
        log_file_2 = args.exp_logpath + "/" + exp_variants[i] + "/output.log"
        with open(log_file_2, "w") as log_2:
            # yaml_path_2 = os.path.abspath(os.path.join(config_path, yaml_files[i]))
            yaml_path_2 = os.path.abspath(os.path.join(args.exp_logpath, exp_variants[i], "exp_conf.yaml"))
            command_2 = base_command + ["--yaml", yaml_path_2]
            processes.append(subprocess.Popen(command_2, stdout=log_2, stderr=subprocess.STDOUT))

    i += 1

    # Wait for both processes to finish before continuing
    for process in processes:
        process.wait()
