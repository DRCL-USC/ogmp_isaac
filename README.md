# Overview

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.2.0-silver)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)

This repository contains the code for the experiments in the paper - [Dynamic Bipedal Loco-manipulation using Oracle Guided Multi-mode Policies with Mode-transition Preference](https://arxiv.org/abs/2410.01030). Check out the project [website](https://indweller.github.io/ogmplm/) for more details.

Authors: Prashanth Ravichandar, Lokesh Krishna, Nikhil Sobanbabu and Quan Nguyen

# Usage

## Installation
1. Install [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html). Activate the conda environment containing the Isaac Lab installation.
2. Clone the repository. Install the project using 
      ```
      python -m pip install -e exts/ogmp_isaac
      ```

## Generate Configurations

Generate the training configurations using the command below. They are saved by default in the `logs/<environment-name>/<experiment-name>/<variant-name>` directory. For example:
```
python scripts/experiment/generate.py --base_path ./exts/ogmp_isaac/config/soccer_base.yaml --vary_path ./exts/ogmp_isaac/config/soccer_vary.yaml
```

NOTE: The USD for Berkeley Humanoid is not included in this repository. Please download the USD from [their repository](https://github.com/HybridRobotics/isaac_berkeley_humanoid) repository and place it in the `exts/ogmp_isaac/assets/robots/berkeley_humanoid/biped` directory.

## Training

1. Deploy all variants using the command below. This will run each variant in the experiment folder. Default is headless mode with 4096 environments.
      ```
      python scripts/experiment/deploy.py --exp_logpath ./logs/soccer/release_experiments
      ```
2. To train a single variant:
      ```
      python scripts/rsl_rl/train.py --yaml_config ./logs/soccer/release_experiments/H1_DC/exp_conf.yaml --headless
      ```

## Evaluation

To evaluate the trained model:
```
python scripts/rsl_rl/play.py --yaml_config ./logs/soccer/release_experiments/H1_DC/exp_conf.yaml --num_envs 1 --visualize --visualize_goalpost
```

# Citation

If you use this code, please cite the following paper:
```
@misc{ravichandar2024dynamicbipedallocomanipulationusing,
      title={Dynamic Bipedal Loco-manipulation using Oracle Guided Multi-mode Policies with Mode-transition Preference}, 
      author={Prashanth Ravichandar and Lokesh Krishna and Nikhil Sobanbabu and Quan Nguyen},
      year={2024},
      eprint={2410.01030},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.01030}, 
}
```
For the theory on oracle guided policy optimization, refer to the paper [OGMP: Oracle Guided Multi-mode Policies for Agile and Versatile Robot Control](https://arxiv.org/abs/2403.04205).

# License and Disclaimer

The template used in this project is based on [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate) developed by the Isaac Lab Project Developers, which is licensed under the MIT License.

All other content in this repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.