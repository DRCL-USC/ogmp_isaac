# Install the package

`python -m pip install -e exts/ogmp_isaac`

# Training

`python scripts/rsl_rl/train.py --headless --yaml_config \<path to exp_config.yaml\>`

# Evaluation

`python scripts/rsl_rl/play.py --yaml_config \<path to trained config (not exp_config)\> --num_envs <number of envs>`

Defaults to 4096 envs
