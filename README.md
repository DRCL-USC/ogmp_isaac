# Install the package

`cd exts/ogmplm && pip install -e . && cd -`

# Training

`python scripts/rsl_rl/train.py --headless --yaml_config \<path to exp_config.yaml\>`

# Evaluation

`python scripts/rsl_rl/play.py --yaml_config \<path to trained config (not exp_config)\> --num_envs <number of envs>`

Defaults to 4096 envs
