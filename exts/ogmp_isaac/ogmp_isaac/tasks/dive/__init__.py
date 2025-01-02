import gymnasium as gym

import ogmplm.agents as agents

from .dive_env import DiveEnv, DiveEnvCfg
from .dive_env2 import DiveEnv2, DiveEnv2Cfg

##
# Register Gym environments.
##

gym.register(
    id="Dive-v0",
    entry_point="ogmplm.tasks.dive:DiveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DiveEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.PPORunnerCfg,
    },
)

gym.register(
    id="Dive-v1",
    entry_point="ogmplm.tasks.dive:DiveEnv2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DiveEnv2Cfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.PPORunnerCfg,
    },
)
