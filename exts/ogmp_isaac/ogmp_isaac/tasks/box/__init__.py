import gymnasium as gym

import ogmplm.agents as agents

from .flat_box_env import FlatBoxEnv, FlatBoxEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Box-v0",
    entry_point="ogmplm.tasks.box:FlatBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlatBoxEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.PPORunnerCfg,
    },
)
