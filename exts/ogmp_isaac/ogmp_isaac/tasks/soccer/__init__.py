import gymnasium as gym

import ogmp_isaac.agents as agents

from .soccer_env import SoccerEnv, SoccerEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Soccer-v0",
    entry_point="ogmp_isaac.tasks.soccer:SoccerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SoccerEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.PPORunnerCfg,
    },
)
