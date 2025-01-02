from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from .recurrent_ac_cfg import RslRlPpoActorCriticRecurrentCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2000  # 10000
    save_interval = 100
    experiment_name = "exp_name"
    run_name = "run_name"
    empirical_normalization = True
    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     class_name="ActorCriticRecurrent",
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[400, 200, 100],
    #     critic_hidden_dims=[400, 200, 100],
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_size=256,
    #     rnn_num_layers=1,
    # )
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=1.0,
        actor_hidden_dims=[200, 100],
        critic_hidden_dims=[200, 100],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=128,
        rnn_num_layers=1,
    )
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[400, 200, 100],
    #     critic_hidden_dims=[400, 200, 100],
    #     activation="elu",
    # )
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[128, 128],
    #     critic_hidden_dims=[128, 128],
    #     activation="elu",
    # )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
