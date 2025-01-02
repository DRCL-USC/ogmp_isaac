from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    rnn_type: str = MISSING
    """The type of RNN to use."""

    rnn_hidden_size: int = MISSING
    """The hidden size of the RNN."""

    rnn_num_layers: int = MISSING
    """The number of layers of the RNN."""
