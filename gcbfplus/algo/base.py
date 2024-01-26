from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Tuple

from gcbfplus.utils.typing import Action, Params, PRNGKey, Array
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.trainer.data import Rollout
from gcbfplus.env.base import MultiAgentEnv


class MultiAgentController(ABC):

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            action_dim: int,
            n_agents: int
    ):
        self._env = env
        self._node_dim = node_dim
        self._edge_dim = edge_dim
        self._action_dim = action_dim
        self._n_agents = n_agents

    @property
    def node_dim(self) -> int:
        return self._node_dim

    @property
    def edge_dim(self) -> int:
        return self._edge_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @abstractproperty
    def config(self) -> dict:
        pass

    @abstractproperty
    def actor_params(self) -> Params:
        pass

    @abstractmethod
    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        pass

    @abstractmethod
    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        pass

    @abstractmethod
    def update(self, rollout: Rollout, step: int) -> dict:
        pass

    @abstractmethod
    def save(self, save_dir: str, step: int):
        pass

    @abstractmethod
    def load(self, load_dir: str, step: int):
        pass
