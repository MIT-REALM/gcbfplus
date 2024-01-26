import jax.tree_util as jtu
import numpy as np

from abc import ABC, abstractproperty, abstractmethod
from .data import Rollout
from .utils import jax2np, np2jax
from ..utils.utils import tree_merge
from ..utils.typing import Array


class Buffer(ABC):

    def __init__(self, size: int):
        self._size = size

    @abstractmethod
    def append(self, rollout: Rollout):
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Rollout:
        pass

    @abstractproperty
    def length(self) -> int:
        pass


class ReplayBuffer(Buffer):

    def __init__(self, size: int):
        super(ReplayBuffer, self).__init__(size)
        self._buffer = None

    def append(self, rollout: Rollout):
        if self._buffer is None:
            self._buffer = jax2np(rollout)
        else:
            self._buffer = tree_merge([self._buffer, jax2np(rollout)])
        if self._buffer.length > self._size:
            self._buffer = jtu.tree_map(lambda x: x[-self._size:], self._buffer)

    def sample(self, batch_size: int) -> Rollout:
        idx = np.random.randint(0, self._buffer.length, batch_size)
        return np2jax(self.get_data(idx))

    def get_data(self, idx: np.ndarray) -> Rollout:
        return jtu.tree_map(lambda x: x[idx], self._buffer)

    @property
    def length(self) -> int:
        if self._buffer is None:
            return 0
        return self._buffer.n_data


class MaskedReplayBuffer:

    def __init__(self, size: int):
        self._size = size
        # (b, T)
        self._buffer = None
        self._safe_mask = None
        self._unsafe_mask = None

    def append(self, rollout: Rollout, safe_mask: Array, unsafe_mask: Array):
        if self._buffer is None:
            self._buffer = jax2np(rollout)
            self._safe_mask = jax2np(safe_mask)
            self._unsafe_mask = jax2np(unsafe_mask)
            # self._mid_mask = jax2np(mid_mask)
        else:
            self._buffer = tree_merge([self._buffer, jax2np(rollout)])
            self._safe_mask = tree_merge([self._safe_mask, jax2np(safe_mask)])
            self._unsafe_mask = tree_merge([self._unsafe_mask, jax2np(unsafe_mask)])
        if self._buffer.length > self._size:
            self._buffer = jtu.tree_map(lambda x: x[-self._size:], self._buffer)
            self._safe_mask = jtu.tree_map(lambda x: x[-self._size:], self._safe_mask)
            self._unsafe_mask = jtu.tree_map(lambda x: x[-self._size:], self._unsafe_mask)

    def sample(self, batch_size: int) -> [Rollout, Array, Array]:
        idx = np.random.randint(0, self._buffer.length, batch_size)
        rollout, safe_mask, unsafe_mask = self.get_data(idx)
        return rollout, safe_mask, unsafe_mask

    def get_data(self, idx: np.ndarray) -> [Rollout, Array, Array]:
        return jtu.tree_map(lambda x: x[idx], self._buffer), self._safe_mask[idx], self._unsafe_mask[idx]

    @property
    def length(self) -> int:
        if self._buffer is None:
            return 0
        return self._buffer.n_data
